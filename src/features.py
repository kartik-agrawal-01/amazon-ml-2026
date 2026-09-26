"""Pair features for the matching model.

Stage A (vectorised, all candidates):  TF-IDF cosines per view + group/rank context
Stage B (python loops, pruned pairs):   string similarities on core names, number/zip
                                         agreement on addresses, legal-form agreement

Everything is deterministic and library-light (numpy/pandas/scipy only) so the same code
gives identical features on the box, the laptop and Claude's sandbox.
"""
from __future__ import annotations

from typing import Dict, List

import os

import numpy as np
import pandas as pd
import scipy.sparse as sp



# ----------------------------------------------------------------- string sims
def jaro_winkler(s1: str, s2: str, p: float = 0.1) -> float:
    if s1 == s2:
        return 1.0
    l1, l2 = len(s1), len(s2)
    if not l1 or not l2:
        return 0.0
    dist = max(max(l1, l2) // 2 - 1, 0)
    m1, m2 = [False] * l1, [False] * l2
    m = 0
    for i, ch in enumerate(s1):
        lo, hi = max(0, i - dist), min(i + dist + 1, l2)
        for j in range(lo, hi):
            if not m2[j] and s2[j] == ch:
                m1[i] = m2[j] = True
                m += 1
                break
    if not m:
        return 0.0
    t, j = 0, 0
    for i in range(l1):
        if m1[i]:
            while not m2[j]:
                j += 1
            if s1[i] != s2[j]:
                t += 1
            j += 1
    t //= 2
    jaro = (m / l1 + m / l2 + (m - t) / m) / 3.0
    pre = 0
    for a, b in zip(s1[:4], s2[:4]):
        if a != b:
            break
        pre += 1
    return jaro + pre * p * (1.0 - jaro)


def _initials(s: str) -> str:
    return "".join(t[0] for t in s.split() if t)


# ------------------------------------------------------------------ stage A
def pair_cosine(Xq: sp.csr_matrix, Xd: sp.csr_matrix, qi: np.ndarray, ci: np.ndarray,
                chunk: int = 500_000) -> np.ndarray:
    out = np.empty(len(qi), dtype=np.float32)
    for st in range(0, len(qi), chunk):
        a, b = Xq[qi[st:st + chunk]], Xd[ci[st:st + chunk]]
        out[st:st + chunk] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    return out


def stage_a(cands: pd.DataFrame, d_src: np.ndarray, q_mats: Dict[str, sp.csr_matrix],
            d_mats: Dict[str, sp.csr_matrix], views: List[str], q_emb=None, d_emb=None) -> pd.DataFrame:
    """Cosines for every view + group/rank context. q/c are row indices into the query block / doc table."""
    qi, ci = cands["q"].values, cands["c"].values
    for v in views:
        cands[f"cos_{v}"] = pair_cosine(q_mats[v], d_mats[v], qi, ci)
    if q_emb is not None and d_emb is not None:
        from .dense import pair_cosine_dense
        cands["cos_dense"] = pair_cosine_dense(q_emb, d_emb, qi, ci)
    cands["c_src"] = d_src[ci].astype(np.int8)
    # a robust "overall" score for ranking/pruning: best cosine over ALL views (dense included)
    cols = [f"cos_{v}" for v in views] + (["cos_dense"] if "cos_dense" in cands else [])
    score = cands[cols].max(axis=1).astype(np.float32)
    cands["score"] = score
    g = cands.groupby(["q", "c_src"], sort=False)
    keys = [cands["q"], cands["c_src"]]
    for col, short in [("score", "score")] + [(f"cos_{v}", v) for v in views]:
        cands[f"r_{short}"] = g[col].rank(ascending=False, method="min").astype(np.float32)
        cands[f"gap_{short}"] = (g[col].transform("max") - cands[col]).astype(np.float32)
    mx = g["score"].transform("max")
    second = score.where(cands["r_score"] >= 2).groupby(keys).transform("max").fillna(0.0)
    n_top = (cands["r_score"] == 1).groupby(keys).transform("sum")
    cands["grp_margin"] = np.where(n_top > 1, 0.0, mx - second).astype(np.float32)
    cands["grp_n"] = g["c"].transform("size").astype(np.int16)
    cands["grp_n_close"] = (score >= mx - 0.05).groupby(keys).transform("sum").astype(np.int16)
    # reverse view (within the block): how does this S1 rank among the S1s competing for the same doc?
    gc = cands.groupby("c", sort=False)
    cands["rq_score"] = gc["score"].rank(ascending=False, method="min").astype(np.float32)
    cands["rq_gap"] = (gc["score"].transform("max") - score).astype(np.float32)
    cands["c_nq"] = gc["q"].transform("size").astype(np.int16)
    cands["mutual_best"] = ((cands["r_score"] == 1) & (cands["rq_score"] == 1)).astype(np.int8)
    return cands


def prune_score(cands: pd.DataFrame) -> np.ndarray:
    """Cheap relevance score used to prune candidates before stage B."""
    return cands["score"].values


def calibrate_prune(score: np.ndarray, y: np.ndarray, keep_recall: float = 0.998) -> float:
    """Largest threshold that keeps `keep_recall` of the positive pairs (fit on train)."""
    pos = np.sort(score[y == 1])
    if len(pos) == 0:
        return 0.0
    return float(pos[int(np.floor((1 - keep_recall) * len(pos)))])


# ------------------------------------------------------------------ stage B
_Q: dict = {}  # query-block column arrays shared with forked workers
_D: dict = {}  # doc-table column arrays
_COLS = ("n_core", "n_name", "legal", "a_nums", "a_zip", "vague", "n_addr", "city", "zip", "state", "country",
         "n_ph", "n_nospace", "indic", "cnt_core_s1", "cnt_core_all", "cnt_ph_s1", "cnt_ph_all", "cnt_nsp_s1",
         "cnt_nsp_all", "cnt_addr_all", "cnt_addr_s1")


def _set_rec(q_rec: pd.DataFrame, d_rec: pd.DataFrame) -> None:
    _Q.clear()
    _D.clear()
    for c in _COLS:
        if c in q_rec:
            _Q[c] = q_rec[c].values
        if c in d_rec:
            _D[c] = d_rec[c].values


_GENERIC_MODE = os.environ.get("AML_GENERIC", "")  # "" = HEAD set, "v2" = v2 set, "v2fr" = v2 + French
_GENERIC_V2 = _GENERIC_MODE in ("v2", "v2fr")

def _stage_b_block(qi: np.ndarray, ci: np.ndarray) -> dict:
    n = len(qi)
    f = {}
    cq, cc = _Q["n_core"][qi], _D["n_core"][ci]
    f["jw_core"] = np.fromiter((jaro_winkler(a, b) for a, b in zip(cq, cc)), np.float32, n)
    f["jw_name"] = np.fromiter((jaro_winkler(a, b) for a, b in zip(_Q["n_name"][qi], _D["n_name"][ci])), np.float32, n)
    tq = [set(a.split()) for a in cq]
    tc = [set(b.split()) for b in cc]
    inter = np.fromiter((len(a & b) for a, b in zip(tq, tc)), np.float32, n)
    lq = np.fromiter((len(a) for a in tq), np.float32, n)
    lc = np.fromiter((len(b) for b in tc), np.float32, n)
    f["tok_jacc"] = inter / np.maximum(lq + lc - inter, 1)
    f["tok_cont"] = inter / np.maximum(np.minimum(lq, lc), 1)
    f["tok_nq"], f["tok_nc"] = lq, lc
    f["core_eq"] = (cq == cc).astype(np.int8)
    f["core_in"] = np.fromiter(((a in b or b in a) if a and b else False for a, b in zip(cq, cc)), np.int8, n)
    f["first_eq"] = np.fromiter(((a.split()[:1] == b.split()[:1]) if a and b else False for a, b in zip(cq, cc)),
                                np.int8, n)
    f["acronym"] = np.fromiter((((" " not in b and len(b) > 1 and _initials(a) == b) or
                                 (" " not in a and len(a) > 1 and _initials(b) == a)) if a and b else False
                                for a, b in zip(cq, cc)), np.int8, n)
    f["len_ratio"] = np.fromiter((min(len(a), len(b)) / max(len(a), len(b), 1) for a, b in zip(cq, cc)), np.float32, n)
    pq, pc = _Q["n_ph"][qi], _D["n_ph"][ci]
    f["ph_eq"] = (pq == pc).astype(np.int8)
    f["jw_ph"] = np.fromiter((jaro_winkler(a, b) for a, b in zip(pq, pc)), np.float32, n)
    tpq = [set(a.split()) for a in pq]
    tpc = [set(b.split()) for b in pc]
    pin = np.fromiter((len(a & b) for a, b in zip(tpq, tpc)), np.float32, n)
    f["ph_jacc"] = pin / np.maximum(np.fromiter((len(a | b) for a, b in zip(tpq, tpc)), np.float32, n), 1)
    sq, sc = _Q["n_nospace"][qi], _D["n_nospace"][ci]
    f["nospace_eq"] = (sq == sc).astype(np.int8)
    f["jw_nospace"] = np.fromiter((jaro_winkler(a, b) for a, b in zip(sq, sc)), np.float32, n)
    f["indic_q"], f["indic_c"] = _Q["indic"][qi].astype(np.int8), _D["indic"][ci].astype(np.int8)
    # global uniqueness (per split+country): how many S1 share this name / how many records share this address
    for c in ("cnt_core_s1", "cnt_ph_s1", "cnt_nsp_s1", "cnt_addr_all", "cnt_addr_s1"):
        if c in _Q and c in _D:
            f[f"q_{c}"] = np.log1p(_Q[c][qi].astype(np.float32))
            f[f"c_{c}"] = np.log1p(_D[c][ci].astype(np.float32))
    if "cnt_core_all" in _D:
        f["c_cnt_core_all"] = np.log1p(_D["cnt_core_all"][ci].astype(np.float32))
    # name-empty / address-empty flags and lengths
    f["addr_empty_c"] = (_D["n_addr"][ci] == "").astype(np.int8)
    f["addr_len_c"] = np.fromiter((len(x) for x in _D["n_addr"][ci]), np.int16, n)
    f["addr_len_q"] = np.fromiter((len(x) for x in _Q["n_addr"][qi]), np.int16, n)
    # extra tokens (name words present on one side only) and how "generic" they are
    # Generic = tokens the vendor generators add/drop/move without changing the entity (measured on same-address
    # train pairs: legal forms, dba/fka/aka constructs, honorifics, corporate filler) — US/India words AND their French
    # counterparts, so a French pair with an extra "groupe"/"sarl"/"fils" gets the same extra_*_content as a US pair
    # with an extra "group"/"llc"/"services" (the model is trained on US/India only).
    generic = {"co", "company", "inc", "llc", "ltd", "limited", "pvt", "private", "corp", "corporation", "group",
               "holdings", "center", "centre", "services", "service", "the", "and", "of", "international", "enterprises",
               "solutions", "partners", "associates", "llp", "pc", "pllc", "lp", "trust", "foundation", "india",
               "dba", "fka", "aka", "formerly", "known", "as", "doing", "business", "ta", "www", "dr", "mr", "shri", "sri",
               "smt", "incorporated", "corp", "holding",
               # France
               "sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "ei", "compagnie", "cie", "societe", "ste", "ets",
               "etablissements", "groupe", "participations", "developpement", "distribution", "associes", "fils",
               "freres", "france", "de", "du", "des", "la", "le", "les", "et"}
    if _GENERIC_V2:  # ablation (day loop): v2's generic set (48bc4e7), to split normalize.py vs features.py effects
        generic = {"co", "company", "inc", "llc", "ltd", "limited", "pvt", "private", "corp", "corporation", "group",
                   "holdings", "center", "centre", "services", "service", "the", "and", "of", "international",
                   "enterprises", "solutions", "partners", "associates", "llp", "pc", "pllc", "lp", "trust", "foundation",
                   "india"}
        if _GENERIC_MODE == "v2fr":  # v2's set + the French words only (no dba/fka/aka/honorific additions)
            generic |= {"sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "ei", "compagnie", "cie", "societe", "ste",
                        "ets", "etablissements", "groupe", "participations", "developpement", "distribution",
                        "associes", "fils", "freres", "france", "de", "du", "des", "la", "le", "les", "et"}
    tq_ = [set(a.split()) for a in cq]
    tc_ = [set(b.split()) for b in cc]
    f["extra_c"] = np.fromiter((len(b - a) for a, b in zip(tq_, tc_)), np.int8, n)
    f["extra_q"] = np.fromiter((len(a - b) for a, b in zip(tq_, tc_)), np.int8, n)
    f["extra_c_content"] = np.fromiter((len((b - a) - generic) for a, b in zip(tq_, tc_)), np.int8, n)
    f["extra_q_content"] = np.fromiter((len((a - b) - generic) for a, b in zip(tq_, tc_)), np.int8, n)

    lq_, lc_ = _Q["legal"][qi], _D["legal"][ci]
    f["legal_eq"] = ((lq_ == lc_) & (lq_ != "")).astype(np.int8)
    f["legal_conflict"] = ((lq_ != lc_) & (lq_ != "") & (lc_ != "")).astype(np.int8)
    f["legal_missing"] = ((lq_ == "") != (lc_ == "")).astype(np.int8)
    nums_q, nums_c = _Q["a_nums"][qi], _D["a_nums"][ci]
    nq = [set(a.split()) for a in nums_q]
    nc = [set(b.split()) for b in nums_c]
    ninter = np.fromiter((len(a & b) for a, b in zip(nq, nc)), np.float32, n)
    nun = np.fromiter((len(a | b) for a, b in zip(nq, nc)), np.float32, n)
    f["num_jacc"] = np.where(nun > 0, ninter / np.maximum(nun, 1), -1).astype(np.float32)  # -1 = no numbers
    f["num_any"] = (ninter > 0).astype(np.int8)
    f["num_both"] = np.fromiter((bool(a) and bool(b) for a, b in zip(nq, nc)), np.int8, n)
    f["num_first_eq"] = np.fromiter(((a.split()[:1] == b.split()[:1]) if a and b else False
                                     for a, b in zip(nums_q, nums_c)), np.int8, n)
    # house-number agreement in detail: first numeric tokens (JW) + numbers on one side only
    nq_first = [a.split()[0] if a else "" for a in nums_q]
    nc_first = [b.split()[0] if b else "" for b in nums_c]
    f["num_first_jw"] = np.fromiter((jaro_winkler(a, b) if a and b else -1.0 for a, b in zip(nq_first, nc_first)), np.float32, n)
    f["num_only_q"] = np.fromiter((len(a - b) for a, b in zip(nq, nc)), np.int8, n)
    f["num_only_c"] = np.fromiter((len(b - a) for a, b in zip(nq, nc)), np.int8, n)
    zq, zc = _Q["a_zip"][qi], _D["a_zip"][ci]
    zeq = np.fromiter((bool(set(a.split()) & set(b.split())) if a and b else False for a, b in zip(zq, zc)), np.int8, n)
    f["zip_eq"] = zeq
    f["zip_conflict"] = ((zq != "") & (zc != "") & (zeq == 0)).astype(np.int8)
    f["vague_q"], f["vague_c"] = _Q["vague"][qi].astype(np.int8), _D["vague"][ci].astype(np.int8)
    f["jw_addr"] = np.fromiter((jaro_winkler(a[:40], b[:40]) for a, b in zip(_Q["n_addr"][qi], _D["n_addr"][ci])),
                               np.float32, n)
    for fld in ("city", "zip", "state", "country"):
        if fld in _Q and fld in _D:
            a, b = _Q[fld][qi], _D[fld][ci]
            f[f"{fld}_fld_eq"] = np.where((a == "") | (b == ""), -1, (a == b).astype(np.int8)).astype(np.int8)
    return f


def _payload(qi: np.ndarray, ci: np.ndarray) -> dict:
    """Gather only the strings a chunk needs (sent to the worker by pickle; workers never touch the big tables)."""
    return {"q": {c: v[qi] for c, v in _Q.items()}, "d": {c: v[ci] for c, v in _D.items()}}


def _worker(payload: dict) -> dict:
    _Q.clear()
    _D.clear()
    _Q.update(payload["q"])
    _D.update(payload["d"])
    n = len(next(iter(_Q.values())))
    idx = np.arange(n)
    return _stage_b_block(idx, idx)


def stage_b(cands: pd.DataFrame, q_rec: pd.DataFrame, d_rec: pd.DataFrame, n_jobs: int = 0,
            chunk: int = 100_000) -> pd.DataFrame:
    """String/number/legal features. Parallel over pair chunks; each worker gets only its chunk's strings
    (avoids copy-on-write blow-up of the parent's tables). n_jobs=0 -> all cores."""
    import multiprocessing as mp
    import os
    _set_rec(q_rec, d_rec)
    qi, ci = cands["q"].values, cands["c"].values
    n_jobs = n_jobs or (os.cpu_count() or 1)
    chunks = [np.arange(a, min(a + chunk, len(qi))) for a in range(0, len(qi), chunk)]
    if n_jobs > 1 and len(chunks) > 1:
        # spawned workers (fresh interpreters): nothing inherited from the parent, so memory = payload only.
        from concurrent.futures import ProcessPoolExecutor  # raises BrokenProcessPool if a worker is OOM-killed
        with ProcessPoolExecutor(min(n_jobs, len(chunks)), mp_context=mp.get_context("spawn")) as ex:
            parts = list(ex.map(_worker, (_payload(qi[ix], ci[ix]) for ix in chunks)))
        feats = {}
        for k in list(parts[0]):  # assemble one feature at a time and free the chunk pieces as we go
            feats[k] = np.concatenate([p.pop(k) for p in parts])
    else:
        feats = _stage_b_block(qi, ci)
    for k, v in feats.items():
        cands[k] = v
    return cands


FEATURE_EXCLUDE = {"q", "c", "y", "fold", "p", "part"}


def feature_columns(df: pd.DataFrame):
    return [c for c in df.columns if c not in FEATURE_EXCLUDE]
