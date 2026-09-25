"""Candidate generation (blocking) — lexical TF-IDF views, built for 12M-record scale.

Design (v2):
  * Vectorisers are fitted ONCE on a text sample, then applied to docs (S2/S3) and to
    query blocks (S1) separately, so a split's doc matrices are built once and query
    blocks stream through them.
  * Top-k uses `sparse_dot_topn` when installed (C++ / multi-threaded, memory-safe);
    otherwise a chunked scipy matmul + argpartition fallback.
  * Optional partition key: queries only see docs with the same key (empty key -> all docs).

Views (L2-normalised TF-IDF):
  name_c3  char 3-grams of the core name (legal forms stripped)  -> typos, spacing
  name_w   word tokens of the core name                          -> rare distinctive words
  addr_c3  char 3-grams of the normalised address                -> same building / look-alikes
  full_c3  char 3-grams of core name + address                   -> chains (same name, many sites)
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

VIEWS = {
    "name_c3": dict(col="n_core", analyzer="char_wb", ngram_range=(3, 3)),
    "name_w": dict(col="n_core", analyzer="word", ngram_range=(1, 1), token_pattern=r"(?u)\b\w+\b"),
    "addr_c3": dict(col="n_addr", analyzer="char_wb", ngram_range=(3, 3)),
    "full_c3": dict(col="n_full", analyzer="char_wb", ngram_range=(3, 3)),
    "name_ph": dict(col="n_ph", analyzer="char_wb", ngram_range=(3, 3)),   # phonetic key: cross-script + typos
}
DEFAULT_VIEWS = list(VIEWS)

try:  # optional fast backend (pip install sparse_dot_topn) — API changed at 1.0
    from sparse_dot_topn import sp_matmul_topn as _sdt_new  # type: ignore
    _sdt_old = None
except ImportError:
    _sdt_new = None
    try:
        from sparse_dot_topn import awesome_cossim_topn as _sdt_old  # type: ignore
    except ImportError:
        _sdt_old = None


def view_text(rec: pd.DataFrame, view: str) -> np.ndarray:
    col = VIEWS[view]["col"]
    if col == "n_full" and "n_full" not in rec:
        rec["n_full"] = (rec["n_core"] + " " + rec["n_addr"]).str.strip()
    return rec[col].fillna("").values


# ---------------------------------------------------------------- vectorisers
def fit_vectorizers(rec: pd.DataFrame, views: Iterable[str] = DEFAULT_VIEWS, max_df: float = 0.2,
                    min_df: int = 2, sample: int = 2_000_000, seed: int = 0) -> Dict[str, TfidfVectorizer]:
    """Fit one TF-IDF vectoriser per view on (a sample of) all records — no labels used."""
    if len(rec) > sample:
        rec = rec.sample(sample, random_state=seed)
    vecs = {}
    for v in views:
        cfg = {k: val for k, val in VIEWS[v].items() if k != "col"}
        vec = TfidfVectorizer(min_df=min_df, max_df=max_df, sublinear_tf=True, dtype=np.float32, **cfg)
        try:
            vec.fit(view_text(rec, v))
        except ValueError:  # empty vocabulary on tiny samples
            vec.fit(["placeholder"])
        vecs[v] = vec
    return vecs


_XF = {"vec": None, "texts": None}


def _xf_chunk(bounds):
    a, b = bounds
    return _XF["vec"].transform(_XF["texts"][a:b]).astype(np.float32)


def transform(vec: TfidfVectorizer, texts: np.ndarray, n_jobs: int = 0, chunk: int = 200_000) -> sp.csr_matrix:
    """vec.transform in parallel chunks (fork). Row order preserved."""
    n = len(texts)
    n_jobs = n_jobs or (os.cpu_count() or 1)
    bounds = [(a, min(a + chunk, n)) for a in range(0, n, chunk)]
    if n_jobs > 1 and len(bounds) > 1 and "fork" in mp.get_all_start_methods():
        _XF["vec"], _XF["texts"] = vec, texts
        with mp.get_context("fork").Pool(min(n_jobs, len(bounds))) as pool:
            parts = pool.map(_xf_chunk, bounds)
        _XF["vec"] = _XF["texts"] = None
        return sp.vstack(parts).tocsr()
    return vec.transform(texts).astype(np.float32).tocsr()


# --------------------------------------------------------------------- top-k
def _topk_fallback(Q: sp.csr_matrix, D: sp.csr_matrix, k: int, min_sim: float, chunk: int):
    DT = D.T.tocsr()
    n = Q.shape[0]
    out_c = np.full((n, k), -1, dtype=np.int64)
    out_s = np.zeros((n, k), dtype=np.float32)
    for st in range(0, n, chunk):
        M = (Q[st:st + chunk] @ DT).tocsr()
        indptr, indices, data = M.indptr, M.indices, M.data
        for i in range(M.shape[0]):
            a, b = indptr[i], indptr[i + 1]
            if b == a:
                continue
            d = data[a:b]
            sel = np.argpartition(-d, k)[:k] if b - a > k else np.arange(b - a)
            sel = sel[np.argsort(-d[sel], kind="stable")]
            out_c[st + i, :len(sel)] = indices[a:b][sel]
            out_s[st + i, :len(sel)] = d[sel]
    return out_c, out_s


def topk_sparse(Q: sp.csr_matrix, D: sp.csr_matrix, k: int, min_sim: float = 0.05, chunk: int = 0,
                n_threads: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Top-k columns of Q @ D.T per row. Returns (row, col, sim, rank) with rank 0 = best."""
    n = Q.shape[0]
    if n == 0 or D.shape[0] == 0:
        z = np.zeros(0, np.int64)
        return z, z, np.zeros(0, np.float32), np.zeros(0, np.int16)
    k = min(k, D.shape[0])
    n_threads = n_threads or (os.cpu_count() or 1)
    if _sdt_new is not None or _sdt_old is not None:
        DT = D.T.tocsr()
        if _sdt_new is not None:
            C = _sdt_new(Q.tocsr(), DT, top_n=k, threshold=min_sim, sort=True, n_threads=n_threads)
        else:
            C = _sdt_old(Q.tocsr(), DT, k, min_sim, use_threads=n_threads > 1, n_jobs=n_threads)
        C = C.tocsr()
        cnt = np.diff(C.indptr)
        row = np.repeat(np.arange(n), cnt)
        rank = (np.arange(C.nnz) - np.repeat(C.indptr[:-1], cnt)).astype(np.int16)
        return row.astype(np.int64), C.indices.astype(np.int64), C.data.astype(np.float32), rank
    if not chunk:  # size chunks so a chunk's product matrix stays around ~60M nonzeros
        col_df = np.bincount(D.indices, minlength=D.shape[1]).astype(np.float64)  # posting length per feature
        Qb = Q.copy()
        Qb.data[:] = 1.0
        est_row = float(np.mean(Qb @ col_df)) if n else 1.0  # expected nonzeros per product row (upper bound)
        chunk = int(min(2048, max(16, 60e6 // max(est_row, 1))))
    out_c, out_s = _topk_fallback(Q, D, k, min_sim, chunk)
    rank = np.broadcast_to(np.arange(k, dtype=np.int16), (n, k))
    row = np.broadcast_to(np.arange(n)[:, None], (n, k))
    keep = (out_c >= 0) & (out_s >= min_sim)
    return row[keep].astype(np.int64), out_c[keep], out_s[keep], rank[keep]


# ------------------------------------------------------------- candidate union
def union_candidates(parts: List[pd.DataFrame], views: List[str]) -> pd.DataFrame:
    """parts: DataFrames with columns q, c, view, rank -> one row per (q, c) with <view>_rank cols."""
    if not parts:
        return pd.DataFrame({"q": np.zeros(0, np.int64), "c": np.zeros(0, np.int64)})
    long = pd.concat(parts, ignore_index=True)
    wide = long.pivot_table(index=["q", "c"], columns="view", values="rank", aggfunc="min")
    wide.columns = [f"{v}_rank" for v in wide.columns]
    wide = wide.fillna(99).astype(np.int16).reset_index()
    for v in views:
        if f"{v}_rank" not in wide:
            wide[f"{v}_rank"] = np.int16(99)
    wide["n_views"] = (wide[[f"{v}_rank" for v in views]] < 99).sum(axis=1).astype(np.int8)
    return wide


def lexical_candidates(q_mats: Dict[str, sp.csr_matrix], d_mats: Dict[str, sp.csr_matrix], d_src: np.ndarray,
                       k: int, views: List[str], q_sel: Optional[np.ndarray] = None,
                       d_sel: Optional[np.ndarray] = None, verbose: bool = False) -> List[pd.DataFrame]:
    """Per-view top-k S2 and S3 docs for the query rows q_sel (default all) among doc rows d_sel.

    Returns long-format parts (q, c, view, rank) with q/c as ROW indices of the query block / doc table.
    """
    parts = []
    q_idx = np.arange(next(iter(q_mats.values())).shape[0]) if q_sel is None else np.asarray(q_sel)
    d_idx = np.arange(len(d_src)) if d_sel is None else np.asarray(d_sel)
    if len(q_idx) == 0 or len(d_idx) == 0:
        return parts
    for tgt in (2, 3):
        c_idx = d_idx[d_src[d_idx] == tgt]
        if len(c_idx) == 0:
            continue
        for v in views:
            t = time.time()
            Q = q_mats[v] if q_sel is None else q_mats[v][q_idx]
            D = d_mats[v][c_idx]
            r, c, s, rk = topk_sparse(Q, D, k)
            parts.append(pd.DataFrame({"q": q_idx[r], "c": c_idx[c], "view": v, "rank": rk}))
            if verbose:
                print(f"      S1->S{tgt} {v:<8} {len(r):>9} pairs ({time.time() - t:.1f}s)", flush=True)
    return parts


# ---------------------------------------------------------------- evaluation
def gt_pairs_block(q_rid: np.ndarray, d_rid: np.ndarray, gt) -> pd.DataFrame:
    """Ground-truth pairs restricted to this query block, as row indices (q, c)."""
    d_pos = pd.Series(np.arange(len(d_rid)), index=d_rid)
    rows = []
    for qi, r in enumerate(q_rid):
        ms = gt.get(r)
        if ms:
            for m in ms:
                p = d_pos.get(m)
                if p is not None:
                    rows.append((qi, int(p)))
    return pd.DataFrame(rows, columns=["q", "c"], dtype=np.int64)


def blocking_recall(cands: pd.DataFrame, gp: pd.DataFrame, n_q: int, ks=(1, 3, 5, 10, 20)) -> pd.DataFrame:
    """Pair recall and 'entity fully covered' rate vs per-view k (uses the <view>_rank columns)."""
    gp = gp.assign(y=1)
    m = cands.merge(gp, on=["q", "c"], how="left")
    m["y"] = m["y"].fillna(0).astype(np.int8)
    rank_cols = [c for c in cands.columns if c.endswith("_rank")]
    best_rank = m[rank_cols].min(axis=1)
    need = gp.groupby("q").size()
    rows = []
    for k in ks:
        sel = m[best_rank < k]
        covered = sel[sel.y == 1].groupby("q").size()
        full = (covered.reindex(need.index, fill_value=0) == need).mean() if len(need) else np.nan
        rows.append(dict(k=k, pairs=len(sel), pairs_per_s1=len(sel) / max(n_q, 1),
                         pair_recall=sel["y"].sum() / max(len(gp), 1), entity_full_recall=full))
    return pd.DataFrame(rows)
