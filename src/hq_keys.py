"""Exact-key candidates + train-calibrated 'sure' match rules + global one-to-one (HQ, 26 Sep).

Why: on the FULL train (US/India, 12M exact-key pairs) some exact-key patterns are near-certain matches, e.g. a
different *invented* name (trade name / domain) at the *same exact address* where no other S1 has that name or that
address is a match 96-98% of the time. v2 finds only 34% of these in France (92% in US): they rank low in the
address views and are cut by the top-10 cascade among France's dense same-name decoys. And v2's one-to-one filter is
block-local, so 13.8K test records were assigned to 2-5 S1s. Details: docs/ENSEMBLE_AND_KEYS.md.

Pure functions over normalised per-country tables (store columns rid, src, country, n_core, n_nospace, n_addr).
Keys use pandas' stable hash (no PYTHONHASHSEED dependence).

Typical wiring (per country, per split), see QUEUE:
    kq, kd = record_keys(q), record_keys(d)                      # q = S1 rows, d = S2/S3 rows of one country
    kp = key_pairs(kq, kd)                                       # exact-key pairs + categories (qi, ci positions)
    train:  rules = calibrate(kp_train, y_train, country)        # P(match) per rule and country, from the GT
    test:   sure = kp[apply_rules(kp, rules, country)]          # pairs whose rule is 'sure' in that country
    -> (a) union `sure` into the post-cascade candidate set (bypasses the top-N cap; they go to stage B + model and
           into candidate_pairs.tsv), and/or (b) force-include them after `decide` (only rules whose
           conservative bound is high, see force_mask).
    -> after ALL blocks of a country: global_one_to_one(decided pairs with p) before writing matching_results.tsv.
"""
from __future__ import annotations

import collections
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

NCATS = ["n_core_eq", "n_nsp_eq", "n_reorder", "n_subset", "n_swap1", "n_disjoint", "n_partial", "n_empty"]
ACATS = ["a_eq", "a_street_eq_num_diff", "a_street_eq_num_missing", "a_num_eq_street_diff", "a_diff", "a_c_empty"]

# Rule = (name category, address category, context condition, name kind). ctx: 'any' | 'not_core' (no other S1 has the
# candidate's core name) | 'ff' (no other S1 has the candidate's core name nor its exact address) | 'not_both'.
RULES = {
    "core_eq|a_eq":            ("n_core_eq", "a_eq", "any", None),
    "reorder|a_eq":            ("n_reorder", "a_eq", "any", None),
    "subset|a_eq":             ("n_subset", "a_eq", "any", None),
    "nsp_eq|a_eq":             ("n_nsp_eq", "a_eq", "not_core", None),
    "core_eq|num_eq":          ("n_core_eq", "a_num_eq_street_diff", "not_both", None),
    "nsp_eq|num_eq":           ("n_nsp_eq", "a_num_eq_street_diff", "ff", None),
    "core_eq|street_eq_nonum": ("n_core_eq", "a_street_eq_num_missing", "any", None),
    "core_eq|c_empty":         ("n_core_eq", "a_c_empty", "not_core", None),
    "swap1|a_eq|invented":     ("n_swap1", "a_eq", "ff", "invented"),
    "swap1|a_eq|realword":     ("n_swap1", "a_eq", "ff", "realword"),
    "partial|a_eq|invented":   ("n_partial", "a_eq", "ff", "invented"),
    "partial|a_eq|realword":   ("n_partial", "a_eq", "ff", "realword"),
    "disjoint|a_eq|invented":  ("n_disjoint", "a_eq", "ff", "invented"),
    "disjoint|a_eq|realword":  ("n_disjoint", "a_eq", "ff", "realword"),
}


def _h(values) -> np.ndarray:
    return pd.util.hash_array(np.asarray(values, dtype=object)).astype(np.int64)


def record_keys(rec: pd.DataFrame) -> pd.DataFrame:
    """Join keys for one country's records (S1 or S2/S3). Needs n_core, n_nospace, n_addr (+ rid, src, country)."""
    core = rec["n_core"].fillna("").astype(str).to_numpy(dtype=object)
    nsp = rec["n_nospace"].fillna("").astype(str).to_numpy(dtype=object) if "n_nospace" in rec else \
        np.array([x.replace(" ", "") for x in core], dtype=object)
    addr = rec["n_addr"].fillna("").astype(str).to_numpy(dtype=object)
    toks = [a.split() for a in addr]
    srt = np.array([" ".join(sorted(t)) for t in toks], dtype=object)
    street = np.array([" ".join(sorted(x for x in t if not x.isdigit())) for t in toks], dtype=object)
    num1 = np.fromiter((next((int(x[:15]) for x in t if x.isdigit()), -1) for t in toks), np.int64, len(toks))
    out = pd.DataFrame({"k_core": _h(core), "k_nsp": _h(nsp), "k_addr": _h(srt), "k_street": _h(street), "num1": num1,
                        "addr_empty": np.fromiter((not t for t in toks), bool, len(toks)),
                        "core_empty": np.fromiter((not c for c in core), bool, len(core))})
    out["n_core"] = core
    for c in ("rid", "src", "country"):
        if c in rec:
            out[c] = rec[c].to_numpy()
    return out


def _tok_rel(a: str, b: str) -> str:
    A, B = a.split(), b.split()
    SA, SB = set(A), set(B)
    if not SA or not SB:
        return "n_empty"
    if SA == SB:
        return "n_reorder"
    if SA < SB or SB < SA:
        return "n_subset"
    if len(SA) == len(SB) and len(SA - SB) == 1:
        return "n_swap1"
    if not SA & SB:
        return "n_disjoint"
    return "n_partial"


def s1_vocab(kq: pd.DataFrame, min_count: int = 5) -> set:
    """Tokens used by >= min_count S1 core names of the country: names built only from these are 'realword'."""
    cnt = collections.Counter(t for x in kq["n_core"].values for t in x.split())
    return {t for t, k in cnt.items() if k >= min_count}


def key_pairs(kq: pd.DataFrame, kd: pd.DataFrame, cap: int = 2000, vocab: Optional[set] = None,
              kq_ctx: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """All (S1 row qi, record row ci) pairs sharing the exact core name, the exact no-space name or the exact address
    (groups with n_q * n_d > cap skipped), with name/address categories and context flags.
    kq_ctx: keys of ALL S1 of the country (default kq). The context flags and the vocabulary MUST come from all S1 of
    the country: with a sampled kq (train pass) the 'another S1 has this name/address' flags would miss most decoys
    (measured: the ff-rule rates drop from 0.96 on the full train to 0.86 on the 8% slice)."""
    parts = []
    for key, skip_col in (("k_core", "core_empty"), ("k_nsp", "core_empty"), ("k_addr", "addr_empty")):
        a = pd.DataFrame({key: kq[key].to_numpy(), "qi": np.arange(len(kq))})[~kq[skip_col].to_numpy()]
        b = pd.DataFrame({key: kd[key].to_numpy(), "ci": np.arange(len(kd))})[~kd[skip_col].to_numpy()]
        na, nb = a[key].value_counts(), b[key].value_counts()
        ok = na.index.intersection(nb.index)
        ok = ok[(na.reindex(ok).to_numpy() * nb.reindex(ok).to_numpy()) <= cap]
        a, b = a[a[key].isin(ok)], b[b[key].isin(ok)]
        parts.append(a.merge(b, on=key)[["qi", "ci"]])
    P = pd.concat(parts, ignore_index=True).drop_duplicates(ignore_index=True)
    qi, ci = P.qi.to_numpy(), P.ci.to_numpy()
    q = {c: kq[c].to_numpy()[qi] for c in ("k_core", "k_nsp", "k_addr", "k_street", "num1")}
    d = {c: kd[c].to_numpy()[ci] for c in ("k_core", "k_nsp", "k_addr", "k_street", "num1", "addr_empty")}
    nc = np.where(q["k_core"] == d["k_core"], "n_core_eq", np.where(q["k_nsp"] == d["k_nsp"], "n_nsp_eq", "")).astype(object)
    rest = np.flatnonzero(nc == "")
    qc, dc = kq["n_core"].to_numpy(), kd["n_core"].to_numpy()
    nc[rest] = [_tok_rel(x, y) for x, y in zip(qc[qi[rest]], dc[ci[rest]])]
    ac = np.full(len(P), "a_diff", dtype=object)
    se = q["k_street"] == d["k_street"]
    both = (q["num1"] >= 0) & (d["num1"] >= 0)
    ac[(q["num1"] >= 0) & (q["num1"] == d["num1"]) & ~se] = "a_num_eq_street_diff"
    ac[se & ~both] = "a_street_eq_num_missing"
    ac[se & both] = "a_street_eq_num_diff"
    ac[q["k_addr"] == d["k_addr"]] = "a_eq"
    ac[d["addr_empty"]] = "a_c_empty"
    kc = kq if kq_ctx is None else kq_ctx
    s1_addr = pd.Series(kc["k_addr"].to_numpy()[~kc["addr_empty"].to_numpy()]).value_counts()
    s1_core = pd.Series(kc["k_core"].to_numpy()[~kc["core_empty"].to_numpy()]).value_counts()
    n_addr = s1_addr.reindex(d["k_addr"]).fillna(0).to_numpy() - (q["k_addr"] == d["k_addr"])
    n_core = s1_core.reindex(d["k_core"]).fillna(0).to_numpy() - (q["k_core"] == d["k_core"])
    P["ncat"], P["acat"] = nc, ac
    P["ctx_addr_other"], P["ctx_core_other"] = n_addr > 0, n_core > 0
    vocab = s1_vocab(kc) if vocab is None else vocab
    kind = np.full(len(P), "", dtype=object)
    sel = np.flatnonzero(np.isin(nc, ["n_swap1", "n_partial", "n_disjoint"]))
    kind[sel] = ["realword" if x.split() and all(t in vocab for t in x.split()) else "invented" for x in dc[ci[sel]]]
    P["kind"] = kind
    P["rule"] = rule_of(P)
    return P


def rule_of(P: pd.DataFrame) -> np.ndarray:
    out = np.full(len(P), "", dtype=object)
    nc, ac, kind = P.ncat.to_numpy(), P.acat.to_numpy(), P.kind.to_numpy()
    co, ao = P.ctx_core_other.to_numpy(), P.ctx_addr_other.to_numpy()
    ctx = {"any": np.ones(len(P), bool), "not_core": ~co, "ff": ~co & ~ao, "not_both": ~(co & ao)}
    for name, (n, a, c, k) in RULES.items():
        m = (nc == n) & (ac == a) & ctx[c] & (out == "")
        if k is not None:
            m &= kind == k
        out[m] = name
    return out


def calibrate(P: pd.DataFrame, y: np.ndarray, min_n: int = 300) -> pd.DataFrame:
    """P(match) per rule on TRAIN key pairs of one country (y = pair is in the GT). Returns rule, n, p."""
    t = pd.DataFrame({"rule": P["rule"].to_numpy(), "y": np.asarray(y, bool)})
    t = t[t.rule != ""].groupby("rule").y.agg(n="size", p="mean").reset_index()
    return t[t.n >= min_n]


def rules_for_country(calibs: Dict[str, pd.DataFrame], country: str) -> Dict[str, float]:
    """rule -> P for `country`; a country unseen in training (France) gets the MINIMUM over training countries."""
    if country in calibs:
        c = calibs[country]
        return dict(zip(c.rule, c.p))
    tabs = [dict(zip(c.rule, c.p)) for c in calibs.values()]
    common = set.intersection(*(set(t) for t in tabs)) if tabs else set()
    return {r: min(t[r] for t in tabs) for r in common}


def apply_rules(P: pd.DataFrame, rule_p: Dict[str, float], p_min: float = 0.96) -> np.ndarray:
    """Mask of key pairs whose rule has calibrated P(match) >= p_min, excluding records that are 'sure' for several S1."""
    ok = np.array([rule_p.get(r, 0.0) >= p_min for r in P["rule"].to_numpy()])
    multi = pd.Series(P["ci"].to_numpy()[ok]).duplicated(keep=False).to_numpy()
    idx = np.flatnonzero(ok)
    ok[idx[multi]] = False
    return ok


def force_mask(P: pd.DataFrame, sure: np.ndarray, decided: np.ndarray, rule_p: Dict[str, float],
               bound_min: float = 0.80) -> np.ndarray:
    """Which sure pairs to force-include after the model decided. Per rule: coverage = share of its sure pairs the model
    already accepted; conservative precision of the rest = (P - coverage) / (1 - coverage) (worst case: every accepted
    pair is true). Force only rules whose bound >= bound_min (break-even of F0.5 for typical set sizes is ~0.73-0.77)."""
    out = np.zeros(len(P), bool)
    rules = P["rule"].to_numpy()
    for r in np.unique(rules[sure]):
        m = sure & (rules == r)
        cov = decided[m].mean()
        if cov >= 1.0:
            continue
        bound = (rule_p.get(r, 0.0) - cov) / (1.0 - cov)
        if bound >= bound_min:
            out |= m & ~decided
    return out


def global_one_to_one(q_rid: np.ndarray, c_rid: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Keep, for every record, only its highest-probability S1 (ties: first). Apply after ALL blocks of a country:
    the ground truth is strictly one-to-one, a record kept for two S1s is at least one guaranteed false positive."""
    order = np.lexsort((-np.asarray(p, float), np.asarray(c_rid)))
    c_sorted = np.asarray(c_rid)[order]
    first = np.r_[True, c_sorted[1:] != c_sorted[:-1]]
    keep = np.zeros(len(order), bool)
    keep[order[first]] = True
    return keep


if __name__ == "__main__":   # tiny self-test
    q = pd.DataFrame({"rid": ["S1-1", "S1-2"], "src": [1, 1], "country": ["fr", "fr"],
                      "n_core": ["movida and freres", "club sportif"], "n_nospace": ["movidaandfreres", "clubsportif"],
                      "n_addr": ["62 rue chanoine larose nantes pdl", "12 rue de la paix lille hdf"]})
    d = pd.DataFrame({"rid": ["S2-1", "S2-2", "S3-3"], "src": [2, 2, 3], "country": ["fr"] * 3,
                      "n_core": ["deltavantagepyra", "club sportif", "movida and freres"],
                      "n_nospace": ["deltavantagepyra", "clubsportif", "movidaandfreres"],
                      "n_addr": ["62 rue chanoine larose nantes pdl", "", "62 rue chanoine larose nantes"]})
    kq, kd = record_keys(q), record_keys(d)
    P = key_pairs(kq, kd, vocab={"club", "sportif", "movida", "and", "freres"})
    print(P)
    print(global_one_to_one(np.array(["a", "b", "a"]), np.array(["x", "x", "y"]), np.array([0.9, 0.95, 0.8])))
