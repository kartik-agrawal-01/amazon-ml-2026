"""Jarvis QUEUE 1b: candidate augmentation that bypasses the cascade's top-N cap.

(i)  exact-key 'sure' pairs (src/hq_keys.py): key pairs between the query S1s and the country's S2/S3 records, with
     the context flags taken from ALL S1 of the country; calibrated on the train pass (P(match) per rule/country),
     rules with P >= p_min are 'sure' (France/unseen countries: the minimum over the train countries).
(ii) reverse blocking: every S2/S3 record queries its top-k S1 among ALL S1 of the country under a few views (GPU
     top-k with the roles swapped). This is also a density-consistent context signal: "is this S1 among the record's
     best S1s over the whole country" (the within-block rq_* stage-A features depend on the block's S1 density).

Both are merged into the pre-cascade union (so they get stage-A features and the cascade sees them); flags:
    key_rule   int8  index+1 of the hq_keys rule of the pair (0 = not an exact-key pair)   [label-free feature]
    rev_<v>_rank int16 rank of the S1 among the record's reverse top-k under view v (99 = not in it)  [feature]
and a keep mask (not a feature) for the pairs that must survive the cascade.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import hq_keys
from .blocking import topk_sparse, transform, view_text

RULE_CODE = {r: i + 1 for i, r in enumerate(hq_keys.RULES)}
REV_VIEWS = ("full_w", "name_c3")


def keys_for(rec: pd.DataFrame) -> pd.DataFrame:
    return hq_keys.record_keys(rec)


def key_pairs_block(kq: pd.DataFrame, kd: pd.DataFrame, kq_ctx: pd.DataFrame, vocab: set) -> pd.DataFrame:
    """Exact-key pairs (qi into kq, ci into kd) with rule labels; context + vocab from all S1 of the country."""
    P = hq_keys.key_pairs(kq, kd, vocab=vocab, kq_ctx=kq_ctx)
    P["key_rule"] = np.array([RULE_CODE.get(r, 0) for r in P["rule"].to_numpy()], dtype=np.int8)
    return P


def reverse_topk(vecs, q_all: pd.DataFrame, d_mats: Dict[str, "object"], d_src: np.ndarray, k: int = 3,
                 views=REV_VIEWS, n_jobs: int = 0, log=print) -> pd.DataFrame:
    """For every doc row: top-k S1 rows of q_all per view. Returns (qa, c, rev_<v>_rank...) one row per pair."""
    parts = []
    for v in views:
        t = time.time()
        Qa = transform(vecs[v], view_text(q_all, v), n_jobs)
        r, c, s, rk = topk_sparse(d_mats[v], Qa, k)   # rows = docs, cols = S1 of the country
        parts.append(pd.DataFrame({"qa": c.astype(np.int64), "c": r.astype(np.int64), "view": v, "rank": rk}))
        log(f"reverse top-{k} {v}: {len(r)} pairs for {d_mats[v].shape[0]} docs x {Qa.shape[0]} S1 ({time.time() - t:.0f}s)")
        del Qa
    if "n_full" in q_all:
        del q_all["n_full"]
    long = pd.concat(parts, ignore_index=True)
    wide = long.pivot_table(index=["qa", "c"], columns="view", values="rank", aggfunc="min")
    wide.columns = [f"rev_{v}_rank" for v in wide.columns]
    wide = wide.fillna(99).astype(np.int16).reset_index()
    for v in views:
        if f"rev_{v}_rank" not in wide:
            wide[f"rev_{v}_rank"] = np.int16(99)
    return wide


def rev_cols(views=REV_VIEWS) -> List[str]:
    return [f"rev_{v}_rank" for v in views]


def merge_aug(union: pd.DataFrame, rev_b: Optional[pd.DataFrame], key_b: Optional[pd.DataFrame],
              view_rank_cols: List[str], views=REV_VIEWS) -> pd.DataFrame:
    """Outer-merge block-local reverse pairs (q, c, rev_*) and key pairs (q, c, key_rule, sure) into the union.
    New pairs get view ranks 99 / n_views 0. Adds columns key_rule, rev_*_rank, aug_sure (bool, dropped before
    stage-A features), src_fwd (1 = pair came from forward top-k)."""
    u = union.copy()
    u["src_fwd"] = np.int8(1)
    add = []
    if rev_b is not None and len(rev_b):
        add.append(rev_b[["q", "c"] + rev_cols(views)])
    if key_b is not None and len(key_b):
        add.append(key_b[["q", "c", "key_rule", "aug_sure"]])
    if add:
        a = add[0]
        for b in add[1:]:
            a = a.merge(b, on=["q", "c"], how="outer")
        u = u.merge(a, on=["q", "c"], how="outer")
    for col in view_rank_cols:
        u[col] = u[col].fillna(99).astype(np.int16)
    u["n_views"] = u["n_views"].fillna(0).astype(np.int8)
    u["src_fwd"] = u["src_fwd"].fillna(0).astype(np.int8)
    for col in rev_cols(views):
        u[col] = u[col].fillna(99).astype(np.int16) if col in u else np.int16(99)
    u["key_rule"] = u["key_rule"].fillna(0).astype(np.int8) if "key_rule" in u else np.int8(0)
    u["aug_sure"] = u["aug_sure"].astype("boolean").fillna(False).astype(bool) if "aug_sure" in u else False
    u = u.sort_values(["q", "c"], kind="stable").reset_index(drop=True)
    return u
