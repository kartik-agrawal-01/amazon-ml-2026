"""Jarvis QUEUE 2a: re-rank the cascade pool with the cross-encoder before stage B (src/pipeline.py --ce-dir).

The pool of an S1 is the same set `dump_pool` writes (top-40 by the cascade score pa + every pair the cascade kept).
Pool pairs are ranked by blend = w * sigmoid(ce) + (1 - w) * pa and the top --ce-top are kept; key-sure / reverse-sure
pairs stay kept as before (the caller ORs them in). Pairs outside the pool are never kept (their pa is below the
top-40 of their S1 and the cascade dropped them).
  train: ce = the out-of-fold logit from `python -m src.jv_ce train` (<ce-dir>/train_ce.parquet), matched by (s1, cand);
  test:  ce = mean sigmoid of the fold models <ce-dir>/fold*, scored in-process on the GPU per block.
"""
from __future__ import annotations

import glob
import os
from typing import Optional

import numpy as np
import pandas as pd

POOL_TOP = 40


def pool_mask(q: np.ndarray, pa: np.ndarray, keep: np.ndarray, top: int = POOL_TOP) -> np.ndarray:
    """Same selection as pipeline.dump_pool: rank < top by pa within the S1, or kept by the cascade."""
    o = np.lexsort((-pa, q))
    qs = q[o]
    st = np.r_[0, np.flatnonzero(qs[1:] != qs[:-1]) + 1] if len(o) else np.zeros(0, np.int64)
    rank = np.arange(len(o)) - np.repeat(st, np.diff(np.r_[st, len(o)]))
    m = np.zeros(len(q), bool)
    m[o] = rank < top
    return m | keep


def top_by_blend(q: np.ndarray, blend: np.ndarray, top: int, floor: float = 0.0) -> np.ndarray:
    """Keep the top `top` pairs per S1 by blend with blend >= floor; NaN blend (not in the pool) is never kept."""
    b = np.where(np.isnan(blend), -np.inf, blend)
    o = np.lexsort((-b, q))
    qs = q[o]
    st = np.r_[0, np.flatnonzero(qs[1:] != qs[:-1]) + 1] if len(o) else np.zeros(0, np.int64)
    rank = np.arange(len(o)) - np.repeat(st, np.diff(np.r_[st, len(o)]))
    k = np.zeros(len(q), bool)
    k[o] = (rank < top) & np.isfinite(b[o]) & (b[o] >= floor)
    return k


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class TrainCE:
    """Out-of-fold CE probabilities for train pairs, looked up by (S1 row, doc row) of the pipeline's memmaps."""

    def __init__(self, ce_dir: str):
        t = pd.read_parquet(os.path.join(ce_dir, "train_ce.parquet"), columns=["s1", "cand", "country", "ce"])
        t["ce_p"] = sigmoid(t["ce"].to_numpy()).astype(np.float32)
        self.t = t.drop(columns=["ce"])

    def lookup(self, country: str, q_rid: np.ndarray, d_rid: np.ndarray, qi: np.ndarray, ci: np.ndarray) -> np.ndarray:
        """ce_p for pairs (q_rid[qi], d_rid[ci]); NaN where the pair is not in the CE table."""
        t = self.t  # record ids are global, so pairs of other countries simply find no row (-1)
        tq = pd.Index(np.asarray(q_rid).astype(str)).get_indexer(t["s1"].to_numpy())
        tc = pd.Index(np.asarray(d_rid).astype(str)).get_indexer(t["cand"].to_numpy())
        ok = (tq >= 0) & (tc >= 0)
        n_d = np.int64(len(d_rid))
        tk = tq[ok].astype(np.int64) * n_d + tc[ok]
        s = pd.Series(t["ce_p"].to_numpy()[ok], index=tk)
        s = s[~s.index.duplicated()]
        return s.reindex(qi.astype(np.int64) * n_d + ci.astype(np.int64)).to_numpy(dtype=np.float32)


class TestCE:
    """Mean sigmoid of the fold models on (S1 text, doc text) pairs, "n_name | n_addr" per side (as in jv_ce)."""

    def __init__(self, ce_dir: str, max_len: int = 96, bs: int = 2048):
        from .jv_ce import Encoder
        paths = sorted(glob.glob(os.path.join(ce_dir, "fold*")))
        if not paths:
            raise FileNotFoundError(f"no fold models in {ce_dir}")
        self.encs = [Encoder(p, max_len, "cuda") for p in paths]
        self.bs = bs

    def score(self, q_df: pd.DataFrame, d_df: pd.DataFrame, qi: np.ndarray, ci: np.ndarray) -> np.ndarray:
        from .jv_ce import predict
        uq, iq = np.unique(qi, return_inverse=True)
        uc, ic = np.unique(ci, return_inverse=True)
        tq = (q_df["n_name"].fillna("").to_numpy()[uq].astype(str) + " | " + q_df["n_addr"].fillna("").to_numpy()[uq].astype(str))
        tc = (d_df["n_name"].fillna("").to_numpy()[uc].astype(str) + " | " + d_df["n_addr"].fillna("").to_numpy()[uc].astype(str))
        toks = self.encs[0].tokenize(list(tq) + list(tc))
        ta, tb = iq.astype(np.int64), ic.astype(np.int64) + len(uq)
        return np.mean([sigmoid(predict(e, toks, ta, tb, self.bs)) for e in self.encs], axis=0).astype(np.float32)


def blend(pa: np.ndarray, ce_p: np.ndarray, w: float) -> np.ndarray:
    return (w * ce_p + (1.0 - w) * pa).astype(np.float32)
