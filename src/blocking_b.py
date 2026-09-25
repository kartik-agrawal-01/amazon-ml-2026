"""Track B blocking: exact-KEY candidates (pandas merges) + DENSE (MiniLM, GPU) candidates.

Both produce the same long-format parts as blocking.lexical_candidates: DataFrames (q, c, view, rank) with q/c as
ROW indices of the query block / doc table, so blocking.union_candidates() and the cascade can consume them unchanged.

Key blocking (within one country, no country vocabulary, no hard-coded country logic):
  k_core     exact normalised core name (legal suffix stripped)
  k_nsp      exact core name without spaces (word-split / joined variants)
  k_ph       exact phonetic key of the core name (typos, transliterations)
  k_tok_zip  first name token + first postcode/pin found in the address
  k_tok_city first name token + city field
A doc-side bucket larger than `cap` records is dropped (too common to be discriminative: the dense/lexical views
cover those); every query keeps at most `per_q` docs per key per source.

Dense blocking: MiniLM (all-MiniLM-L6-v2, Apache-2.0) fp16 unit embeddings of "<n_name> | <n_addr>", exact top-k by
chunked matmul on the GPU per source (src/dense.py).
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

KEY_VIEWS = ["k_core", "k_nsp", "k_ph", "k_tok_zip", "k_tok_city"]


def _first_tok(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.split(n=1).str[0].fillna("")


def key_columns(rec: pd.DataFrame) -> Dict[str, pd.Series]:
    """The blocking keys of a normalised record table (empty string = no key)."""
    core = rec["n_core"].fillna("").astype(str)
    tok = _first_tok(core)
    zip1 = _first_tok(rec["a_zip"]) if "a_zip" in rec else pd.Series("", index=rec.index)
    city = rec["city"].fillna("").astype(str).str.strip().str.lower() if "city" in rec else pd.Series("", index=rec.index)
    keys = {
        "k_core": core,
        "k_nsp": rec["n_nospace"].fillna("").astype(str) if "n_nospace" in rec else core.str.replace(" ", "", regex=False),
        "k_ph": rec["n_ph"].fillna("").astype(str) if "n_ph" in rec else pd.Series("", index=rec.index),
        "k_tok_zip": np.where((tok != "") & (zip1 != ""), tok + " " + zip1, ""),
        "k_tok_city": np.where((tok != "") & (city != ""), tok + " " + city, ""),
    }
    return {k: pd.Series(np.asarray(v, dtype=object), index=rec.index) for k, v in keys.items()}


def key_candidates(q: pd.DataFrame, d: pd.DataFrame, d_src: np.ndarray, views: List[str] = KEY_VIEWS,
                   cap: int = 100, per_q: int = 30, q_sel: Optional[np.ndarray] = None,
                   verbose: bool = False) -> List[pd.DataFrame]:
    """Exact-key matches of the query rows (q_sel, default all) against the doc table, per source and key view."""
    qk, dk = key_columns(q), key_columns(d)
    q_idx = np.arange(len(q)) if q_sel is None else np.asarray(q_sel)
    parts = []
    for v in views:
        t = time.time()
        dser = dk[v]
        dkeys = pd.DataFrame({"key": dser.to_numpy(), "c": np.arange(len(d)), "src": d_src})
        dkeys = dkeys[dkeys["key"] != ""]
        # bucket cap per (key, source): a key shared by > cap docs of one source is not discriminative
        sz = dkeys.groupby(["key", "src"])["c"].transform("size")
        dkeys = dkeys[sz <= cap]
        qkeys = pd.DataFrame({"key": qk[v].to_numpy()[q_idx], "q": q_idx})
        qkeys = qkeys[qkeys["key"] != ""]
        m = qkeys.merge(dkeys, on="key", how="inner")
        m["rank"] = m.groupby(["q", "src"]).cumcount().astype(np.int16) if len(m) else np.zeros(0, np.int16)
        m = m[m["rank"] < per_q]
        n = 0
        for tgt in (2, 3):
            mm = m[m["src"] == tgt]
            parts.append(pd.DataFrame({"q": mm["q"].to_numpy(np.int64), "c": mm["c"].to_numpy(np.int64),
                                       "view": v, "rank": mm["rank"].to_numpy(np.int16)}))
            n += len(mm)
        if verbose:
            print(f"      key {v:<11} {n:>9} pairs ({time.time() - t:.1f}s)", flush=True)
    return parts


def dense_text_norm(rec: pd.DataFrame, max_chars: int = 160) -> np.ndarray:
    """Normalised 'name | address' text for the bi-encoder (no raw text needed in the store)."""
    return (rec["n_name"].fillna("").astype(str).str.strip() + " | " +
            rec["n_addr"].fillna("").astype(str).str.strip()).str.slice(0, max_chars).to_numpy()


def encode_table(rec: pd.DataFrame, model, cache_path: Optional[str] = None, batch_size: int = 1024,
                 verbose: bool = True) -> Tuple[np.ndarray, float]:
    """fp16 unit embeddings of a record table (+ encode seconds, 0 when served from cache)."""
    from .dense import encode
    t = time.time()
    emb = encode(dense_text_norm(rec), model, batch_size=batch_size, cache_path=cache_path, verbose=verbose)
    return emb, time.time() - t


def dense_parts(q_emb: np.ndarray, d_emb: np.ndarray, d_src: np.ndarray, k: int, q_sel: Optional[np.ndarray] = None,
                device: Optional[str] = None, verbose: bool = False, min_sim: float = 0.3,
                sim_budget_bytes: float = 1.5e9) -> List[pd.DataFrame]:
    """Exact dense top-k per source (view 'dense'), same output format as key_candidates."""
    from .dense import topk_dense
    parts = []
    q_idx = np.arange(len(q_emb)) if q_sel is None else np.asarray(q_sel)
    d_idx = np.arange(len(d_emb))
    for tgt in (2, 3):
        c_idx = d_idx[d_src == tgt]
        if len(q_idx) == 0 or len(c_idx) == 0:
            continue
        t = time.time()
        r, c, s, rk = topk_dense(q_emb[q_idx], d_emb[c_idx], k, min_sim=min_sim, device=device,
                                 sim_budget_bytes=sim_budget_bytes)
        parts.append(pd.DataFrame({"q": q_idx[r], "c": c_idx[c], "view": "dense", "rank": rk, "sim": s}))
        if verbose:
            print(f"      S1->S{tgt} dense    {len(r):>9} pairs ({time.time() - t:.1f}s)", flush=True)
    return parts
