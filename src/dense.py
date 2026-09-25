"""Dense (embedding) candidate view — the scalable blocking route once a GPU is available.

  encode()      sentence-transformers bi-encoder (MiniLM / bge-small, or our fine-tuned copy),
                fp16 unit vectors, cached to disk as .npy
  topk_dense()  EXACT top-k by inner product. torch on GPU when available (250K queries vs
                4M docs x 384 dims ~ seconds on a 16 GB card), chunked numpy on CPU otherwise.
No FAISS dependency: exact search is cheap enough at this size on a GPU and has no recall loss.

Text fed to the encoder: "<name>, <address>" (raw casing; models were trained on natural text).
"""
from __future__ import annotations

import os
import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def dense_text(rec: pd.DataFrame, max_chars: int = 200) -> np.ndarray:
    return (rec["name"].fillna("").astype(str).str.strip() + ", " +
            rec["addr"].fillna("").astype(str).str.strip()).str.slice(0, max_chars).values


def get_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_encoder(model_name: str = DEFAULT_MODEL, device: Optional[str] = None):
    from sentence_transformers import SentenceTransformer
    device = device or get_device()
    m = SentenceTransformer(model_name, device=device)
    m.max_seq_length = 64
    return m


def encode(texts: np.ndarray, model, batch_size: int = 1024, cache_path: Optional[str] = None,
           verbose: bool = True) -> np.ndarray:
    """Unit-normalised fp16 embeddings (n, d). Uses/creates cache_path (.npy) when given."""
    if cache_path and os.path.exists(cache_path):
        emb = np.load(cache_path, mmap_mode="r")
        if emb.shape[0] == len(texts):
            return np.asarray(emb)
        if verbose:
            print(f"    cache {cache_path} has {emb.shape[0]} rows, need {len(texts)} -> re-encoding")
    t = time.time()
    emb = model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,
                       show_progress_bar=False).astype(np.float16)
    if verbose:
        print(f"    encoded {len(texts)} texts in {time.time() - t:.0f}s ({len(texts) / max(time.time() - t, 1e-9):.0f}/s)")
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.save(cache_path, emb)
    return emb


def topk_dense(Q: np.ndarray, D: np.ndarray, k: int, min_sim: float = 0.3, device: Optional[str] = None,
               q_chunk: int = 0, sim_budget_bytes: float = 1.5e9) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact top-k inner product of each Q row against all D rows. Returns (row, col, sim, rank).

    q_chunk is sized so the (q_chunk x m) similarity matrix stays under `sim_budget_bytes`
    (1.5 GB by default: 5.5M docs -> ~136 queries per chunk; a fixed 2048 would need 22 GB and OOM a 16 GB GPU).
    """
    n, m = len(Q), len(D)
    if n == 0 or m == 0:
        z = np.zeros(0, np.int64)
        return z, z, np.zeros(0, np.float32), np.zeros(0, np.int16)
    k = min(k, m)
    device = device or get_device()
    if not q_chunk:
        bytes_per = 2 if device == "cuda" else 4
        q_chunk = int(max(16, min(4096, sim_budget_bytes // (m * bytes_per))))
    out_c = np.empty((n, k), dtype=np.int64)
    out_s = np.empty((n, k), dtype=np.float32)
    if device == "cuda":
        import torch
        Dt = torch.from_numpy(np.ascontiguousarray(D)).to(device="cuda", dtype=torch.float16)
        for st in range(0, n, q_chunk):
            q = torch.from_numpy(np.ascontiguousarray(Q[st:st + q_chunk])).to(device="cuda", dtype=torch.float16)
            s, i = torch.topk(q @ Dt.T, k, dim=1)
            out_s[st:st + q_chunk] = s.float().cpu().numpy()
            out_c[st:st + q_chunk] = i.cpu().numpy()
        del Dt
        torch.cuda.empty_cache()
    else:
        D32 = np.asarray(D, dtype=np.float32)
        for st in range(0, n, q_chunk):
            s = np.asarray(Q[st:st + q_chunk], dtype=np.float32) @ D32.T
            idx = np.argpartition(-s, k - 1, axis=1)[:, :k] if k < m else np.tile(np.arange(m), (len(s), 1))
            vals = np.take_along_axis(s, idx, axis=1)
            o = np.argsort(-vals, axis=1, kind="stable")
            out_c[st:st + q_chunk] = np.take_along_axis(idx, o, axis=1)
            out_s[st:st + q_chunk] = np.take_along_axis(vals, o, axis=1)
    rank = np.broadcast_to(np.arange(k, dtype=np.int16), (n, k))
    row = np.broadcast_to(np.arange(n)[:, None], (n, k))
    keep = out_s >= min_sim
    return row[keep].astype(np.int64), out_c[keep], out_s[keep], rank[keep]


def dense_candidates(q_emb: np.ndarray, d_emb: np.ndarray, d_src: np.ndarray, k: int,
                     q_sel: Optional[np.ndarray] = None, d_sel: Optional[np.ndarray] = None,
                     device: Optional[str] = None, verbose: bool = False):
    """Long-format parts (q, c, view='dense', rank) for query rows q_sel among doc rows d_sel."""
    parts = []
    q_idx = np.arange(len(q_emb)) if q_sel is None else np.asarray(q_sel)
    d_idx = np.arange(len(d_emb)) if d_sel is None else np.asarray(d_sel)
    for tgt in (2, 3):
        c_idx = d_idx[d_src[d_idx] == tgt]
        if len(q_idx) == 0 or len(c_idx) == 0:
            continue
        t = time.time()
        r, c, s, rk = topk_dense(q_emb[q_idx], d_emb[c_idx], k, device=device)
        parts.append(pd.DataFrame({"q": q_idx[r], "c": c_idx[c], "view": "dense", "rank": rk}))
        if verbose:
            print(f"      S1->S{tgt} dense    {len(r):>9} pairs ({time.time() - t:.1f}s)", flush=True)
    return parts


def pair_cosine_dense(q_emb: np.ndarray, d_emb: np.ndarray, qi: np.ndarray, ci: np.ndarray,
                      chunk: int = 1_000_000) -> np.ndarray:
    out = np.empty(len(qi), dtype=np.float32)
    for st in range(0, len(qi), chunk):
        a = np.asarray(q_emb[qi[st:st + chunk]], dtype=np.float32)
        b = np.asarray(d_emb[ci[st:st + chunk]], dtype=np.float32)
        out[st:st + chunk] = np.einsum("ij,ij->i", a, b)
    return out
