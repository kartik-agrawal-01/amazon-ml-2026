"""End-to-end pipeline, per country and block-wise so it fits the shared 15 GB box (~5 GB for us),
with the CASCADE candidate filter (docs/CASCADE_INTEGRATION.md) as the last candidate-generation stage.

Flow
  1. build/load the per-country store (src/store.py): normalised records of each split as one parquet per
     country; ground truth as a (s1, m) pair table. Only one country is in RAM at a time.
  2. fit TF-IDF vectorisers on a text sample drawn across splits/countries (~2.5M rows)
  3. TRAIN, phase A: for each country: sample S1 (--train-s1 split proportionally to the countries' S1
     counts) -> blocks -> lexical/dense candidates -> stage A -> labels. Stage-A features of ALL countries
     go to a float32 memmap; ids/labels stay as small arrays.
     Cascade: cheap GBDT on the stage-A features (3-fold GroupKFold OOF by S1) -> keep top --cascade-top
     pairs per S1 with probability >= --cascade-floor (--no-cascade: the old per-country score prune).
     Phase B: for each country again: stage B on the kept pairs only -> features to a second memmap.
     Then GroupKFold OOF of the full model, rule sweep on OOF, final model.
  4. TEST: for each country: blocks -> candidates -> stage A -> cascade -> stage B -> predict -> decide ->
     rows appended to matching_results.tsv + candidate_pairs.tsv (= the kept pairs) as they are produced.
Group/rank context features are computed WITHIN a block (block size is the same for train and test).

Run from the repo root:
  python -m src.pipeline --data-dir data --out-dir output --run-name v2 [--dense]
Quick dev loop:  --train-s1 50000 --test-limit 50000
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .blocking import (DEFAULT_VIEWS, blocking_recall, fit_vectorizers, gt_pairs_block, lexical_candidates,
                       transform, union_candidates, view_text)
from .cascade import candidate_stats, cascade_keep, fit_cascade, stage_a_columns
from .features import calibrate_prune, feature_columns, prune_score, stage_a, stage_b
from .metric import macro_f05, read_matches_tsv
from .model import decide, feature_importance, make_model, oof_predict, subsample_negatives, sweep_rules
from .store import (KEEP_COLS, RAW_COLS, allocate, ensure_store, gt_dict, load_country, load_gt_pairs,
                    text_sample)

T0 = time.time()
ID_DTYPE = "S16"  # fixed-width bytes for record ids (S1-xxxxxxxx / S2-xxxxxxxxx fit easily)
ROW_CHUNK = 1_000_000  # rows per to_numpy / predict_proba chunk (bounds the float32 copies)


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def peak_rss_mb() -> int:
    """Peak resident set size of THIS process (VmHWM), MB. Workers are spawned and small."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return -1


def release_memory() -> None:
    """Hand freed heap back to the OS after dropping a big table (glibc keeps fragmented arenas otherwise)."""
    gc.collect()
    try:
        import pyarrow as pa
        pa.default_memory_pool().release_unused()
    except Exception:  # noqa: BLE001
        pass
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # noqa: BLE001
        pass


def split_tables(rec: pd.DataFrame):
    """(S1 rows, S2+S3 rows). Rows are stored S1 first, so these are positional slices (zero-copy for arrow)."""
    src = rec["src"].to_numpy()
    n1 = int((src == 1).sum())
    if n1 and np.all(src[:n1] == 1) and (n1 == len(src) or np.all(src[n1:] != 1)):
        q = rec.iloc[:n1].reset_index(drop=True)
        d = rec.iloc[n1:].reset_index(drop=True)
    else:
        q = rec[rec.src == 1].reset_index(drop=True)
        d = rec[rec.src != 1].reset_index(drop=True)
    return q, d


def ids_bytes(s: pd.Series) -> np.ndarray:
    """Record ids as fixed-width bytes (16 B each instead of ~60 B Python strings)."""
    if int(s.str.len().max()) > np.dtype(ID_DTYPE).itemsize:
        raise ValueError(f"record ids longer than {ID_DTYPE}")
    return np.asarray(s.astype(object), dtype=ID_DTYPE)


def blocks(n: int, block_size: int):
    for st in range(0, n, block_size):
        yield np.arange(st, min(st + block_size, n))


def write_rows(fh, df: pd.DataFrame, cols: List[str]) -> int:
    """Append df[cols] as float32 rows to an open binary file, ROW_CHUNK rows at a time. Returns rows written."""
    for st in range(0, len(df), ROW_CHUNK):
        fh.write(np.ascontiguousarray(df[cols].iloc[st:st + ROW_CHUNK].to_numpy(dtype=np.float32)).tobytes())
    return len(df)


def predict_chunked(model, cands: pd.DataFrame, feats: List[str]) -> np.ndarray:
    out = np.empty(len(cands), dtype=np.float32)
    for st in range(0, len(cands), ROW_CHUNK):
        out[st:st + ROW_CHUNK] = model.predict_proba(cands[feats].iloc[st:st + ROW_CHUNK].to_numpy(dtype=np.float32))[:, 1]
    return out


def stats_from_counts(counts: np.ndarray, n_kept: int = None, n_true: int = None) -> Dict:
    """candidate_stats() equivalent from per-S1 candidate counts (aggregated over blocks/countries)."""
    out = dict(pairs=int(counts.sum()), n_s1=int(len(counts)), mean=float(counts.mean()) if len(counts) else 0.0,
               median=float(np.median(counts)) if len(counts) else 0.0,
               p90=float(np.percentile(counts, 90)) if len(counts) else 0.0, max=int(counts.max()) if len(counts) else 0,
               zero_share=float((counts == 0).mean()) if len(counts) else 0.0)
    if n_kept is not None and n_true:
        out["pair_recall"] = float(n_kept / n_true)
    return out


# ------------------------------------------------------- candidate cache (fast lane)
class CandCache:
    """Pre-cascade candidate union (q, c, <view>_rank, n_views) per (split, country, block), saved as parquet.

    Everything downstream of top-k (stage A cosines + group/rank context, cascade, stage B, model, rule) is a pure
    function of this union and the store, so a cached union makes feature/model/rule changes cheap (the GPU top-k
    is ~80% of a full-data run). The union is keyed by record ids: q rows are checked against the block's S1 ids,
    doc rows are remapped by id if the store was rebuilt (e.g. new normalisation) and rows moved."""

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self._drid_map = {}

    @staticmethod
    def _fname(tag: str) -> str:
        return "".join(ch if ch.isalnum() else "_" for ch in tag)

    def _paths(self, tag: str, bi: int):
        b = os.path.join(self.root, f"{self._fname(tag)}__b{bi:03d}")
        return b + ".parquet", b + "_qrid.npy"

    def ensure_docs(self, tag: str, d_rid: np.ndarray) -> None:
        """Doc id order of the country the cache was built on (written once); remap table if the store differs."""
        p = os.path.join(self.root, f"{self._fname(tag)}__docs.npy")
        cur = np.asarray(d_rid, dtype=ID_DTYPE)
        if not os.path.exists(p):
            np.save(p, cur)
            self._drid_map[tag] = None
            return
        old = np.load(p)
        if len(old) == len(cur) and np.array_equal(old, cur):
            self._drid_map[tag] = None
            return
        idx = pd.Index(cur).get_indexer(old)  # old doc row -> current doc row (-1: doc gone)
        log(f"cand-cache {tag}: doc order differs from the cached store -> remapping ({int((idx < 0).sum())} docs missing)")
        self._drid_map[tag] = idx

    def load(self, tag: str, bi: int, q_rid: np.ndarray) -> Optional[pd.DataFrame]:
        pq, pr = self._paths(tag, bi)
        if not (os.path.exists(pq) and os.path.exists(pr)):
            return None
        if not np.array_equal(np.load(pr), np.asarray(q_rid, dtype=ID_DTYPE)):
            raise RuntimeError(f"cand-cache {tag} block {bi}: S1 ids differ from the cached block (block size / sample changed?)")
        u = pd.read_parquet(pq)
        u["q"] = u["q"].astype(np.int64)
        c = u["c"].to_numpy().astype(np.int64)
        m = self._drid_map.get(tag)
        if m is not None:
            c = m[c]
            ok = c >= 0
            if not ok.all():
                u, c = u[ok].reset_index(drop=True), c[ok]
        u["c"] = c
        return u

    def save(self, tag: str, bi: int, q_rid: np.ndarray, union: pd.DataFrame) -> None:
        pq, pr = self._paths(tag, bi)
        if self._drid_map.get(tag) is not None:
            return  # built on another store layout; never overwrite the reference cache
        out = union.copy()
        out["q"] = out["q"].astype(np.int32)
        out["c"] = out["c"].astype(np.int32)
        out.to_parquet(pq + ".tmp", index=False)
        np.save(pr, np.asarray(q_rid, dtype=ID_DTYPE))
        os.replace(pq + ".tmp", pq)


# --------------------------------------------------------------- block step
class CountryContext:
    """Doc matrices (+ dense embeddings) of ONE country's S2/S3 records, plus candidates + stage A per query block."""

    def __init__(self, tag, q, d, vecs, views, dense_model, cache_dir, n_jobs, device, q_emb_all=None):
        self.tag, self.q, self.d, self.views = tag, q, d, views
        self.d_src = d["src"].to_numpy().astype(np.int8)
        self.d_rid = d["rid"].to_numpy(dtype=object)  # numpy: 1600x faster than ArrowStringArray in per-S1 loops
        self.vecs, self.n_jobs, self.device = vecs, n_jobs, device
        t = time.time()
        self.d_mats = {v: transform(vecs[v], view_text(d, v), n_jobs) for v in views}
        if "n_full" in d:  # built on the fly by view_text; not kept in RAM (~80 MB per 1M docs)
            del d["n_full"]
        log(f"{tag}: doc matrices {[(v, m.shape, m.nnz) for v, m in self.d_mats.items()]} ({time.time() - t:.0f}s)")
        self.d_emb = self.q_emb_all = None
        if dense_model is not None:
            from .dense import dense_text, encode
            fname = "".join(ch if ch.isalnum() else "_" for ch in tag)
            self.d_emb = encode(dense_text(d), dense_model, cache_path=os.path.join(cache_dir, f"{fname}_docs_emb.npy"))
            self.q_emb_all = q_emb_all if q_emb_all is not None else encode(
                dense_text(q), dense_model, cache_path=os.path.join(cache_dir, f"{fname}_s1_emb.npy"))
            log(f"{tag}: dense embeddings ready (docs {self.d_emb.shape}, S1 {self.q_emb_all.shape})")

    def block(self, q_idx: np.ndarray, k: int, verbose=False, cache: Optional[CandCache] = None, bi: int = 0,
              union_only: bool = False):
        """Candidates + stage A for one query block. Returns (cands, q_block, q_emb).
        cache: load the pre-cascade union from the candidate cache (top-k skipped) or fill it after top-k.
        union_only: return the union without stage A (cache building)."""
        qb = self.q.iloc[q_idx].reset_index(drop=True)
        q_mats = {v: transform(self.vecs[v], view_text(qb, v), self.n_jobs) for v in self.views}
        q_emb = None
        views = list(self.views)
        if self.d_emb is not None:
            q_emb = np.asarray(self.q_emb_all[q_idx])
            views.append("dense")
        q_rid = qb["rid"].to_numpy(dtype=object)
        cands = cache.load(self.tag, bi, q_rid) if cache is not None else None
        if cands is None:
            parts = lexical_candidates(q_mats, self.d_mats, self.d_src, k, self.views, verbose=verbose,
                                       n_threads=self.n_jobs)  # shared box: 20 OpenMP threads under load only add contention
            if self.d_emb is not None:
                from .dense import dense_candidates
                parts += dense_candidates(q_emb, self.d_emb, self.d_src, k, device=self.device, verbose=verbose)
            cands = union_candidates(parts, views)
            del parts
            if cache is not None:
                cache.save(self.tag, bi, q_rid, cands)
        elif verbose:
            log(f"{self.tag} block {bi}: union loaded from the candidate cache ({len(cands)} pairs, top-k skipped)")
        if union_only:
            return cands, qb, q_emb
        cands = stage_a(cands, self.d_src, q_mats, self.d_mats, self.views, q_emb, self.d_emb)
        return cands, qb, q_emb


def need_cols(dense: bool) -> List[str]:
    """Columns loaded per country: everything in the store except n_full (rebuilt transiently by view_text)."""
    return [c for c in KEEP_COLS if c != "n_full"] + (RAW_COLS if dense else [])


def load_train_country(meta, c, cols, n_c, seed, dense_model, cache_dir):
    """One train country: (sampled S1 table, doc table, sample positions, S1 embeddings of the sample or None)."""
    rec = load_country(meta, c, cols)
    q_all, d = split_tables(rec)
    del rec
    q_emb_all = None
    if n_c < len(q_all):
        q = q_all.sample(n_c, random_state=seed)
        sel = q.index.to_numpy()
        q = q.reset_index(drop=True)
    else:
        q, sel = q_all, None
    if dense_model is not None:
        from .dense import dense_text, encode
        fname = "".join(ch if ch.isalnum() else "_" for ch in f"train/{c}")
        emb = encode(dense_text(q_all), dense_model, cache_path=os.path.join(cache_dir, f"{fname}_s1_emb.npy"))
        q_emb_all = np.asarray(emb) if sel is None else np.asarray(emb)[sel]
        del emb
    del q_all
    return q, d, sel, q_emb_all


# ---------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--k", type=int, default=10, help="top-k per view per source")
    ap.add_argument("--views", default="name_c3,name_w,addr_c3,name_ph,full_w,addr_w",
                    help="comma list of lexical views (full_w = name+address words is the strongest single view; "
                         "full_c3 is the costliest and off by default)")
    ap.add_argument("--max-df", type=float, default=0.05, help="drop n-grams present in more than this share of records")
    ap.add_argument("--dense", action="store_true", help="add bi-encoder dense view (needs sentence-transformers)")
    ap.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--block-size", type=int, default=100_000, help="S1 queries per block (train and test)")
    ap.add_argument("--train-s1", type=int, default=150_000, help="S1 entities used for training, split across countries (0 = all)")
    ap.add_argument("--neg-rate", type=float, default=0.5, help="keep this share of negative pairs when FITTING (weighted 1/rate); OOF/rules use all pairs")
    ap.add_argument("--test-limit", type=int, default=0, help="only predict the first N test S1 rows (dev)")
    ap.add_argument("--cascade-top", type=int, default=10, help="cascade: keep at most this many candidates per S1")
    ap.add_argument("--cascade-floor", type=float, default=0.002, help="cascade: drop candidates below this probability")
    ap.add_argument("--no-cascade", action="store_true", help="fall back to the per-country score-threshold prune")
    ap.add_argument("--keep-recall", type=float, default=0.998, help="(--no-cascade only) prune keeps this pair recall")
    ap.add_argument("--vec-sample", type=int, default=2_500_000, help="rows of text the vectorisers are fitted on")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=0)
    ap.add_argument("--topk-device", default="auto", help="auto | cuda | cpu: where the sparse TF-IDF top-k runs "
                    "(cuda: exact same cosines via cuSPARSE, ~4x faster than 8 CPU threads on this box)")
    ap.add_argument("--stage-b-jobs", type=int, default=0, help="workers for stage B (default: n_jobs // 2; each worker ~400 MB)")
    ap.add_argument("--rule", default="auto", help="auto | thr | expf")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--no-one2one", action="store_true")
    ap.add_argument("--test-gt", default="", help="hidden test GT to score (synthetic runs only)")
    ap.add_argument("--run-name", default="")
    ap.add_argument("--load-model", default=None, help="path of a model.joblib from a previous run with the same --views/--seed/--vec-sample: skip training and go straight to the test pass")
    ap.add_argument("--skip-test", action="store_true")
    ap.add_argument("--cand-cache", default="", help="fast lane: dir of cached pre-cascade candidate unions per block "
                    "(filled on the first run, top-k skipped on later runs with the same views/k/block size/train sample)")
    ap.add_argument("--cands-only", action="store_true", help="only build the --cand-cache (train sample + test), then stop")
    ap.add_argument("--vec-cache", default="", help="joblib file of the fitted vectorisers (loaded if present, else fitted + saved)")
    ap.add_argument("--save-probs", action="store_true", help="write per-pair test probabilities to <out>/test_probs_<country>.parquet")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    from . import blocking as _blocking
    _blocking.TOPK_DEVICE["device"] = a.topk_device
    sb_jobs = a.stage_b_jobs or max(1, (a.n_jobs or (os.cpu_count() or 1)) // 2)
    log(f"top-k device: {a.topk_device} (cuda available: {_blocking._cuda_ok()}) | stage-B workers: {sb_jobs}")
    views = [v for v in a.views.split(",") if v]
    use_cascade = not a.no_cascade
    lgb_jobs = a.n_jobs or -1  # shared box: LightGBM with all 20 threads spin-waits itself to a crawl under load
    report = dict(args=vars(a), env=dict(python=platform.python_version(), machine=platform.node()), peak_rss_mb={})
    device = None
    dense_model = None
    if a.dense:
        from .dense import get_device, load_encoder
        device = get_device()
        dense_model = load_encoder(a.dense_model, device)
        log(f"dense model {a.dense_model} on {device}")
    cols = need_cols(a.dense)

    # 1. per-country stores (built once; a rerun with the same --cache-dir skips normalisation)
    meta_te = ensure_store(a.data_dir, "test", a.cache_dir, a.n_jobs, keep_raw=a.dense)
    release_memory()
    meta_tr = ensure_store(a.data_dir, "train", a.cache_dir, a.n_jobs, keep_raw=a.dense)
    release_memory()
    report["peak_rss_mb"]["store"] = peak_rss_mb()
    gt_pairs = load_gt_pairs(meta_tr)
    n_multi = int((gt_pairs["m"].value_counts() > 1).sum())
    one2one = (not a.no_one2one) and n_multi == 0
    log(f"GT: {meta_tr['gt']['n_s1']} S1, {len(gt_pairs)} pairs; S2/S3 ids linked to >1 S1: {n_multi} -> "
        f"one-to-one filter {'ON' if one2one else 'OFF'}")
    report["gt"] = dict(n_s1=meta_tr["gt"]["n_s1"], singleton_rate=meta_tr["gt"]["singleton_rate"], multi_s1_ids=n_multi)
    header = tuple(meta_tr["gt"].get("header") or ("source1_id", "matched_ids"))

    # 2. vectorisers on a text sample across splits + countries (all sources; no labels)
    vecs = None
    vec_key = dict(views=views, max_df=a.max_df, seed=a.seed, vec_sample=a.vec_sample, store=os.path.abspath(a.cache_dir))
    if a.vec_cache and os.path.exists(a.vec_cache):
        import joblib
        vc = joblib.load(a.vec_cache)
        if vc.get("key") == vec_key:
            vecs = vc["vecs"]
            log(f"vectorisers loaded from {a.vec_cache}")
        else:
            log(f"vec-cache {a.vec_cache} was fitted with {vc.get('key')} != {vec_key} -> refitting")
    if vecs is None:
        sample = text_sample([meta_tr, meta_te], a.vec_sample, a.seed)
        vecs = fit_vectorizers(sample, views, max_df=a.max_df, seed=a.seed)
        log(f"vectorisers (fit on {len(sample)} rows): {[(v, len(vec.vocabulary_)) for v, vec in vecs.items()]}")
        del sample
        if a.vec_cache:
            import joblib
            joblib.dump(dict(key=vec_key, vecs=vecs), a.vec_cache)
    release_memory()
    cache = CandCache(a.cand_cache) if a.cand_cache else None

    if a.cands_only:
        if cache is None:
            raise SystemExit("--cands-only needs --cand-cache")
        alloc = allocate({c: meta_tr["countries"][c]["n_s1"] for c in sorted(meta_tr["countries"])}, a.train_s1)
        jobs = [("train", c) for c in sorted(meta_tr["countries"]) if alloc.get(c, 0) > 0] + \
               [("test", c) for c in sorted(meta_te["countries"])]
        for split, c in jobs:
            t_c = time.time()
            if split == "train":
                q, d, _, q_emb_all = load_train_country(meta_tr, c, cols, alloc[c], a.seed, dense_model, a.cache_dir)
            else:
                rec = load_country(meta_te, c, cols)
                q, d = split_tables(rec)
                del rec
                q_emb_all = None
                if a.test_limit:
                    q = q[q["pos"].to_numpy() < a.test_limit].reset_index(drop=True)
            ctx = CountryContext(f"{split}/{c}", q, d, vecs, views, dense_model, a.cache_dir, a.n_jobs, device, q_emb_all)
            ctx.d = None
            del d
            cache.ensure_docs(ctx.tag, ctx.d_rid)
            for bi, q_idx in enumerate(blocks(len(q), a.block_size)):
                u, _, _ = ctx.block(q_idx, a.k, verbose=(bi == 0), cache=cache, bi=bi, union_only=True)
                log(f"{split}/{c} block {bi}: {len(u)} union pairs cached")
                del u
            del ctx, q
            release_memory()
            log(f"{split}/{c}: cand cache done in {time.time() - t_c:.0f}s | peak RSS so far {peak_rss_mb()} MB")
        log("cands-only: done")
        return

    def _train():
        nonlocal gt_pairs
        # 3a. TRAIN phase A: blocking + stage A per country -> stage-A memmap + labels
        countries_tr = sorted(meta_tr["countries"])
        alloc = allocate({c: meta_tr["countries"][c]["n_s1"] for c in countries_tr}, a.train_s1)
        log(f"train S1 sample per country: {alloc}")
        xa_path = os.path.join(a.out_dir, "train_Xa.f32")
        xa_fh = open(xa_path, "wb")
        a_cols: Optional[List[str]] = None
        n_pairs_a = 0
        y_parts, qg_parts, cl_parts, qrid_parts = [], [], [], []
        rec_curves, gp_total = [], 0
        country_rows: Dict[str, tuple] = {}   # country -> (first pair row, last pair row) in the memmaps
        country_q: Dict[str, tuple] = {}      # country -> (q offset, n sampled S1, sample positions)
        offset = 0
        rng = np.random.default_rng(a.seed)
        for c in countries_tr:
            n_c = alloc.get(c, 0)
            if n_c <= 0:
                continue
            t_c = time.time()
            q, d, sel, q_emb_all = load_train_country(meta_tr, c, cols, n_c, a.seed, dense_model, a.cache_dir)
            gt_c = gt_dict(gt_pairs, q["rid"].tolist())
            log(f"train/{c}: {len(q)} S1 sampled of {meta_tr['countries'][c]['n_s1']}, {len(d)} docs "
                f"({sum(1 for v in gt_c.values() if not v)} singletons in sample)")
            ctx = CountryContext(f"train/{c}", q, d, vecs, views, dense_model, a.cache_dir, a.n_jobs, device, q_emb_all)
            if cache is not None:
                cache.ensure_docs(ctx.tag, ctx.d_rid)
            del d  # phase A never touches the doc strings again (ctx keeps d_src/d_rid); phase B reloads the table
            ctx.d = None
            release_memory()
            row0 = n_pairs_a
            gp_c = 0
            for bi, q_idx in enumerate(blocks(len(q), a.block_size)):
                cands, qb, _ = ctx.block(q_idx, a.k, verbose=(bi == 0), cache=cache, bi=bi)
                gp = gt_pairs_block(qb["rid"].to_numpy(dtype=object), ctx.d_rid, gt_c)
                gp_c += len(gp)
                rec_curves.append(blocking_recall(cands, gp, len(qb), ks=sorted({1, 3, 5, a.k})).assign(n=len(gp)))
                cands = cands.merge(gp.assign(y=np.int8(1)), on=["q", "c"], how="left")
                cands["y"] = cands["y"].fillna(0).astype(np.int8)
                if a_cols is None:
                    a_cols = [col for col in cands.columns if col not in ("q", "c", "y")]  # original stage-A order
                    sa = set(stage_a_columns(cands.drop(columns=["y"])))
                    if sa != set(a_cols):
                        raise RuntimeError(f"stage-A column set mismatch with cascade.stage_a_columns: {sa ^ set(a_cols)}")
                    log(f"stage-A features ({len(a_cols)}): {a_cols}")
                elif [col for col in cands.columns if col not in ("q", "c", "y")] != a_cols:
                    raise RuntimeError("stage-A columns differ between blocks/countries")
                n_pairs_a += write_rows(xa_fh, cands, a_cols)
                y_parts.append(cands["y"].to_numpy().astype(np.int8))
                qg_parts.append(offset + q_idx[cands["q"].to_numpy()].astype(np.int64))
                cl_parts.append(cands["c"].to_numpy().astype(np.int64))
                log(f"train/{c} block {bi} ({len(qb)} S1): {len(cands)} cands, {int(cands.y.sum())}/{len(gp)} GT pairs found")
                del cands, qb, gp
            gp_total += gp_c
            country_rows[c] = (row0, n_pairs_a)
            country_q[c] = (offset, len(q), sel)
            qrid_parts.append(ids_bytes(q["rid"]))
            offset += len(q)
            del ctx, q, gt_c, q_emb_all
            release_memory()
            log(f"train/{c}: phase A done in {time.time() - t_c:.0f}s | peak RSS so far {peak_rss_mb()} MB")
        xa_fh.close()
        if a_cols is None:
            raise RuntimeError("no training pairs produced")
        n_q_tr = offset
        y = np.concatenate(y_parts)
        qg = np.concatenate(qg_parts)
        cl = np.concatenate(cl_parts)
        q_rid_tr = np.concatenate(qrid_parts)
        del y_parts, qg_parts, cl_parts, qrid_parts
        rc = pd.concat(rec_curves)
        rc = rc.groupby("k").apply(lambda g: pd.Series(dict(pairs=g.pairs.sum(), pairs_per_s1=g.pairs.sum() / n_q_tr,
                                                           pair_recall=(g.pair_recall * g.n).sum() / max(g.n.sum(), 1),
                                                           entity_full_recall=(g.entity_full_recall * g.n).sum() / max(g.n.sum(), 1)))
                                   ).reset_index()
        print(rc.to_string(index=False))
        report["blocking_recall"] = rc.to_dict(orient="records")
        report["peak_rss_mb"]["train_phase_a"] = peak_rss_mb()

        # 3b. cascade (or score prune) on the stage-A memmap
        Xa = np.memmap(xa_path, dtype=np.float32, mode="r", shape=(n_pairs_a, len(a_cols)))
        before = candidate_stats(qg, n_q_tr, y, gp_total)
        report["cands_before_cascade"] = before
        cascade_model = None
        thr_prune: Dict[str, float] = {}
        if use_cascade:
            t = time.time()
            pa, cascade_model = fit_cascade(Xa, y, qg, 3, a.seed, n_jobs=lgb_jobs)
            keep = cascade_keep(qg, pa, a.cascade_top, a.cascade_floor)
            log(f"cascade fitted on {n_pairs_a} pairs x {len(a_cols)} stage-A features in {time.time() - t:.0f}s")
            del pa
        else:
            keep = np.zeros(n_pairs_a, dtype=bool)
            score_col = a_cols.index("score")
            for c, (r0, r1) in country_rows.items():
                sc = np.asarray(Xa[r0:r1, score_col])
                thr_prune[c] = calibrate_prune(sc, y[r0:r1], a.keep_recall)
                keep[r0:r1] = sc >= thr_prune[c]
                log(f"train/{c}: prune at score>={thr_prune[c]:.3f} ({int(keep[r0:r1].sum())}/{r1 - r0} pairs kept)")
        after = candidate_stats(qg[keep], n_q_tr, y[keep], gp_total)
        report["cands_after_cascade"] = after
        report["prune"] = dict(method="cascade" if use_cascade else "score", per_country=thr_prune,
                               cascade_top=a.cascade_top, cascade_floor=a.cascade_floor,
                               train_pairs=int(keep.sum()), train_pair_recall=after.get("pair_recall"))
        log(f"cascade: before mean {before['mean']:.1f} / median {before['median']:.0f} per S1, "
            f"after mean {after['mean']:.1f} / median {after['median']:.0f}, train pair recall {after.get('pair_recall', float('nan')):.4f} "
            f"(p90 {after['p90']:.0f}, max {after['max']}, zero-cand share {after['zero_share']:.4f})")
        report["peak_rss_mb"]["cascade"] = peak_rss_mb()

        # 3c. TRAIN phase B: stage B on the kept pairs, per country -> full feature memmap
        feat_path = os.path.join(a.out_dir, "train_X.f32")
        feat_fh = open(feat_path, "wb")
        feats: Optional[List[str]] = None
        n_pairs_tr = 0
        yb_parts, qgb_parts, cid_parts = [], [], []
        for c in countries_tr:
            if c not in country_rows:
                continue
            t_c = time.time()
            r0, r1 = country_rows[c]
            q_off, n_c, sel = country_q[c]
            idx = r0 + np.flatnonzero(keep[r0:r1])
            q, d, sel2, _ = load_train_country(meta_tr, c, cols, n_c, a.seed, None, a.cache_dir)
            assert (sel is None and sel2 is None) or np.array_equal(sel, sel2), "train sample not reproducible"
            cand_c = pd.DataFrame(np.asarray(Xa[idx]), columns=a_cols)
            cand_c["q"] = qg[idx] - q_off
            cand_c["c"] = cl[idx]
            cand_c["y"] = y[idx]
            d_rid_c = ids_bytes(d["rid"])
            cand_c = stage_b(cand_c, q, d, sb_jobs)
            fc = feature_columns(cand_c)
            fc = [f for f in fc if f not in ("score",)] + ["score"]
            if feats is None:
                feats = fc
                log(f"features ({len(feats)}): {feats}")
            elif fc != feats:
                raise RuntimeError(f"feature columns differ between countries: {set(fc) ^ set(feats)}")
            n_pairs_tr += write_rows(feat_fh, cand_c, feats)
            yb_parts.append(cand_c["y"].to_numpy().astype(np.int8))
            qgb_parts.append(q_off + cand_c["q"].to_numpy().astype(np.int64))
            cid_parts.append(d_rid_c[cand_c["c"].to_numpy()])
            log(f"train/{c}: stage B on {len(cand_c)} kept pairs ({len(cand_c) / max(n_c, 1):.1f} per S1) in {time.time() - t_c:.0f}s")
            del cand_c, q, d, d_rid_c
            release_memory()
        feat_fh.close()
        del Xa, keep, cl
        try:
            os.remove(xa_path)
        except OSError:
            pass
        if feats is None:
            raise RuntimeError("no training pairs kept")
        y = np.concatenate(yb_parts)
        qg = np.concatenate(qgb_parts)
        cid = np.concatenate(cid_parts)
        del yb_parts, qgb_parts, cid_parts
        report["peak_rss_mb"]["train_phase_b"] = peak_rss_mb()

        # 3d. OOF + rule sweep + final model (features read back from the memmap)
        X = np.memmap(feat_path, dtype=np.float32, mode="r", shape=(n_pairs_tr, len(feats)))
        oof, kind = oof_predict(X, y, qg, a.folds, a.seed, neg_rate=a.neg_rate, n_jobs=lgb_jobs)
        from sklearn.metrics import average_precision_score, roc_auc_score
        log(f"OOF ({kind}): AUC={roc_auc_score(y, oof):.5f} AP={average_precision_score(y, oof):.5f}")
        report["peak_rss_mb"]["oof"] = peak_rss_mb()
        q_rid_str = q_rid_tr.astype(str)
        d_uniq, c_codes = np.unique(cid, return_inverse=True)
        d_rid_str = d_uniq.astype(str)
        pairs = pd.DataFrame({"q": qg, "c": c_codes.astype(np.int64), "p": oof})
        gt_tr = gt_dict(gt_pairs, q_rid_str.tolist())
        sweep_n = min(n_q_tr, 100_000)
        sw_q = np.sort(rng.choice(n_q_tr, sweep_n, replace=False))
        sw_mask = np.isin(pairs["q"].values, sw_q)
        tab, best = sweep_rules(pairs[sw_mask], q_rid_str, d_rid_str, gt_tr, one2one_ok=one2one, q_subset=sw_q)
        print(tab.head(10).to_string(index=False))
        for lbl, sub in [("best global threshold", tab[tab.rule == "thr"]), ("expected-F", tab[tab.rule == "expf"])]:
            if len(sub):
                print(f"  {lbl}: {sub.iloc[0].to_dict()}")
        if a.rule != "auto":
            cfg = tab[(tab.rule == a.rule) & (tab.one2one == one2one)]
            if a.rule == "thr":
                cfg = cfg.iloc[(cfg.thr - a.thr).abs().argsort()[:1]]
            best = cfg.iloc[0].to_dict()
        thr_best = best["thr"] if best["rule"] == "thr" else 0.0
        log(f"OOF macro F0.5 (chosen) = {best['f05']:.5f} with {best}")
        report["oof"] = dict(model=kind, auc=float(roc_auc_score(y, oof)), rules=tab.to_dict(orient="records"), chosen=best)
        print("  OOF breakdown:")
        macro_f05(gt_tr, decide(pairs, q_rid_str, d_rid_str, best["rule"], thr_best, bool(best["one2one"])), verbose=True)
        pd.DataFrame({"s1": q_rid_str[qg], "cand": d_rid_str[c_codes], "y": y, "p": oof}).to_csv(
            os.path.join(a.out_dir, "oof_pairs.tsv.gz"), sep="\t", index=False)
        del pairs, gt_tr, d_uniq, c_codes, d_rid_str, cid, sw_mask
        release_memory()
        model, _ = make_model(a.seed, n_jobs=lgb_jobs)
        fit_idx, fit_w = subsample_negatives(y, a.neg_rate, a.seed)
        model.fit(np.ascontiguousarray(X[fit_idx]), y[fit_idx], sample_weight=fit_w)
        imp = feature_importance(model, feats)
        if len(imp):
            report["importance_top25"] = imp.head(25).round(4).to_dict()
            print("  top features:", imp.head(15).round(3).to_dict())
        try:
            import joblib
            joblib.dump(dict(model=model, feats=feats, a_cols=a_cols, cascade_model=cascade_model,
                             cascade=dict(top=a.cascade_top, floor=a.cascade_floor, enabled=use_cascade),
                             thr_prune=thr_prune, best=best, views=views), os.path.join(a.out_dir, "model.joblib"))
        except Exception as e:  # noqa: BLE001
            log(f"model not saved: {e}")
        del X, y, qg, oof, fit_idx, fit_w, q_rid_tr, q_rid_str, gt_pairs
        release_memory()
        try:
            os.remove(feat_path)
        except OSError:
            pass
        report["peak_rss_mb"]["model"] = peak_rss_mb()
        log(f"model fitted | peak RSS so far {peak_rss_mb()} MB")
        return model, feats, a_cols, cascade_model, thr_prune, best, thr_best

    if a.load_model:
        import joblib
        mdl = joblib.load(a.load_model)
        if list(mdl['views']) != list(views):
            raise SystemExit(f"--load-model views {mdl['views']} != --views {views}")
        model, feats, a_cols, cascade_model = mdl['model'], mdl['feats'], mdl['a_cols'], mdl['cascade_model']
        thr_prune, best = mdl['thr_prune'], mdl['best']
        use_cascade = bool(mdl['cascade']['enabled'])
        a.cascade_top, a.cascade_floor = mdl['cascade']['top'], mdl['cascade']['floor']
        thr_best = best['thr'] if best['rule'] == 'thr' else 0.0
        report['oof'] = dict(chosen=best, loaded_from=a.load_model)
        report['loaded_model'] = a.load_model
        del gt_pairs
        release_memory()
        log(f"model loaded from {a.load_model}: {len(feats)} features, cascade={'on' if use_cascade else 'off'} "
            f"(top {a.cascade_top}, floor {a.cascade_floor}), rule {best} | training skipped")
    else:
        model, feats, a_cols, cascade_model, thr_prune, best, thr_best = _train()

    # 4. TEST pass, one country at a time, rows written as they are produced
    if a.skip_test:
        report["runtime_s"] = round(time.time() - T0, 1)
        json.dump(report, open(os.path.join(a.out_dir, "report.json"), "w"), indent=2, default=str)
        log("skip-test: done")
        return
    thr_min = min(thr_prune.values()) if thr_prune else 0.0
    countries_te = sorted(meta_te["countries"])
    fh_m = open(os.path.join(a.out_dir, "matching_results.tsv"), "w")
    fh_c = open(os.path.join(a.out_dir, "candidate_pairs.tsv"), "w")
    fh_m.write(f"{header[0]}\t{header[1]}\n")
    fh_c.write(f"{header[0]}\tcandidate_entity_ids\n")
    per_country, count_parts, test_cands = [], [], {}
    n_s1_te = n_empty = n_pred = n_pairs_te = 0
    pred_for_gt: Dict[str, frozenset] = {}
    for c in countries_te:
        t_c = time.time()
        rec = load_country(meta_te, c, cols)
        q, d = split_tables(rec)
        del rec
        if a.test_limit:
            q = q[q["pos"].to_numpy() < a.test_limit].reset_index(drop=True)
        if len(q) == 0:
            del q, d
            continue
        thr_c = None
        if not use_cascade:
            thr_c = thr_prune.get(c)
            if thr_c is None:
                thr_c = thr_min
                log(f"test/{c}: country unseen in training -> using the smallest train prune threshold {thr_c:.3f}")
        ctx = CountryContext(f"test/{c}", q, d, vecs, views, dense_model, a.cache_dir, a.n_jobs, device)
        if cache is not None:
            cache.ensure_docs(ctx.tag, ctx.d_rid)
        prob_parts = []
        c_s1 = c_empty = c_pred = c_pairs = 0
        c_counts = []
        for bi, q_idx in enumerate(blocks(len(q), a.block_size)):
            cands, qb, _ = ctx.block(q_idx, a.k, verbose=(bi == 0), cache=cache, bi=bi)
            n_before = len(cands)
            if use_cascade:
                pa = predict_chunked(cascade_model, cands, a_cols)
                keep_b = cascade_keep(cands["q"].to_numpy(), pa, a.cascade_top, a.cascade_floor)
                del pa
            else:
                keep_b = prune_score(cands) >= thr_c
            cands = cands[keep_b].reset_index(drop=True)
            del keep_b
            counts = np.bincount(cands["q"].to_numpy(), minlength=len(qb))
            c_counts.append(counts)
            cands = stage_b(cands, qb, d, sb_jobs)
            cands["p"] = predict_chunked(model, cands, feats)
            qb_rid = qb["rid"].to_numpy(dtype=object)
            if a.save_probs:
                prob_parts.append(pd.DataFrame({"s1": qb_rid[cands["q"].to_numpy()], "cand": ctx.d_rid[cands["c"].to_numpy()],
                                                "p": cands["p"].to_numpy(), "score": cands["score"].to_numpy()}))
            sets = decide(cands, qb_rid, ctx.d_rid, best["rule"], thr_best, bool(best["one2one"]))
            cand_rows = [""] * len(qb)
            for qq, grp in cands.groupby("q")["c"]:
                cand_rows[qq] = ",".join(ctx.d_rid[grp.values])
            for i, r in enumerate(qb_rid):
                m = sets[r]
                fh_m.write(f"{r}\t{','.join(sorted(m))}\n")
                fh_c.write(f"{r}\t{cand_rows[i]}\n")
                c_pred += len(m)
                c_empty += not m
                if a.test_gt:
                    pred_for_gt[r] = m
            c_s1 += len(qb)
            c_pairs += len(cands)
            log(f"test/{c} block {bi} ({len(qb)} S1): {n_before} cands -> {len(cands)} kept ({len(cands) / len(qb):.1f}/S1), "
                f"{sum(1 for r in qb_rid if sets[r])} S1 with matches")
            del cands, sets, cand_rows, qb
        fh_m.flush()
        fh_c.flush()
        if prob_parts:
            pd.concat(prob_parts, ignore_index=True).to_parquet(os.path.join(a.out_dir, f"test_probs_{c}.parquet"), index=False)
        del prob_parts
        counts_c = np.concatenate(c_counts)
        count_parts.append(counts_c)
        test_cands[c] = stats_from_counts(counts_c)
        per_country.append(dict(country=c, n_s1=c_s1, empty_rate=c_empty / max(c_s1, 1), mean_matches=c_pred / max(c_s1, 1),
                                cand_mean=test_cands[c]["mean"], cand_median=test_cands[c]["median"],
                                prune_threshold=thr_c if thr_c is not None else np.nan))
        n_s1_te += c_s1
        n_empty += c_empty
        n_pred += c_pred
        n_pairs_te += c_pairs
        del ctx, q, d
        release_memory()
        log(f"test/{c}: done in {time.time() - t_c:.0f}s | peak RSS so far {peak_rss_mb()} MB")
    fh_m.close()
    fh_c.close()
    per = pd.DataFrame(per_country).set_index("country")
    print(per.to_string())
    report["test_pred_by_country"] = per.reset_index().to_dict(orient="records")
    all_counts = np.concatenate(count_parts) if count_parts else np.zeros(0, np.int64)
    report["test_candidates"] = dict(overall=stats_from_counts(all_counts), per_country=test_cands)
    report["test_pred"] = dict(n_s1=n_s1_te, empty_rate=n_empty / max(n_s1_te, 1), mean_matches=n_pred / max(n_s1_te, 1),
                               cand_pairs=n_pairs_te, cand_pairs_per_s1=n_pairs_te / max(n_s1_te, 1))
    ov = report["test_candidates"]["overall"]
    log(f"test: {n_s1_te} S1 rows written | predicted-empty={n_empty / max(n_s1_te, 1):.1%} | "
        f"mean matches={n_pred / max(n_s1_te, 1):.2f} | candidates/S1 mean {ov['mean']:.1f} median {ov['median']:.0f} "
        f"p90 {ov['p90']:.0f} max {ov['max']}")
    if a.test_gt:
        tgt = read_matches_tsv(a.test_gt)
        tgt = {r: v for r, v in tgt.items() if r in pred_for_gt}
        report["test_f05_hidden"] = macro_f05(tgt, pred_for_gt, verbose=True)
    report["peak_rss_mb"]["test"] = peak_rss_mb()
    report["runtime_s"] = round(time.time() - T0, 1)
    with open(os.path.join(a.out_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    if a.run_name:
        import shutil
        rd = os.path.join("runs", a.run_name)
        os.makedirs(rd, exist_ok=True)
        for f in ("report.json", "matching_results.tsv"):
            shutil.copy(os.path.join(a.out_dir, f), os.path.join(rd, f))
        log(f"copied report + matching_results to {rd}/ (commit report.json + stdout.txt, not the tsv)")
    log(f"done | peak RSS {peak_rss_mb()} MB")


if __name__ == "__main__":
    main()
