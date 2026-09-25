"""End-to-end pipeline, per country and block-wise so it fits the shared 15 GB box (~5 GB for us).

Flow
  1. build/load the per-country store (src/store.py): normalised records of each split as one parquet per
     country; ground truth as a (s1, m) pair table. Only one country is in RAM at a time.
  2. fit TF-IDF vectorisers on a text sample drawn across splits/countries (~2.5M rows)
  3. TRAIN: for each country: sample S1 (--train-s1 split proportionally to the countries' S1 counts) ->
     blocks -> candidates -> stage A -> labels -> prune threshold calibrated PER COUNTRY (--keep-recall) ->
     stage B; features go to a float32 memmap, ids are kept as fixed-width bytes. Then GroupKFold OOF,
     rule sweep on OOF, final model.
  4. TEST: for each country (an unseen country uses the smallest train prune threshold): blocks -> candidates
     -> stage A -> prune -> stage B -> predict -> decide -> rows appended to matching_results.tsv +
     candidate_pairs.tsv as they are produced.
Modelling logic (views, features, model, decision rule) is unchanged from the previous single-table version;
group/rank context features are computed WITHIN a block (block size is the same for train and test).

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
from .features import calibrate_prune, feature_columns, prune_score, stage_a, stage_b
from .metric import macro_f05, read_matches_tsv
from .model import decide, feature_importance, make_model, oof_predict, subsample_negatives, sweep_rules
from .store import (KEEP_COLS, RAW_COLS, allocate, ensure_store, gt_dict, load_country, load_gt_pairs,
                    text_sample)

T0 = time.time()
ID_DTYPE = "S16"  # fixed-width bytes for record ids (S1-xxxxxxxx / S2-xxxxxxxxx fit easily)


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


# --------------------------------------------------------------- block step
class CountryContext:
    """Doc matrices (+ dense embeddings) of ONE country's S2/S3 records, plus candidates + stage A per query block."""

    def __init__(self, tag, q, d, vecs, views, dense_model, cache_dir, n_jobs, device, q_emb_all=None):
        self.tag, self.q, self.d, self.views = tag, q, d, views
        self.d_src = d["src"].to_numpy().astype(np.int8)
        self.d_rid = d["rid"].values
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

    def block(self, q_idx: np.ndarray, k: int, verbose=False):
        """Candidates + stage A for one query block. Returns (cands, q_block, q_emb)."""
        qb = self.q.iloc[q_idx].reset_index(drop=True)
        q_mats = {v: transform(self.vecs[v], view_text(qb, v), self.n_jobs) for v in self.views}
        parts = lexical_candidates(q_mats, self.d_mats, self.d_src, k, self.views, verbose=verbose)
        q_emb = None
        views = list(self.views)
        if self.d_emb is not None:
            from .dense import dense_candidates
            q_emb = np.asarray(self.q_emb_all[q_idx])
            parts += dense_candidates(q_emb, self.d_emb, self.d_src, k, device=self.device, verbose=verbose)
            views.append("dense")
        cands = union_candidates(parts, views)
        del parts
        cands = stage_a(cands, self.d_src, q_mats, self.d_mats, self.views, q_emb, self.d_emb)
        return cands, qb, q_emb


def need_cols(dense: bool) -> List[str]:
    """Columns loaded per country: everything in the store except n_full (rebuilt transiently by view_text)."""
    return [c for c in KEEP_COLS if c != "n_full"] + (RAW_COLS if dense else [])


def predict_chunked(model, cands: pd.DataFrame, feats: List[str], chunk: int = 1_000_000) -> np.ndarray:
    out = np.empty(len(cands), dtype=np.float32)
    for st in range(0, len(cands), chunk):
        out[st:st + chunk] = model.predict_proba(cands[feats].iloc[st:st + chunk].to_numpy(dtype=np.float32))[:, 1]
    return out


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
    ap.add_argument("--keep-recall", type=float, default=0.998)
    ap.add_argument("--vec-sample", type=int, default=2_500_000, help="rows of text the vectorisers are fitted on")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=0)
    ap.add_argument("--rule", default="auto", help="auto | thr | expf")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--no-one2one", action="store_true")
    ap.add_argument("--test-gt", default="", help="hidden test GT to score (synthetic runs only)")
    ap.add_argument("--run-name", default="")
    ap.add_argument("--skip-test", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    views = [v for v in a.views.split(",") if v]
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
    sample = text_sample([meta_tr, meta_te], a.vec_sample, a.seed)
    vecs = fit_vectorizers(sample, views, max_df=a.max_df, seed=a.seed)
    log(f"vectorisers (fit on {len(sample)} rows): {[(v, len(vec.vocabulary_)) for v, vec in vecs.items()]}")
    del sample
    release_memory()

    # 3. TRAIN pass, one country at a time
    countries_tr = sorted(meta_tr["countries"])
    alloc = allocate({c: meta_tr["countries"][c]["n_s1"] for c in countries_tr}, a.train_s1)
    log(f"train S1 sample per country: {alloc}")
    all_views = views + (["dense"] if a.dense else [])
    feat_path = os.path.join(a.out_dir, "train_X.f32")
    feat_fh = open(feat_path, "wb")
    feats: Optional[List[str]] = None
    n_pairs_tr = 0
    y_parts, qg_parts, cid_parts, qrid_parts = [], [], [], []
    rec_curves, thr_prune, prune_info = [], {}, {}
    gp_total = 0
    offset = 0
    rng = np.random.default_rng(a.seed)
    for c in countries_tr:
        n_c = alloc.get(c, 0)
        if n_c <= 0:
            continue
        t_c = time.time()
        rec = load_country(meta_tr, c, cols)
        q_all, d = split_tables(rec)
        del rec
        q_emb_all = None
        if n_c < len(q_all):
            q = q_all.sample(n_c, random_state=a.seed)
            sel = q.index.to_numpy()
            q = q.reset_index(drop=True)
        else:
            q, sel = q_all, None
        if dense_model is not None:
            from .dense import dense_text, encode
            fname = "".join(ch if ch.isalnum() else "_" for ch in f"train/{c}")
            emb = encode(dense_text(q_all), dense_model, cache_path=os.path.join(a.cache_dir, f"{fname}_s1_emb.npy"))
            q_emb_all = np.asarray(emb) if sel is None else np.asarray(emb)[sel]
            del emb
        del q_all
        gt_c = gt_dict(gt_pairs, q["rid"].tolist())
        log(f"train/{c}: {len(q)} S1 sampled of {meta_tr['countries'][c]['n_s1']}, {len(d)} docs "
            f"({sum(1 for v in gt_c.values() if not v)} singletons in sample)")
        ctx = CountryContext(f"train/{c}", q, d, vecs, views, dense_model, a.cache_dir, a.n_jobs, device, q_emb_all)
        cand_parts, gp_c = [], 0
        for bi, q_idx in enumerate(blocks(len(q), a.block_size)):
            cands, qb, _ = ctx.block(q_idx, a.k, verbose=(bi == 0))
            gp = gt_pairs_block(qb["rid"].values, ctx.d_rid, gt_c)
            gp_c += len(gp)
            rec_curves.append(blocking_recall(cands, gp, len(qb), ks=sorted({1, 3, 5, a.k})).assign(n=len(gp)))
            cands = cands.merge(gp.assign(y=np.int8(1)), on=["q", "c"], how="left")
            cands["y"] = cands["y"].fillna(0).astype(np.int8)
            cands["q"] = q_idx[cands["q"].values]  # back to row indices of q
            cand_parts.append(cands)
            log(f"train/{c} block {bi} ({len(qb)} S1): {len(cands)} cands, {int(cands.y.sum())}/{len(gp)} GT pairs found")
        gp_total += gp_c
        cand_c = pd.concat(cand_parts, ignore_index=True) if len(cand_parts) > 1 else cand_parts[0]
        del cand_parts
        thr_c = calibrate_prune(prune_score(cand_c), cand_c["y"].values, a.keep_recall)
        n0 = len(cand_c)
        cand_c = cand_c[prune_score(cand_c) >= thr_c].reset_index(drop=True)
        thr_prune[c] = thr_c
        prune_info[c] = dict(threshold=thr_c, pairs_before=n0, pairs_after=len(cand_c),
                             pair_recall=float(cand_c.y.sum() / max(gp_c, 1)))
        log(f"train/{c}: prune at score>={thr_c:.3f}: {n0}->{len(cand_c)}; pair recall after pruning "
            f"{cand_c.y.sum() / max(gp_c, 1):.4f}")
        d_rid_c = ids_bytes(d["rid"])
        del ctx  # doc matrices / embeddings are not needed for stage B
        release_memory()
        cand_c = stage_b(cand_c, q, d, a.n_jobs)
        fc = feature_columns(cand_c)
        fc = [f for f in fc if f not in ("score",)] + ["score"]
        if feats is None:
            feats = fc
            log(f"features ({len(feats)}): {feats}")
        elif fc != feats:
            raise RuntimeError(f"feature columns differ between countries: {set(fc) ^ set(feats)}")
        for st in range(0, len(cand_c), 1_000_000):  # float32 rows -> memmap file, 1M rows at a time
            feat_fh.write(np.ascontiguousarray(cand_c[feats].iloc[st:st + 1_000_000].to_numpy(dtype=np.float32)).tobytes())
        n_pairs_tr += len(cand_c)
        y_parts.append(cand_c["y"].to_numpy().astype(np.int8))
        qg_parts.append(offset + cand_c["q"].to_numpy().astype(np.int64))
        cid_parts.append(d_rid_c[cand_c["c"].to_numpy()])
        qrid_parts.append(ids_bytes(q["rid"]))
        offset += len(q)
        del cand_c, q, d, d_rid_c, gt_c
        release_memory()
        log(f"train/{c}: done in {time.time() - t_c:.0f}s | peak RSS so far {peak_rss_mb()} MB")
    feat_fh.close()
    report["peak_rss_mb"]["train_candidates"] = peak_rss_mb()
    if feats is None:
        raise RuntimeError("no training pairs produced")
    n_q_tr = offset
    rc = pd.concat(rec_curves)
    rc = rc.groupby("k").apply(lambda g: pd.Series(dict(pairs=g.pairs.sum(), pairs_per_s1=g.pairs.sum() / n_q_tr,
                                                       pair_recall=(g.pair_recall * g.n).sum() / max(g.n.sum(), 1),
                                                       entity_full_recall=(g.entity_full_recall * g.n).sum() / max(g.n.sum(), 1)))
                               ).reset_index()
    print(rc.to_string(index=False))
    report["blocking_recall"] = rc.to_dict(orient="records")
    report["prune"] = dict(per_country=prune_info, train_pairs=n_pairs_tr,
                           train_pair_recall=float(sum(p.sum() for p in y_parts) / max(gp_total, 1)))
    log(f"prune thresholds per country: { {c: round(t, 4) for c, t in thr_prune.items()} }; train pairs {n_pairs_tr}; "
        f"pair recall after pruning {report['prune']['train_pair_recall']:.4f}")

    # 3b. OOF + rule sweep + final model (features read back from the memmap)
    X = np.memmap(feat_path, dtype=np.float32, mode="r", shape=(n_pairs_tr, len(feats)))
    y = np.concatenate(y_parts)
    qg = np.concatenate(qg_parts)
    cid = np.concatenate(cid_parts)
    q_rid_tr = np.concatenate(qrid_parts)
    del y_parts, qg_parts, cid_parts, qrid_parts
    lgb_jobs = a.n_jobs or -1  # shared box: LightGBM with all 20 threads spin-waits itself to a crawl under load
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
        joblib.dump(dict(model=model, feats=feats, thr_prune=thr_prune, best=best, views=views), os.path.join(a.out_dir, "model.joblib"))
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

    # 4. TEST pass, one country at a time, rows written as they are produced
    if a.skip_test:
        report["runtime_s"] = round(time.time() - T0, 1)
        json.dump(report, open(os.path.join(a.out_dir, "report.json"), "w"), indent=2, default=str)
        log("skip-test: done")
        return
    thr_min = min(thr_prune.values())
    countries_te = sorted(meta_te["countries"])
    fh_m = open(os.path.join(a.out_dir, "matching_results.tsv"), "w")
    fh_c = open(os.path.join(a.out_dir, "candidate_pairs.tsv"), "w")
    fh_m.write(f"{header[0]}\t{header[1]}\n")
    fh_c.write(f"{header[0]}\tcandidate_entity_ids\n")
    per_country = []
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
        thr_c = thr_prune.get(c)
        if thr_c is None:
            thr_c = thr_min
            log(f"test/{c}: country unseen in training -> using the smallest train prune threshold {thr_c:.3f}")
        ctx = CountryContext(f"test/{c}", q, d, vecs, views, dense_model, a.cache_dir, a.n_jobs, device)
        c_s1 = c_empty = c_pred = c_pairs = 0
        for bi, q_idx in enumerate(blocks(len(q), a.block_size)):
            cands, qb, _ = ctx.block(q_idx, a.k, verbose=(bi == 0))
            cands = cands[prune_score(cands) >= thr_c].reset_index(drop=True)
            cands = stage_b(cands, qb, d, a.n_jobs)
            cands["p"] = predict_chunked(model, cands, feats)
            qb_rid = qb["rid"].values
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
            log(f"test/{c} block {bi} ({len(qb)} S1): {len(cands)} cands after prune, "
                f"{sum(1 for r in qb_rid if sets[r])} S1 with matches")
            del cands, sets, cand_rows, qb
        fh_m.flush()
        fh_c.flush()
        per_country.append(dict(country=c, n_s1=c_s1, empty_rate=c_empty / max(c_s1, 1), mean_matches=c_pred / max(c_s1, 1),
                                cand_pairs_per_s1=c_pairs / max(c_s1, 1), prune_threshold=thr_c))
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
    report["test_pred"] = dict(n_s1=n_s1_te, empty_rate=n_empty / max(n_s1_te, 1), mean_matches=n_pred / max(n_s1_te, 1),
                               cand_pairs=n_pairs_te, cand_pairs_per_s1=n_pairs_te / max(n_s1_te, 1))
    log(f"test: {n_s1_te} S1 rows written | predicted-empty={n_empty / max(n_s1_te, 1):.1%} | "
        f"mean matches={n_pred / max(n_s1_te, 1):.2f}")
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
