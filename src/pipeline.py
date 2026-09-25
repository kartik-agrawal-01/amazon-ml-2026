"""End-to-end pipeline, block-wise so it scales to ~12M records on a 10 GB box.

Flow
  1. load + normalise each split (cached to cache/<split>_norm.pkl)
  2. fit TF-IDF vectorisers on a text sample; build DOC matrices (S2+S3) per split
  3. optional dense view: encode all records with a bi-encoder (GPU), cached as .npy
  4. TRAIN: stream S1 query blocks (a sample of --train-s1 entities) -> candidates -> stage A
     -> keep everything; calibrate pruning on train; stage B; 5-fold GroupKFold OOF;
     pick the decision rule on OOF; fit final model
  5. TEST: stream all S1 blocks -> candidates -> stage A -> prune -> stage B -> predict -> decide
     -> write matching_results.tsv + candidate_pairs.tsv incrementally
Group/rank context features are computed WITHIN a block (block size is the same for train and test).

Run from the repo root:
  python -m src.pipeline --data-dir data --out-dir output --run-name v1 [--dense] [--partition-col city]
Quick dev loop:  --train-s1 50000 --test-limit 50000
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import platform
import time

import numpy as np
import pandas as pd

from .blocking import (DEFAULT_VIEWS, blocking_recall, fit_vectorizers, gt_pairs_block, lexical_candidates,
                       transform, union_candidates, view_text)
from .data import describe, load_split
from .features import calibrate_prune, feature_columns, prune_score, stage_a, stage_b
from .metric import macro_f05, read_matches_tsv
from .model import check_one_to_one, decide, feature_importance, make_model, oof_predict, subsample_negatives, sweep_rules
from .normalize import add_normalized

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ------------------------------------------------------------------ loading
def add_uniqueness(rec: pd.DataFrame) -> pd.DataFrame:
    """Global (per split, per country) duplicate counts: how many S1 records share this core name / phonetic
    key / no-space name, and how many records of any source share this address. Resolves empty-address and
    trade-name candidates: a name that matches exactly ONE S1 entity is a near-certain match."""
    key_ctry = rec["country"].fillna("") if "country" in rec else pd.Series("", index=rec.index)
    s1 = rec["src"] == 1
    for col, name in (("n_core", "core"), ("n_ph", "ph"), ("n_nospace", "nsp")):
        k = key_ctry + "|" + rec[col].fillna("")
        cnt_s1 = k[s1].value_counts()
        rec[f"cnt_{name}_s1"] = k.map(cnt_s1).fillna(0).astype(np.int32)
        rec[f"cnt_{name}_all"] = k.map(k.value_counts()).astype(np.int32)
    ka = key_ctry + "|" + rec["n_addr"].fillna("")
    rec["cnt_addr_all"] = ka.map(ka.value_counts()).astype(np.int32)
    rec.loc[rec["n_addr"].fillna("") == "", "cnt_addr_all"] = 0
    rec["cnt_addr_s1"] = ka.map(ka[s1].value_counts()).fillna(0).astype(np.int32)
    return rec


def load_norm(data_dir: str, split: str, cache_dir: str, n_jobs: int):
    path = os.path.join(cache_dir, f"{split}_norm.pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            rec, gt, info = pickle.load(fh)
        log(f"{split}: loaded normalised cache {path} ({len(rec)} rows)")
        if "cnt_core_s1" not in rec:
            rec = add_uniqueness(rec)
        return rec, gt, info
    rec, gt, info = load_split(data_dir, split)
    describe(rec, gt, info, split)
    rec = add_normalized(rec, n_jobs=n_jobs)
    rec["n_full"] = (rec["n_core"] + " " + rec["n_addr"]).str.strip()
    rec = add_uniqueness(rec)
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump((rec, gt, info), fh, protocol=pickle.HIGHEST_PROTOCOL)
    log(f"{split}: normalised {len(rec)} rows -> cached {path}")
    return rec, gt, info


def release_memory() -> None:
    """Hand freed heap back to the OS after dropping a big table (glibc keeps fragmented arenas otherwise;
    the box only has ~10 GB for us, shared with other users)."""
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
    src = rec["src"].to_numpy()
    n1 = int((src == 1).sum())
    if n1 and np.all(src[:n1] == 1) and (n1 == len(src) or np.all(src[n1:] != 1)):
        # sources are stacked S1,S2,S3 -> positional slices (zero-copy for arrow columns; boolean masks copy ~5 GB)
        q = rec.iloc[:n1].reset_index(drop=True)
        d = rec.iloc[n1:].reset_index(drop=True)
    else:
        q = rec[rec.src == 1].reset_index(drop=True)
        d = rec[rec.src != 1].reset_index(drop=True)
    return q, d


def partition_key(df: pd.DataFrame, col: str) -> np.ndarray:
    if not col or col == "none":
        return np.full(len(df), "", dtype=object)
    if col == "auto":
        for c in ("country", "state", "city"):
            if c in df and (df[c] != "").mean() > 0.9 and df[c].nunique() > 1:
                col = c
                break
        else:
            return np.full(len(df), "", dtype=object)
    return df[col].fillna("").astype(str).values


def make_blocks(q: pd.DataFrame, part_col: str, block_size: int):
    """Yield (partition_value, query_row_indices) with each block <= block_size rows."""
    keys = partition_key(q, part_col)
    order = np.argsort(keys, kind="stable")
    keys_sorted = keys[order]
    bounds = np.flatnonzero(keys_sorted[1:] != keys_sorted[:-1]) + 1
    for grp in np.split(order, bounds):
        val = keys[grp[0]]
        for st in range(0, len(grp), block_size):
            yield val, grp[st:st + block_size]


# --------------------------------------------------------------- block step
class SplitContext:
    """Everything needed to generate candidates + stage-A features for one split."""

    def __init__(self, name, q, d, vecs, views, dense_model, cache_dir, part_col, n_jobs, device):
        self.name, self.q, self.d, self.views, self.part_col = name, q, d, views, part_col
        self.d_src = d["src"].values.astype(np.int8)
        self.d_rid = d["rid"].values
        self.q_rid = q["rid"].values
        self.vecs, self.n_jobs, self.device = vecs, n_jobs, device
        t = time.time()
        self.d_mats = {v: transform(vecs[v], view_text(d, v), n_jobs) for v in views}
        log(f"{name}: doc matrices {[(v, m.shape, m.nnz) for v, m in self.d_mats.items()]} ({time.time() - t:.0f}s)")
        self.d_emb = self.q_emb_all = None
        if dense_model is not None:
            from .dense import dense_text, encode
            self.d_emb = encode(dense_text(d), dense_model, cache_path=os.path.join(cache_dir, f"{name}_docs_emb.npy"))
            self.q_emb_all = encode(dense_text(q), dense_model, cache_path=os.path.join(cache_dir, f"{name}_s1_emb.npy"))
            log(f"{name}: dense embeddings ready (docs {self.d_emb.shape}, S1 {self.q_emb_all.shape})")
        self.d_keys = partition_key(d, part_col)
        self.d_by_key = None
        if part_col and part_col != "none":
            self.d_by_key = pd.Series(np.arange(len(d))).groupby(self.d_keys).apply(lambda s: s.values).to_dict()

    def doc_subset(self, key):
        if self.d_by_key is None or key == "":
            return None  # all docs
        return self.d_by_key.get(key, np.zeros(0, np.int64))

    def block(self, q_idx: np.ndarray, key, k: int, verbose=False):
        """Candidates + stage A for one query block. Returns (cands, q_block, q_mats, q_emb)."""
        qb = self.q.iloc[q_idx].reset_index(drop=True)
        q_mats = {v: transform(self.vecs[v], view_text(qb, v), self.n_jobs) for v in self.views}
        d_sel = self.doc_subset(key)
        parts = lexical_candidates(q_mats, self.d_mats, self.d_src, k, self.views, d_sel=d_sel, verbose=verbose)
        q_emb = None
        views = list(self.views)
        if self.d_emb is not None:
            from .dense import dense_candidates
            q_emb = np.asarray(self.q_emb_all[q_idx])
            parts += dense_candidates(q_emb, self.d_emb, self.d_src, k, d_sel=d_sel, device=self.device, verbose=verbose)
            views.append("dense")
        cands = union_candidates(parts, views)
        cands = stage_a(cands, self.d_src, q_mats, self.d_mats, self.views, q_emb, self.d_emb)
        return cands, qb, q_emb


def write_grouped(fh, q_rid, sets, header_written):
    for r in q_rid:
        fh.write(f"{r}\t{','.join(sorted(sets.get(r, ())))}\n")


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
    ap.add_argument("--partition-col", default="auto", help="none | auto | <column>; auto -> country when present")
    ap.add_argument("--block-size", type=int, default=100_000, help="S1 queries per block (train and test)")
    ap.add_argument("--train-s1", type=int, default=150_000, help="S1 entities used for training (0 = all)")
    ap.add_argument("--neg-rate", type=float, default=0.5, help="keep this share of negative pairs when FITTING (weighted 1/rate); OOF/rules use all pairs")
    ap.add_argument("--test-limit", type=int, default=0, help="only predict the first N test S1 (dev)")
    ap.add_argument("--keep-recall", type=float, default=0.998)
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
    report = dict(args=vars(a), env=dict(python=platform.python_version(), machine=platform.node()))
    device = None
    dense_model = None
    if a.dense:
        from .dense import get_device, load_encoder
        device = get_device()
        dense_model = load_encoder(a.dense_model, device)
        log(f"dense model {a.dense_model} on {device}")

    # 1. load + normalise (test first, keep only a text sample of it; train stays in memory)
    te, _, info_te = load_norm(a.data_dir, "test", a.cache_dir, a.n_jobs)
    te_sample = te.sample(min(len(te), 1_000_000), random_state=a.seed)[["n_core", "n_addr", "n_full", "n_ph"]]
    del te
    release_memory()
    tr, gt, info_tr = load_norm(a.data_dir, "train", a.cache_dir, a.n_jobs)
    n_multi = check_one_to_one(gt)
    one2one = (not a.no_one2one) and n_multi == 0
    log(f"GT: S2/S3 ids linked to >1 S1: {n_multi} -> one-to-one filter {'ON' if one2one else 'OFF'}")
    report["gt"] = dict(n_s1=len(gt), singleton_rate=float(np.mean([len(v) == 0 for v in gt.values()])), multi_s1_ids=n_multi)

    # 2. vectorisers on a text sample of train + test (all sources; no labels)
    tr_sample = tr.sample(min(len(tr), 1_500_000), random_state=a.seed)[["n_core", "n_addr", "n_full", "n_ph"]]
    vecs = fit_vectorizers(pd.concat([tr_sample, te_sample], ignore_index=True), views, max_df=a.max_df, seed=a.seed)
    del tr_sample, te_sample
    log(f"vectorisers: {[(v, len(vec.vocabulary_)) for v, vec in vecs.items()]}")

    # 3. TRAIN pass
    q_tr, d_tr = split_tables(tr)
    del tr
    release_memory()
    if a.train_s1 and a.train_s1 < len(q_tr):
        q_tr = q_tr.sample(a.train_s1, random_state=a.seed).reset_index(drop=True)
    ctx = SplitContext("train", q_tr, d_tr, vecs, views, dense_model, a.cache_dir, a.partition_col, a.n_jobs, device)
    all_views = views + (["dense"] if a.dense else [])
    cand_parts, rec_curves, gp_total = [], [], 0
    for bi, (key, q_idx) in enumerate(make_blocks(q_tr, a.partition_col, a.block_size)):
        cands, qb, _ = ctx.block(q_idx, key, a.k, verbose=(bi == 0))
        gp = gt_pairs_block(qb["rid"].values, ctx.d_rid, gt)
        gp_total += len(gp)
        rec_curves.append(blocking_recall(cands, gp, len(qb), ks=sorted({1, 3, 5, a.k})).assign(n=len(gp)))
        cands = cands.merge(gp.assign(y=np.int8(1)), on=["q", "c"], how="left")
        cands["y"] = cands["y"].fillna(0).astype(np.int8)
        cands["q"] = q_idx[cands["q"].values]  # back to row indices of q_tr
        cand_parts.append(cands)
        log(f"train block {bi} ({key!r}, {len(qb)} S1): {len(cands)} cands, {int(cands.y.sum())}/{len(gp)} GT pairs found")
    cand_tr = pd.concat(cand_parts, ignore_index=True)
    del cand_parts
    rc = pd.concat(rec_curves)
    rc = rc.groupby("k").apply(lambda g: pd.Series(dict(pairs=g.pairs.sum(), pairs_per_s1=g.pairs.sum() / len(q_tr),
                                                       pair_recall=(g.pair_recall * g.n).sum() / max(g.n.sum(), 1),
                                                       entity_full_recall=(g.entity_full_recall * g.n).sum() / max(g.n.sum(), 1)))
                               ).reset_index()
    print(rc.to_string(index=False))
    report["blocking_recall"] = rc.to_dict(orient="records")
    thr_prune = calibrate_prune(prune_score(cand_tr), cand_tr["y"].values, a.keep_recall)
    n0 = len(cand_tr)
    cand_tr = cand_tr[prune_score(cand_tr) >= thr_prune].reset_index(drop=True)
    log(f"prune at score>={thr_prune:.3f}: train {n0}->{len(cand_tr)}; pair recall after pruning "
        f"{cand_tr.y.sum() / max(gp_total, 1):.4f}")
    report["prune"] = dict(threshold=thr_prune, train_pairs=len(cand_tr), train_pair_recall=float(cand_tr.y.sum() / max(gp_total, 1)))
    q_rid_tr, d_rid_tr = ctx.q_rid, ctx.d_rid
    del ctx  # free doc matrices / embeddings before the memory-heavy training stage
    release_memory()
    cand_tr = stage_b(cand_tr, q_tr, d_tr, a.n_jobs)
    feats = feature_columns(cand_tr)
    feats = [f for f in feats if f not in ("score",)] + ["score"]
    log(f"features ({len(feats)}): {feats}")

    X = np.ascontiguousarray(cand_tr[feats].to_numpy(dtype=np.float32))
    y = cand_tr["y"].values
    cand_tr = cand_tr[["q", "c", "y"]].copy()  # drop feature columns (X holds them)
    gc.collect()
    oof, kind = oof_predict(X, y, cand_tr["q"].values, a.folds, a.seed, neg_rate=a.neg_rate)
    cand_tr["p"] = oof
    from sklearn.metrics import average_precision_score, roc_auc_score
    log(f"OOF ({kind}): AUC={roc_auc_score(y, oof):.5f} AP={average_precision_score(y, oof):.5f}")
    sweep_n = min(len(q_rid_tr), 100_000)
    sw_q = np.sort(np.random.default_rng(a.seed).choice(len(q_rid_tr), sweep_n, replace=False))
    sw_mask = np.isin(cand_tr["q"].values, sw_q)
    tab, best = sweep_rules(cand_tr[sw_mask], q_rid_tr, d_rid_tr, gt, one2one_ok=one2one, q_subset=sw_q)
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
    gt_sub = {r: gt.get(r, frozenset()) for r in q_rid_tr}
    print("  OOF breakdown:")
    macro_f05(gt_sub, decide(cand_tr, q_rid_tr, d_rid_tr, best["rule"], thr_best, bool(best["one2one"])), verbose=True)
    pd.DataFrame({"s1": q_rid_tr[cand_tr.q], "cand": d_rid_tr[cand_tr.c], "y": y, "p": oof}).to_csv(
        os.path.join(a.out_dir, "oof_pairs.tsv.gz"), sep="\t", index=False)
    model, _ = make_model(a.seed)
    fit_idx, fit_w = subsample_negatives(y, a.neg_rate, a.seed)
    model.fit(X[fit_idx], y[fit_idx], sample_weight=fit_w)
    imp = feature_importance(model, feats)
    if len(imp):
        report["importance_top25"] = imp.head(25).round(4).to_dict()
        print("  top features:", imp.head(15).round(3).to_dict())
    try:
        import joblib
        joblib.dump(dict(model=model, feats=feats, thr_prune=thr_prune, best=best, views=views), os.path.join(a.out_dir, "model.joblib"))
    except Exception as e:  # noqa: BLE001
        log(f"model not saved: {e}")
    del cand_tr, X, d_tr, q_tr
    release_memory()

    # 4. TEST pass (streamed)
    header = ("source1_id", "matched_ids")
    if "gt" in info_tr["files"]:
        header = tuple(pd.read_csv(info_tr["files"]["gt"], sep="\t", nrows=0).columns[:2])
    if a.skip_test:
        report["runtime_s"] = round(time.time() - T0, 1)
        json.dump(report, open(os.path.join(a.out_dir, "report.json"), "w"), indent=2, default=str)
        log("skip-test: done")
        return
    te, _, _ = load_norm(a.data_dir, "test", a.cache_dir, a.n_jobs)
    q_te, d_te = split_tables(te)
    del te
    release_memory()
    if a.test_limit:
        q_te = q_te.iloc[:a.test_limit].reset_index(drop=True)
    ctx = SplitContext("test", q_te, d_te, vecs, views, dense_model, a.cache_dir, a.partition_col, a.n_jobs, device)
    pred_rows = np.empty(len(q_te), dtype=object)
    cand_rows = np.empty(len(q_te), dtype=object)
    n_pairs = n_pred = 0
    for bi, (key, q_idx) in enumerate(make_blocks(q_te, a.partition_col, a.block_size)):
        cands, qb, _ = ctx.block(q_idx, key, a.k)
        cands = cands[prune_score(cands) >= thr_prune].reset_index(drop=True)
        cands = stage_b(cands, qb, d_te, a.n_jobs)
        cands["p"] = model.predict_proba(cands[feats].to_numpy(dtype=np.float32))[:, 1]
        sets = decide(cands, qb["rid"].values, ctx.d_rid, best["rule"], thr_best, bool(best["one2one"]))
        for qi, r in zip(q_idx, qb["rid"].values):
            pred_rows[qi] = ",".join(sorted(sets[r]))
            n_pred += len(sets[r])
        cand_rows[q_idx] = ""
        for qq, grp in cands.groupby("q")["c"]:
            cand_rows[q_idx[qq]] = ",".join(ctx.d_rid[grp.values])
        n_pairs += len(cands)
        log(f"test block {bi} ({key!r}, {len(qb)} S1): {len(cands)} cands after prune, "
            f"{sum(1 for r in qb['rid'].values if sets[r])} S1 with matches")
    q_rid = q_te["rid"].values
    with open(os.path.join(a.out_dir, "matching_results.tsv"), "w") as fh:
        fh.write(f"{header[0]}\t{header[1]}\n")
        for r, m in zip(q_rid, pred_rows):
            fh.write(f"{r}\t{m or ''}\n")
    with open(os.path.join(a.out_dir, "candidate_pairs.tsv"), "w") as fh:
        fh.write(f"{header[0]}\tcandidate_entity_ids\n")
        for r, m in zip(q_rid, cand_rows):
            fh.write(f"{r}\t{m or ''}\n")
    empty = float(np.mean([not m for m in pred_rows]))
    report["test_pred"] = dict(n_s1=len(q_rid), empty_rate=empty, mean_matches=n_pred / max(len(q_rid), 1),
                               cand_pairs=n_pairs, cand_pairs_per_s1=n_pairs / max(len(q_rid), 1))
    if "country" in q_te:  # sanity per country (France is unseen in train: watch its empty rate / mean matches)
        n_m = np.array([m.count(",") + 1 if m else 0 for m in pred_rows])
        per = pd.DataFrame({"country": q_te["country"].values, "n": n_m}).groupby("country")["n"].agg(
            n_s1="size", empty_rate=lambda x: float((x == 0).mean()), mean_matches="mean")
        print(per.to_string())
        report["test_pred_by_country"] = per.reset_index().to_dict(orient="records")
    log(f"test: {len(q_rid)} S1 rows written | predicted-empty={empty:.1%} | mean matches={n_pred / max(len(q_rid), 1):.2f}")
    if a.test_gt:
        tgt = read_matches_tsv(a.test_gt)
        pred = {r: frozenset(m.split(",")) if m else frozenset() for r, m in zip(q_rid, pred_rows)}
        tgt = {r: v for r, v in tgt.items() if r in pred}
        report["test_f05_hidden"] = macro_f05(tgt, pred, verbose=True)
    report["runtime_s"] = round(time.time() - T0, 1)
    with open(os.path.join(a.out_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    if a.run_name:
        import shutil
        rd = os.path.join("runs", a.run_name)
        os.makedirs(rd, exist_ok=True)
        for f in ("report.json", "matching_results.tsv"):
            shutil.copy(os.path.join(a.out_dir, f), os.path.join(rd, f))
        log(f"copied report + matching_results to {rd}/ (commit this folder)")
    log("done")


if __name__ == "__main__":
    main()
