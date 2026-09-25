"""Candidate-budget study: how small can candidate_pairs.tsv be without losing F0.5?

Runs the train pass of the pipeline on a slice up to stage A (all blocking candidates, BEFORE pruning),
dumps the candidate table, then evaluates pruning strategies OOF:
  A. global score threshold (current)                       -> recall / cands per S1
  B. per-S1 top-N by score
  C. cascade: stage-A-only GBDT (OOF by S1) -> keep p_a >= t  (and/or top-N by p_a)
For each strategy/setting we report mean & median candidates per S1, pair recall, and the macro-F0.5
obtained by applying the existing full-model OOF probabilities (out_slice/oof_pairs.tsv.gz) to the
surviving pairs (pairs pruned away become misses). This is a faithful lower bound: the full model would
be retrained on the smaller set in production.

Usage: python scripts/cascade_study.py --data-dir data_slice --cache-dir cache_slice --oof out_slice/oof_pairs.tsv.gz
"""
import argparse
import gc
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.blocking import fit_vectorizers, gt_pairs_block  # noqa: E402
from src.features import prune_score  # noqa: E402
from src.metric import macro_f05, read_matches_tsv  # noqa: E402
from src.pipeline import SplitContext, load_norm, make_blocks, split_tables  # noqa: E402


def f05_from_pairs(df_keep, gt):
    """Apply the full-model OOF decision (thr 0.80 + one-to-one) on the kept pairs."""
    d = df_keep[df_keep.p >= 0.80]
    if len(d):
        d = d[d.p >= d.groupby("cand").p.transform("max")]
    pred = {s1: frozenset(g) for s1, g in d.groupby("s1").cand}
    return macro_f05(gt, {k: pred.get(k, frozenset()) for k in gt})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data_slice")
    ap.add_argument("--cache-dir", default="cache_slice")
    ap.add_argument("--oof", default="out_slice/oof_pairs.tsv.gz")
    ap.add_argument("--train-s1", type=int, default=40000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max-df", type=float, default=0.01)
    ap.add_argument("--views", default="name_c3,name_w,addr_c3,name_ph,full_w,addr_w")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dump", default="out_slice/stageA_cands.pkl")
    a = ap.parse_args()
    views = a.views.split(",")
    t0 = time.time()
    if os.path.exists(a.dump):
        cand = pd.read_pickle(a.dump)
        print(f"loaded {a.dump}: {len(cand)} pairs")
    else:
        te, _, _ = load_norm(a.data_dir, "test", a.cache_dir, 2)
        te_s = te.sample(min(len(te), 300_000), random_state=a.seed)[["n_core", "n_addr", "n_full", "n_ph"]]
        del te
        tr, gt, _ = load_norm(a.data_dir, "train", a.cache_dir, 2)
        tr_s = tr.sample(min(len(tr), 600_000), random_state=a.seed)[["n_core", "n_addr", "n_full", "n_ph"]]
        vecs = fit_vectorizers(pd.concat([tr_s, te_s]), views, max_df=a.max_df, seed=a.seed)
        del tr_s, te_s
        q_tr, d_tr = split_tables(tr)
        del tr
        gc.collect()
        q_tr = q_tr.sample(a.train_s1, random_state=a.seed).reset_index(drop=True)
        ctx = SplitContext("train", q_tr, d_tr, vecs, views, None, a.cache_dir, "auto", 2, None)
        parts = []
        for bi, (key, q_idx) in enumerate(make_blocks(q_tr, "auto", 60000)):
            cands, qb, _ = ctx.block(q_idx, key, a.k)
            gp = gt_pairs_block(qb["rid"].values, ctx.d_rid, gt)
            cands = cands.merge(gp.assign(y=np.int8(1)), on=["q", "c"], how="left")
            cands["y"] = cands["y"].fillna(0).astype(np.int8)
            cands["s1"] = qb["rid"].values[cands.q.values]
            cands["cand"] = ctx.d_rid[cands.c.values]
            cands["n_true"] = cands["s1"].map(lambda r: len(gt.get(r, ())))
            cands = cands.drop(columns=["q", "c"])
            parts.append(cands)
            print(f"block {bi} ({key}, {len(qb)} S1): {len(cands)} cands, {int(cands.y.sum())}/{len(gp)} GT pairs ({time.time() - t0:.0f}s)")
        cand = pd.concat(parts, ignore_index=True)
        cand.to_pickle(a.dump)
        del ctx, d_tr
        gc.collect()
    gt_all = read_matches_tsv(os.path.join(a.data_dir, "train", "train_ground_truth.tsv"))
    s1s = cand.s1.unique()
    gt = {k: gt_all.get(k, frozenset()) for k in s1s}
    n_s1 = len(s1s)
    n_true_pairs = sum(len(v) for v in gt.values())
    oof = pd.read_csv(a.oof, sep="\t", dtype={"s1": str, "cand": str})
    cand = cand.merge(oof[["s1", "cand", "p"]], on=["s1", "cand"], how="left")
    print(f"\n{len(cand)} blocking pairs for {n_s1} S1 | true pairs {n_true_pairs} | blocking recall {cand.y.sum() / n_true_pairs:.4f} "
          f"| cands/S1 mean {len(cand) / n_s1:.1f} median {cand.groupby('s1').size().median():.0f}")
    cand["p"] = cand["p"].fillna(0.0)  # pairs outside the v3 pruned set never reached the model
    base_f = f05_from_pairs(cand, gt)
    print(f"F0.5 with the full v3 set (prune 0.51): {base_f:.5f}")

    def report(name, keep):
        sub = cand[keep]
        per = sub.groupby("s1").size().reindex(s1s, fill_value=0)
        print(f"  {name:<34} cands/S1 mean {per.mean():5.1f} median {per.median():4.0f} p90 {per.quantile(.9):4.0f} | "
              f"pair recall {sub.y.sum() / n_true_pairs:.4f} | F0.5 {f05_from_pairs(sub, gt):.5f}")

    score = prune_score(cand)
    print("\nA. global score threshold")
    for t in (0.0, 0.51, 0.6, 0.7, 0.8, 0.9):
        report(f"score >= {t}", score >= t)
    print("\nB. per-S1 top-N by score")
    rank = cand.groupby("s1")["score"].rank(ascending=False, method="first")
    for n in (60, 40, 30, 20, 15, 10):
        report(f"top-{n} per S1", rank <= n)

    print("\nC. cascade: stage-A-only model (OOF by S1)")
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    drop = {"y", "s1", "cand", "n_true", "p"}
    feats = [c for c in cand.columns if c not in drop]
    X = cand[feats].to_numpy(dtype=np.float32)
    y = cand.y.values
    pa = np.zeros(len(cand), np.float32)
    for f, (tr_i, va_i) in enumerate(GroupKFold(3).split(X, y, cand.s1.values)):
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=50,
                                           random_state=a.seed).fit(X[tr_i], y[tr_i])
        pa[va_i] = m.predict_proba(X[va_i])[:, 1]
    cand["pa"] = pa
    from sklearn.metrics import average_precision_score
    print(f"  stage-A model AP {average_precision_score(y, pa):.5f} ({len(feats)} features)")
    for t in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
        report(f"p_a >= {t}", pa >= t)
    rank_a = cand.groupby("s1")["pa"].rank(ascending=False, method="first")
    for n in (30, 20, 15, 12, 10, 8):
        report(f"top-{n} per S1 by p_a", rank_a <= n)
    for t, n in ((0.01, 20), (0.02, 15), (0.05, 12), (0.02, 10)):
        report(f"p_a >= {t} & top-{n}", (pa >= t) & (rank_a <= n))
    cand.to_pickle(a.dump)


if __name__ == "__main__":
    main()
