"""QUEUE 7: model capacity on the fast lane — retrain LightGBM variants on a --feat-cache (full-density train pass)
and compare OOF macro F0.5 per country (GroupKFold by S1, same folds/neg-rate as the pipeline).

Needs <feat-cache>/train_X.f32 + train_meta.npz + train_feats.json (written by src.pipeline --feat-cache, e.g. v3's
feats_v3). Rule per variant: global threshold (sweep 0.05..0.95) + global one-to-one, scored on ALL train S1 of the pass
(S1 without candidates count, GT sizes from train_ground_truth.tsv) — the pipeline's thr rule on the full sample.
This is a HEAVY job (LightGBM, 5 folds x variants; ~1.5-3 GB RSS): run it alone, detached, like a slice screen.
Usage: python scripts/day/model_capacity.py feats_v3 [--variants base,big,seed3] [--n-jobs 8] > runs/day/model_capacity.md
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.model import make_model, subsample_negatives  # noqa: E402

TR = "data/data_extracted/student_resource/dataset/train/"
BETA2 = 0.25


def variant_models(name, seed, n_jobs):
    """List of (unfitted) models whose probabilities are averaged."""
    import lightgbm as lgb
    if name == "base":
        return [make_model(seed, n_jobs=n_jobs)[0]]
    if name == "seed3":
        return [make_model(seed + 1000 * i, n_jobs=n_jobs)[0] for i in range(3)]
    if name == "big":
        return [lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.03, num_leaves=127, min_child_samples=40,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                   n_jobs=n_jobs, random_state=seed, verbose=-1, deterministic=True,
                                   force_row_wise=True)]
    if name == "deep":
        return [lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=255, min_child_samples=20,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=2.0,
                                   n_jobs=n_jobs, random_state=seed, verbose=-1, deterministic=True,
                                   force_row_wise=True)]
    raise SystemExit(f"unknown variant {name}")


def oof(X, y, qg, name, folds, seed, neg_rate, n_jobs):
    from sklearn.model_selection import GroupKFold
    out = np.zeros(len(y), dtype=np.float32)
    for f, (tr, va) in enumerate(GroupKFold(n_splits=folds).split(X, y, qg)):
        sub, w = subsample_negatives(y[tr], neg_rate, seed + f)
        idx = tr[sub]
        Xf = np.ascontiguousarray(X[idx])
        ms = variant_models(name, seed + f, n_jobs)
        for m in ms:
            m.fit(Xf, y[idx], sample_weight=w)
            out[va] += m.predict_proba(X[va])[:, 1] / len(ms)
        del Xf
        print(f"    {name} fold {f} done", file=sys.stderr, flush=True)
    return out


def macro_f(q, c, p, y, n_q, n_true, thr):
    """Mean per-S1 F0.5 (PS singleton rule) for thr + global one-to-one (max-p S1 per doc), vectorised."""
    best = pd.Series(p).groupby(c).transform("max").to_numpy()
    sel = (p >= best) & (p >= thr)
    npred = np.bincount(q[sel], minlength=n_q)
    tp = np.bincount(q[sel], weights=y[sel], minlength=n_q)
    fn = n_true - tp
    fp = npred - tp
    den = (1 + BETA2) * tp + BETA2 * fn + fp
    f = np.where(tp > 0, (1 + BETA2) * tp / np.maximum(den, 1e-9), 0.0)
    f = np.where(n_true == 0, (npred == 0).astype(float), f)
    return f, npred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("feat_cache")
    ap.add_argument("--variants", default="base,big,seed3")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--neg-rate", type=float, default=0.5)
    ap.add_argument("--n-jobs", type=int, default=8)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.feat_cache, "train_feats.json")))
    z = np.load(os.path.join(a.feat_cache, "train_meta.npz"), allow_pickle=True)
    y, qg, cid, q_rid = z["y"].astype(np.int8), z["qg"].astype(np.int64), z["cid"], z["q_rid"].astype(str)
    n_q = int(meta["n_q"])
    X = np.memmap(os.path.join(a.feat_cache, "train_X.f32"), dtype=np.float32, mode="r",
                  shape=(int(meta["n_rows"]), len(meta["feats"])))
    _, c = np.unique(cid, return_inverse=True)
    ids = set(q_rid)
    country, n_true_d = {}, {}
    for ch in pd.read_csv(TR + "train_source1.tsv", sep="\t", usecols=["entity_id", "country"], chunksize=500_000,
                          dtype=str):
        ch = ch[ch["entity_id"].isin(ids)]
        country.update(zip(ch["entity_id"], ch["country"]))
    for ch in pd.read_csv(TR + "train_ground_truth.tsv", sep="\t", chunksize=500_000, dtype=str):
        ch = ch[ch.iloc[:, 0].isin(ids)]
        for s, m in zip(ch.iloc[:, 0], ch.iloc[:, 1]):
            n_true_d[s] = sum(1 for x in str(m).split(",") if x and x != "nan")
    n_true = np.array([n_true_d.get(s, 0) for s in q_rid], dtype=np.float64)
    ctry = np.array([country.get(s, "?") for s in q_rid])
    print(f"# Model capacity (QUEUE 7) on `{a.feat_cache}`\n\n{len(y):,} pairs x {X.shape[1]} features, {n_q:,} train S1, "
          f"{a.folds}-fold GroupKFold, neg-rate {a.neg_rate}. Rule: global thr + global one-to-one (best thr per variant; "
          f"also at base's thr).\n")
    print("| variant | fit+OOF min | best thr | OOF F0.5 ALL | " + " | ".join(f"{k} F / pred-empty"
                                                                           for k in sorted(set(ctry))) +
          " | ALL F at base thr |")
    print("|---|---|---|---|" + "---|" * len(set(ctry)) + "---|")
    thr_base = None
    for name in a.variants.split(","):
        t = time.time()
        p = oof(X, y, qg, name, a.folds, a.seed, a.neg_rate, a.n_jobs)
        np.save(os.path.join(a.feat_cache, f"oof_{name}.npy"), p)
        grid = np.round(np.arange(0.05, 0.951, 0.05), 2)
        scores = [(macro_f(qg, c, p, y, n_q, n_true, th)[0].mean(), th) for th in grid]
        fbest, thb = max(scores)
        if thr_base is None:
            thr_base = thb
        f, npred = macro_f(qg, c, p, y, n_q, n_true, thb)
        cells = [f"{f[ctry == k].mean():.5f} / {(npred[ctry == k] == 0).mean():.2%}" for k in sorted(set(ctry))]
        fb = macro_f(qg, c, p, y, n_q, n_true, thr_base)[0].mean()
        print(f"| {name} | {(time.time() - t) / 60:.1f} | {thb:.2f} | {fbest:.5f} | " + " | ".join(cells) +
              f" | {fb:.5f} |", flush=True)


if __name__ == "__main__":
    main()
