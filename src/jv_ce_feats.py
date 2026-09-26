"""Jarvis QUEUE 2b: add cross-encoder features to a --feat-cache -> new feat-cache for `python -m src.rescore`.

  python -m src.jv_ce_feats --feat-cache /home/cache_jv/j1_feats --ce-dir /home/pools/jv1_ce --out /home/cache_jv/j2_feats

Features (computed over the whole pool of the S1, so they see the candidates the cascade dropped too):
  ce_p    sigmoid of the cross-encoder logit (train: out-of-fold, test: mean of the fold models)
  ce_rank rank of the pair within its S1 by ce_p (0 = best)
  ce_gap  best ce_p of the S1 minus this pair's ce_p
Pairs missing from the pool get NaN (LightGBM routes them as missing).
"""
import argparse
import glob
import json
import os
import shutil

import numpy as np
import pandas as pd

from .jv_ce import log

NEW = ["ce_p", "ce_rank", "ce_gap"]


def ce_table(df: pd.DataFrame, prob: np.ndarray) -> pd.DataFrame:
    t = pd.DataFrame({"s1": df["s1"].to_numpy(), "cand": df["cand"].to_numpy(), "ce_p": prob.astype(np.float32)})
    g = t.groupby("s1")["ce_p"]
    t["ce_rank"] = (g.rank(ascending=False, method="first") - 1).astype(np.float32)
    t["ce_gap"] = (g.transform("max") - t["ce_p"]).astype(np.float32)
    return t


def lookup(tab: pd.DataFrame, s1: np.ndarray, cand: np.ndarray) -> np.ndarray:
    key = pd.MultiIndex.from_arrays([tab["s1"].to_numpy(), tab["cand"].to_numpy()])
    idx = key.get_indexer(pd.MultiIndex.from_arrays([s1, cand]))
    out = np.full((len(s1), len(NEW)), np.nan, np.float32)
    ok = idx >= 0
    out[ok] = tab[NEW].to_numpy()[idx[ok]]
    return out, int((~ok).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-cache", required=True)
    ap.add_argument("--ce-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=1_000_000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    info = json.load(open(os.path.join(a.feat_cache, "train_feats.json")))
    feats, n = info["feats"], info["n_rows"]
    tm = np.load(os.path.join(a.feat_cache, "train_meta.npz"))
    s1 = tm["q_rid"][tm["qg"]].astype(str)
    cand = tm["cid"].astype(str)
    tr = pd.read_parquet(os.path.join(a.ce_dir, "train_ce.parquet"), columns=["s1", "cand", "ce"])
    ext, miss = lookup(ce_table(tr, 1 / (1 + np.exp(-tr["ce"].to_numpy()))), s1, cand)
    del tr, s1, cand
    log(f"train: {n} pairs, {miss} missing from the ce pool")
    X = np.memmap(os.path.join(a.feat_cache, "train_X.f32"), dtype=np.float32, mode="r", shape=(n, len(feats)))
    with open(os.path.join(a.out, "train_X.f32"), "wb") as fh:
        for s in range(0, n, a.chunk):
            fh.write(np.ascontiguousarray(np.hstack([X[s:s + a.chunk], ext[s:s + a.chunk]]), dtype=np.float32).tobytes())
    shutil.copy(os.path.join(a.feat_cache, "train_meta.npz"), a.out)
    info2 = dict(info, feats=feats + NEW)
    json.dump(info2, open(os.path.join(a.out, "train_feats.json"), "w"))
    for f in sorted(glob.glob(os.path.join(a.feat_cache, "test__*__b*.parquet"))):
        base = os.path.basename(f)
        c_blk = base[len("test__"):-len(".parquet")]
        df = pd.read_parquet(f)
        ce = pd.read_parquet(os.path.join(a.ce_dir, f"test_ce__{c_blk}.parquet"), columns=["s1", "cand", "ce_p"])
        e, miss = lookup(ce_table(ce, ce["ce_p"].to_numpy()), df["s1"].to_numpy(), df["cand"].to_numpy())
        for j, col in enumerate(NEW):
            df.insert(len(df.columns) - 1, col, e[:, j])  # before "p"
        df.to_parquet(os.path.join(a.out, base), index=False)
        shutil.copy(f[:-len(".parquet")] + "_qrid.npy", a.out)
        log(f"{base}: {len(df)} pairs, {miss} missing from the ce pool")
    log("done")


if __name__ == "__main__":
    main()
