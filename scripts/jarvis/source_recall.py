"""QUEUE 1 report: GT pairs recovered by each candidate source on the full-density train pass (from the --cand-cache).

  python scripts/jarvis/source_recall.py --cand /home/cache_jv/j1_cand --store /home/cache_jv/store [--country india]

Per country (finished forward blocks only): GT pairs of the cached S1 blocks found by
  v2 = 4 views (name_c3, name_w, addr_c3, full_w) at rank < 10 | the 6-view forward union at k 15 | name_ph alone |
  name_ph only (no other view) | reverse top-3 (all / new to the forward union) | reverse-sure (rev_best >= 2: count per S1 and precision).
Key-rule pairs are not cached; the pipeline log prints how many are new to the union ("key pairs: ... new").
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

V2 = ["name_c3", "name_w", "addr_c3", "full_w"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--country", default="")
    ap.add_argument("--split", default="train")
    ap.add_argument("--bypass", type=int, default=2)
    a = ap.parse_args()
    gt = pd.read_parquet(os.path.join(a.store, "train_gt_pairs.parquet"))
    gt_set = set(zip(gt["s1"].to_numpy(), gt["m"].to_numpy()))
    gt_per_s1 = gt.groupby("s1").size()
    rows = []
    countries = [a.country] if a.country else sorted({re.match(rf"{a.split}_(\w+?)__", os.path.basename(p)).group(1)
                                                      for p in glob.glob(os.path.join(a.cand, f"{a.split}_*__b*.parquet"))})
    for c in countries:
        docs = np.load(os.path.join(a.cand, f"{a.split}_{c}__docs.npy"), allow_pickle=True).astype(str)
        s1_all = pd.read_parquet(os.path.join(a.store, f"{a.split}__{c}.parquet"), columns=["rid", "src"])
        s1_all = s1_all.loc[s1_all["src"] == 1, "rid"].to_numpy().astype(str)
        revp = glob.glob(os.path.join(a.cand, f"{a.split}_{c}__rev3_*.parquet"))
        rev = pd.read_parquet(revp[0]) if revp else None
        if rev is not None:
            rev = rev.assign(s1=s1_all[rev["qa"].to_numpy()], m=docs[rev["c"].to_numpy()])
        blocks = sorted(p for p in glob.glob(os.path.join(a.cand, f"{a.split}_{c}__b*.parquet")) if "__aug_" not in p)
        tot = dict(n_s1=0, gt=0, v2=0, fwd6=0, name_ph=0, ph_only=0, fwd_n=0, rev=0, rev_new=0, union=0, union_n=0, sure_n=0, sure_y=0,
                   sure_new=0)
        for p in blocks:
            qrid = np.load(p.replace(".parquet", "_qrid.npy"), allow_pickle=True).astype(str)
            u = pd.read_parquet(p)
            s1 = qrid[u["q"].to_numpy()]
            m = docs[u["c"].to_numpy()]
            y = np.fromiter((k in gt_set for k in zip(s1, m)), bool, len(u))
            v2 = np.zeros(len(u), bool)
            for v in V2:
                v2 |= u[f"{v}_rank"].to_numpy() < 10
            ph = u["name_ph_rank"].to_numpy() < 99  # 99 = not in the view's top k
            others = np.zeros(len(u), bool)
            for v in ("name_c3", "name_w", "addr_c3", "full_w", "addr_w"):
                others |= u[f"{v}_rank"].to_numpy() < 99
            n_gt = int(gt_per_s1.reindex(qrid).fillna(0).sum())
            tot["n_s1"] += len(qrid)
            tot["gt"] += n_gt
            tot["v2"] += int((y & v2).sum())
            tot["fwd6"] += int(y.sum())
            tot["name_ph"] += int((y & ph).sum())
            tot["ph_only"] += int((y & ph & ~others).sum())
            tot["fwd_n"] += len(u)
            if rev is not None:
                rb = rev[rev["s1"].isin(set(qrid))]
                fwd_keys = set(zip(s1, m))
                ry = np.fromiter((k in gt_set for k in zip(rb["s1"], rb["m"])), bool, len(rb))
                new = ~np.fromiter((k in fwd_keys for k in zip(rb["s1"], rb["m"])), bool, len(rb))
                sure = rb["rev_best"].to_numpy() >= a.bypass
                tot["rev"] += int(ry.sum())
                tot["rev_new"] += int((ry & new).sum())
                tot["union"] += int(y.sum()) + int((ry & new).sum())
                tot["union_n"] += len(u) + int(new.sum())
                tot["sure_n"] += int(sure.sum())
                tot["sure_y"] += int((sure & ry).sum())
                tot["sure_new"] += int((sure & ry & new).sum())
        g, n = max(tot["gt"], 1), max(tot["n_s1"], 1)
        rows.append(dict(country=c, blocks=len(blocks), s1=tot["n_s1"], gt_pairs=tot["gt"],
                         rec_v2_4views_k10=tot["v2"] / g, rec_fwd_6views_k15=tot["fwd6"] / g, rec_name_ph=tot["name_ph"] / g, rec_ph_only=tot["ph_only"] / g,
                         rec_reverse=tot["rev"] / g, rec_rev_new=tot["rev_new"] / g, rec_union=tot["union"] / g,
                         fwd_per_s1=tot["fwd_n"] / n, union_per_s1=tot["union_n"] / n,
                         rev_sure_per_s1=tot["sure_n"] / n, rev_sure_prec=tot["sure_y"] / max(tot["sure_n"], 1),
                         rev_sure_new_gt=tot["sure_new"] / g))
    df = pd.DataFrame(rows)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
