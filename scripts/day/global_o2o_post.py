"""QUEUE 2 check on the fast lane: apply the global one-to-one to a finished test pass post hoc (same logic as the
inline pipeline/rescore version) and report multi-owner records before/after + pairs removed per country.
Usage: python scripts/day/global_o2o_post.py <matching_results.tsv> <dir with test_probs_<country>.parquet> [out.tsv]
Memory ~1-2 GB on the full data (probs filtered to the decided pairs): run it when no heavy job is running."""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.hq_keys import global_one_to_one  # noqa: E402

mr = pd.read_csv(sys.argv[1], sep="\t", dtype=str, keep_default_na=False)
long = mr.assign(m=mr.iloc[:, 1].str.split(",")).explode("m")
long = long[long["m"].fillna("") != ""][[mr.columns[0], "m"]].rename(columns={mr.columns[0]: "s1"})
print(f"pairs {len(long)}; records under >1 S1 before: {int((long['m'].value_counts() > 1).sum())}")
parts = []
for f in sorted(glob.glob(os.path.join(sys.argv[2], "test_probs_*.parquet"))):
    c = os.path.basename(f)[len("test_probs_"):-len(".parquet")]
    pr = pd.read_parquet(f, columns=["s1", "cand", "p"]).rename(columns={"cand": "m"})
    sub = long.merge(pr, on=["s1", "m"], how="inner").assign(country=c)
    parts.append(sub)
    del pr
dec = pd.concat(parts, ignore_index=True)
assert len(dec) == len(long), f"decided pairs without probability: {len(long) - len(dec)}"
keep = np.zeros(len(dec), bool)
for c, g in dec.groupby("country"):
    k = global_one_to_one(g["s1"].to_numpy(), g["m"].to_numpy(), g["p"].to_numpy())
    keep[g.index.to_numpy()] = k
    print(f"{c}: removed {int((~k).sum())} of {len(g)} decided pairs; S1 losing a match {g.loc[~k, 's1'].nunique()}")
kept = dec[keep]
print(f"records under >1 S1 after: {int((kept['m'].value_counts() > 1).sum())}")
if len(sys.argv) > 3:
    agg = kept.groupby("s1")["m"].apply(lambda x: ",".join(sorted(x)))
    out = pd.DataFrame({mr.columns[0]: mr.iloc[:, 0], mr.columns[1]: mr.iloc[:, 0].map(agg).fillna("")})
    out.to_csv(sys.argv[3], sep="\t", index=False)
    print(f"written {sys.argv[3]}")
