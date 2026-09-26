"""Offline decision-threshold sweep on a hidden-holdout run made with --save-probs (light, slice-sized).
Approximates the pipeline's 'thr' rule: p >= 0.02, one-to-one per candidate record (max-p S1, global per country),
then p >= thr. Ignores key-rule forcing (--key-rules), so compare runs made without it.
Usage: PYTHONPATH=. python scripts/day/thr_sweep.py <data_dir> <out_dir> [thr ...]"""
import glob
import sys

import numpy as np
import pandas as pd

from src.metric import macro_f05, read_matches_tsv

dd, od = sys.argv[1:3]
thrs = [float(t) for t in sys.argv[3:]] or [round(t, 2) for t in np.arange(0.5, 0.951, 0.05)]
truth = read_matches_tsv(f"{dd}/test_ground_truth_HIDDEN.tsv")
parts = []
for f in sorted(glob.glob(f"{od}/test_probs_*.parquet")):
    d = pd.read_parquet(f, columns=["s1", "cand", "p"])
    d = d[d["p"] >= 0.02]
    parts.append(d[d["p"] >= d.groupby("cand")["p"].transform("max")])
df = pd.concat(parts, ignore_index=True)
for t in thrs:
    sel = df[df["p"] >= t]
    pred = {k: frozenset(v) for k, v in sel.groupby("s1")["cand"]}
    print(f"thr {t:.2f}: macro F0.5 {macro_f05(truth, pred):.4f}  pairs {len(sel)}")
