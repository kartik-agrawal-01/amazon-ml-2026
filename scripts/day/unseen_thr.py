"""QUEUE 2d: label-free threshold rules for an unseen target country, scored on a cross-country screen made with
--save-probs (light, slice-sized). Same decision approximation as thr_sweep.py (p >= 0.02, one-to-one per candidate
record, then p >= t; key-rule forcing ignored).
  R1 count matching: t where target pred matches/S1 = (source OOF pred/GT matches per S1 at t_oof) x target GT/S1
     (label-free: GT/S1 assumed 3.46 as in US/India train; the true value is also shown)
  R2 empty matching: t where target pred-empty rate = source OOF pred-empty rate at t_oof
Usage: PYTHONPATH=. python scripts/day/unseen_thr.py <data_dir> <out_dir> [assumed_gt_per_s1=3.46]"""
import glob
import json
import sys

import numpy as np
import pandas as pd

from src.metric import macro_f05, read_matches_tsv

dd, od = sys.argv[1:3]
gt_assumed = float(sys.argv[3]) if len(sys.argv) > 3 else 3.46
grid = np.round(np.arange(0.02, 0.991, 0.01), 2)


def o2o(d):
    d = d[d["p"] >= 0.02]
    return d[d["p"] >= d.groupby("cand")["p"].transform("max")]


def stats(d, s1_all, t):
    """matches/S1 and empty rate over the S1 set s1_all at threshold t."""
    cnt = d.loc[d["p"] >= t, "s1"].value_counts()
    cnt = cnt.reindex(s1_all, fill_value=0)
    return float(cnt.mean()), float((cnt == 0).mean())


rep = json.load(open(f"{od}/report.json"))
t_oof = rep["oof"]["chosen"]["thr"]
# ---- source (OOF)
oof = o2o(pd.read_csv(f"{od}/oof_pairs.tsv.gz", sep="\t", usecols=["s1", "cand", "p"]))
tr_gt = read_matches_tsv(f"{dd}/train/train_ground_truth.tsv")
src_s1 = pd.read_csv(f"{od}/oof_pairs.tsv.gz", sep="\t", usecols=["s1"])["s1"].unique()
src_gt_per = float(np.mean([len(tr_gt.get(s, ())) for s in src_s1]))
src_pred_per, src_empty = stats(oof, src_s1, t_oof)
ratio = src_pred_per / src_gt_per
# ---- target (test)
truth = read_matches_tsv(f"{dd}/test_ground_truth_HIDDEN.tsv")
tst_s1 = pd.read_csv(f"{dd}/test/test_source1.tsv", sep="\t", usecols=["entity_id"])["entity_id"].unique()
tgt_gt_per = float(np.mean([len(truth.get(s, ())) for s in tst_s1]))
df = pd.concat([o2o(pd.read_parquet(f, columns=["s1", "cand", "p"]))
                for f in sorted(glob.glob(f"{od}/test_probs_*.parquet"))], ignore_index=True)


def f_at(t):
    sel = df[df["p"] >= t]
    return macro_f05(truth, {k: frozenset(v) for k, v in sel.groupby("s1")["cand"]})


curve = {t: stats(df, tst_s1, t) for t in grid}


def solve(target, idx):  # grid t whose statistic is closest to target (monotone in t)
    return min(grid, key=lambda t: abs(curve[t][idx] - target))


t_r1 = solve(ratio * gt_assumed, 0)
t_r1_true = solve(ratio * tgt_gt_per, 0)
t_r2 = solve(src_empty, 1)
coarse = np.round(np.r_[0.02, 0.05, 0.10, 0.15, 0.20, 0.25, np.arange(0.30, 0.951, 0.05)], 2)
fs = {t: f_at(t) for t in sorted(set(coarse) | {t_oof, t_r1, t_r1_true, t_r2})}
t_best = max(fs, key=fs.get)
# refine hidden best on the fine grid around the coarse best
for t in grid[(grid >= t_best - 0.05) & (grid <= t_best + 0.05)]:
    if t not in fs:
        fs[t] = f_at(t)
t_best = max(fs, key=fs.get)

print(f"# {od} (data {dd})")
print(f"source OOF: t_oof {t_oof:.2f}, pred/S1 {src_pred_per:.3f}, GT/S1 {src_gt_per:.3f} (ratio {ratio:.3f}), "
      f"empty {src_empty:.4f}, S1 {len(src_s1)}")
print(f"target: S1 {len(tst_s1)}, GT/S1 {tgt_gt_per:.3f} (assumed {gt_assumed}), reported hidden F {rep['test_f05_hidden']:.4f}")
print("\n| t | hidden F0.5 | pred/S1 | empty |\n|---|---|---|---|")
for t in coarse:
    print(f"| {t:.2f} | {fs[t]:.4f} | {curve[t][0]:.3f} | {curve[t][1]:.4f} |")
print("\n| rule | t | hidden F0.5 | Δ vs best |\n|---|---|---|---|")
for name, t in [("OOF-chosen", t_oof), ("hidden best", t_best), (f"R1 count (GT/S1 {gt_assumed})", t_r1),
                ("R1 count (true GT/S1)", t_r1_true), ("R2 empty", t_r2)]:
    print(f"| {name} | {t:.2f} | {fs[t]:.4f} | {fs[t] - fs[t_best]:+.4f} |")

# ---- per target country (label-free rules applied per test country; lower-only = min(t_oof, rule))
cty = pd.read_csv(f"{dd}/test/test_source1.tsv", sep="\t", usecols=["entity_id", "country"])
cty = dict(zip(cty["entity_id"], cty["country"].str.lower()))
df["c"] = df["s1"].map(cty)
tmaps = {"OOF-chosen": {}, "R1": {}, "R2": {}}
print("\n| country | S1 | GT/S1 | R1 t | R2 t |\n|---|---|---|---|---|")
for c in sorted(set(cty.values())):
    s1c = [s for s, v in cty.items() if v == c]
    dc = df[df["c"] == c]
    cur = {t: stats(dc, s1c, t) for t in grid}
    r1 = min(grid, key=lambda t: abs(cur[t][0] - ratio * gt_assumed))
    r2 = min(grid, key=lambda t: abs(cur[t][1] - src_empty))
    tmaps["OOF-chosen"][c], tmaps["R1"][c], tmaps["R2"][c] = t_oof, r1, r2
    print(f"| {c} | {len(s1c)} | {np.mean([len(truth.get(s, ())) for s in s1c]):.3f} | {r1:.2f} | {r2:.2f} |")
for r in ("R1", "R2"):
    tmaps[r + " lower-only"] = {c: min(t_oof, t) for c, t in tmaps[r].items()}


def f_map(tm):
    sel = df[df["p"] >= df["c"].map(tm)]
    return macro_f05(truth, {k: frozenset(v) for k, v in sel.groupby("s1")["cand"]})


print("\n| per-country rule | thresholds | hidden F0.5 | Δ vs OOF-chosen |\n|---|---|---|---|")
f0 = f_map(tmaps["OOF-chosen"])
for r, tm in tmaps.items():
    f = f_map(tm)
    print(f"| {r} | {', '.join(f'{c} {t:.2f}' for c, t in tm.items())} | {f:.4f} | {f - f0:+.4f} |")
