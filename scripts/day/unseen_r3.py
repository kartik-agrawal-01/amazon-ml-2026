"""QUEUE 2d (iii) R3 sure-key matching: pick the target country's threshold t so that its recall of the 'sure' exact-key
pairs (src/hq_keys key_pairs + apply_rules, p_min 0.96; rules calibrated on the source's train key pairs, unseen target
country -> min over train countries as in the pipeline) equals the source's recall of its own sure pairs at t_oof (OOF).
Label-free on the target. Same decision approximation as unseen_thr.py (p >= 0.02, one-to-one per candidate, p >= t).
Light: slice-sized stores only.
Usage: PYTHONPATH=. python scripts/day/unseen_r3.py <data_dir> <cache_dir> <out_dir> [p_min=0.96]"""
import glob
import json
import sys

import numpy as np
import pandas as pd

from src.hq_keys import apply_rules, calibrate, key_pairs, record_keys, rules_for_country, s1_vocab
from src.metric import macro_f05, read_matches_tsv

dd, cd, od = sys.argv[1:4]
p_min = float(sys.argv[4]) if len(sys.argv) > 4 else 0.96
grid = np.round(np.arange(0.02, 0.991, 0.01), 2)
COLS = ["rid", "src", "n_core", "n_nospace", "n_addr"]


def o2o(d):
    d = d[d["p"] >= 0.02]
    return d[d["p"] >= d.groupby("cand")["p"].transform("max")]


def keyed(store, q_rids=None):
    """(sure pair frame [s1, cand, rule], per-country kq/kd) for one country store; q_rids restricts the queried S1."""
    r = pd.read_parquet(store, columns=COLS)
    s1, docs = r[r["src"] == 1].reset_index(drop=True), r[r["src"] != 1].reset_index(drop=True)
    kq_all, kd = record_keys(s1), record_keys(docs)
    q = s1 if q_rids is None else s1[s1["rid"].isin(q_rids)].reset_index(drop=True)
    kq = kq_all if q_rids is None else kq_all[s1["rid"].isin(q_rids).to_numpy()].reset_index(drop=True)
    P = key_pairs(kq, kd, vocab=s1_vocab(kq_all), kq_ctx=kq_all)
    del kd, kq
    P["s1"] = q["rid"].to_numpy()[P.qi.to_numpy()]
    P["cand"] = docs["rid"].to_numpy()[P.ci.to_numpy()]
    return P


def recall(sure, dec, t):
    got = dec.loc[dec["p"] >= t, ["s1", "cand"]].assign(hit=1)
    return float(sure.merge(got, on=["s1", "cand"], how="left")["hit"].fillna(0).mean()) if len(sure) else float("nan")


rep = json.load(open(f"{od}/report.json"))
t_oof = rep["oof"]["chosen"]["thr"]
# ---- source: calibrate rules per train country on its OOF S1 (context: all its S1), sure pairs, recall at t_oof
oof = pd.read_csv(f"{od}/oof_pairs.tsv.gz", sep="\t", usecols=["s1", "cand", "p"])
oof_s1 = set(oof["s1"].unique())
oof = o2o(oof)
tr_gt = read_matches_tsv(f"{dd}/train/train_ground_truth.tsv")
calibs, src_sure = {}, []
for f in sorted(glob.glob(f"{cd}/train__*.parquet")):
    c = f.split("__")[-1][:-8]
    P = keyed(f, oof_s1)
    y = np.array([cn in tr_gt.get(s, ()) for s, cn in zip(P["s1"], P["cand"])])
    calibs[c] = calibrate(P, y)
    sure = P[apply_rules(P, dict(zip(calibs[c].rule, calibs[c].p)), p_min)][["s1", "cand", "rule"]]
    print(f"source {c}: {len(P)} key pairs, {len(sure)} sure ({sure.s1.nunique()} S1), "
          f"sure rules {sorted(calibs[c][calibs[c].p >= p_min].rule)}; sure recall at t_oof {recall(sure, oof, t_oof):.4f}")
    src_sure.append(sure.assign(c=c))
src_sure = pd.concat(src_sure, ignore_index=True)
src_rec = recall(src_sure, oof, t_oof)
src_n = src_rec / recall(src_sure, oof, 0.02)  # R3n: recall relative to the floor (reachable sure pairs only)
print(f"source pooled sure recall at t_oof {t_oof:.2f}: {src_rec:.4f} (relative to floor {src_n:.4f})")
# ---- target
truth = read_matches_tsv(f"{dd}/test_ground_truth_HIDDEN.tsv")
df = pd.concat([o2o(pd.read_parquet(f, columns=["s1", "cand", "p"])).assign(c=f.split("_")[-1][:-8])
                for f in sorted(glob.glob(f"{od}/test_probs_*.parquet"))], ignore_index=True)


def f_map(tm):
    sel = df[df["p"] >= df["c"].map(tm)]
    return macro_f05(truth, {k: frozenset(v) for k, v in sel.groupby("s1")["cand"]})


tm = {"OOF-chosen": {}, "R3": {}, "R3 own-country": {}, "R3n": {}}
print("\n| country | sure pairs | sure prec (hidden) | recall at t_oof | R3 t (pooled src) | R3 t (own src) | R3n t |\n|---|---|---|---|---|---|---|")
for f in sorted(glob.glob(f"{cd}/test__*.parquet")):
    c = f.split("__")[-1][:-8]
    P = keyed(f)
    rp = rules_for_country(calibs, c)
    sure = P[apply_rules(P, rp, p_min)][["s1", "cand"]]
    prec = float(np.mean([cn in truth.get(s, ()) for s, cn in zip(sure["s1"], sure["cand"])])) if len(sure) else float("nan")
    dc = df[df["c"] == c]
    cur = {t: recall(sure, dc, t) for t in grid}
    own = recall(src_sure[src_sure.c == c], oof, t_oof) if c in calibs else src_rec
    r3 = min(grid, key=lambda t: abs(cur[t] - src_rec))
    r3o = min(grid, key=lambda t: abs(cur[t] - own))
    r3n = min(grid, key=lambda t: abs(cur[t] / cur[0.02] - src_n))
    tm["OOF-chosen"][c], tm["R3"][c], tm["R3 own-country"][c], tm["R3n"][c] = t_oof, r3, r3o, r3n
    print(f"| {c} | {len(sure)} | {prec:.4f} | {cur[round(t_oof, 2)]:.4f} | {r3:.2f} | {r3o:.2f} | {r3n:.2f} |")
    del P
for r in ("R3", "R3 own-country", "R3n"):
    tm[r + " lower-only"] = {c: min(t_oof, t) for c, t in tm[r].items()}
f0 = f_map(tm["OOF-chosen"])
print("\n| per-country rule | thresholds | hidden F0.5 | Δ vs OOF-chosen |\n|---|---|---|---|")
for r, m in tm.items():
    f = f_map(m)
    print(f"| {r} | {', '.join(f'{c} {t:.2f}' for c, t in m.items())} | {f:.4f} | {f - f0:+.4f} |")
