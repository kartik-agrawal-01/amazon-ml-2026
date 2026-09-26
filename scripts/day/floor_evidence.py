"""NEXT 3 (session 8): does a low decision threshold cost more at FULL density than on the slice? (light, streaming)

In-distribution OOF curves per country: F0.5(t) - F0.5(t_ref) for the v2 full-density train pass vs a slice OOF.
Rule = pipeline 'thr' rule approximation: p >= 0.02, one-to-one per candidate (max-p S1), then p >= t.
S1 without candidates are absent from the dumps (same for every t, so the deltas are unaffected in sign).
Usage: python scripts/day/floor_evidence.py <train_dir> <oof.tsv.gz> [t_ref]
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.metric import f05  # noqa: E402

tr, oof_path = sys.argv[1:3]
t_ref = float(sys.argv[3]) if len(sys.argv) > 3 else 0.70
T = [0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, t_ref, 0.85]

oof = pd.read_csv(oof_path, sep="\t", dtype={"s1": str, "cand": str, "y": np.int8, "p": np.float32})
s1_ids = set(oof["s1"].unique())
country, gt = {}, {}
for ch in pd.read_csv(f"{tr}/train_source1.tsv", sep="\t", usecols=["entity_id", "country"], chunksize=500_000, dtype=str):
    ch = ch[ch["entity_id"].isin(s1_ids)]
    country.update(zip(ch["entity_id"], ch["country"]))
for ch in pd.read_csv(f"{tr}/train_ground_truth.tsv", sep="\t", chunksize=500_000, dtype=str):
    ch = ch[ch.iloc[:, 0].isin(s1_ids)]
    for s, m in zip(ch.iloc[:, 0], ch.iloc[:, 1]):
        gt[s] = frozenset(x for x in str(m).split(",") if x and x != "nan")
d = oof[oof["p"] >= 0.02]
d = d[d["p"] >= d.groupby("cand")["p"].transform("max")]
s1c = pd.Series(country)
res = {}
for t in sorted(set(T)):
    pred = d[d["p"] >= t].groupby("s1")["cand"].agg(frozenset).to_dict()
    f = {s: f05(gt.get(s, frozenset()), pred.get(s, frozenset())) for s in s1_ids}
    fs = pd.Series(f)
    res[t] = fs.groupby(s1c.reindex(fs.index).fillna("?")).mean().to_dict() | {"ALL": fs.mean()}
cs = sorted(res[t_ref])
print(f"`{oof_path}`: {len(s1_ids):,} S1, {len(oof):,} pairs; t_ref {t_ref}")
print("| t | " + " | ".join(cs) + " |\n|---|" + "---|" * len(cs))
for t in sorted(res):
    print(f"| {t:.2f} | " + " | ".join(f"{res[t][c]:.4f} ({res[t][c] - res[t_ref][c]:+.4f})" for c in cs) + " |")
