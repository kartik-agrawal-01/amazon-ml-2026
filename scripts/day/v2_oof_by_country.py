"""QUEUE 2c: v2's train-pass OOF per country from output_v2_train/oof_pairs.tsv.gz (light, streaming).

Rule = v2's chosen rule (thr 0.70 + one-to-one over the whole OOF sample). The S1 set = the S1 ids present in the OOF
dump (train S1 with >= 1 candidate after the cascade; S1 without candidates are not in the dump).
Usage: python scripts/day/v2_oof_by_country.py [oof.tsv.gz] [thr] > runs/day/v2_oof_by_country.md
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.metric import f05  # noqa: E402

TR = "data/data_extracted/student_resource/dataset/train/"
oof_path = sys.argv[1] if len(sys.argv) > 1 else "output_v2_train/oof_pairs.tsv.gz"
thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.70

oof = pd.read_csv(oof_path, sep="\t", dtype={"s1": str, "cand": str, "y": np.int8, "p": np.float32})
s1_ids = set(oof["s1"].unique())
country = {}
for ch in pd.read_csv(TR + "train_source1.tsv", sep="\t", usecols=["entity_id", "country"], chunksize=500_000,
                      dtype=str):
    ch = ch[ch["entity_id"].isin(s1_ids)]
    country.update(zip(ch["entity_id"], ch["country"]))
gt = {}
for ch in pd.read_csv(TR + "train_ground_truth.tsv", sep="\t", chunksize=500_000, dtype=str):
    ch = ch[ch.iloc[:, 0].isin(s1_ids)]
    for s, m in zip(ch.iloc[:, 0], ch.iloc[:, 1]):
        gt[s] = frozenset(x for x in str(m).split(",") if x and x != "nan")

best = oof.groupby("cand")["p"].transform("max")
sel = oof[(oof["p"] >= best) & (oof["p"] >= thr)]
pred = sel.groupby("s1")["cand"].agg(frozenset).to_dict()

rows = []
for s in s1_ids:
    t, p = gt.get(s, frozenset()), pred.get(s, frozenset())
    rows.append((country.get(s, "?"), f05(t, p), len(t), len(p), len(t & p)))
df = pd.DataFrame(rows, columns=["country", "f", "n_true", "n_pred", "tp"])
cand_recall = oof.groupby(oof["s1"].map(country))["y"].sum()

print(f"# v2 OOF per country (QUEUE 2c)\n\nSource: `{oof_path}` ({len(oof):,} OOF pairs, {len(s1_ids):,} S1 with >= 1 "
      f"candidate). Rule: thr {thr:.2f} + one-to-one over the whole OOF sample (v2's chosen rule). S1 without any "
      f"candidate are not in the dump (so these numbers are slightly optimistic). Note v2's logged OOF 0.9623 was "
      f"computed on a 100K-S1 subsample.\n")
print("| country | S1 | OOF F0.5 | true matches/S1 (GT) | pred matches/S1 | pred-empty | GT-empty | TP/GT (recall) | "
      "precision | GT pairs in candidates |")
print("|---|---|---|---|---|---|---|---|---|---|")
for c, g in list(df.groupby("country")) + [("ALL", df)]:
    ncand = cand_recall.sum() if c == "ALL" else cand_recall.get(c, 0)
    print(f"| {c} | {len(g):,} | {g.f.mean():.5f} | {g.n_true.mean():.3f} | {g.n_pred.mean():.3f} | "
          f"{(g.n_pred == 0).mean():.2%} | {(g.n_true == 0).mean():.2%} | {g.tp.sum() / max(g.n_true.sum(), 1):.4f} | "
          f"{g.tp.sum() / max(g.n_pred.sum(), 1):.4f} | {ncand / max(g.n_true.sum(), 1):.4f} |")
print("\nPer GT bucket (mean F0.5 / pred-empty):\n")
print("| country | 0 true | 1 true | 2 true | 3+ true |")
print("|---|---|---|---|---|")
for c, g in list(df.groupby("country")) + [("ALL", df)]:
    cells = []
    for lo, hi in [(0, 0), (1, 1), (2, 2), (3, 99)]:
        m = g[(g.n_true >= lo) & (g.n_true <= hi)]
        cells.append(f"{m.f.mean():.4f} / {(m.n_pred == 0).mean():.1%} (n={len(m):,})" if len(m) else "-")
    print(f"| {c} | " + " | ".join(cells) + " |")
