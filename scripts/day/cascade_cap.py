"""QUEUE 8 evidence: what does a lower per-S1 candidate cap cost, per country, at full density?

Proxy on v2's full-density train OOF (output_v2_train/oof_pairs.tsv.gz: the pairs that survived v2's cascade top-10,
with the FINAL model's OOF p). For cap c we keep the top-c pairs per S1 by the final p, then v2's rule (thr + global
one-to-one). The real cascade ranks with a weaker stage-A model, so this is an OPTIMISTIC bound on the loss: if even
this bound loses F0.5 for a country, the cut is off for that country.
Usage: python scripts/day/cascade_cap.py [oof.tsv.gz] [thr] > runs/day/cascade_cap.md
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

oof["country"] = oof["s1"].map(country)
oof["rank"] = oof.groupby("s1")["p"].rank(method="first", ascending=False).astype(int)
n_true = {c: sum(len(gt.get(s, ())) for s in s1_ids if country.get(s) == c) for c in ("India", "US")}
print(f"# Candidate cap per S1 (QUEUE 8) — optimistic proxy on v2's full-density train OOF\n\nSource `{oof_path}` "
      f"({len(oof):,} pairs, {len(s1_ids):,} S1), rule thr {thr:.2f} + global one-to-one. Top-c per S1 by the FINAL "
      f"model's OOF p (the real cascade ranks worse -> real losses are larger).\n")
print("| cap | country | cands/S1 | GT pairs kept (of GT) | TP pairs lost vs cap 10 | OOF F0.5 | ΔF vs cap 10 |")
print("|---|---|---|---|---|---|---|")
base = {}
for cap in (10, 9, 8, 7, 6, 5):
    o = oof[oof["rank"] <= cap]
    best = o.groupby("cand")["p"].transform("max")
    sel = o[(o["p"] >= best) & (o["p"] >= thr)]
    pred = sel.groupby("s1")["cand"].agg(frozenset).to_dict()
    for c in ("India", "US"):
        ss = [s for s in s1_ids if country.get(s) == c]
        f = float(np.mean([f05(gt.get(s, frozenset()), pred.get(s, frozenset())) for s in ss]))
        tp = int(sel[(sel["country"] == c)]["y"].sum())
        oc = o[o["country"] == c]
        if cap == 10:
            base[c] = (f, tp)
        print(f"| {cap} | {c} | {len(oc) / len(ss):.2f} | {oc['y'].sum() / n_true[c]:.4f} | {base[c][1] - tp:,} | "
              f"{f:.5f} | {f - base[c][0]:+.5f} |")
print("\nTrue pairs by final-p rank within the S1 (share of the country's GT pairs in the candidates):\n")
print("| rank | India | US |")
print("|---|---|---|")
for r in range(1, 11):
    cells = [f"{oof[(oof['rank'] == r) & (oof['country'] == c)]['y'].sum() / n_true[c]:.4f}" for c in ("India", "US")]
    print(f"| {r} | " + " | ".join(cells) + " |")
