"""Offline check of AML_PH=2 on India (light): ph-key equality on India GT pairs (all / v2-blocked) before vs after,
and how much the new key merges distinct names (decoy proxy). Run with AML_PH=2 set:
AML_PH=2 python scripts/day/ph_check.py > runs/day/ph_check.md"""
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.normalize import split_legal
from src.translit import phonetic_key

assert os.environ.get("AML_PH") == "2"
TR = "data/data_extracted/student_resource/dataset/train/"
qrid = np.load("cands_v2/train_india__b000_qrid.npy").astype(str)
docs = np.load("cands_v2/train_india__docs.npy").astype(str)
un = pd.read_parquet("cands_v2/train_india__b000.parquet", columns=["q", "c"])
un = set(zip(qrid[un.q.values], docs[un.c.values]))
s1_set = set(qrid)
gt = []
for ch in pd.read_csv(TR + "train_ground_truth.tsv", sep="\t", chunksize=500_000, dtype=str):
    ch = ch[ch.iloc[:, 0].isin(s1_set)].dropna()
    gt += [(s, x) for s, m in zip(ch.iloc[:, 0], ch.iloc[:, 1]) for x in m.split(",") if x]
gt = pd.DataFrame(gt, columns=["s1", "cand"])
gt["blocked"] = [p not in un for p in zip(gt.s1, gt.cand)]
need = set(gt.s1) | set(gt.cand)
parts, sample = [], []
rng = np.random.default_rng(0)
for b in pq.ParquetFile("cache/train__india.parquet").iter_batches(batch_size=500_000,
                                                                  columns=["rid", "n_name", "n_ph", "indic"]):
    d = b.to_pandas()
    parts.append(d[d.rid.isin(need)])
    sample.append(d.iloc[rng.choice(len(d), min(len(d), 40_000), replace=False)])
st = pd.concat(parts).drop_duplicates("rid").set_index("rid")
st["ph2"] = [phonetic_key(split_legal(n)[0]) for n in st.n_name.fillna("")]
L, R = st.reindex(gt.s1.values), st.reindex(gt.cand.values)
gt["eq_old"] = L.n_ph.values == R.n_ph.values
gt["eq_new"] = L.ph2.values == R.ph2.values
gt["indic"] = R.indic.values == 1
print("# AML_PH=2 offline check (India train S1 of v2's full-density pass)\n")
print("| GT subset | pairs | ph equal (v2/HEAD key) | ph equal (AML_PH=2) |\n|---|---|---|---|")
for lbl, m in [("all", gt.index == gt.index), ("blocked by v2", gt.blocked), ("cand Indic", gt.indic),
               ("blocked & cand Indic", gt.blocked & gt.indic), ("blocked & ASCII", gt.blocked & ~gt.indic)]:
    g = gt[m]
    print(f"| {lbl} | {len(g):,} | {g.eq_old.mean():.2%} | {g.eq_new.mean():.2%} |")
sm = pd.concat(sample)
sm["ph2"] = [phonetic_key(split_legal(n)[0]) for n in sm.n_name.fillna("")]
print(f"\nDecoy proxy on a random {len(sm):,}-record sample of the India store: distinct ph keys "
      f"{sm.n_ph.nunique():,} (old) vs {sm.ph2.nunique():,} (AML_PH=2); mean records per key "
      f"{len(sm) / sm.n_ph.nunique():.3f} vs {len(sm) / sm.ph2.nunique():.3f}.")
