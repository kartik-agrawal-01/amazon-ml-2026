"""Offline decision-rule sweep on a train-pass OOF dump: separate threshold for candidates with an EMPTY address.

Light (< 1.5 GB): streams the stores for the OOF candidate rids only.
Usage: python scripts/day/oof_rule_sweep.py [oof.tsv.gz] [cache_dir] > runs/day/oof_rule_sweep.md
"""
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, ".")
from src.metric import f05  # noqa: E402

TR = "data/data_extracted/student_resource/dataset/train/"
oof_path = sys.argv[1] if len(sys.argv) > 1 else "output_v2_train/oof_pairs.tsv.gz"
cache = sys.argv[2] if len(sys.argv) > 2 else "cache"

oof = pd.read_csv(oof_path, sep="\t", dtype={"s1": str, "cand": str})
need = set(oof.cand) | set(oof.s1)
info = []
for ctry in ["us", "india"]:
    for b in pq.ParquetFile(f"{cache}/train__{ctry}.parquet").iter_batches(batch_size=500_000,
                                                                          columns=["rid", "n_addr", "country"]):
        d = b.to_pandas()
        d = d[d.rid.isin(need)]
        info.append(pd.DataFrame({"rid": d.rid.values, "empty": d.n_addr.fillna("").str.strip().values == "",
                                  "country": d.country.values}))
info = pd.concat(info).drop_duplicates("rid").set_index("rid")
oof["empty"] = info["empty"].reindex(oof.cand.values).fillna(False).values
oof["country"] = info["country"].reindex(oof.s1.values).values
s1_ids = oof.s1.unique()
gt = {}
for ch in pd.read_csv(TR + "train_ground_truth.tsv", sep="\t", chunksize=500_000, dtype=str):
    ch = ch[ch.iloc[:, 0].isin(set(s1_ids))]
    for s, m in zip(ch.iloc[:, 0], ch.iloc[:, 1]):
        gt[s] = frozenset(x for x in str(m).split(",") if x and x != "nan")
best = oof.groupby("cand")["p"].transform("max")
o2o = oof[oof.p >= best]
ctry_of = oof.drop_duplicates("s1").set_index("s1")["country"]


def score(t_full, t_empty):
    thr = np.where(o2o["empty"].values, t_empty, t_full)
    sel = o2o[o2o.p.values >= thr]
    pred = sel.groupby("s1")["cand"].agg(frozenset).to_dict()
    f = pd.Series({s: f05(gt.get(s, frozenset()), pred.get(s, frozenset())) for s in s1_ids})
    return f.groupby(ctry_of.reindex(f.index).values).mean().to_dict() | {"ALL": f.mean()}


print(f"# OOF rule sweep: separate threshold for empty-address candidates\n\n`{oof_path}`; empty-address share of "
      f"OOF pairs {oof['empty'].mean():.2%}, of positives {oof.loc[oof.y == 1, 'empty'].mean():.2%}.\n")
print("| thr (addr present) | thr (empty addr) | ALL | India | US |\n|---|---|---|---|---|")
for tf in [0.65, 0.70, 0.75]:
    for te in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        r = score(tf, te)
        print(f"| {tf:.2f} | {te:.2f} | {r['ALL']:.5f} | {r.get('India', np.nan):.5f} | {r.get('US', np.nan):.5f} |")
        sys.stdout.flush()
