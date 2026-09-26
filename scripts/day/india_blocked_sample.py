"""Sample India GT pairs that v2's full-density blocking missed (not in the union), with normalised fields. Light.
Usage: python scripts/day/india_blocked_sample.py [n] > runs/day/india_blocked_sample.md"""
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
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
miss = [p for p in gt if p not in un]
rng = np.random.default_rng(0)
miss = [miss[i] for i in rng.choice(len(miss), min(N, len(miss)), replace=False)]
need = {x for p in miss for x in p}
cols = ["rid", "indic", "n_name", "n_ph", "n_addr", "city", "zip"]
parts = []
for b in pq.ParquetFile("cache/train__india.parquet").iter_batches(batch_size=500_000, columns=cols):
    d = b.to_pandas()
    parts.append(d[d.rid.isin(need)])
st = pd.concat(parts).drop_duplicates("rid").set_index("rid")
raw = {}
for f in ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv"]:
    for ch in pd.read_csv(TR + f, sep="\t", chunksize=500_000, dtype=str):
        ch = ch[ch.entity_id.isin(need)]
        raw.update({r.entity_id: (r.business_name, r.business_address) for r in ch.itertuples()})
print(f"# {len(miss)} random India GT pairs missed by v2 blocking (full density)\n")
for s, c in miss:
    print(f"- S1 raw: {raw.get(s)}\n  C  raw: {raw.get(c)}")
    for k, r in [("S1", s), ("C ", c)]:
        x = st.loc[r] if r in st.index else None
        if x is not None:
            print(f"  {k} norm: name=[{x.n_name}] ph=[{x.n_ph}] addr=[{x.n_addr}] city={x.city} zip={x.zip} indic={x.indic}")
