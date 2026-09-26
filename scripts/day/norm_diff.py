"""Which normalised fields change between v2's and HEAD's normalize.py for given raw records (light).
Setup: git show 7defba7~1:src/normalize.py > src/_normalize_old_tmp.py (delete after). Input: name<TAB>address lines."""
import sys
import pandas as pd
from src import normalize as new
from src import _normalize_old_tmp as old

recs = [l.rstrip("\n").split("\t") for l in open(sys.argv[1])]
df = pd.DataFrame({"name": [r[0] for r in recs], "address": [r[1] for r in recs]})
for tag, m in (("old", old), ("new", new)):
    df[f"name_{tag}"] = df["name"].map(m.fold)
    df[f"core_legal_{tag}"] = df[f"name_{tag}"].map(lambda s: m.split_legal(s))
    df[f"addr_{tag}"] = df["address"].map(m.norm_addr)
for _, r in df.iterrows():
    ch = [k for k in ("name", "core_legal", "addr") if r[f"{k}_old"] != r[f"{k}_new"]]
    print(f"{r['name'][:40]!r} | changed={ch}")
    for k in ch:
        print(f"    {k}: {r[f'{k}_old']!r}\n      -> {r[f'{k}_new']!r}")
