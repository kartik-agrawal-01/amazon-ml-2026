"""Offline check of src.pipeline.adapt_threshold on a --save-probs run: PYTHONPATH=. python scripts/day/check_adapt.py <data> <out>"""
import glob, json, sys
import numpy as np, pandas as pd
from src.metric import macro_f05, read_matches_tsv
from src.pipeline import adapt_threshold
dd, od = sys.argv[1:3]; floor = float(sys.argv[3]) if len(sys.argv) > 3 else 0.02
rep = json.load(open(f"{od}/report.json")); thr = rep["oof"]["chosen"]["thr"]
oof = pd.read_csv(f"{od}/oof_pairs.tsv.gz", sep="\t", usecols=["s1", "cand", "p"])
s1_tr = oof["s1"].unique(); o = oof[oof["p"] >= 0.02]; o = o[o["p"] >= o.groupby("cand")["p"].transform("max")]
kept = set(o.loc[o["p"] >= thr, "s1"]); src_empty = 1 - len(kept) / len(s1_tr)
truth = read_matches_tsv(f"{dd}/test_ground_truth_HIDDEN.tsv")
t1 = pd.read_csv(f"{dd}/test/test_source1.tsv", sep="\t", usecols=["entity_id", "country"])
pred = {}
for f in sorted(glob.glob(f"{od}/test_probs_*.parquet")):
    c = f.split("test_probs_")[1][:-8]
    d = pd.read_parquet(f, columns=["s1", "cand", "p"]); d = d[d["p"] >= 0.02]
    d = d[d["p"] >= d.groupby("cand")["p"].transform("max")]
    sets = {s: frozenset() for s in t1.loc[t1["country"].str.lower() == c, "entity_id"]}
    for s, g in d.groupby("s1")["cand"]:
        sets[s] = frozenset(g)
    t = adapt_threshold(sets, d["s1"].to_numpy(), d["cand"].to_numpy(), d["p"].to_numpy(), thr, src_empty, floor)
    print(f"{c}: src_empty {src_empty:.4f} thr {thr} -> adapt {t:.2f}"); pred.update(sets)
print(f"hidden F0.5 with adapt: {macro_f05(truth, pred):.4f} (reported {rep['test_f05_hidden']:.4f})")
