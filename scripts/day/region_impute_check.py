"""QUEUE 6 (region imputation, evidence only): per country store, share of S2/S3 with empty state but a city, and how many
of them get a state from S1's city -> state table (modal state, purity >= 0.9, >= 3 S1). Also, for S2/S3 that HAVE a state,
the table's accuracy on them (a label-free check). Usage: python scripts/day/region_impute_check.py <store.parquet>..."""
import sys

import pandas as pd

for f in sys.argv[1:]:
    r = pd.read_parquet(f, columns=["src", "city", "state"])
    s1, d = r[r.src == 1], r[r.src != 1]
    t = s1[(s1.city != "") & (s1.state != "")].groupby(["city", "state"]).size().rename("n").reset_index()
    tot = t.groupby("city").n.transform("sum")
    t = t[t.n == t.groupby("city").n.transform("max")].assign(pur=t.n / tot, tot=tot).drop_duplicates("city")
    t = t[(t.pur >= 0.9) & (t.tot >= 3)].set_index("city")["state"]
    miss = (d.state == "")
    has_city = miss & (d.city != "")
    imp = d.loc[has_city, "city"].map(t).notna()
    chk = d[(d.state != "") & d.city.isin(t.index)]
    acc = (chk.city.map(t) == chk.state).mean() if len(chk) else float("nan")
    print(f"{f}: S1 {len(s1)} (state empty {(s1.state == '').mean():.3f}), S2/S3 {len(d)}: state empty {miss.mean():.3f}, "
          f"of which city present {has_city.sum() / max(miss.sum(), 1):.3f}, imputable {imp.sum()} "
          f"({imp.sum() / max(miss.sum(), 1):.3f} of empty); table {len(t)} cities, accuracy on known-state docs {acc:.3f}")
