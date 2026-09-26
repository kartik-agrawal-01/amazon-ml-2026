"""QUEUE 6 (region imputation, evidence): France S2/S3 addresses without a region/departement component. City -> region
table from test S1 ('..., <city>, <region>'), purity >= 0.9 and >= 3 S1; how many region-less S2/S3 have a last
component that is a known city, and the table's accuracy on S2/S3 that DO carry a region (label-free sanity check).
Streams the test TSVs (France rows only). Usage: PYTHONPATH=. python scripts/day/fr_region_impute.py"""
import collections

import pandas as pd

from src.normalize import _FR_CANON, fold

D = "data/data_extracted/student_resource/dataset/test"
codes = set(_FR_CANON.values())


def comps(a):
    out = []
    for raw in str(a).split(","):
        c = fold(raw)
        if c:
            out.append(_FR_CANON.get(c, c))
    return out


def rows(src):
    for ch in pd.read_csv(f"{D}/test_source{src}.tsv", sep="\t", usecols=["business_address", "country"],
                          chunksize=500_000, dtype=str):
        yield from ch.loc[ch.country == "France", "business_address"].fillna("")


cnt = collections.defaultdict(collections.Counter)
n1 = n1r = 0
for a in rows(1):
    c = comps(a)
    n1 += 1
    if len(c) >= 2 and c[-1] in codes:
        n1r += 1
        cnt[c[-2]][c[-1]] += 1
tab = {city: k.most_common(1)[0][0] for city, k in cnt.items()
       if sum(k.values()) >= 3 and k.most_common(1)[0][1] / sum(k.values()) >= 0.9}
print(f"S1 France {n1}, ending in a region {n1r / n1:.3f}; cities {len(cnt)}, usable (>=3 S1, purity>=0.9) {len(tab)}")
for src in (2, 3):
    n = nr = nimp = ok = chk = 0
    for a in rows(src):
        c = comps(a)
        n += 1
        if c and c[-1] in codes:
            nr += 1
            if len(c) >= 2 and c[-2] in tab:
                chk += 1
                ok += tab[c[-2]] == c[-1]
        elif c and c[-1] in tab:
            nimp += 1
    print(f"S{src} France {n}: with region {nr / n:.3f}; region-less {n - nr} of which imputable (last comp = known city) "
          f"{nimp} ({nimp / max(n - nr, 1):.3f}); table accuracy on region-carrying docs {ok / max(chk, 1):.3f} (n {chk})")
