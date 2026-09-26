"""QUEUE 9: does jw_fast (rapidfuzz Jaro in C++ + our Winkler prefix term, always applied) reproduce
features.jaro_winkler bit-exactly after the float32 cast? Real pairs from a slice store; speed of both.
Usage: PYTHONPATH=. python scripts/day/jw_rapidfuzz_check.py <store.parquet> <candidate_pairs.tsv> [N=100000]"""
import sys
import time

import numpy as np
import pandas as pd
from rapidfuzz.distance import Jaro

from src.features import jaro_winkler

_jaro = Jaro.similarity


def jw_fast(s1, s2, p=0.1):
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    j = _jaro(s1, s2)
    pre = 0
    for a, b in zip(s1[:4], s2[:4]):
        if a != b:
            break
        pre += 1
    return j + pre * p * (1.0 - j)


store, cp = sys.argv[1:3]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 100_000
r = pd.read_parquet(store, columns=["rid", "src", "n_core", "n_name", "n_addr", "n_nospace", "n_ph"])
q, d = r[r.src == 1].set_index("rid"), r[r.src != 1].set_index("rid")
c = pd.read_csv(cp, sep="\t", dtype=str, nrows=N // 5)
c.columns = ["s1", "m"]
c = c.assign(cand=c.m.str.split(",")).explode("cand")
c = c[c.s1.isin(q.index) & c.cand.isin(d.index)].head(N)
for col, cut in (("n_core", None), ("n_name", None), ("n_nospace", None), ("n_ph", None), ("n_addr", 40)):
    A, B = q.loc[c.s1, col].fillna("").to_numpy(), d.loc[c.cand, col].fillna("").to_numpy()
    if cut:
        A, B = [a[:cut] for a in A], [b[:cut] for b in B]
    t = time.time(); x = np.fromiter((jaro_winkler(a, b) for a, b in zip(A, B)), np.float32, len(A)); t1 = time.time() - t
    t = time.time(); y = np.fromiter((jw_fast(a, b) for a, b in zip(A, B)), np.float32, len(A)); t2 = time.time() - t
    bad = np.flatnonzero(x != y)
    print(f"{col}{'[:40]' if cut else ''}: {len(A)} pairs, exact-equal {1 - len(bad) / len(A):.6f}, "
          f"max |diff| {np.abs(x - y).max():.2e}, py {t1:.2f}s vs rapidfuzz {t2:.2f}s ({t1 / max(t2, 1e-9):.1f}x)")
    for i in bad[:3]:
        print(f"   {A[i]!r} | {B[i]!r} -> {x[i]!r} vs {y[i]!r}")
