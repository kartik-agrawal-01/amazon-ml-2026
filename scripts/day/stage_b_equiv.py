"""Bitwise check: features._stage_b_block with rapidfuzz Jaro vs the pure-Python fallback (_jaro = None = old code),
all features, real candidate pairs. Usage: PYTHONPATH=. python scripts/day/stage_b_equiv.py <store> <candidate_pairs.tsv> [N]"""
import sys
import time

import numpy as np
import pandas as pd

from src import features as F

store, cp = sys.argv[1:3]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 100_000
r = pd.read_parquet(store)
q, d = r[r.src == 1].reset_index(drop=True), r[r.src != 1].reset_index(drop=True)
del r
qpos, dpos = pd.Series(np.arange(len(q)), index=q.rid), pd.Series(np.arange(len(d)), index=d.rid)
c = pd.read_csv(cp, sep="\t", dtype=str, nrows=N // 4)
c.columns = ["s1", "m"]
c = c.assign(cand=c.m.str.split(",")).explode("cand")
c = c[c.s1.isin(qpos.index) & c.cand.isin(dpos.index)].head(N)
qi, ci = qpos[c.s1].to_numpy(), dpos[c.cand].to_numpy()
F._set_rec(q, d)
assert F._jaro is not None, "rapidfuzz not importable"
t = time.time(); new = F._stage_b_block(qi, ci); t_new = time.time() - t
rf, F._jaro = F._jaro, None
t = time.time(); old = F._stage_b_block(qi, ci); t_old = time.time() - t
F._jaro = rf
bad = [k for k in old if not np.array_equal(np.asarray(old[k]), np.asarray(new[k]), equal_nan=True)]
print(f"{store}: {len(qi)} pairs, {len(old)} features, differing: {bad or 'none'}; "
      f"stage B {t_old:.1f}s -> {t_new:.1f}s ({1 - t_new / t_old:.0%} less)")
