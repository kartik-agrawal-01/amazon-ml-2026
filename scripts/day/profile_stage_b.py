"""QUEUE 9 (profile only): cProfile of features._stage_b_block on N real candidate pairs of a slice store (single process).
Generator expressions show up as <genexpr> with their line number in src/features.py -> which features cost most.
Usage: PYTHONPATH=. python scripts/day/profile_stage_b.py <store.parquet> <candidate_pairs.tsv> [N=100000]"""
import cProfile
import pstats
import sys
import time

import numpy as np
import pandas as pd

from src import features as F

store, cp = sys.argv[1:3]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 100_000
r = pd.read_parquet(store)
q, d = r[r.src == 1].reset_index(drop=True), r[r.src != 1].reset_index(drop=True)
qpos, dpos = pd.Series(np.arange(len(q)), index=q.rid), pd.Series(np.arange(len(d)), index=d.rid)
c = pd.read_csv(cp, sep="\t", dtype=str, nrows=N // 5)
c.columns = ["s1", "m"]
c = c.assign(cand=c.m.str.split(",")).explode("cand")
c = c[c.s1.isin(qpos.index) & c.cand.isin(dpos.index)].head(N)
qi, ci = qpos[c.s1].to_numpy(), dpos[c.cand].to_numpy()
F._set_rec(q, d)
t = time.time()
pr = cProfile.Profile()
pr.enable()
out = F._stage_b_block(qi, ci)
pr.disable()
el = time.time() - t
print(f"{len(qi)} pairs, {len(out)} features, {el:.1f}s single process ({1e6 * el / len(qi):.1f} us/pair)")
st = pstats.Stats(pr)
rows = sorted(((v[3], k) for k, v in st.stats.items()), reverse=True)  # cumulative time
lines = open(F.__file__).read().splitlines()
print("\n| cum s | share | function (features.py line: code) |\n|---|---|---|")
for tt, (fn, ln, name) in rows[:30]:
    if name == "_stage_b_block" or "features.py" not in fn and not name.startswith("<"):
        continue
    code = lines[ln - 1].strip()[:90] if fn.endswith("features.py") else ""
    print(f"| {tt:.2f} | {tt / el:.1%} | {name} @{ln}: `{code}` |")
