"""Unit test of the reverse-blocking helpers (src/pipeline.py: reverse_pairs, rev_by_query, rev_block, add_rev_pairs)."""
import numpy as np, pandas as pd, scipy.sparse as sp
from src import blocking
from src.pipeline import reverse_pairs, rev_by_query, rev_block, add_rev_pairs
blocking.TOPK_DEVICE["device"] = "cpu"
rng = np.random.default_rng(0)
nq, nd, f = 50, 80, 30
Q = {v: sp.random(nq, f, 0.3, "csr", random_state=i) for i, v in enumerate(["a", "b"])}
D = {v: sp.random(nd, f, 0.3, "csr", random_state=10 + i) for i, v in enumerate(["a", "b"])}
from sklearn.preprocessing import normalize
Q = {v: normalize(m) for v, m in Q.items()}; D = {v: normalize(m) for v, m in D.items()}
rev = reverse_pairs(Q, D, ["a", "b"], 3, 1)
# brute force
exp = {}
for v in ["a", "b"]:
    S = (D[v] @ Q[v].T).toarray()
    for c in range(nd):
        o = np.argsort(-S[c], kind="stable")[:3]
        for r, qa in enumerate(o):
            if S[c, qa] >= 0.05:
                e = exp.setdefault((qa, c), [99, 0, 0]); e[0] = min(e[0], r); e[1] += 1; e[2] += r == 0
got = {(int(a), int(c)): [int(x), int(y), int(z)] for a, c, x, y, z in rev[["qa", "c", "rev_rank", "rev_n", "rev_best"]].itertuples(index=False)}
print("pairs got", len(got), "exp", len(exp), "same keys frac", len(set(got) & set(exp)) / max(len(exp), 1))
agree = np.mean([got[k] == exp[k] for k in set(got) & set(exp)])
print("value agreement on common keys", agree)
assert set(got) == set(exp) and agree == 1.0
sel = np.sort(rng.choice(nq, 20, replace=False)); q_of_all = np.full(nq, -1); q_of_all[sel] = np.arange(20)
rq = rev_by_query(rev, q_of_all)
assert set(rq["q"]) <= set(range(20)) and len(rq) == int(np.isin(rev["qa"], sel).sum())
b = rev_block(rq, np.arange(10, 20), 2)
assert b["q"].between(0, 9).all() and len(b) == int((rq["q"] >= 10).sum())
u = pd.DataFrame({"q": np.array([0, 1], np.int64), "c": np.array([int(b.c.iloc[0]), 999], np.int64),
                  "a_rank": np.int16([0, 1]), "b_rank": np.int16([99, 2]), "n_views": np.int8([1, 2])})
b0 = b.copy(); b0.loc[b0.index[0], "q"] = 0
m = add_rev_pairs(u, b0)
print(m.head(), m.dtypes.to_dict(), sep="\n")
assert len(m) == len(u) + len(b0) - int(((b0.q == 0) & (b0.c == u.c[0])).sum() + ((b0.q == 1) & (b0.c == 999)).sum())
print("OK")
