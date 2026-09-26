"""Calibrated P(match) for every exact-key join pair of the test set (label-free scorer input).

Train: every (S1, S2/S3) pair of the FULL US/India train that shares an exact key (train_calib_pairs.pkl, 12.0M pairs, y
from GT) -> cell = (country, name relation, address relation, ctx_core_other, ctx_addr_other, kind) -> P(y), n.
kind = invented / realword for swap1 / partial / disjoint names (S1 vocabulary = tokens used by >= 5 S1 names).
Test: the same cell for every test join pair (test_join_pairs.pkl, 11.5M). France has no labels -> P_fr = mean of
the US and India cell rates (p_lo = min of the two). Cells with n < 50 fall back to the cell without kind.
Output: test_pairs_p.pkl (q_rid, c_rid, cty, p, p_lo, cell)."""
import collections, gc, time
import numpy as np, pandas as pd

t0 = time.time()
KIND_N = ["n_swap1", "n_partial", "n_disjoint"]


def rid_int(ids):
    s = pd.Series(ids, dtype=object)
    return (s.str[1].astype(np.int64) * 10_000_000_000 + s.str[3:].astype(np.int64)).to_numpy()


def kinds_for(D, split):
    """invented / realword for the swap1/partial/disjoint rows of D (else '')."""
    sel = np.flatnonzero(D.ncat.astype(str).isin(KIND_N).values)
    need = set(D.c_rid.values[sel].tolist())
    s1 = pd.read_pickle(f"keys_{split}_s1.pkl")[["country", "n_core"]]
    V = {c: {t for t, k in collections.Counter(t for x in g.n_core.values for t in x.split()).items() if k >= 5}
         for c, g in s1.groupby("country")}
    del s1; gc.collect()
    core, cty = {}, {}
    for s in (2, 3):
        d = pd.read_pickle(f"keys_{split}_s{s}.pkl")[["id", "country", "n_core"]]
        r = rid_int(d.id.values)
        m = np.fromiter((x in need for x in r.tolist()), bool, len(r))
        core.update(zip(r[m].tolist(), d.n_core.values[m]))
        cty.update(zip(r[m].tolist(), d.country.values[m]))
        del d, r, m; gc.collect()
    kind = np.full(len(D), "", dtype=object)
    kind[sel] = ["realword" if core[c].split() and all(t in V[cty[c]] for t in core[c].split()) else "invented"
                 for c in D.c_rid.values[sel].tolist()]
    return kind


def cell_frame(D, kind):
    return pd.DataFrame({"ncat": D.ncat.astype(str).values, "acat": D.acat.astype(str).values,
                         "cco": D.ctx_core_other.values, "cao": D.ctx_addr_other.values, "kind": kind})


# ---- train cells
D = pd.read_pickle("train_calib_pairs.pkl")
C = cell_frame(D, kinds_for(D, "train"))
C["cty"] = D.cty.values.astype(np.int8)
C["y"] = D.y.values
del D; gc.collect()
keys6 = ["cty", "ncat", "acat", "cco", "cao", "kind"]
keys5 = ["cty", "ncat", "acat", "cco", "cao"]
T6 = C.groupby(keys6, observed=True).y.agg(n="size", p="mean").reset_index()
T5 = C.groupby(keys5, observed=True).y.agg(n="size", p="mean").reset_index()
del C; gc.collect()
T6.to_csv("cells_train.csv", index=False)
print(f"train cells: {len(T6)} (with kind), {len(T5)} (without) | {time.time()-t0:.0f}s", flush=True)

# ---- test pairs
T = pd.read_pickle("test_join_pairs.pkl")
X = cell_frame(T, kinds_for(T, "test"))
X["cty"] = T.cty.values.astype(np.int8)
X["q_rid"] = T.q_rid.values
X["c_rid"] = T.c_rid.values
del T; gc.collect()
print(f"test kinds done | {time.time()-t0:.0f}s", flush=True)


def lookup(frame, table, keys, cty_src):
    """P and n of the cell of `frame` rows in `table` for train country cty_src."""
    t = table[table.cty == cty_src].drop(columns="cty")
    m = frame[keys[1:]].merge(t, on=keys[1:], how="left")
    return m.p.to_numpy(), m.n.fillna(0).to_numpy()


out_p = np.full(len(X), np.nan)
out_lo = np.full(len(X), np.nan)
for src_cty in (0, 1, 2):
    idx = np.flatnonzero(X.cty.values == src_cty)
    sub = X.iloc[idx]
    ps, ns = [], []
    for tc in ((src_cty,) if src_cty < 2 else (0, 1)):
        p6, n6 = lookup(sub, T6, keys6, tc)
        p5, n5 = lookup(sub, T5, keys5, tc)
        p = np.where(n6 >= 50, p6, np.where(n5 >= 50, p5, np.nan))
        ps.append(p); ns.append(np.where(n6 >= 50, n6, n5))
    if len(ps) == 1:
        out_p[idx] = ps[0]; out_lo[idx] = ps[0]
    else:  # France: mean / min of the US and India rates (NaN-aware)
        P = np.vstack(ps)
        out_p[idx] = np.nanmean(np.where(np.isnan(P), np.nan, P), axis=0)
        out_lo[idx] = np.nanmin(P, axis=0)
cell = (X.ncat.str[2:] + "|" + X.acat.str[2:] + np.where(X.cco, "|core_o", "") + np.where(X.cao, "|addr_o", "")
        + np.where(X.kind != "", "|" + X.kind, ""))
R = pd.DataFrame({"q_rid": X.q_rid.values, "c_rid": X.c_rid.values, "cty": X.cty.values, "p": out_p, "p_lo": out_lo,
                  "cell": cell.astype("category")})
R = R[~np.isnan(R.p.values)].reset_index(drop=True)
R.to_pickle("test_pairs_p.pkl")
print(f"test pairs with a calibrated P: {len(R)} of {len(X)} | by country {R.cty.value_counts().to_dict()} | "
      f"{time.time()-t0:.0f}s")
print(R.groupby("cty").p.describe().round(3).to_string())
