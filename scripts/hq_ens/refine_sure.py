import sys, collections, numpy as np, pandas as pd, gc
def rid_int(ids):
    s = pd.Series(ids, dtype=object); return (s.str[1].astype(np.int64) * 10_000_000_000 + s.str[3:].astype(np.int64)).to_numpy()
T = pd.read_pickle("ens/test_sure.pkl")
sub = T.cat.isin(["n_disjoint | a_eq", "n_swap1 | a_eq", "n_partial | a_eq"]).values
need = set(T.c_rid.values[sub].tolist())
te1 = pd.read_pickle("ens/keys_test_s1.pkl")[["country", "n_core"]]
V = {c: {t for t, k in collections.Counter(t for x in g.n_core.values for t in x.split()).items() if k >= 5} for c, g in te1.groupby("country")}
del te1; gc.collect()
core = {}; ccty = {}
for s in (2, 3):
    d = pd.read_pickle(f"ens/keys_test_s{s}.pkl")[["id", "country", "n_core"]]
    r = rid_int(d.id.values); m = np.fromiter((x in need for x in r.tolist()), bool, len(r))
    core.update(zip(r[m].tolist(), d.n_core.values[m])); ccty.update(zip(r[m].tolist(), d.country.values[m]))
    del d, r, m; gc.collect()
kinds = np.full(len(T), "", dtype=object)
idx = np.flatnonzero(sub)
kinds[idx] = ["realword" if core[c].split() and all(t in V[ccty[c]] for t in core[c].split()) else "invented" for c in T.c_rid.values[idx].tolist()]
T["kind"] = kinds
drop = ((T.cat == "n_disjoint | a_eq") & (T.kind == "realword")) | ((T.cty == 1) & (T.cat == "n_core_eq | a_street_eq_num_missing"))
T = T[~drop].copy()
P = {("n_core_eq | a_eq", ""): 1.0, ("n_reorder | a_eq", ""): 1.0, ("n_subset | a_eq", ""): 0.998, ("n_nsp_eq | a_eq", ""): 1.0,
     ("n_core_eq | a_num_eq_street_diff", ""): 0.97, ("n_nsp_eq | a_num_eq_street_diff", ""): 0.99,
     ("n_core_eq | a_street_eq_num_missing", ""): 0.985, ("n_core_eq | a_c_empty", ""): 0.978,
     ("n_disjoint | a_eq", "invented"): 0.963, ("n_swap1 | a_eq", "invented"): 0.97, ("n_swap1 | a_eq", "realword"): 0.90,
     ("n_partial | a_eq", "invented"): 0.961, ("n_partial | a_eq", "realword"): 0.88}
T["p"] = [P[(c, k)] for c, k in zip(T.cat.values, T.kind.values)]
T.to_pickle("ens/test_sure_refined.pkl")
print("refined sure pairs", len(T), T.groupby("cty").size().to_dict())
