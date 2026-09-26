import pickle, sys, collections, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/aml")
def load_core(split):
    d1 = pd.read_pickle(f"ens/keys_{split}_s1.pkl")[["id", "country", "n_core"]]
    dr = pd.concat([pd.read_pickle(f"ens/keys_{split}_s{s}.pkl")[["id", "country", "n_core"]] for s in (2, 3)], ignore_index=True)
    return d1, dr
def rid_int(ids):
    s = pd.Series(ids, dtype=object); return (s.str[1].astype(np.int64) * 10_000_000_000 + s.str[3:].astype(np.int64)).to_numpy()
def vocab(d1):
    v = {}
    for c, g in d1.groupby("country"):
        cnt = collections.Counter(t for x in g.n_core.values for t in x.split())
        v[c] = {t for t, k in cnt.items() if k >= 5}
    return v
def kind(core, V):
    toks = core.split()
    if not toks: return "empty"
    return "realword" if all(t in V for t in toks) else "invented"
# ---- train calibration for n_disjoint | a_eq (+ swap1/partial) by realword/invented
tr1, trr = load_core("train")
Vtr = vocab(tr1)
core_by_rid = pd.Series(trr.n_core.values, index=rid_int(trr.id.values))
cty_by_rid = pd.Series(trr.country.values, index=rid_int(trr.id.values))
D = pd.read_pickle("ens/train_calib_pairs.pkl")
D["cat"] = D.ncat.astype(str) + " | " + D.acat.astype(str)
D = pd.read_pickle("ens/train_calib_pairs.pkl")
D["cat"] = D.ncat.astype(str) + " | " + D.acat.astype(str)
sel = D[D.cat.isin(["n_disjoint | a_eq", "n_swap1 | a_eq", "n_partial | a_eq"]) & ~D.ctx_core_other & ~D.ctx_addr_other].copy()
cores = core_by_rid.reindex(sel.c_rid.values).values; ctys = cty_by_rid.reindex(sel.c_rid.values).values
sel["kind"] = [kind(x, Vtr[c]) for x, c in zip(cores, ctys)]
print("TRAIN (ctx F/F):"); print(sel.groupby(["cat", "cty", "kind"]).y.agg(n="size", p="mean").round(3).to_string())
del D, sel, trr, core_by_rid, cty_by_rid
# ---- test: France / US / India additions by kind
te1, ter = load_core("test")
Vte = vocab(te1)
core_te = pd.Series(ter.n_core.values, index=rid_int(ter.id.values)); cty_te = pd.Series(ter.country.values, index=rid_int(ter.id.values))
T = pd.read_pickle("ens/test_sure.pkl")
T = T[T.cat.isin(["n_disjoint | a_eq", "n_swap1 | a_eq", "n_partial | a_eq"])].copy()
cores = core_te.reindex(T.c_rid.values).values; ctys = cty_te.reindex(T.c_rid.values).values
T["kind"] = [kind(x, Vte[c]) for x, c in zip(cores, ctys)]
T["missed_v2"] = ~T.in_v2
print("\nTEST sure pairs (ctx F/F) by kind: coverage by v2 / soha")
print(T.groupby(["cat", "cty", "kind"]).agg(n=("in_v2", "size"), v2=("in_v2", "mean"), so=("in_so", "mean")).round(3).to_string())
fr = T[(T.cty == 2) & (T.cat == "n_disjoint | a_eq")]
fr.assign(core=core_te.reindex(fr.c_rid.values).values).to_pickle("ens/fr_disjoint_sure.pkl")
