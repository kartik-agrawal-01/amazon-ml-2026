"""P(match | pair category) on the FULL US/India train, via exact-key joins (no blocking), new normalisation.
Memory-lean: integer ids, hashed legal, n_core strings only fetched for joined pairs."""
import sys, time, gc
import numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/aml")
from ens.cats import tok_rel
t0 = time.time()

def rid_int(ids):
    s = pd.Series(ids, dtype=object)
    src = s.str[1].astype(np.int64)
    return (src * 10_000_000_000 + s.str[3:].astype(np.int64)).to_numpy()

def load(path):
    d = pd.read_pickle(path)
    out = pd.DataFrame({"rid": rid_int(d.id.values), "cty": np.select([d.country.values == "india", d.country.values == "france"], [1, 2], 0).astype(np.int8)})
    for k in ("k_core", "k_nsp", "k_addr", "k_street", "num1"):
        out[k] = d[k].to_numpy()
    out["addr_empty"] = d.addr_empty.to_numpy()
    out["k_legal"] = np.fromiter((hash(x) for x in d.legal.values), np.int64, len(d))
    out["legal_empty"] = (d.legal.values == "")
    core = d.n_core.to_numpy()
    del d; gc.collect()
    return out, core

S1, core1 = load("ens/keys_test_s1.pkl")
R2, core2 = load("ens/keys_test_s2.pkl"); R3, core3 = load("ens/keys_test_s3.pkl")
R = pd.concat([R2, R3], ignore_index=True); coreR = np.concatenate([core2, core3]); del R2, R3, core2, core3; gc.collect()
S1["row"] = np.arange(len(S1)); R["row"] = np.arange(len(R))
print("loaded", len(S1), len(R), f"{time.time()-t0:.0f}s", flush=True)
owner = pd.Series(dtype=np.int64)
cnt_addr = S1.groupby(["cty", "k_addr"]).size().rename("n")
cnt_core = S1.groupby(["cty", "k_core"]).size().rename("n")

def join(key, cap=2000):
    ga = S1.groupby(["cty", key]).size().rename("na"); gb = R.groupby(["cty", key]).size().rename("nb")
    g = pd.concat([ga, gb], axis=1, join="inner"); g = g[g.na * g.nb <= cap]
    a = S1[["cty", key, "row"]].merge(g[[]].reset_index(), on=["cty", key])
    b = R[["cty", key, "row"]].merge(g[[]].reset_index(), on=["cty", key])
    m = a.merge(b, on=["cty", key], suffixes=("_q", "_c"))[["row_q", "row_c"]]
    print(f"  join {key}: {len(m)} pairs {time.time()-t0:.0f}s", flush=True)
    return m

pairs = pd.concat([join("k_core"), join("k_nsp"), join("k_addr")], ignore_index=True).drop_duplicates()
print("union pairs", len(pairs), f"{time.time()-t0:.0f}s", flush=True)
out = []
NC = ["n_core_eq", "n_nsp_eq", "n_reorder", "n_subset", "n_swap1", "n_disjoint", "n_partial", "n_empty"]
AC = ["a_eq", "a_street_eq_num_diff", "a_street_eq_num_missing", "a_num_eq_street_diff", "a_diff", "a_c_empty"]
for lo in range(0, len(pairs), 1_500_000):
    pr = pairs.iloc[lo:lo + 1_500_000]
    rq = pr.row_q.values; rc = pr.row_c.values
    cols = ["cty", "rid", "k_core", "k_nsp", "k_addr", "k_street", "num1", "addr_empty", "k_legal", "legal_empty"]
    q = {k: S1[k].values[rq] for k in cols}; c = {k: R[k].values[rc] for k in cols}
    D = pd.DataFrame({"cty": q["cty"]})
    D["q_rid"] = q["rid"]; D["c_rid"] = c["rid"]
    nc = np.where(q["k_core"] == c["k_core"], "n_core_eq", np.where(q["k_nsp"] == c["k_nsp"], "n_nsp_eq", "")).astype(object)
    rest = np.flatnonzero(nc == "")
    nc[rest] = [tok_rel(x, y) for x, y in zip(core1[rq[rest]], coreR[rc[rest]])]
    D["ncat"] = pd.Categorical(nc, categories=NC)
    ac = np.full(len(D), "a_diff", dtype=object)
    se = q["k_street"] == c["k_street"]
    bothnum = (q["num1"] >= 0) & (c["num1"] >= 0)
    ac[(q["num1"] >= 0) & (q["num1"] == c["num1"]) & ~se] = "a_num_eq_street_diff"
    ac[se & ~bothnum] = "a_street_eq_num_missing"
    ac[se & bothnum] = "a_street_eq_num_diff"
    ac[q["k_addr"] == c["k_addr"]] = "a_eq"
    ac[c["addr_empty"]] = "a_c_empty"
    D["acat"] = pd.Categorical(ac, categories=AC)
    n_addr = cnt_addr.reindex(pd.MultiIndex.from_arrays([q["cty"], c["k_addr"]])).fillna(0).to_numpy()
    D["ctx_addr_other"] = (n_addr - (q["k_addr"] == c["k_addr"])) > 0
    n_core = cnt_core.reindex(pd.MultiIndex.from_arrays([q["cty"], c["k_core"]])).fillna(0).to_numpy()
    D["ctx_core_other"] = (n_core - (q["k_core"] == c["k_core"])) > 0
    D["legal_rel"] = pd.Categorical(np.where(q["legal_empty"] | c["legal_empty"], "l_missing", np.where(q["k_legal"] == c["k_legal"], "l_eq", "l_diff")), categories=["l_eq", "l_diff", "l_missing"])
    out.append(D)
    print(f"  categorised {lo + len(pr)} {time.time()-t0:.0f}s", flush=True)
del S1, R, pairs; gc.collect()
D = pd.concat(out, ignore_index=True)
D = D[~((D.acat == "a_c_empty") & ~D.ncat.isin(["n_core_eq", "n_nsp_eq"]))]
D.to_pickle("ens/test_join_pairs.pkl")
print("saved", len(D), f"{time.time()-t0:.0f}s", flush=True)
