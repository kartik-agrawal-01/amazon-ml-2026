import pickle, numpy as np, pandas as pd, time
t0 = time.time()
v2, so, cty = pickle.load(open("ens/res.pkl", "rb"))
def rid(i): return int(i[1]) * 10_000_000_000 + int(i[3:])
def pairs_df(d, name):
    q, c = [], []
    for s, m in d.items():
        r = rid(s)
        for x in m: q.append(r); c.append(rid(x))
    return pd.DataFrame({"q_rid": np.array(q, np.int64), "c_rid": np.array(c, np.int64), name: True})
V = pairs_df(v2, "in_v2"); S = pairs_df(so, "in_so")
print("pred pairs", len(V), len(S), f"{time.time()-t0:.0f}s", flush=True)
T = pd.read_pickle("ens/test_join_pairs.pkl")
T["cat"] = T.ncat.astype(str) + " | " + T.acat.astype(str)
ctxF = ~T.ctx_core_other & ~T.ctx_addr_other
sure = ((T.cat == "n_core_eq | a_eq") | (T.cat == "n_reorder | a_eq") | (T.cat == "n_subset | a_eq")
        | ((T.cat == "n_nsp_eq | a_eq") & ~T.ctx_core_other)
        | ((T.cat == "n_core_eq | a_num_eq_street_diff") & ~(T.ctx_core_other & T.ctx_addr_other))
        | ((T.cat == "n_nsp_eq | a_num_eq_street_diff") & ctxF)
        | (T.cat == "n_core_eq | a_street_eq_num_missing")
        | ((T.cat == "n_swap1 | a_eq") & ctxF) | ((T.cat == "n_partial | a_eq") & ctxF) | ((T.cat == "n_disjoint | a_eq") & ctxF)
        | ((T.cat == "n_core_eq | a_c_empty") & ~T.ctx_core_other))
T = T[sure].copy()
T = T.merge(V, on=["q_rid", "c_rid"], how="left").merge(S, on=["q_rid", "c_rid"], how="left")
T[["in_v2", "in_so"]] = T[["in_v2", "in_so"]].fillna(False).astype(bool)
# is the record assigned by v2 / soha to ANOTHER S1?
v2_owner = V.groupby("c_rid").q_rid.first(); so_owner = S.set_index("c_rid").q_rid
T["v2_other"] = v2_owner.reindex(T.c_rid.values).notna().values & ~T.in_v2.values
T["so_other"] = so_owner.reindex(T.c_rid.values).notna().values & ~T.in_so.values
# a record can be 'sure' for several S1 (ambiguous) -> flag
T["multi_sure"] = T.groupby("c_rid").q_rid.transform("size").values > 1
T.to_pickle("ens/test_sure.pkl")
names = {0: "us", 1: "india", 2: "france"}
nS1 = {"us": 663106, "india": 809986, "france": 259452}
pd.set_option("display.width", 250)
for k, cn in names.items():
    d = T[T.cty == k]
    print(f"\n== {cn}: sure pairs {len(d)} ({len(d)/nS1[cn]:.2f}/S1) | v2 covers {d.in_v2.mean():.3%} | soha covers {d.in_so.mean():.3%} | "
          f"missed by both {(~d.in_v2 & ~d.in_so).mean():.3%} = {(~d.in_v2 & ~d.in_so).sum()/nS1[cn]:.4f}/S1 | "
          f"missed by v2 {(~d.in_v2).sum()/nS1[cn]:.4f}/S1 (of which record given to another S1 by v2: {(d.v2_other).sum()/nS1[cn]:.4f}, multi-sure {(~d.in_v2 & d.multi_sure).sum()/nS1[cn]:.4f})")
    g = d.groupby("cat").agg(n=("in_v2", "size"), v2=("in_v2", "mean"), so=("in_so", "mean"))
    g["miss_both_per_S1"] = d.groupby("cat").apply(lambda x: (~x.in_v2 & ~x.in_so).sum()).values / nS1[cn]
    g["miss_v2_per_S1"] = d.groupby("cat").apply(lambda x: (~x.in_v2).sum()).values / nS1[cn]
    print(g.sort_values("n", ascending=False).round(4).to_string())
