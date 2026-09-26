"""ens2 = v2 -> global one-to-one dedup -> + train-calibrated 'sure' exact-key pairs v2 missed (lean memory)."""
import pickle, sys, collections, gc, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/aml")
from ens.load import read_res, SUB
def s1id(r): return f"S1-{r - 10_000_000_000}"
def cid(r): return f"S{r // 10_000_000_000}-{r % 10_000_000_000}"
_, v2 = read_res(f"{SUB}/v2_matching_results.tsv")
owners = collections.defaultdict(list)
for s, m in v2.items():
    for x in m: owners[x].append(s)
multi = {x: ss for x, ss in owners.items() if len(ss) > 1}; del owners; gc.collect()
_, so = read_res(f"{SUB}/soha_matching_results.tsv")
so_owner = {x: s for s, m in so.items() for x in m if x in multi}; del so; gc.collect()
T = pd.read_pickle("ens/test_sure_refined.pkl")[["q_rid", "c_rid", "cty", "cat", "kind", "p", "multi_sure"]]
sure_key = set((s1id(q), cid(c)) for q, c in zip(T.q_rid.values.tolist(), T.c_rid.values.tolist()) if cid(c) in multi)
res = {s: set(m) for s, m in v2.items()}; del v2; gc.collect()
n_rm = 0; how = collections.Counter(); rm_by = collections.Counter()
cty = pd.read_pickle("ens/keys_test_s1.pkl")[["id", "country"]]; cty = dict(zip(cty.id.values, cty.country.values))
for x, ss in multi.items():
    so_ = [s for s in ss if (s, x) in sure_key]
    if len(so_) == 1: keep = so_[0]; how["sure"] += 1
    elif so_owner.get(x) in ss: keep = so_owner[x]; how["soha"] += 1
    else: keep = None; how["drop_all"] += 1
    for s in ss:
        if s != keep: res[s].discard(x); n_rm += 1; rm_by[cty[s]] += 1
print("dedup:", dict(how), "| pairs removed", n_rm, dict(rm_by))
assigned = {x for m in res.values() for x in m}
T = T[~T.multi_sure]
ALLOW = {  # (cat, cty) -> conservative P(true | v2 excluded) lower bound, from the coverage table (>= 0.80 only)
    ("n_core_eq | a_eq", 2): 0.970, ("n_core_eq | a_num_eq_street_diff", 2): 0.943, ("n_subset | a_eq", 2): 0.966,
    ("n_disjoint | a_eq", 2): 0.917, ("n_partial | a_eq", 2): 0.811, ("n_nsp_eq | a_eq", 2): 1.0,
    ("n_core_eq | a_street_eq_num_missing", 2): 0.998, ("n_nsp_eq | a_num_eq_street_diff", 2): 0.995, ("n_reorder | a_eq", 2): 0.928,
    ("n_core_eq | a_eq", 1): 0.919, ("n_core_eq | a_c_empty", 1): 0.848, ("n_nsp_eq | a_num_eq_street_diff", 1): 0.859,
    ("n_nsp_eq | a_eq", 1): 1.0, ("n_reorder | a_eq", 1): 0.957,
    ("n_core_eq | a_eq", 0): 0.886, ("n_core_eq | a_street_eq_num_missing", 0): 0.976, ("n_nsp_eq | a_eq", 0): 1.0,
    ("n_nsp_eq | a_num_eq_street_diff", 0): 0.977}
key = list(zip(T.cat.values, T.cty.values))
T = T[[k in ALLOW for k in key]].copy()
T["p_rule"] = T.p.values
T["p"] = [ALLOW[k] for k in zip(T.cat.values, T.cty.values)]
names = {0: "us", 1: "india", 2: "france"}
nS1 = {"us": 663106, "india": 809986, "france": 259452}
gain = collections.defaultdict(float); cnt = collections.Counter(); bycat = collections.defaultdict(float); bycnt = collections.Counter()
T = T.sort_values("q_rid")
for q, c, p, k, cat, kind in zip(T.q_rid.values.tolist(), T.c_rid.values.tolist(), T.p.values.tolist(), T.cty.values.tolist(), T.cat.values.tolist(), T.kind.values.tolist()):
    s, x = s1id(q), cid(c)
    if x in res[s] or x in assigned: continue
    kk = len(res[s])
    g = 0.25 / (1.25 * kk + 0.25); l = 1.0 / (1.25 * kk + 1.0)
    e = p * g - (1 - p) * l
    key = (names[k], cat + (" [" + kind + "]" if kind else ""))
    gain[names[k]] += e; cnt[names[k]] += 1; bycat[key] += e; bycnt[key] += 1
    res[s].add(x); assigned.add(x)
for c in ("us", "india", "france"):
    print(f"  {c}: adds {cnt[c]} ({cnt[c]/nS1[c]:.3f}/S1) | expected dF(country) {gain[c]/nS1[c]:+.4f} | expected dLB {gain[c]/1732544:+.4f}")
print("  total expected dLB from adds:", f"{sum(gain.values())/1732544:+.4f}")
for key, e in sorted(bycat.items(), key=lambda z: -z[1])[:14]: print(f"    {key[0]:7} {key[1]:50} n={bycnt[key]:6d}  dLB {e/1732544:+.5f}")
pickle.dump(res, open("ens/ens2c_sets.pkl", "wb"))
