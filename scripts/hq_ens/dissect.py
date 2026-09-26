"""Dissect a new submission vs v2 and Soha-1 (label-free): per-source count distributions vs the train GT, and the
cells (exact-key relation) of the pairs it adds / drops, with the train rate of each cell."""
import pickle, sys, collections, time
import numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/aml/ens")
from load import read_res
t0 = time.time()
B = 10_000_000_000
NEW = sys.argv[1]
v2, so1, cty = pickle.load(open("/home/claude/aml/ens/res.pkl", "rb"))
_, s2 = read_res(NEW)
print(f"loaded {time.time()-t0:.0f}s", flush=True)

# --- train GT per-source distribution (S2 count, S3 count per S1) by country
gt = pd.read_csv("/home/claude/aml/data/train/train_ground_truth.tsv", sep="\t", dtype=str)
print("GT columns:", list(gt.columns), len(gt))
s1col = [c for c in gt.columns if "source1" in c.lower() or c.lower().startswith("s1")][0]
mcol = [c for c in gt.columns if c != s1col][0]
tr1 = pd.read_csv("/home/claude/aml/data/train/train_source1.tsv", sep="\t", dtype=str, usecols=["entity_id", "country"])
tcty = dict(zip(tr1.entity_id, tr1.country.str.lower()))
def src_counts(d, cmap):
    rows = []
    for s, m in d.items():
        ids = [x for x in (m if isinstance(m, (list, tuple, set)) else str(m).split(",")) if x and x != "nan"]
        n2 = sum(1 for x in ids if x.startswith("S2")); n3 = sum(1 for x in ids if x.startswith("S3"))
        rows.append((cmap.get(s, "?"), n2, n3))
    return pd.DataFrame(rows, columns=["cty", "n2", "n3"])
gtd = {s: (str(m).split(",") if isinstance(m, str) and m else []) for s, m in zip(gt[s1col], gt[mcol].fillna(""))}
G = src_counts(gtd, tcty)
def dist(D, name):
    out = []
    for c, g in D.groupby("cty"):
        n2 = g.n2.clip(upper=4).value_counts(normalize=True).reindex(range(5), fill_value=0)
        n3 = g.n3.clip(upper=4).value_counts(normalize=True).reindex(range(5), fill_value=0)
        out.append(dict(file=name, cty=c, **{f"S2={k}": round(v * 100, 1) for k, v in n2.items()},
                        **{f"S3={k}": round(v * 100, 1) for k, v in n3.items()},
                        mean2=round(g.n2.mean(), 3), mean3=round(g.n3.mean(), 3)))
    return out
tab = dist(G, "train GT")
for name, d in (("v2", v2), ("soha1", so1), ("soha2", s2)):
    tab += dist(src_counts(d, cty), name)
pd.set_option("display.width", 250)
T = pd.DataFrame(tab).sort_values(["cty", "file"])
print(T.to_string(index=False), flush=True)

# --- identical sets soha2 vs soha1 / v2 per country
for name, ref in (("soha1", so1), ("v2", v2)):
    same = collections.Counter(); tot = collections.Counter()
    for s, m in s2.items():
        c = cty[s]; tot[c] += 1
        same[c] += set(m) == set(ref[s])
    print(f"identical sets soha2 vs {name}:", {c: round(same[c] / tot[c], 4) for c in tot})

# --- adds / drops vs v2 and vs soha1 by exact-key cell
P = pd.read_pickle("/home/claude/aml/ens/test_pairs_p.pkl")[["q_rid", "c_rid", "cty", "p", "cell"]]
def pairs(d):
    q, c = [], []
    for s, m in d.items():
        r = int(s[3:]) + B
        for x in m:
            q.append(r); c.append(int(x[1]) * B + int(x[3:]))
    return pd.DataFrame({"q_rid": np.array(q, np.int64), "c_rid": np.array(c, np.int64)})
X = pairs(s2).assign(inX=True)
N1 = {"us": 663_106, "india": 809_986, "france": 259_452}
CN = {0: "us", 1: "india", 2: "france"}
qc = pd.Series({int(s[3:]) + B: {"us": 0, "india": 1, "france": 2}[c] for s, c in cty.items()})
for name, ref in (("v2", v2), ("soha1", so1)):
    R = pairs(ref).assign(inR=True)
    U = R.merge(X, on=["q_rid", "c_rid"], how="outer")
    U[["inR", "inX"]] = U[["inR", "inX"]].fillna(False).astype(bool)
    U = U[U.inR ^ U.inX]
    U["cty"] = qc.reindex(U.q_rid.values).values
    U = U.merge(P[["q_rid", "c_rid", "p", "cell"]], on=["q_rid", "c_rid"], how="left")
    U["cell"] = U.cell.astype(str).where(U.cell.notna(), "(no exact key)")
    U["side"] = np.where(U.inX, "soha2_only", f"{name}_only")
    for k in (0, 1, 2):
        u = U[U.cty == k]
        g = u.groupby(["cell", "side"]).size().unstack(fill_value=0)
        g = (g / N1[CN[k]]).round(4)
        g["p_train"] = u.groupby("cell").p.mean().round(3)
        g["tot"] = g.drop(columns="p_train").sum(axis=1)
        print(f"\n== {CN[k]}: pairs per S1 only in one file (soha2 vs {name}), top cells")
        print(g.sort_values("tot", ascending=False).drop(columns="tot").head(14).to_string())
print(f"done {time.time()-t0:.0f}s")
