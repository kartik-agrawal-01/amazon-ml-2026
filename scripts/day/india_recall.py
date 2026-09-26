"""QUEUE 5a: decompose India's missed GT pairs on v2's FULL-DENSITY train pass (light, < 1 GB).

Inputs (all read-only): cands_v2/train_india__b000* (raw union of v2's views for the 60K India train S1, rebuilt by
gate A with v2 code: same 3,588,823 pairs as v2), output_v2_train/oof_pairs.tsv.gz (post-cascade candidates + OOF p),
cache/train__india.parquet (v2 normalised store), train GT.
Stages per GT pair: blocked (not in union) / cascade cut (in union, not in OOF) / model FN (p < thr or lost one-to-one)
/ TP. Plus model FPs per category.
Usage: python scripts/day/india_recall.py > runs/day/india_recall.md
"""
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

THR = 0.70
TR = "data/data_extracted/student_resource/dataset/train/"

qrid = np.load("cands_v2/train_india__b000_qrid.npy").astype(str)
docs = np.load("cands_v2/train_india__docs.npy").astype(str)
un = pd.read_parquet("cands_v2/train_india__b000.parquet", columns=["q", "c", "n_views"])
un = pd.DataFrame({"s1": qrid[un.q.values], "cand": docs[un.c.values], "n_views": un.n_views.values})
del docs
s1_set = set(qrid)

oof = pd.read_csv("output_v2_train/oof_pairs.tsv.gz", sep="\t", dtype={"s1": str, "cand": str})
oof = oof[oof.s1.isin(s1_set)]
best = oof.groupby("cand")["p"].transform("max")
oof["pred"] = (oof.p >= best) & (oof.p >= THR)

gt = []
for ch in pd.read_csv(TR + "train_ground_truth.tsv", sep="\t", chunksize=500_000, dtype=str):
    ch = ch[ch.iloc[:, 0].isin(s1_set)].dropna()
    for s, m in zip(ch.iloc[:, 0], ch.iloc[:, 1]):
        gt += [(s, x) for x in m.split(",") if x]
gt = pd.DataFrame(gt, columns=["s1", "cand"])
gt["y"] = 1

need = set(gt.s1) | set(gt.cand) | set(oof.cand)
cols = ["rid", "indic", "n_name", "n_core", "n_addr", "legal", "city"]
parts = []
for b in pq.ParquetFile("cache/train__india.parquet").iter_batches(batch_size=500_000, columns=cols):
    d = b.to_pandas()
    parts.append(d[d.rid.isin(need)])
st = pd.concat(parts).drop_duplicates("rid").set_index("rid")
del parts


def jac(a, b):
    a, b = set(str(a).split()), set(str(b).split())
    return len(a & b) / len(a | b) if a | b else 0.0


def tag(df):
    L = st.reindex(df.s1.values)
    R = st.reindex(df.cand.values)
    out = pd.DataFrame(index=df.index)
    out["script"] = np.where(R.indic.values == 1, "cand Indic", np.where(L.indic.values == 1, "S1 Indic", "both ASCII"))
    out["cand_addr"] = np.where(R.n_addr.fillna("").str.strip().values == "", "empty", "present")
    nj = np.array([jac(a, b) for a, b in zip(L.n_name.values, R.n_name.values)])
    aj = np.array([jac(a, b) for a, b in zip(L.n_addr.values, R.n_addr.values)])
    out["name_jac"] = pd.cut(nj, [-0.01, 0.0, 0.25, 0.5, 0.75, 0.999, 1.0], labels=["0", "0-.25", ".25-.5", ".5-.75",
                                                                                   ".75-1", "1"]).astype(str)
    out["addr_jac"] = pd.cut(aj, [-0.01, 0.0, 0.25, 0.5, 0.75, 1.0], labels=["0", "0-.25", ".25-.5", ".5-.75",
                                                                            ".75-1"]).astype(str)
    la, ra = L.legal.fillna("").values, R.legal.fillna("").values
    out["legal"] = np.where((la == "") & (ra == ""), "none", np.where(la == ra, "same", np.where(
        (la == "") | (ra == ""), "one side", "differ")))
    return out


g = gt.merge(un[["s1", "cand"]].assign(in_union=True), on=["s1", "cand"], how="left")
g = g.merge(oof[["s1", "cand", "p", "pred"]], on=["s1", "cand"], how="left")
g["stage"] = np.select([g.in_union.isna() & g.p.isna(), g.p.isna(), ~g.pred.astype(bool)],
                       ["blocked", "cascade cut", "model FN"], "TP")
g = g.join(tag(g))
fp = oof[oof.pred & (oof.y == 0)].copy()
fp = fp.join(tag(fp))

n = len(g)
print(f"# India recall decomposition (QUEUE 5a) — v2 full-density train pass\n\n{len(s1_set):,} India train S1, "
      f"{n:,} GT pairs; union {len(un):,} pairs ({len(un) / len(s1_set):.1f}/S1), post-cascade {len(oof):,} "
      f"({len(oof) / len(s1_set):.1f}/S1). Rule thr {THR} + one-to-one (v2). FPs: {len(fp):,}.\n")
print("| stage | pairs | share of GT |\n|---|---|---|")
for s in ["blocked", "cascade cut", "model FN", "TP"]:
    k = int((g.stage == s).sum())
    print(f"| {s} | {k:,} | {k / n:.2%} |")
print(f"| (model FP) | {len(fp):,} | {len(fp) / n:.2%} |")
for dim in ["script", "cand_addr", "name_jac", "addr_jac", "legal"]:
    t = pd.crosstab(g[dim], g.stage).reindex(columns=["blocked", "cascade cut", "model FN", "TP"], fill_value=0)
    t["GT"] = t.sum(axis=1)
    t["FP"] = fp[dim].value_counts().reindex(t.index).fillna(0).astype(int)
    print(f"\n## by {dim}\n\n| {dim} | GT | share of GT | blocked | cascade cut | model FN | TP rate | FP | lost pairs (share of all lost) |")
    print("|---|---|---|---|---|---|---|---|---|")
    lost_all = (g.stage != "TP").sum()
    for idx, r in t.sort_values("GT", ascending=False).iterrows():
        lost = r.GT - r.TP
        print(f"| {idx} | {r.GT:,} | {r.GT / n:.1%} | {r.blocked / r.GT:.1%} | {r['cascade cut'] / r.GT:.1%} | "
              f"{r['model FN'] / r.GT:.1%} | {r.TP / r.GT:.1%} | {r.FP:,} | {lost:,} ({lost / lost_all:.1%}) |")
print("\nBlocked GT pairs per S1-with-a-blocked-pair: "
      f"{(g.stage == 'blocked').sum() / max(g[g.stage == 'blocked'].s1.nunique(), 1):.2f}; S1 with >= 1 blocked pair: "
      f"{g[g.stage == 'blocked'].s1.nunique():,}")
cut = g[g.stage == "cascade cut"]
print(f"Cascade-cut GT pairs: {len(cut):,}. Model FN p quantiles (10/50/90%): "
      f"{np.round(g[g.stage == 'model FN'].p.quantile([.1, .5, .9]).values, 3).tolist()}")
sys.stdout.flush()
