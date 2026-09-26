"""QUEUE 5a (part 1): which GT pairs does the full-density BLOCKING miss (pre-cascade union, v2 views, k=10)?
Reads the fast-lane candidate cache (cands_v2/train_<country>__b*.parquet + _qrid.npy + __docs.npy), the train GT and
the train store (only rid/n_core/n_addr/indic of the involved records). Light: streams via pyarrow filters.
Usage: python scripts/day/blocking_miss.py --country india [--cands cands_v2] [--cache cache] [--out runs/day/india_recall.md]
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def jac(a: str, b: str) -> float:
    x, y = set(a.split()), set(b.split())
    return len(x & y) / len(x | y) if x or y else 1.0


def read_filtered(path: str, cols, key: str, values: pa.Array) -> pd.DataFrame:
    """Rows of a parquet file whose `key` is in `values`, read batch by batch (keeps RSS low)."""
    out = []
    for b in pq.ParquetFile(path).iter_batches(batch_size=500_000, columns=cols):
        out.append(b.filter(pc.is_in(b[key], value_set=values)))
    return pa.Table.from_batches(out).to_pandas() if out else pd.DataFrame(columns=cols)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default="india")
    ap.add_argument("--cands", default="cands_v2")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    c = a.country
    docs = np.load(os.path.join(a.cands, f"train_{c}__docs.npy"))
    order = np.argsort(docs)
    docs_sorted = docs[order]
    parts, qrids, off = [], [], 0
    for f in sorted(glob.glob(os.path.join(a.cands, f"train_{c}__b*.parquet"))):
        u = pd.read_parquet(f)
        qr = np.load(f.replace(".parquet", "_qrid.npy"))
        u["q"] = u["q"].astype(np.int64) + off
        off += len(qr)
        parts.append(u)
        qrids.append(qr)
    u = pd.concat(parts, ignore_index=True)
    qrid = np.concatenate(qrids)
    rank_cols = [x for x in u.columns if x.endswith("_rank")]
    print(f"{c}: {len(qrid)} S1, union {len(u)} pairs ({len(u) / len(qrid):.1f}/S1)")

    gt = read_filtered(os.path.join(a.cache, "train_gt_pairs.parquet"), ["s1", "m"], "s1", pa.array(qrid.astype(str)))
    q_of = pd.Series(np.arange(len(qrid)), index=qrid.astype(str))
    gt["q"] = q_of.reindex(gt["s1"].to_numpy()).to_numpy()
    m = gt["m"].to_numpy().astype("S16")
    pos = np.searchsorted(docs_sorted, m)
    pos = np.minimum(pos, len(docs_sorted) - 1)
    found = docs_sorted[pos] == m
    gt["c"] = np.where(found, order[pos], -1)
    print(f"GT pairs of the sample: {len(gt)} ({(~found).sum()} partners not among the country's docs)")
    gt = gt[found].copy()
    hit = gt.merge(u, on=["q", "c"], how="left", indicator=True)
    gt["in_union"] = (hit["_merge"] == "both").to_numpy()
    for col in rank_cols:
        gt[col] = hit[col].fillna(99).to_numpy()
    gt["best_rank"] = gt[rank_cols].min(axis=1)

    # record fields of the involved rids only
    need = pa.array(np.unique(np.concatenate([gt["s1"].to_numpy().astype(str), gt["m"].to_numpy().astype(str)])))
    meta = json.load(open(os.path.join(a.cache, "train_meta.json")))
    path = meta["countries"][c]["path"]
    if not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(a.cache, os.path.basename(path))
    t = read_filtered(path, ["rid", "n_core", "n_ph", "n_addr", "indic"], "rid", need).set_index("rid")
    for side, col in (("q", "s1"), ("d", "m")):
        f = t.reindex(gt[col].to_numpy())
        gt[f"{side}_core"] = f["n_core"].fillna("").astype(str).to_numpy()
        gt[f"{side}_ph"] = f["n_ph"].fillna("").astype(str).to_numpy()
        gt[f"{side}_addr"] = f["n_addr"].fillna("").astype(str).to_numpy()
        gt[f"{side}_indic"] = f["indic"].fillna(0).astype(bool).to_numpy()
    gt["name_j"] = [jac(x, y) for x, y in zip(gt["q_core"], gt["d_core"])]
    gt["ph_j"] = [jac(x, y) for x, y in zip(gt["q_ph"], gt["d_ph"])]
    gt["addr_j"] = [jac(x, y) for x, y in zip(gt["q_addr"], gt["d_addr"])]
    gt["cat_indic"] = gt["q_indic"] | gt["d_indic"]
    gt["cat_d_addr_empty"] = gt["d_addr"].str.len() == 0
    gt["cat_name_eq"] = gt["q_core"] == gt["d_core"]
    bins = [-0.01, 0, 0.25, 0.5, 0.75, 0.999, 1.0]
    gt["name_j_bin"] = pd.cut(gt["name_j"], bins).astype(str)
    gt["addr_j_bin"] = pd.cut(gt["addr_j"], bins).astype(str)
    gt["ph_j_bin"] = pd.cut(gt["ph_j"], bins).astype(str)
    gt["indic_x_ph"] = gt["cat_indic"].map({True: "indic", False: "latin"}) + "|" + gt["ph_j_bin"]

    lines = [f"# Blocking misses — train/{c} full density (v2 views, k=10, pre-cascade union)", "",
             f"S1 {len(qrid)}, GT pairs {len(gt)}, union {len(u) / len(qrid):.1f}/S1, "
             f"**pair recall {gt['in_union'].mean():.4f}** (misses {int((~gt['in_union']).sum())}, "
             f"{(~gt['in_union']).sum() / len(qrid):.3f}/S1)", ""]
    for by in ("cat_indic", "cat_d_addr_empty", "cat_name_eq", "name_j_bin", "addr_j_bin", "ph_j_bin", "indic_x_ph"):
        g = gt.groupby(by)["in_union"].agg(n="size", recall="mean")
        g["misses"] = (g["n"] * (1 - g["recall"])).round().astype(int)
        g["share_of_misses"] = g["misses"] / max(int((~gt["in_union"]).sum()), 1)
        lines += [f"## by {by}", "", "```\n" + g.reset_index().to_string(index=False, float_format="%.3f") + "\n```", ""]
    ex = gt[~gt["in_union"]].sample(min(25, int((~gt["in_union"]).sum())), random_state=0)
    lines += ["## sample misses (S1 core | cand core || S1 addr | cand addr)", ""]
    lines += [f"- `{r.q_core}` | `{r.d_core}` || `{r.q_addr[:60]}` | `{r.d_addr[:60]}`" for r in ex.itertuples()]
    txt = "\n".join(lines) + "\n"
    print(txt)
    if a.out:
        open(a.out, "w").write(txt)


if __name__ == "__main__":
    main()
