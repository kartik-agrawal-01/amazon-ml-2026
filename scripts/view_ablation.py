"""Per-view blocking recall on a sample of train S1 per country: which lexical views can be dropped when the
full run must finish in time. Prints, per country: recall of every single view, of all views, leave-one-out,
and of the 3 base views (name_c3,name_w,addr_c3) + each extra view.

  python scripts/view_ablation.py --n-s1 5000 --countries india,us --n-jobs 4 --topk-device cuda
"""
from __future__ import annotations

import argparse
import itertools
import time

import numpy as np
import pandas as pd

from src import blocking as _blocking
from src.blocking import fit_vectorizers, gt_pairs_block, lexical_candidates, transform, view_text
from src.store import KEEP_COLS, ensure_store, gt_dict, load_gt_pairs, load_country, text_sample

T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--views", default="name_c3,name_w,addr_c3,name_ph,full_w,addr_w")
    ap.add_argument("--base", default="name_c3,name_w,addr_c3")
    ap.add_argument("--countries", default="india,us")
    ap.add_argument("--n-s1", type=int, default=5000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max-df", type=float, default=0.01)
    ap.add_argument("--vec-sample", type=int, default=600_000)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--topk-device", default="auto")
    a = ap.parse_args()
    _blocking.TOPK_DEVICE["device"] = a.topk_device
    views = a.views.split(",")
    base = a.base.split(",")
    cols = [c for c in KEEP_COLS if c != "n_full"]
    meta_te = ensure_store(a.data_dir, "test", a.cache_dir, a.n_jobs)
    meta_tr = ensure_store(a.data_dir, "train", a.cache_dir, a.n_jobs)
    gt_pairs = load_gt_pairs(meta_tr)
    sample = text_sample([meta_tr, meta_te], a.vec_sample, a.seed)
    vecs = fit_vectorizers(sample, views, max_df=a.max_df, seed=a.seed)
    log(f"vectorisers fitted on {len(sample)} rows: {[(v, len(vec.vocabulary_)) for v, vec in vecs.items()]}")
    del sample
    summary = {}
    for c in a.countries.split(","):
        rec = load_country(meta_tr, c, cols)
        src = rec["src"].to_numpy()
        n1 = int((src == 1).sum())
        q_all, d = rec.iloc[:n1].reset_index(drop=True), rec.iloc[n1:].reset_index(drop=True)
        del rec
        q = q_all.sample(min(a.n_s1, len(q_all)), random_state=a.seed).reset_index(drop=True)
        del q_all
        gt_c = gt_dict(gt_pairs, q["rid"].tolist())
        d_src = d["src"].to_numpy().astype(np.int8)
        t = time.time()
        d_mats = {v: transform(vecs[v], view_text(d, v), a.n_jobs) for v in views}
        log(f"{c}: {len(q)} S1, {len(d)} docs; doc matrices in {time.time() - t:.0f}s")
        q_mats = {v: transform(vecs[v], view_text(q, v), 1) for v in views}
        t = time.time()
        parts = lexical_candidates(q_mats, d_mats, d_src, a.k, views, verbose=True, n_threads=a.n_jobs)
        t_block = time.time() - t
        gp = gt_pairs_block(q["rid"].values, d["rid"].values, gt_c)
        gp_set = set(zip(gp["q"].tolist(), gp["c"].tolist()))
        found = {v: set() for v in views}
        for p in parts:
            v = p["view"].iloc[0]
            found[v] |= set(zip(p["q"].tolist(), p["c"].tolist()))
        n_gt = len(gp_set)
        rows = []

        def rec_of(vs):
            s = set()
            for v in vs:
                s |= found[v]
            return len(s & gp_set) / max(n_gt, 1), len(s) / len(q)

        for v in views:
            r, cps = rec_of([v])
            rows.append(dict(subset=f"only {v}", recall=r, cands_per_s1=cps))
        r, cps = rec_of(views)
        rows.append(dict(subset="ALL", recall=r, cands_per_s1=cps))
        for v in views:
            r, cps = rec_of([x for x in views if x != v])
            rows.append(dict(subset=f"ALL - {v}", recall=r, cands_per_s1=cps))
        r, cps = rec_of(base)
        rows.append(dict(subset=f"base {'+'.join(base)}", recall=r, cands_per_s1=cps))
        extras = [v for v in views if v not in base]
        for v in extras:
            r, cps = rec_of(base + [v])
            rows.append(dict(subset=f"base + {v}", recall=r, cands_per_s1=cps))
        for v1, v2 in itertools.combinations(extras, 2):
            r, cps = rec_of(base + [v1, v2])
            rows.append(dict(subset=f"base + {v1} + {v2}", recall=r, cands_per_s1=cps))
        tab = pd.DataFrame(rows)
        print(f"\n=== {c}: {len(q)} S1, {n_gt} GT pairs, blocking {t_block:.0f}s for {len(views)} views x 2 sources "
              f"({1000 * t_block / len(q) / (2 * len(views)):.2f} ms/query/pass)")
        print(tab.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        summary[c] = tab
        del d, d_mats, q_mats, parts, found
    log("done")


if __name__ == "__main__":
    main()
