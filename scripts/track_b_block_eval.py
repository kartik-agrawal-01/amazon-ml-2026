"""Track B blocking study on the slice: KEY + DENSE (MiniLM, GPU) candidates vs the TF-IDF views, same S1 sample.

Per country prints pair recall / cands per S1 / seconds for: each key view, all keys, dense@k, keys+dense,
tfidf (given views, k), tfidf+dense, and the encoder throughput (-> full-data projection).

  PYTHONPATH=. python scripts/track_b_block_eval.py --data-dir data_slice --cache-dir cache_slice \
      --n-s1 20000 --n-jobs 2 --views name_c3,name_w,addr_c3,full_w --k 10 --dense-k 10
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

from src import blocking as _blocking
from src.blocking import fit_vectorizers, gt_pairs_block, lexical_candidates, transform, view_text
from src.blocking_b import KEY_VIEWS, dense_parts, encode_table, key_candidates
from src.store import KEEP_COLS, ensure_store, gt_dict, load_country, load_gt_pairs, text_sample

T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data_slice")
    ap.add_argument("--cache-dir", default="cache_slice")
    ap.add_argument("--split", default="train")
    ap.add_argument("--countries", default="india,us")
    ap.add_argument("--n-s1", type=int, default=20000)
    ap.add_argument("--views", default="name_c3,name_w,addr_c3,full_w")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--dense-k", type=int, default=10)
    ap.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--max-df", type=float, default=0.01)
    ap.add_argument("--vec-sample", type=int, default=300_000)
    ap.add_argument("--cap", type=int, default=100)
    ap.add_argument("--per-q", type=int, default=30)
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sim-budget", type=float, default=0.5e9)
    ap.add_argument("--no-tfidf", action="store_true")
    a = ap.parse_args()
    _blocking.TOPK_DEVICE["device"] = "cuda"
    views = a.views.split(",")
    cols = list(dict.fromkeys(list(KEEP_COLS) + ["n_full", "n_name"]))
    meta_tr = ensure_store(a.data_dir, "train", a.cache_dir, a.n_jobs)
    meta = meta_tr if a.split == "train" else ensure_store(a.data_dir, a.split, a.cache_dir, a.n_jobs)
    gt_pairs = load_gt_pairs(meta_tr)
    vecs = None
    if not a.no_tfidf:
        sample = text_sample([meta_tr], a.vec_sample, a.seed)
        vecs = fit_vectorizers(sample, views, max_df=a.max_df, seed=a.seed)
        log(f"vectorisers fitted on {len(sample)} rows")
        del sample
    from src.dense import load_encoder
    model = load_encoder(a.dense_model, "cuda")
    log(f"encoder {a.dense_model} on cuda")
    for c in a.countries.split(","):
        rec = load_country(meta, c, cols)
        src = rec["src"].to_numpy()
        n1 = int((src == 1).sum())
        q_all, d = rec.iloc[:n1].reset_index(drop=True), rec.iloc[n1:].reset_index(drop=True)
        del rec
        rng = np.random.default_rng(a.seed)
        q_sel = np.sort(rng.choice(len(q_all), min(a.n_s1, len(q_all)), replace=False))
        q = q_all  # keep the full S1 table (embeddings of all S1 are what the full run would compute)
        gt_c = gt_dict(gt_pairs, q["rid"].to_numpy()[q_sel].tolist())
        d_src = d["src"].to_numpy().astype(np.int8)
        gp = gt_pairs_block(q["rid"].to_numpy()[q_sel], d["rid"].to_numpy(), gt_c)
        gp_set = set(zip(q_sel[gp["q"].to_numpy()].tolist(), gp["c"].tolist()))
        n_gt = len(gp_set)
        log(f"{a.split}/{c}: {len(q_sel)} of {len(q)} S1, {len(d)} docs, {n_gt} GT pairs")
        found, secs = {}, {}

        def add(parts, name, t):
            s = set()
            for p in parts:
                s |= set(zip(p["q"].tolist(), p["c"].tolist()))
            found[name] = s
            secs[name] = t

        # --- keys
        t = time.time()
        kparts = key_candidates(q, d, d_src, KEY_VIEWS, cap=a.cap, per_q=a.per_q, q_sel=q_sel, verbose=True)
        t_keys = time.time() - t
        for v in KEY_VIEWS:
            add([p for p in kparts if len(p) and p["view"].iloc[0] == v], f"key {v}", np.nan)
        add(kparts, "KEYS (all)", t_keys)
        # --- dense
        d_emb, t_enc_d = encode_table(d, model, os.path.join(a.cache_dir, f"emb_{a.split}_{c}_docs.npy"))
        q_emb, t_enc_q = encode_table(q, model, os.path.join(a.cache_dir, f"emb_{a.split}_{c}_s1.npy"))
        enc_rate = (len(d) + len(q)) / max(t_enc_d + t_enc_q, 1e-9) if (t_enc_d + t_enc_q) > 1 else np.nan
        t = time.time()
        dparts = dense_parts(q_emb, d_emb, d_src, a.dense_k, q_sel=q_sel, device="cuda", verbose=True,
                             sim_budget_bytes=a.sim_budget)
        t_dense = time.time() - t
        add(dparts, f"DENSE k={a.dense_k}", t_dense)
        for kk in (3, 5):
            add([p[p["rank"] < kk] for p in dparts], f"dense k={kk}", np.nan)
        add(kparts + dparts, "KEYS + DENSE", t_keys + t_dense)
        add(kparts + [p[p["rank"] < 5] for p in dparts], "KEYS + dense k=5", np.nan)
        # --- tfidf
        if vecs is not None:
            t = time.time()
            d_mats = {v: transform(vecs[v], view_text(d, v), a.n_jobs) for v in views}
            t_dm = time.time() - t
            q_mats = {v: transform(vecs[v], view_text(q.iloc[q_sel], v), 1) for v in views}
            t = time.time()
            lparts = lexical_candidates(q_mats, d_mats, d_src, a.k, views, verbose=True, n_threads=a.n_jobs)
            t_lex = time.time() - t
            for p in lparts:
                p["q"] = q_sel[p["q"].to_numpy()]
            add(lparts, f"TFIDF {len(views)} views k={a.k}", t_lex)
            add(lparts + dparts, "TFIDF + DENSE", t_lex + t_dense)
            add(lparts + kparts, "TFIDF + KEYS", t_lex + t_keys)
            add(lparts + kparts + dparts, "TFIDF + KEYS + DENSE", t_lex + t_keys + t_dense)
            log(f"{c}: tfidf doc matrices {t_dm:.0f}s ({len(views)} views, {a.n_jobs} jobs)")
            del d_mats, q_mats
        rows = [dict(subset=n, recall=len(s & gp_set) / max(n_gt, 1), cands_per_s1=len(s) / len(q_sel),
                     secs=secs[n], ms_per_s1=1000 * secs[n] / len(q_sel)) for n, s in found.items()]
        print(f"\n=== {a.split}/{c}: {len(q_sel)} S1 vs {len(d)} docs, {n_gt} GT pairs | encoder {enc_rate:.0f} rec/s "
              f"(docs {t_enc_d:.0f}s, S1 {t_enc_q:.0f}s; 0 = cached)")
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(flush=True)
        del d, q_all, q, d_emb, q_emb, kparts, dparts, found
    log("done")


if __name__ == "__main__":
    main()
