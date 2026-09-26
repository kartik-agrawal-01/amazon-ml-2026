"""Fast lane, model + decision-rule stage only: retrain (or load) the matcher on the cached train pair features and
re-predict the cached test pair features written by `python -m src.pipeline ... --feat-cache DIR`.

Candidates, stage A/B features and the cascade are taken as they are in DIR, so this is exact for model / rule /
feature-subset changes (minutes instead of hours). Feature or blocking changes need src.pipeline (with --cand-cache).

  python -m src.rescore --feat-cache feats_v3 --cache-dir cache_n3 --out-dir output_v3b [--drop-feats a,b] [--save-probs]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

from .model import decide, feature_importance, make_model, oof_predict, subsample_negatives, sweep_rules
from .pipeline import decided_pairs, global_o2o, log, peak_rss_mb, predict_chunked, release_memory, stats_from_counts
from .store import gt_dict, load_gt_pairs, load_meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-cache", required=True)
    ap.add_argument("--cache-dir", default="cache", help="store with the train GT")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--neg-rate", type=float, default=0.5)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--rule", default="auto")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--drop-feats", default="", help="comma list of features left out of the model")
    ap.add_argument("--load-model", default=None, help="model.joblib to use instead of retraining")
    ap.add_argument("--save-probs", action="store_true")
    ap.add_argument("--no-global-o2o", action="store_true", help="one-to-one per block only (v2 behaviour)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    fc = a.feat_cache
    info = json.load(open(os.path.join(fc, "train_feats.json")))
    all_feats = info["feats"]
    drop = {f for f in a.drop_feats.split(",") if f}
    feats = [f for f in all_feats if f not in drop]
    report = dict(args=vars(a), feats=feats, peak_rss_mb={})

    if a.load_model:
        import joblib
        mdl = joblib.load(a.load_model)
        model, feats, best = mdl["model"], mdl["feats"], mdl["best"]
    else:
        meta_tr = load_meta(a.cache_dir, "train")
        gt_pairs = load_gt_pairs(meta_tr)
        one2one = int((gt_pairs["m"].value_counts() > 1).sum()) == 0
        tm = np.load(os.path.join(fc, "train_meta.npz"))
        y, qg, cid, q_rid_tr = tm["y"], tm["qg"], tm["cid"], tm["q_rid"]
        Xall = np.memmap(os.path.join(fc, "train_X.f32"), dtype=np.float32, mode="r", shape=(info["n_rows"], len(all_feats)))
        col = [all_feats.index(f) for f in feats]
        X = np.ascontiguousarray(Xall[:, col]) if len(col) != len(all_feats) else Xall
        log(f"train features {X.shape} ({len(drop)} dropped)")
        oof, kind = oof_predict(X, y, qg, a.folds, a.seed, neg_rate=a.neg_rate, n_jobs=a.n_jobs)
        from sklearn.metrics import roc_auc_score
        q_rid_str = q_rid_tr.astype(str)
        d_uniq, c_codes = np.unique(cid, return_inverse=True)
        d_rid_str = d_uniq.astype(str)
        pairs = pd.DataFrame({"q": qg, "c": c_codes.astype(np.int64), "p": oof})
        gt_tr = gt_dict(gt_pairs, q_rid_str.tolist())
        n_q_tr = int(info["n_q"])
        rng = np.random.default_rng(a.seed)
        sw_q = np.sort(rng.choice(n_q_tr, min(n_q_tr, 100_000), replace=False))
        tab, best = sweep_rules(pairs[np.isin(pairs["q"].values, sw_q)], q_rid_str, d_rid_str, gt_tr,
                                one2one_ok=one2one, q_subset=sw_q)
        print(tab.head(10).to_string(index=False))
        if a.rule != "auto":
            cfg = tab[(tab.rule == a.rule) & (tab.one2one == one2one)]
            if a.rule == "thr":
                cfg = cfg.iloc[(cfg.thr - a.thr).abs().argsort()[:1]]
            best = cfg.iloc[0].to_dict()
        log(f"OOF ({kind}): AUC={roc_auc_score(y, oof):.5f} | OOF macro F0.5 (chosen) = {best['f05']:.5f} with {best}")
        report["oof"] = dict(model=kind, auc=float(roc_auc_score(y, oof)), chosen=best)
        del pairs, gt_tr, oof
        release_memory()
        model, _ = make_model(a.seed, n_jobs=a.n_jobs)
        fit_idx, fit_w = subsample_negatives(y, a.neg_rate, a.seed)
        model.fit(np.ascontiguousarray(X[fit_idx]), y[fit_idx], sample_weight=fit_w)
        imp = feature_importance(model, feats)
        if len(imp):
            report["importance_top25"] = imp.head(25).round(4).to_dict()
        import joblib
        joblib.dump(dict(model=model, feats=feats, best=best), os.path.join(a.out_dir, "rescore_model.joblib"))
        del X, Xall
        release_memory()
    thr_best = best["thr"] if best["rule"] == "thr" else 0.0

    # test: one cached block at a time, rows written in the original S1 order
    files = sorted(glob.glob(os.path.join(fc, "test__*__b*.parquet")))
    if not files:
        raise SystemExit(f"no cached test blocks in {fc}")
    fh_m = open(os.path.join(a.out_dir, "matching_results.tsv"), "w")
    fh_c = open(os.path.join(a.out_dir, "candidate_pairs.tsv"), "w")
    fh_m.write("source1_entity_id\tmatched_entity_ids\n")
    fh_c.write("source1_entity_id\tcandidate_entity_ids\n")
    per, counts_all, probs = {}, [], {}
    buf = dict(c=None, sets={}, order=[], rows=[], dp=[])

    def flush():
        """Write one country's rows after the global one-to-one over all its blocks."""
        c = buf["c"]
        if c is None:
            return
        n_rm = 0
        if buf["dp"]:
            n_rm = global_o2o(buf["sets"], *(np.concatenate([d[i] for d in buf["dp"]]) for i in range(3)))
            log(f"{c}: global one-to-one removed {n_rm} pairs")
        st = per.setdefault(c, dict(n_s1=0, empty=0, pred=0, o2o_removed=n_rm))
        for q_rid, cand_rows in zip(buf["order"], buf["rows"]):
            for i, r in enumerate(q_rid):
                m = buf["sets"][r]
                fh_m.write(f"{r}\t{','.join(sorted(m))}\n")
                fh_c.write(f"{r}\t{cand_rows[i]}\n")
                st["n_s1"] += 1
                st["empty"] += not m
                st["pred"] += len(m)
        buf.update(c=None, sets={}, order=[], rows=[], dp=[])

    for f in files:
        c = re.match(r"test__(.+)__b\d+\.parquet", os.path.basename(f)).group(1)
        if c != buf["c"]:
            flush()
            buf["c"] = c
        df = pd.read_parquet(f)
        q_rid = np.load(f[: -len(".parquet")] + "_qrid.npy").astype(str).astype(object)
        q = pd.Index(q_rid).get_indexer(df["s1"].to_numpy())
        d_uniq, cc = np.unique(df["cand"].to_numpy(), return_inverse=True)
        pairs = pd.DataFrame({"q": q.astype(np.int64), "c": cc.astype(np.int64)})
        pairs["p"] = predict_chunked(model, df, feats)
        sets = decide(pairs, q_rid, d_uniq.astype(object), best["rule"], thr_best, bool(best["one2one"]))
        cand_rows = [""] * len(q_rid)
        for qq, grp in df.groupby(q, sort=False)["cand"]:
            cand_rows[qq] = ",".join(grp.values)
        if not a.no_global_o2o:
            buf["dp"].append(decided_pairs(sets, pairs, q_rid, d_uniq.astype(object)))
        buf["sets"].update(sets)
        buf["order"].append(q_rid)
        buf["rows"].append(cand_rows)
        counts_all.append(np.bincount(q, minlength=len(q_rid)))
        if a.save_probs:
            probs.setdefault(c, []).append(pd.DataFrame({"s1": df["s1"], "cand": df["cand"], "p": pairs["p"],
                                                         "score": df["score"]}))
        log(f"{os.path.basename(f)}: {len(q_rid)} S1, {len(df)} pairs, {sum(1 for r in q_rid if sets[r])} with matches")
        del df, pairs, sets
    flush()
    fh_m.close()
    fh_c.close()
    for c, parts in probs.items():
        pd.concat(parts, ignore_index=True).to_parquet(os.path.join(a.out_dir, f"test_probs_{c}.parquet"), index=False)
    tab = pd.DataFrame({c: dict(n_s1=v["n_s1"], empty_rate=v["empty"] / max(v["n_s1"], 1),
                                mean_matches=v["pred"] / max(v["n_s1"], 1), o2o_removed=v["o2o_removed"])
                        for c, v in per.items()}).T
    print(tab.to_string())
    report["test_pred_by_country"] = tab.reset_index().rename(columns={"index": "country"}).to_dict(orient="records")
    report["test_candidates"] = dict(overall=stats_from_counts(np.concatenate(counts_all)))
    report["peak_rss_mb"]["rescore"] = peak_rss_mb()
    json.dump(report, open(os.path.join(a.out_dir, "report.json"), "w"), indent=2, default=str)
    log("rescore: done")


if __name__ == "__main__":
    main()
