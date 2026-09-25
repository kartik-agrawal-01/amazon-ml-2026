"""Post-hoc decision-rule tuning on OOF pair probabilities (cheap, CPU, seconds-minutes).

Reads output/oof_pairs.tsv.gz (s1, cand, y, p) written by the pipeline, plus the train ground truth,
and evaluates on macro-F0.5 (over the S1 entities present in the OOF file):
  1. global threshold sweep (fine grid)                    [baseline rule]
  2. expected-F0.5 selection with isotonic-calibrated p    [cross-fitted calibration]
  3. 2nd-stage GROUP model: HistGB on per-pair context features derived from stage-1 p
     (rank in (S1,source) group, gap to group max, group sum/count, reverse best p of the candidate,
     candidate's competition, ...), cross-fitted by S1 -> p2, then threshold / expected-F on p2
  4. per-country thresholds (if --records given to map S1 -> country)
Prints a table; writes the best config to output/rule_config.json for src.pipeline --rule-config.
Usage: python scripts/tune_rules.py --oof output/oof_pairs.tsv.gz --gt <train_ground_truth.tsv> [--records cache/train_norm.pkl]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.metric import expected_f05_topk, macro_f05, read_matches_tsv  # noqa: E402


def decide_thr(df, thr, one2one=True):
    d = df[df.p >= thr]
    if one2one and len(d):
        d = d[d.p >= d.groupby("cand").p.transform("max")]
    return {s1: frozenset(g) for s1, g in d.groupby("s1").cand}


def decide_expf(df, pcol="p", one2one=True, min_p=0.02):
    d = df[df[pcol] >= min_p]
    if one2one and len(d):
        d = d[d[pcol] >= d.groupby("cand")[pcol].transform("max")]
    out = {}
    d = d.sort_values(["s1", pcol], ascending=[True, False])
    for s1, g in d.groupby("s1", sort=False):
        idx, _ = expected_f05_topk(g[pcol].values, max_k=10)
        if len(idx):
            out[s1] = frozenset(g.cand.values[idx])
    return out


def score(gt, pred):
    return macro_f05(gt, {k: pred.get(k, frozenset()) for k in gt})


def group_features(df, pcol="p"):
    """Context features from stage-1 probabilities."""
    f = pd.DataFrame(index=df.index)
    f["p"] = df[pcol]
    f["src"] = df.cand.str[1].astype(int)
    g = df.groupby(["s1", "src"], sort=False)[pcol]
    f["r_p"] = g.rank(ascending=False, method="min")
    f["gap_max"] = g.transform("max") - df[pcol]
    f["grp_sum"] = g.transform("sum")
    f["grp_n"] = g.transform("size")
    f["grp_n05"] = (df[pcol] >= 0.5).groupby([df.s1, df.src]).transform("sum")
    f["grp_n09"] = (df[pcol] >= 0.9).groupby([df.s1, df.src]).transform("sum")
    ga = df.groupby("s1", sort=False)[pcol]
    f["ent_max"] = ga.transform("max")
    f["ent_sum"] = ga.transform("sum")
    f["ent_n05"] = (df[pcol] >= 0.5).groupby(df.s1).transform("sum")
    gc = df.groupby("cand", sort=False)[pcol]
    f["cand_max_other"] = gc.transform("max") - df[pcol]  # >0 -> another S1 wants this candidate more
    f["cand_n"] = gc.transform("size")
    f["cand_rank"] = gc.rank(ascending=False, method="min")
    f["logit"] = np.log(np.clip(df[pcol], 1e-6, 1 - 1e-6) / (1 - np.clip(df[pcol], 1e-6, 1 - 1e-6)))
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="output/oof_pairs.tsv.gz")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--records", default="", help="cache/train_norm.pkl to get S1 country")
    ap.add_argument("--out", default="output/rule_config.json")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    df = pd.read_csv(a.oof, sep="\t", dtype={"s1": str, "cand": str})
    df["src"] = df.cand.str[1].astype(int)
    gt_all = read_matches_tsv(a.gt)
    s1s = df.s1.unique()
    gt = {k: gt_all.get(k, frozenset()) for k in s1s}
    print(f"OOF pairs {len(df)} | S1 entities {len(s1s)} | positives {int(df.y.sum())}")
    results = []

    # 1. global threshold
    best_thr, best = None, -1
    for t in np.round(np.arange(0.30, 0.96, 0.02), 2):
        sc = score(gt, decide_thr(df, t))
        results.append(("thr", t, sc))
        if sc > best:
            best, best_thr = sc, t
    print(f"1. global threshold: best thr={best_thr} F0.5={best:.5f}")

    # 2. isotonic calibration (cross-fitted by S1) + expected-F
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import GroupKFold
    df["p_cal"] = 0.0
    for tr, va in GroupKFold(5).split(df, df.y, df.s1):
        iso = IsotonicRegression(out_of_bounds="clip").fit(df.p.values[tr], df.y.values[tr])
        df.loc[df.index[va], "p_cal"] = iso.predict(df.p.values[va])
    sc_exp_raw = score(gt, decide_expf(df, "p"))
    sc_exp_cal = score(gt, decide_expf(df, "p_cal"))
    print(f"2. expected-F: raw p {sc_exp_raw:.5f} | calibrated p {sc_exp_cal:.5f}")
    results += [("expf_raw", None, sc_exp_raw), ("expf_cal", None, sc_exp_cal)]

    # 3. second-stage group model
    from sklearn.ensemble import HistGradientBoostingClassifier
    X = group_features(df)
    df["p2"] = 0.0
    for tr, va in GroupKFold(5).split(X, df.y, df.s1):
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=50,
                                           random_state=a.seed).fit(X.iloc[tr], df.y.values[tr])
        df.loc[df.index[va], "p2"] = m.predict_proba(X.iloc[va])[:, 1]
    best2_thr, best2 = None, -1
    for t in np.round(np.arange(0.30, 0.96, 0.02), 2):
        sc = score(gt, decide_thr(df.assign(p=df.p2), t))
        if sc > best2:
            best2, best2_thr = sc, t
    sc_exp2 = score(gt, decide_expf(df, "p2"))
    print(f"3. 2nd-stage group model: best thr={best2_thr} F0.5={best2:.5f} | expected-F on p2 {sc_exp2:.5f}")
    results += [("stage2_thr", best2_thr, best2), ("stage2_expf", None, sc_exp2)]

    # 4. per-country thresholds on the better of p / p2
    if a.records and os.path.exists(a.records):
        import pickle
        rec = pickle.load(open(a.records, "rb"))[0]
        ctry = rec.set_index("rid")["country"]
        df["country"] = df.s1.map(ctry).fillna("")
        pcol = "p2" if best2 > best else "p"
        for c, g in df.groupby("country"):
            gt_c = {k: gt[k] for k in g.s1.unique()}
            bt, bs = None, -1
            for t in np.round(np.arange(0.30, 0.96, 0.02), 2):
                sc = score(gt_c, decide_thr(g.assign(p=g[pcol]), t))
                if sc > bs:
                    bs, bt = sc, t
            print(f"4. {c}: n={len(gt_c)} best thr={bt} F0.5={bs:.5f} ({pcol})")

    tab = pd.DataFrame(results, columns=["rule", "thr", "f05"]).sort_values("f05", ascending=False)
    print(tab.head(8).to_string(index=False))
    cfg = tab.iloc[0].to_dict()
    json.dump(cfg, open(a.out, "w"), indent=2, default=str)
    print("best ->", cfg, "written to", a.out)


if __name__ == "__main__":
    main()
