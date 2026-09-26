"""Metric-aware decision rule for macro F0.5 (HQ, 26 Sep). Works on ANY model's pair probabilities.

The leaderboard averages F0.5 over Source-1 entities, singletons included (empty list = 1.0, any match = 0.0). A single
global threshold on pair probabilities ignores two things the metric rewards:
  * the FIRST match of an S1 deserves a different bar than the 2nd/3rd: if the S1 has any true match, an empty list
    scores 0, so accepting its best candidate is worth more than adding a 3rd pair to a set that is already right;
  * countries differ in calibration (France is unseen in training: the model is less confident there).
Rule (per country): one-to-one first (each S2/S3 record keeps only its highest-p S1), then per S1
    keep its top pair if p >= t_first; keep further pairs if p >= t_rest and the top pair was kept.
t_first / t_rest are tuned per training country on a labelled validation split (macro F0.5 over ALL validation S1,
singletons included). An unseen country (France) gets the train countries' rule shifted down by delta, where delta is
chosen label-free so that its predicted matches per S1 equal the train countries' average on the test set
(bounded by --max-delta); pass --no-shift to keep the plain rule.

    tune:   python -m src.hq_decide tune  --val val_pairs.parquet --val-s1 val_s1.txt --gt train_ground_truth.tsv --out rule.json
    apply:  python -m src.hq_decide apply --test test_pairs.parquet --test-s1 test_source1.tsv --rule rule.json --out matching_results.tsv
Pair files: columns s1, cand, country, p (+ nothing else needed; parquet or TSV). val_s1.txt: one validation S1 id per
line, optionally 'id<TAB>country' (S1s without candidates count too). test_source1.tsv: the official file (all S1 must get a row).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

GRID_FIRST = np.round(np.arange(0.30, 0.96, 0.05), 2)
GRID_REST = np.round(np.arange(0.50, 0.96, 0.05), 2)


def read_pairs(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, sep="\t", dtype={"s1": str, "cand": str})
    df = df[["s1", "cand", "country", "p"]].copy()
    df["country"] = df["country"].astype(str).str.strip().str.lower()
    return df


def one_to_one(df: pd.DataFrame) -> pd.DataFrame:
    """Each S2/S3 record keeps only its highest-p S1 (the ground truth is strictly one-to-one)."""
    df = df.sort_values(["cand", "p"], ascending=[True, False])
    return df[~df["cand"].duplicated()].copy()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = one_to_one(df)
    df = df.sort_values(["s1", "p"], ascending=[True, False]).reset_index(drop=True)
    df["rank"] = df.groupby("s1").cumcount().to_numpy()
    top = df["p"].where(df["rank"] == 0).groupby(df["s1"]).transform("max")
    df["p_top"] = top.to_numpy()
    return df


def keep_mask(df: pd.DataFrame, t_first: float, t_rest: float) -> np.ndarray:
    p, r, pt = df["p"].to_numpy(), df["rank"].to_numpy(), df["p_top"].to_numpy()
    top_kept = pt >= t_first
    return ((r == 0) & top_kept) | ((r > 0) & top_kept & (p >= t_rest))


def macro_f05(df: pd.DataFrame, kept: np.ndarray, y: np.ndarray, s1_all: pd.Index, n_true: pd.Series) -> float:
    """Macro F0.5 over s1_all; df rows are candidate pairs with label y; n_true = true matches per S1 (all S1)."""
    tp = pd.Series(y & kept).groupby(df["s1"].to_numpy()).sum()
    npred = pd.Series(kept).groupby(df["s1"].to_numpy()).sum()
    tp = tp.reindex(s1_all, fill_value=0).to_numpy(float)
    npred = npred.reindex(s1_all, fill_value=0).to_numpy(float)
    ng = n_true.reindex(s1_all, fill_value=0).to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.where(npred > 0, tp / npred, 0.0)
        R = np.where(ng > 0, tp / ng, 0.0)
        F = np.where(tp > 0, 1.25 * P * R / (0.25 * P + R), 0.0)
    F = np.where(ng == 0, (npred == 0).astype(float), F)
    return float(F.mean())


def load_gt(path: str) -> dict:
    gt = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return {s: set(x for x in m.split(",") if x) for s, m in zip(gt.iloc[:, 0], gt.iloc[:, 1])}


def tune(a) -> None:
    df = prepare(read_pairs(a.val))
    gt = load_gt(a.gt)
    lines = [x.rstrip("\r\n").split("\t") for x in open(a.val_s1, encoding="utf-8") if x.strip()]
    s1_list = [x[0].strip() for x in lines]
    y = np.fromiter((c in gt.get(s, ()) for s, c in zip(df["s1"].to_numpy(), df["cand"].to_numpy())), bool, len(df))
    n_true = pd.Series({s: len(gt.get(s, ())) for s in s1_list})
    cty_of = df.groupby("s1")["country"].first().to_dict()
    # optional 2nd column = country (then S1 without any candidate are placed in their country; otherwise they are
    # left out of the per-country F, which only shifts its level, not the choice of thresholds)
    cty_of.update({x[0].strip(): x[1].strip().lower() for x in lines if len(x) > 1 and x[1].strip()})
    rule = {"countries": {}, "grid": {"first": GRID_FIRST.tolist(), "rest": GRID_REST.tolist()}}
    for c in sorted(df["country"].unique()):
        m = (df["country"] == c).to_numpy()
        sub = df[m].reset_index(drop=True)
        s1_c = pd.Index([s for s in s1_list if cty_of.get(s) == c])
        base = macro_f05(sub, keep_mask(sub, a.base_t, a.base_t), y[m], s1_c, n_true)
        best = (base, a.base_t, a.base_t)
        for tf in GRID_FIRST:
            for tr in GRID_REST:
                f = macro_f05(sub, keep_mask(sub, tf, tr), y[m], s1_c, n_true)
                if f > best[0] + 1e-9:
                    best = (f, float(tf), float(tr))
        rule["countries"][c] = dict(t_first=best[1], t_rest=best[2], f_val=round(best[0], 6), f_base=round(base, 6),
                                    n_s1=int(len(s1_c)))
        print(f"{c}: base thr {a.base_t} F={base:.5f} | best t_first={best[1]} t_rest={best[2]} F={best[0]:.5f} "
              f"({(best[0]-base)*100:+.3f} pt) on {len(s1_c)} S1")
    json.dump(rule, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


def apply(a) -> None:
    rule = json.load(open(a.rule))
    tab = rule["countries"]
    df = prepare(read_pairs(a.test))
    known = [c for c in tab if c in set(df["country"])]
    avg_first = float(np.mean([tab[c]["t_first"] for c in tab]))
    avg_rest = float(np.mean([tab[c]["t_rest"] for c in tab]))
    kept = np.zeros(len(df), bool)
    s1_count = df.groupby("country")["s1"].nunique()
    for c in known:
        m = (df["country"] == c).to_numpy()
        kept[m] = keep_mask(df[m], tab[c]["t_first"], tab[c]["t_rest"])
    unseen = sorted(set(df["country"]) - set(tab))
    if known:
        target = float(np.mean([kept[(df["country"] == c).to_numpy()].sum() / s1_count[c] for c in known]))
    for c in unseen:
        m = (df["country"] == c).to_numpy()
        sub = df[m]
        delta = 0.0
        if not a.no_shift and known:
            lo, hi = 0.0, a.max_delta
            if keep_mask(sub, avg_first, avg_rest).sum() / s1_count[c] < target:
                for _ in range(20):
                    mid = (lo + hi) / 2
                    k = keep_mask(sub, avg_first - mid, avg_rest - mid).sum() / s1_count[c]
                    lo, hi = (mid, hi) if k < target else (lo, mid)
                delta = hi if keep_mask(sub, avg_first - hi, avg_rest - hi).sum() / s1_count[c] <= target else lo
        kept[m] = keep_mask(sub, avg_first - delta, avg_rest - delta)
        print(f"{c} (unseen): t_first={avg_first - delta:.3f} t_rest={avg_rest - delta:.3f} (delta {delta:.3f}), "
              f"matches/S1 {kept[m].sum() / s1_count[c]:.3f} (train-country average {target if known else float('nan'):.3f})")
    for c in known:
        m = (df["country"] == c).to_numpy()
        print(f"{c}: t_first={tab[c]['t_first']} t_rest={tab[c]['t_rest']}, matches/S1 {kept[m].sum() / s1_count[c]:.3f}")
    sel = df[kept]
    out = sel.groupby("s1")["cand"].apply(lambda x: ",".join(sorted(x)))
    s1 = pd.read_csv(a.test_s1, sep="\t", dtype=str, usecols=["entity_id"], keep_default_na=False, quoting=3)["entity_id"]
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s in s1.to_numpy():
            f.write(f"{s}\t{out.get(s, '')}\n")
    print(f"wrote {a.out}: {len(s1)} S1, {int(kept.sum())} pairs")


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("tune")
    t.add_argument("--val", required=True)
    t.add_argument("--val-s1", required=True)
    t.add_argument("--gt", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--base-t", type=float, default=0.80)
    p = sp.add_parser("apply")
    p.add_argument("--test", required=True)
    p.add_argument("--test-s1", required=True)
    p.add_argument("--rule", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-delta", type=float, default=0.30)
    p.add_argument("--no-shift", action="store_true")
    a = ap.parse_args()
    tune(a) if a.cmd == "tune" else apply(a)


if __name__ == "__main__":
    main()
