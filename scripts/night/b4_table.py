"""B4 summary: per-country hidden-holdout F0.5 for slice_mix (train US+India), xc_us (train US -> India holdout),
xc_in (train India -> US holdout) + the cross-country drop. Reads runs/<run>/report.json and <out_dir>/matching_results.tsv.
Usage: python scripts/night/b4_table.py   (from repo root, aml env)"""
import json, os, sys
import pandas as pd
sys.path.insert(0, ".")
from src.metric import read_matches_tsv, macro_f05

RUNS = [("slice_mix", "data_slice", "output_slice"), ("xc_us", "data_xc_us", "output_xc_us"), ("xc_in", "data_xc_in", "output_xc_in")]
rows, per_cc = [], {}
for run, dd, od in RUNS:
    rp = f"runs/{run}/report.json"
    if not os.path.exists(rp):
        rows.append(dict(run=run, note="no report.json")); continue
    r = json.load(open(rp))
    br = {b["k"]: b["pair_recall"] for b in r.get("blocking_recall", [])}
    row = dict(run=run, train_s1=r["gt"].get("n_s1") if isinstance(r.get("gt"), dict) else "?",
               block_recall_k10=round(br.get(10, float("nan")), 4),
               cands_after=round(r["cands_after_cascade"]["mean"], 2), oof_f05=round(r["oof"]["chosen"]["f05"], 4),
               hidden_f05=round(r.get("test_f05_hidden", float("nan")), 4), runtime_min=round(r["runtime_s"] / 60, 1),
               peak_rss_mb=max(r["peak_rss_mb"].values()))
    for p in r.get("test_pred_by_country", []):
        row[f"{p['country']}_empty"] = round(p["empty_rate"], 3); row[f"{p['country']}_mean_m"] = round(p["mean_matches"], 2)
    mr = f"{od}/matching_results.tsv"
    if os.path.exists(mr) and os.path.exists(f"{dd}/test_ground_truth_HIDDEN.tsv"):
        pred = read_matches_tsv(mr); gt = read_matches_tsv(f"{dd}/test_ground_truth_HIDDEN.tsv")
        s1 = pd.read_csv(f"{dd}/test/test_source1.tsv", sep="\t", dtype=str, keep_default_na=False, usecols=["entity_id", "country"])
        for c, ids in s1.groupby("country")["entity_id"]:
            g = {k: gt[k] for k in ids if k in gt}
            f = macro_f05(g, pred); row[f"{c.lower()}_f05"] = round(f, 4); per_cc[(run, c.lower())] = f
    rows.append(row)
df = pd.DataFrame(rows)
print(df.to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string(index=False))
for run, c in (("xc_us", "india"), ("xc_in", "us")):
    if (run, c) in per_cc and ("slice_mix", c) in per_cc:
        print(f"cross-country drop on {c} holdout: {run} {per_cc[(run, c)]:.4f} vs slice_mix {per_cc[('slice_mix', c)]:.4f} "
              f"=> {per_cc[(run, c)] - per_cc[('slice_mix', c)]:+.4f}")
