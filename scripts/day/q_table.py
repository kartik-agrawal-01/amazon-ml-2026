"""Q = 0.5*mix + 0.3*xc_us + 0.2*xc_in from runs/day_<tag>_<run>/report.json (missing runs -> baseline B4 value).
Usage: python scripts/day/q_table.py <tag> [<tag> ...]"""
import json
import os
import sys

BASE = dict(slice_mix=0.9805, xc_us=0.9339, xc_in=0.9810)
W = dict(slice_mix=0.5, xc_us=0.3, xc_in=0.2)
for tag in sys.argv[1:]:
    vals, extra = {}, []
    for run in W:
        p = f"runs/day_{tag}_{run}/report.json"
        if os.path.exists(p):
            r = json.load(open(p))
            vals[run] = r["test_f05_hidden"]
            extra.append(f"{run}: F={vals[run]:.4f} cands/S1={r['test_candidates']['overall']['mean']:.2f} "
                         f"runtime={r['runtime_s'] / 60:.1f}min rss={max(r['peak_rss_mb'].values())}MB")
    q = sum(W[k] * vals.get(k, BASE[k]) for k in W)
    q0 = sum(W[k] * BASE[k] for k in W)
    miss = [k for k in W if k not in vals]
    print(f"[{tag}] Q={q:.4f} (dQ vs Q0 {q - q0:+.4f}; baseline used for {miss or 'none'})")
    for e in extra:
        print("   ", e)
