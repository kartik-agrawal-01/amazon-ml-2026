"""Row-level agreement of two matching_results.tsv (+ optional candidate_pairs.tsv) files. Usage:
python scripts/day/compare_matches.py ref_matching.tsv new_matching.tsv [ref_cands.tsv new_cands.tsv]"""
import sys

import pandas as pd


def load(p):
    df = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
    return dict(zip(df.iloc[:, 0], df.iloc[:, 1].map(lambda s: frozenset(x for x in s.split(",") if x))))


def cmp(a, b, name):
    ka, kb = set(a), set(b)
    common = ka & kb
    same = sum(a[k] == b[k] for k in common)
    print(f"{name}: ref {len(ka)} rows, new {len(kb)} rows, common {len(common)}, identical {same} "
          f"({same / max(len(ka), 1):.5%} of ref)")
    return same / max(len(ka), 1)


r = cmp(load(sys.argv[1]), load(sys.argv[2]), "matching_results")
if len(sys.argv) > 4:
    cmp(load(sys.argv[3]), load(sys.argv[4]), "candidate_pairs")
print("GATE", "PASS" if r >= 0.999 else "FAIL")
