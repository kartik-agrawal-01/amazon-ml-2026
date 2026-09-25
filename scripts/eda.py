"""First-look EDA on the real dataset. Prints everything we need to adapt the pipeline.

Usage (repo root):  python scripts/eda.py --data-dir data [--n-examples 25]
"""
import argparse
import os
import random
import re
import sys
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data import describe, find_files, load_split  # noqa: E402
from src.model import check_one_to_one  # noqa: E402
from src.normalize import add_normalized  # noqa: E402


def script_of(ch):
    try:
        return unicodedata.name(ch).split()[0]
    except ValueError:
        return "UNKNOWN"


def text_profile(s: pd.Series, label: str):
    s = s.fillna("").astype(str)
    lens = s.str.len()
    non_ascii = s.map(lambda x: any(ord(c) > 127 for c in x))
    scripts = Counter(script_of(c) for x in s.sample(min(len(s), 5000), random_state=0) for c in x if c.isalpha())
    tot = sum(scripts.values()) or 1
    top_scripts = {k: round(v / tot, 3) for k, v in scripts.most_common(5)}
    print(f"    {label:<5} empty={np.mean(lens == 0):.1%} len p5/50/95={np.percentile(lens, [5, 50, 95]).round().tolist()} "
          f"non-ascii rows={non_ascii.mean():.1%} scripts={top_scripts}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n-examples", type=int, default=25)
    a = ap.parse_args()
    random.seed(0)

    print("== files found")
    for split, d in find_files(a.data_dir).items():
        for k, p in d.items():
            print(f"  {split:<5} {str(k):<3} {p}  ({os.path.getsize(p) / 1e6:.1f} MB)")
    others = [p for p in (os.path.join(r, f) for r, _, fs in os.walk(a.data_dir) for f in fs)
              if not p.lower().endswith((".tsv", ".csv"))]
    print("  other files:", others[:30])

    tr, gt, info_tr = load_split(a.data_dir, "train")
    te, _, info_te = load_split(a.data_dir, "test")
    describe(tr, gt, info_tr, "train")
    describe(te, None, info_te, "test")
    for split, info in (("train", info_tr), ("test", info_te)):
        for s, sch in info["schema"].items():
            print(f"  {split} S{s} columns: {sch['all_columns']}")

    print("\n== text profile (raw)")
    for s in (1, 2, 3):
        print(f"  S{s}")
        sub = tr[tr.src == s]
        text_profile(sub["name"], "name")
        text_profile(sub["addr"], "addr")
        ids = sub["rid"].head(3).tolist()
        print(f"    id examples: {ids} | id pattern: {Counter(re.sub(r'[0-9]', '9', x) for x in sub.rid.head(2000)).most_common(3)}")

    trn = add_normalized(tr)
    print("\n== normalised token stats (train)")
    for s in (1, 2, 3):
        sub = trn[trn.src == s]
        last_name = Counter(x.split()[-1] for x in sub["n_name"] if x)
        legal = Counter(sub["legal"])
        last_addr = Counter(x.split()[-1] for x in sub["n_addr"] if x)
        first_addr = Counter(x.split()[0] for x in sub["n_addr"] if x)
        print(f"  S{s} last name tokens : {last_name.most_common(25)}")
        print(f"  S{s} legal forms found: {legal.most_common(12)}")
        print(f"  S{s} last addr tokens : {last_addr.most_common(20)}")
        print(f"  S{s} first addr tokens: {first_addr.most_common(15)}")
        print(f"  S{s} addr has digits={np.mean(sub.a_nums != ''):.1%} long-number(zip?)={np.mean(sub.a_zip != ''):.1%} "
              f"vague/landmark={sub.vague.mean():.1%}")

    print("\n== ground truth structure")
    n_true = pd.Series({k: len(v) for k, v in gt.items()})
    print(f"  singleton share = {np.mean(n_true == 0):.2%}  (= score of an all-empty submission)")
    per_src = Counter(m.split('-')[0] if '-' in m else '?' for ms in gt.values() for m in ms)
    print(f"  match ids by prefix: {dict(per_src)}")
    by_src = pd.DataFrame([{"n2": sum(1 for m in ms if m in set2), "n3": sum(1 for m in ms if m in set3)}
                           for set2, set3 in [(set(tr.rid[tr.src == 2]), set(tr.rid[tr.src == 3]))]
                           for ms in gt.values()])
    print(f"  per S1: #S2 matches dist {by_src.n2.clip(upper=3).value_counts().sort_index().to_dict()} | "
          f"#S3 matches dist {by_src.n3.clip(upper=3).value_counts().sort_index().to_dict()}")
    matched = set(m for ms in gt.values() for m in ms)
    for s in (2, 3):
        ids = set(tr.rid[tr.src == s])
        print(f"  S{s}: {len(ids & matched)}/{len(ids)} records matched to some S1 ({len(ids & matched) / max(len(ids), 1):.1%}); "
              f"GT ids missing from S{s} file: {len([m for m in matched if m.startswith(f'S{s}') and m not in ids])}")
    print(f"  S2/S3 ids under >1 S1 (1-to-1 violations): {check_one_to_one(gt)}")

    print("\n== matched examples (S1 | match)")
    recd = tr.set_index("rid")
    keys = [k for k, v in gt.items() if v]
    for k in random.sample(keys, min(a.n_examples, len(keys))):
        r1 = recd.loc[k]
        for m in sorted(gt[k]):
            if m in recd.index:
                r2 = recd.loc[m]
                print(f"  {k}: {r1['name'][:40]:<40} | {r1['addr'][:45]:<45}\n  {m}: {r2['name'][:40]:<40} | {r2['addr'][:45]:<45}\n")
    print("== singleton examples (S1 with no match)")
    single = [k for k, v in gt.items() if not v]
    for k in random.sample(single, min(8, len(single))):
        r1 = recd.loc[k]
        print(f"  {k}: {r1['name'][:45]:<45} | {r1['addr'][:60]}")


if __name__ == "__main__":
    main()
