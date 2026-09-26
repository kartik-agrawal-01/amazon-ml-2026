"""Box-side per-country table for a full-data submission vs v2 (QUEUE 3c France table; scripts/hq_ens/score_sub.py needs
HQ-only pickles and does not run on the box). Per country: empty rate, matches/S1, records under 2+ S1, identical sets vs
the reference; then, for the countries in --keys, coverage of the exact-key pairs by hq_keys rule (key_pairs over all S1
of the country; rules marked * are 'sure' for an unseen country at p_min 0.96 in --rules-from's log line, if given).
Usage: PYTHONPATH=. python scripts/day/country_table.py <matching_results.tsv> [--ref output_v2/matching_results.tsv]
          [--store-dir cache_n3] [--keys france] [--rules-from runs/v3/stdout.txt]"""
import argparse
import ast
import re

import numpy as np
import pandas as pd

from src.hq_keys import key_pairs, record_keys, s1_vocab

ap = argparse.ArgumentParser()
ap.add_argument("sub")
ap.add_argument("--ref", default="output_v2/matching_results.tsv")
ap.add_argument("--store-dir", default="cache_n3")
ap.add_argument("--keys", default="france")
ap.add_argument("--rules-from", default="")
a = ap.parse_args()
S1 = "data/data_extracted/student_resource/dataset/test/test_source1.tsv"


def enc(ids):
    """'S1-123' / 'S2-123' / 'S3-123' -> int64 (source * 1e10 + number): ~10x less RAM than strings."""
    ids = pd.Series(ids, dtype=str)
    return ids.str[1].astype(np.int64).to_numpy() * 10_000_000_000 + ids.str[3:].astype(np.int64).to_numpy()


def pairs(path):
    """(s1, cand) pairs of a matching_results file, streamed."""
    out = []
    for ch in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=300_000):
        ch.columns = ["s1", "m"]
        ch = ch[ch.m != ""]
        e = ch.assign(cand=ch.m.str.split(",")).explode("cand")
        out.append(pd.DataFrame({"s1": enc(e.s1), "cand": enc(e.cand)}))
    return pd.concat(out, ignore_index=True)


cty = pd.read_csv(S1, sep="\t", usecols=["entity_id", "country"], dtype=str)
cty = pd.Series(cty.country.str.lower().to_numpy(), index=enc(cty.entity_id))
X, R = pairs(a.sub), pairs(a.ref)
for d in (X, R):
    d["c"] = d.s1.map(cty)
print(f"# {a.sub} vs {a.ref}\n")
print("| country | S1 | empty X (ref) | matches/S1 X (ref) | recs under 2+ S1 X (ref) | identical sets | pairs only X / only ref |")
print("|---|---|---|---|---|---|---|")
for c in sorted(cty.unique()):
    n = int((cty == c).sum())
    x, r = X[X.c == c], R[R.c == c]
    sx, sr = x.groupby("s1").cand.apply(frozenset), r.groupby("s1").cand.apply(frozenset)
    ids = cty.index[cty == c]
    same = (sx.reindex(ids).fillna(0).astype(str) == sr.reindex(ids).fillna(0).astype(str)).mean()
    m = x.merge(r, on=["s1", "cand"], how="outer", indicator=True)["_merge"].value_counts()
    print(f"| {c} | {n} | {1 - x.s1.nunique() / n:.4f} ({1 - r.s1.nunique() / n:.4f}) | {len(x) / n:.3f} ({len(r) / n:.3f}) "
          f"| {int((x.cand.value_counts() > 1).sum())} ({int((r.cand.value_counts() > 1).sum())}) | {same:.4f} "
          f"| {m.get('left_only', 0)} / {m.get('right_only', 0)} |")

sure = set()
if a.rules_from:
    for line in open(a.rules_from, errors="replace"):
        mm = re.search(r"test/(\w+): key rules .*sure: (\[.*\])", line)
        if mm:
            sure |= {(mm.group(1), r) for r in ast.literal_eval(mm.group(2))}
for c in [k for k in a.keys.split(",") if k]:
    r = pd.read_parquet(f"{a.store_dir}/test__{c}.parquet", columns=["rid", "src", "n_core", "n_nospace", "n_addr"])
    s1, docs = r[r.src == 1].reset_index(drop=True), r[r.src != 1].reset_index(drop=True)
    del r
    kq, kd = record_keys(s1), record_keys(docs)
    P = key_pairs(kq, kd, vocab=s1_vocab(kq))
    del kq, kd
    P = pd.DataFrame({"s1": enc(s1.rid.to_numpy()[P.qi.to_numpy()]), "cand": enc(docs.rid.to_numpy()[P.ci.to_numpy()]),
                      "rule": P.rule.to_numpy()})
    P = P[P.rule != ""]
    for name, d in (("X", X), ("ref", R)):
        P[name] = P.merge(d[["s1", "cand"]].assign(h=1), on=["s1", "cand"], how="left")["h"].fillna(0).to_numpy()
    t = P.groupby("rule").agg(n=("X", "size"), cov_X=("X", "mean"), cov_ref=("ref", "mean")).sort_values("n", ascending=False)
    t.index = [f"{i}{'*' if (c, i) in sure else ''}" for i in t.index]
    print(f"\n## {c}: exact-key pair coverage by rule ({len(P)} key pairs with a rule; * = sure in the run log)\n")
    print("| rule | pairs | coverage X | coverage ref |\n|---|---|---|---|")
    for i, row in t.iterrows():
        print(f"| {i} | {int(row.n)} | {row.cov_X:.3f} | {row.cov_ref:.3f} |")
