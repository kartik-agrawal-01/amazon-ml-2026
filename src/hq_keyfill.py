"""Exact-key fill: add high-precision exact-key matches that a submission is missing (HQ, 26 Sep).

Post-processing step on top of any model's matching_results.tsv (built for Soha-2 -> 'Kartik2'). Deterministic, uses
only the competition data:
  1. normalise every test record (src/normalize.py: fold + unleet + split_legal, norm_addr);
  2. per country, exact-key pairs S1 x (S2 u S3) and their relation (src/hq_keys.key_pairs; context flags from ALL S1 of
     the country), e.g. rule 'disjoint|a_eq|invented' = different invented name, SAME exact address, and no other S1
     has that name or that address;
  3. rule precision P = measured on the FULL labelled train set (docs/ENSEMBLE_AND_KEYS.md s3; constants below);
     records that fall under a rule for 2+ S1 are skipped;
  4. per (country, rule), coverage = share of the rule's pairs the submission already contains. The pairs it misses
     are added only if their WORST-CASE precision (P' - coverage) / (1 - coverage) >= --bound-min (default 0.80; the
     F0.5 break-even is ~0.73-0.77), with P' = 1 - false_mult * (1 - P) (default 2: test has ~2x the unmatched
     S2/S3 records per S1 of train), and only if the record is not already assigned to another S1.
Nothing is removed. Every added pair is also appended to candidate_pairs.tsv (--candidates), so matches stay a subset
of the candidate set.

    python -m src.hq_keyfill --test-dir data/test --matching in.tsv --out out.tsv \
        [--candidates candidate_pairs.tsv --out-candidates candidate_pairs_out.tsv] [--false-mult 2 --bound-min 0.8]
"""
from __future__ import annotations

import argparse
import collections
import os
import re
import time

import numpy as np
import pandas as pd

from .hq_keys import key_pairs, record_keys
from .normalize import _is_compact, fold, norm_addr, split_legal, unleet

# Rule precision on the full labelled train set (US / India; exact-key joins, new normaliser). One value per rule, as
# used for the uploaded file; 'disjoint|a_eq|realword' (US 0.49) is never used, 'core_eq|street_eq_nonum' not in India
# (0.755 there).
RULE_P = {
    "core_eq|a_eq": 1.0, "reorder|a_eq": 1.0, "subset|a_eq": 0.998, "nsp_eq|a_eq": 1.0,
    "core_eq|num_eq": 0.97, "nsp_eq|num_eq": 0.99, "core_eq|street_eq_nonum": 0.985, "core_eq|c_empty": 0.978,
    "disjoint|a_eq|invented": 0.963, "swap1|a_eq|invented": 0.97, "swap1|a_eq|realword": 0.90,
    "partial|a_eq|invented": 0.961, "partial|a_eq|realword": 0.88,
}
EXCLUDE = {("india", "core_eq|street_eq_nonum")}
MIN_RULE_PAIRS = 200


B = 10_000_000_000


def _keys_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise one chunk of raw records and keep only compact join keys (+ the core name for name relations)."""
    names = df["business_name"].astype(str).tolist()
    core = [split_legal(unleet(fold(nm)), bool(_is_compact.match(nm)))[0] for nm in names]
    rec = pd.DataFrame({"n_core": core, "n_nospace": [c.replace(" ", "") for c in core],
                        "n_addr": [norm_addr(a) for a in df["business_address"].astype(str).tolist()]})
    k = record_keys(rec)
    ids = df["entity_id"].astype(str)
    k["rid"] = (ids.str[1].astype(np.int64) * B + ids.str[3:].astype(np.int64)).to_numpy()
    k["cty"] = df["country"].astype(str).str.strip().str.lower().to_numpy()
    return k


def load_keys(path: str, n_jobs: int) -> pd.DataFrame:
    reader = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3, chunksize=50_000,
                         usecols=["entity_id", "business_name", "business_address", "country"])
    if n_jobs > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(n_jobs, mp_context=mp.get_context("spawn")) as ex:
            parts = list(ex.map(_keys_chunk, reader))
    else:
        parts = [_keys_chunk(c) for c in reader]
    return pd.concat(parts, ignore_index=True)


def rid_of(x: str) -> int:
    return int(x[1]) * B + int(x[3:])


def id_of(r: int) -> str:
    return f"S{r // B}-{r % B}"


def read_pairs(path: str):
    """(header, s1 rids, record rids) of a matching file, as int64 arrays."""
    qs, cs = [], []
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n")
        for line in f:
            s1, _, m = line.rstrip("\r\n").partition("\t")
            if m:
                r = rid_of(s1)
                for x in m.split(","):
                    if x:
                        qs.append(r)
                        cs.append(rid_of(x))
    return header, np.array(qs, np.int64), np.array(cs, np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--matching", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--candidates", default=None)
    ap.add_argument("--out-candidates", default=None)
    ap.add_argument("--false-mult", type=float, default=2.0)
    ap.add_argument("--bound-min", type=float, default=0.80)
    ap.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--report", default=None, help="optional TSV with the per-(country, rule) table")
    a = ap.parse_args()
    t0 = time.time()
    if os.environ.get("AML_PH"):
        print("WARNING: AML_PH is set; the Kartik2 build used the default normaliser (AML_PH unset)", flush=True)

    header, sq, sc = read_pairs(a.matching)
    sub = pd.DataFrame({"q": sq, "c": sc, "have": True})
    owned = pd.Index(np.unique(sc))
    print(f"matching: {len(sub)} pairs ({time.time()-t0:.0f}s)", flush=True)
    q = load_keys(os.path.join(a.test_dir, "test_source1.tsv"), a.n_jobs)
    d = pd.concat([load_keys(os.path.join(a.test_dir, f"test_source{k}.tsv"), a.n_jobs) for k in (2, 3)],
                  ignore_index=True)
    print(f"keys: {len(q)} S1, {len(d)} S2/S3 records ({time.time()-t0:.0f}s)", flush=True)

    adds, table = [], []
    for c in sorted(q.cty.unique()):
        kq, kd = q[q.cty == c].reset_index(drop=True), d[d.cty == c].reset_index(drop=True)
        P = key_pairs(kq, kd, kq_ctx=kq)
        P = P[P.rule != ""]
        P = P[~P.ci.duplicated(keep=False).to_numpy()]      # record under a rule for 2+ S1: ambiguous, skip
        P = P[P.rule.isin(list(RULE_P)).to_numpy() & ~P.rule.map(lambda r: (c, r) in EXCLUDE).to_numpy()]
        R = pd.DataFrame({"q": kq.rid.to_numpy()[P.qi.to_numpy()], "c": kd.rid.to_numpy()[P.ci.to_numpy()],
                          "rule": P.rule.to_numpy()})
        R = R.merge(sub, on=["q", "c"], how="left")
        R["have"] = R["have"].fillna(False).astype(bool)
        R["owned"] = pd.Index(R.c.to_numpy()).isin(owned)
        for r, g in R.groupby("rule"):
            n, cov = len(g), float(g.have.mean())
            p = 1.0 - a.false_mult * (1.0 - RULE_P[r])
            bound = (p - cov) / (1.0 - cov) if cov < 1.0 else 0.0
            ok = bound >= a.bound_min and n >= MIN_RULE_PAIRS
            table.append(dict(country=c, rule=r, pairs=n, coverage=round(cov, 4), p_test=round(p, 4),
                              bound=round(bound, 4), eligible=ok))
            if ok:
                adds.append(g[~g.have & ~g.owned].assign(country=c))
        print(f"  {c}: {len(R)} rule pairs ({time.time()-t0:.0f}s)", flush=True)
        del kq, kd, P, R
    T = pd.DataFrame(table).sort_values(["country", "pairs"], ascending=[True, False])
    print(T.to_string(index=False))
    if a.report:
        T.to_csv(a.report, sep="\t", index=False)

    A = pd.concat(adds, ignore_index=True) if adds else pd.DataFrame(columns=["q", "c", "rule", "country"])
    A = A.drop_duplicates("c")                                  # a record goes to one S1 at most
    print(A.groupby(["country", "rule"]).size().to_string())
    print(f"added pairs: {len(A)}", flush=True)
    add_map = collections.defaultdict(list)
    for qq, cc in zip(A.q.to_numpy(), A.c.to_numpy()):
        add_map[id_of(int(qq))].append(id_of(int(cc)))

    with open(a.matching, encoding="utf-8") as fi, open(a.out, "w", encoding="utf-8", newline="\n") as fo:
        fo.write(fi.readline().rstrip("\r\n") + "\n")
        for line in fi:
            s, _, m = line.rstrip("\r\n").partition("\t")
            ids = [x for x in m.split(",") if x] if m else []
            ids = sorted(set(ids + add_map.get(s, [])))
            fo.write(f"{s}\t{','.join(ids)}\n")
    if a.candidates and a.out_candidates:
        with open(a.candidates, encoding="utf-8") as fi, open(a.out_candidates, "w", encoding="utf-8", newline="\n") as fo:
            fo.write(fi.readline().rstrip("\r\n") + "\n")
            for line in fi:
                s, _, m = line.rstrip("\r\n").partition("\t")
                cand = [x for x in m.split(",") if x] if m else []
                have = set(cand)
                fo.write(f"{s}\t{','.join(cand + [x for x in add_map.get(s, []) if x not in have])}\n")
    print(f"wrote {a.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
