"""Density-preserving slice of the dataset for fast development (streams files; low memory).

Keeps a fraction f of Source-1 entities, ALL of their ground-truth matches, and the same
fraction f of every other Source-2/3 record (so distractor density stays realistic).
Optionally restrict to one country. Writes the same layout under --out:
  <out>/train/train_source{1,2,3}.tsv, train_ground_truth.tsv
  <out>/test/test_source{1,2,3}.tsv  (test S1 fraction f, S2/S3 fraction f)

Usage:  python scripts/make_slice.py --data-dir data --out data_slice --frac 0.08 [--country India]
"""
import argparse
import glob
import hashlib
import os


LO = 0.0


def keep(id_: str, frac: float) -> bool:
    """Deterministic: hash(id) in [LO, LO + frac)."""
    h = int(hashlib.md5(id_.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return LO <= h < LO + frac


def find(data_dir, pattern):
    hits = [p for p in glob.glob(os.path.join(data_dir, "**", pattern), recursive=True) if "parts" not in p.split(os.sep)]
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]


def filter_file(src, dst, want, frac, country, id_col=0, country_col=3):
    n_in = n_out = 0
    with open(src, encoding="utf-8") as fi, open(dst, "w", encoding="utf-8") as fo:
        fo.write(fi.readline())
        for line in fi:
            n_in += 1
            parts = line.rstrip("\n").split("\t")
            rid = parts[id_col]
            if country and len(parts) > country_col and parts[country_col] != country:
                continue
            if (want is not None and rid in want) or keep(rid, frac):
                fo.write(line)
                n_out += 1
    print(f"  {os.path.basename(dst)}: {n_out}/{n_in} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="data_slice")
    ap.add_argument("--frac", type=float, default=0.08)
    ap.add_argument("--country", default="")
    ap.add_argument("--lo", type=float, default=0.0, help="hash-range start (use a different value for a disjoint holdout slice)")
    ap.add_argument("--train-as-test", action="store_true", help="write the slice as the TEST split (+ hidden GT) instead")
    a = ap.parse_args()
    global LO
    LO = a.lo
    for split in ("train", "test"):
        os.makedirs(os.path.join(a.out, split), exist_ok=True)
    if a.train_as_test:
        return holdout(a)
    # --- train: choose S1 by hash, keep their matches
    s1_path = find(a.data_dir, "train_source1.tsv")
    gt_path = find(a.data_dir, "train_ground_truth.tsv")
    chosen = set()
    with open(s1_path, encoding="utf-8") as fi:
        next(fi)
        for line in fi:
            parts = line.rstrip("\n").split("\t")
            if a.country and parts[3] != a.country:
                continue
            if keep(parts[0], a.frac):
                chosen.add(parts[0])
    must = set()
    with open(gt_path, encoding="utf-8") as fi, open(os.path.join(a.out, "train", "train_ground_truth.tsv"), "w", encoding="utf-8") as fo:
        fo.write(fi.readline())
        for line in fi:
            s1, _, rest = line.rstrip("\n").partition("\t")
            if s1 in chosen:
                fo.write(line)
                if rest:
                    must.update(rest.split(","))
    print(f"train: {len(chosen)} S1 chosen, {len(must)} matched S2/S3 ids kept")
    filter_file(s1_path, os.path.join(a.out, "train", "train_source1.tsv"), chosen, 0.0, a.country)
    for s in (2, 3):
        filter_file(find(a.data_dir, f"train_source{s}.tsv"), os.path.join(a.out, "train", f"train_source{s}.tsv"), must, a.frac, a.country)
    # --- test: plain fraction of each file (no labels)
    for s in (1, 2, 3):
        try:
            filter_file(find(a.data_dir, f"test_source{s}.tsv"), os.path.join(a.out, "test", f"test_source{s}.tsv"), None, a.frac, a.country)
        except FileNotFoundError:
            print(f"  test_source{s}.tsv not found - skipped")


def holdout(a):
    """Write a train-derived slice as data/test (S1, S2, S3) + <out>/test_ground_truth_HIDDEN.tsv."""
    s1_path = find(a.data_dir, "train_source1.tsv")
    gt_path = find(a.data_dir, "train_ground_truth.tsv")
    chosen = set()
    with open(s1_path, encoding="utf-8") as fi:
        next(fi)
        for line in fi:
            parts = line.rstrip("\n").split("\t")
            if (not a.country or parts[3] == a.country) and keep(parts[0], a.frac):
                chosen.add(parts[0])
    must = set()
    with open(gt_path, encoding="utf-8") as fi, open(os.path.join(a.out, "test_ground_truth_HIDDEN.tsv"), "w", encoding="utf-8") as fo:
        fo.write(fi.readline())
        for line in fi:
            s1, _, rest = line.rstrip("\n").partition("\t")
            if s1 in chosen:
                fo.write(line)
                if rest:
                    must.update(rest.split(","))
    print(f"holdout-as-test: {len(chosen)} S1 chosen, {len(must)} matched ids kept")
    filter_file(s1_path, os.path.join(a.out, "test", "test_source1.tsv"), chosen, 0.0, a.country)
    for s in (2, 3):
        filter_file(find(a.data_dir, f"train_source{s}.tsv"), os.path.join(a.out, "test", f"test_source{s}.tsv"), must, a.frac, a.country)


if __name__ == "__main__":
    main()
