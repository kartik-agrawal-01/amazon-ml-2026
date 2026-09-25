"""Cross-country (B4) slices derived from an existing data_slice/ by country filtering (streaming, ~no RAM).
  <out>/train/*  = data_slice/train rows with country == TRAIN_COUNTRY (S1, S2, S3, GT rows of those S1)
  <out>/test/*   = data_slice/test  rows with country == TEST_COUNTRY  + <out>/test_ground_truth_HIDDEN.tsv
Usage: python scripts/night/make_xc_from_slice.py --slice data_slice --out data_xc_us --train-country US --test-country India
"""
import argparse, os

ap = argparse.ArgumentParser()
ap.add_argument("--slice", default="data_slice")
ap.add_argument("--out", required=True)
ap.add_argument("--train-country", required=True)
ap.add_argument("--test-country", required=True)
a = ap.parse_args()

def filter_src(src, dst, country):
    ids = set(); n = 0
    with open(src, encoding="utf-8") as fi, open(dst, "w", encoding="utf-8") as fo:
        fo.write(fi.readline())
        for line in fi:
            f = line.rstrip("\n").split("\t")
            if len(f) > 3 and f[3] == country:
                fo.write(line); ids.add(f[0]); n += 1
    print(f"{dst}: {n} rows"); return ids

def filter_gt(src, dst, s1_ids):
    n = 0
    with open(src, encoding="utf-8") as fi, open(dst, "w", encoding="utf-8") as fo:
        fo.write(fi.readline())
        for line in fi:
            if line.split("\t", 1)[0] in s1_ids:
                fo.write(line); n += 1
    print(f"{dst}: {n} rows")

for split, country in (("train", a.train_country), ("test", a.test_country)):
    os.makedirs(os.path.join(a.out, split), exist_ok=True)
    s1 = filter_src(os.path.join(a.slice, split, f"{split}_source1.tsv"), os.path.join(a.out, split, f"{split}_source1.tsv"), country)
    for s in (2, 3):
        filter_src(os.path.join(a.slice, split, f"{split}_source{s}.tsv"), os.path.join(a.out, split, f"{split}_source{s}.tsv"), country)
    if split == "train":
        filter_gt(os.path.join(a.slice, "train", "train_ground_truth.tsv"), os.path.join(a.out, "train", "train_ground_truth.tsv"), s1)
    else:
        filter_gt(os.path.join(a.slice, "test_ground_truth_HIDDEN.tsv"), os.path.join(a.out, "test_ground_truth_HIDDEN.tsv"), s1)
print("XC_OK")
