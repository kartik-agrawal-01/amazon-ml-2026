"""Emergency fallback: turn a PARTIAL output/ (pipeline died mid-test) into a valid submission.
Rows already written keep their predictions; every other test S1 gets an empty prediction and an
empty candidate list. Usage:
  python scripts/night/salvage_partial.py --out-dir output --dest output_salvage
Then run the official validator on <dest>/matching_results.tsv + candidate_pairs.tsv.
"""
import argparse, csv, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--out-dir", default="output")
ap.add_argument("--dest", default="output_salvage")
ap.add_argument("--test-s1", default="data/data_extracted/student_resource/dataset/test/test_source1.tsv")
a = ap.parse_args()
os.makedirs(a.dest, exist_ok=True)

def read_partial(path):
    d = {}
    if not os.path.exists(path):
        return d, None
    with open(path, newline="") as fh:
        rd = csv.reader(fh, delimiter="\t")
        header = next(rd, None)
        for row in rd:
            if len(row) >= 1 and row[0]:
                d[row[0]] = row[1] if len(row) > 1 else ""
    return d, header

m, mh = read_partial(os.path.join(a.out_dir, "matching_results.tsv"))
c, ch = read_partial(os.path.join(a.out_dir, "candidate_pairs.tsv"))
# a torn last line: drop it (keep only ids present in both files) - candidates must be a superset of matches
common = set(m) & set(c)
n_written = n_filled = 0
with open(a.test_s1, newline="") as fh, \
     open(os.path.join(a.dest, "matching_results.tsv"), "w") as fm, \
     open(os.path.join(a.dest, "candidate_pairs.tsv"), "w") as fc:
    rd = csv.reader(fh, delimiter="\t")
    hdr = next(rd)
    idx = hdr.index("entity_id")
    fm.write("source1_entity_id\tmatched_entity_ids\n")
    fc.write("source1_entity_id\tcandidate_entity_ids\n")
    for row in rd:
        r = row[idx]
        if r in common:
            fm.write(f"{r}\t{m[r]}\n"); fc.write(f"{r}\t{c[r]}\n"); n_written += 1
        else:
            fm.write(f"{r}\t\n"); fc.write(f"{r}\t\n"); n_filled += 1
print(f"salvaged: {n_written} S1 with model predictions, {n_filled} S1 filled empty -> {a.dest}/")
