"""Build the final submission zip in the layout the organisers require.

  <team>_submission.zip
  ├── output/matching_results.tsv, output/candidate_pairs.tsv
  ├── code/business_entity_resolution/{src/, scripts/, README.md, requirements.txt}
  └── Documentation_template.md   (our filled-in copy: docs/Documentation_template.md)

Usage (repo root):  python scripts/make_package.py --team <team_name> --output output
Runs the official validator first and refuses to package a failing output.
"""
import argparse
import glob
import os
import subprocess
import sys
import zipfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--output", default="output", help="folder holding matching_results.tsv + candidate_pairs.tsv")
    ap.add_argument("--data-root", default="data/data_extracted/student_resource")
    ap.add_argument("--doc", default="docs/Documentation_template.md")
    ap.add_argument("--skip-validate", action="store_true")
    a = ap.parse_args()
    m, c = os.path.join(a.output, "matching_results.tsv"), os.path.join(a.output, "candidate_pairs.tsv")
    for p in (m, c, a.doc, "README.md", "requirements.txt"):
        if not os.path.exists(p):
            sys.exit(f"missing {p}")
    if not a.skip_validate:
        cmd = [sys.executable, os.path.join(a.data_root, "utils", "validate_submission.py"), "--matching", m,
               "--candidate", c, "--test-dir", os.path.join(a.data_root, "dataset", "test")]
        print("validating:", " ".join(cmd))
        if subprocess.call(cmd) != 0:
            sys.exit("validator FAILED - not packaging")
    out = f"{a.team}_submission.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(m, "output/matching_results.tsv")
        z.write(c, "output/candidate_pairs.tsv")
        base = "code/business_entity_resolution/"
        for f in glob.glob("src/*.py") + glob.glob("scripts/*.py") + glob.glob("scripts/*.sh"):
            z.write(f, base + f)
        z.write("README.md", base + "README.md")
        z.write("requirements.txt", base + "requirements.txt")
        z.write(a.doc, "Documentation_template.md")
    print("wrote", out, f"({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
