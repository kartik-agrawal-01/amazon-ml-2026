"""Split big TSVs into <=150 MB parts (header repeated) so they can be transferred in pieces.

Usage (from the repo root, any Python 3):
    python scripts/split_tsv.py "data/data_extracted/student_resource/dataset"
Writes  <dataset>/parts/<name>.partNN.tsv  for every .tsv over the limit; leaves originals untouched.
The pipeline itself never reads the parts — they exist only for the file transfer.
"""
import glob
import os
import sys

LIMIT = 150 * 1024 * 1024


def split(path: str, out_dir: str) -> None:
    base = os.path.splitext(os.path.basename(path))[0]
    with open(path, "rb") as fh:
        header = fh.readline()
        part, size, out = 0, 0, None
        for line in fh:
            if out is None or size + len(line) > LIMIT:
                if out:
                    out.close()
                part += 1
                out = open(os.path.join(out_dir, f"{base}.part{part:02d}.tsv"), "wb")
                out.write(header)
                size = len(header)
            out.write(line)
            size += len(line)
        if out:
            out.close()
    print(f"{os.path.basename(path)} -> {part} parts")


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "data/data_extracted/student_resource/dataset"
    out_dir = os.path.join(root, "parts")
    os.makedirs(out_dir, exist_ok=True)
    for p in sorted(glob.glob(os.path.join(root, "**", "*.tsv"), recursive=True)):
        if "parts" in p.replace("\\", "/").split("/"):
            continue
        if os.path.getsize(p) > LIMIT:
            split(p, out_dir)
        else:
            print(f"{os.path.basename(p)}: under limit, skipped")
