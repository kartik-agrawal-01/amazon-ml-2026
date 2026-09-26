"""Pick the QUEUE 7 winner from runs/day/model_capacity.md: a variant whose OOF F0.5 beats base by >= MARGIN in EVERY
country (the NEXT-list rule); among those the best OOF ALL. Prints the name (nothing if no winner) + reasons on stderr.
Usage: python scripts/day/pick_variant.py runs/day/model_capacity.md [--margin 0.0015]"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("md")
    ap.add_argument("--margin", type=float, default=0.0015)
    a = ap.parse_args()
    head, rows = None, {}
    for line in open(a.md):
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [x.strip() for x in line.strip().strip("|").split("|")]
        if cells[0] == "variant":
            head = cells
            continue
        if head is None or len(cells) != len(head):
            continue
        ctry = {h.split()[0]: float(v.split("/")[0]) for h, v in zip(head[4:-1], cells[4:-1])}
        rows[cells[0]] = dict(all=float(cells[3]), ctry=ctry)
    if "base" not in rows:
        print("no base row", file=sys.stderr)
        return
    base = rows["base"]
    win = []
    for name, r in rows.items():
        if name == "base":
            continue
        d = {k: r["ctry"][k] - base["ctry"][k] for k in base["ctry"]}
        ok = all(v >= a.margin for v in d.values())
        print(f"{name}: ALL {r['all'] - base['all']:+.5f} " + " ".join(f"{k} {v:+.5f}" for k, v in d.items())
              + (" -> WIN" if ok else ""), file=sys.stderr)
        if ok:
            win.append((r["all"], name))
    if win:
        print(max(win)[1])


if __name__ == "__main__":
    main()
