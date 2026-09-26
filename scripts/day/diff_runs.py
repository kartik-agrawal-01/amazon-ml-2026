"""Pair-level diff of two runs on the same hidden-holdout slice (light: slice-sized files only).
Usage: python scripts/day/diff_runs.py <data_dir> <out_dir_A> <out_dir_B> [n_examples]
Prints TP/FP gained/lost by B vs A, whether lost pairs were even candidates in B, and examples (raw names/addresses)."""
import sys
import random
import pandas as pd

dd, A, B = sys.argv[1:4]
nex = int(sys.argv[4]) if len(sys.argv) > 4 else 15


def pairs(path):
    df = pd.read_csv(path, sep="\t", dtype=str).dropna()
    return {(s, e) for s, m in zip(df.iloc[:, 0], df.iloc[:, 1]) for e in m.split(",") if e}


cands = pairs  # candidate_pairs.tsv has the same wide format (S1, comma-joined candidates)


gt = pairs(f"{dd}/test_ground_truth_HIDDEN.tsv")
pa, pb = pairs(f"{A}/matching_results.tsv"), pairs(f"{B}/matching_results.tsv")
ca, cb = cands(f"{A}/candidate_pairs.tsv"), cands(f"{B}/candidate_pairs.tsv")
print(f"GT {len(gt)} | A pred {len(pa)} TP {len(pa & gt)} | B pred {len(pb)} TP {len(pb & gt)}")
print(f"cand recall A {len(ca & gt) / len(gt):.4f} B {len(cb & gt) / len(gt):.4f} | cands A {len(ca)} B {len(cb)}")
lost_tp, gain_tp = (pa - pb) & gt, (pb - pa) & gt
lost_fp, gain_fp = (pa - pb) - gt, (pb - pa) - gt
print(f"B vs A: TP lost {len(lost_tp)} (not a B candidate: {len(lost_tp - cb)}) | TP gained {len(gain_tp)} "
      f"(not an A candidate: {len(gain_tp - ca)}) | FP removed {len(lost_fp)} | FP added {len(gain_fp)}")
recs = {}
for s in ("source1", "source2", "source3"):
    df = pd.read_csv(f"{dd}/test/test_{s}.tsv", sep="\t", dtype=str)
    idc = "source1_entity_id" if s == "source1" and "source1_entity_id" in df else ("entity_id" if "entity_id" in df else df.columns[0])
    namec = [c for c in df.columns if "name" in c][0]
    addrc = [c for c in df.columns if "addr" in c]
    df["_a"] = df[addrc].fillna("").astype(str).agg(" | ".join, axis=1) if addrc else ""
    recs.update(dict(zip(df[idc], zip(df[namec].fillna(""), df["_a"]))))
random.seed(0)
LOST_OUT = open("/tmp/lost_pairs.tsv", "w")
for s, e in sorted(lost_tp & cb)[:400]:
    for x in (s, e):
        LOST_OUT.write("\t".join(recs.get(x, ("", ""))) + "\n")
LOST_OUT.close()
for title, S in (("TP lost (in B candidates)", lost_tp & cb), ("TP lost (cut before model)", lost_tp - cb),
                 ("TP gained", gain_tp), ("FP added", gain_fp)):
    print(f"\n== {title}: {len(S)}")
    for s, e in random.sample(sorted(S), min(nex, len(S))):
        print(f"  {recs.get(s)}\n    ~ {recs.get(e)}")


def script_tag(e):
    n = recs.get(e, ("", ""))[0]
    return "nonascii_name" if any(ord(ch) > 127 for ch in n) else ("empty_addr" if not recs.get(e, ("", ""))[1].strip() else "ascii")


print("\n== share by candidate-side tag (lost TP / gained TP / all GT)")
for t in ("nonascii_name", "empty_addr", "ascii"):
    f = lambda S: sum(script_tag(e) == t for _, e in S)
    print(f"  {t:14s} lost {f(lost_tp):5d}  gained {f(gain_tp):5d}  GT {f(gt):6d}  A-TP {f(pa & gt):6d}  B-TP {f(pb & gt):6d}")
print("\n== A: where GT pairs are lost, by tag (not a candidate / candidate but rejected)")
for t in ("nonascii_name", "empty_addr", "ascii"):
    G = {p for p in gt if script_tag(p[1]) == t}
    print(f"  {t:14s} GT {len(G):6d}  not-cand {len(G - ca):6d}  rejected {len((G & ca) - pa):6d}  TP {len(G & pa):6d}")
