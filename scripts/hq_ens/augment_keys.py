"""Post-hoc 'sure key pair' augmentation of ANY submission (generalised ens2c step 2; HQ tool for the jv1 night).

    python augment_keys.py <in.tsv> <out.tsv> [--bound-min 0.80]

For every (rule, country) of the refined 'sure' exact-key pairs (test_sure_refined.pkl: new normaliser, context =
all S1, train rate P per rule, records 'sure' for 2+ S1 excluded), measure the file's own coverage cov and add the
pairs it misses only where the WORST-CASE precision of its misses, (P - cov) / (1 - cov) (= every pair the file
kept is true), is >= --bound-min (F0.5 break-even 0.73-0.77), and the record is not already assigned to another S1.
A file that already covers a rule at ~P gets nothing added there (its misses may be the true negatives). Prints the
adds per rule and country with the expected dLB (marginal F0.5 formula at the worst-case precision)."""
import argparse, collections, os, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from load import read_res  # noqa: E402

NS1 = {0: 663_106, 1: 809_986, 2: 259_452}
CN = {0: "us", 1: "india", 2: "france"}
TOT = 1_732_544
B = 10_000_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--bound-min", type=float, default=0.80)
    ap.add_argument("--false-mult", type=float, default=1.0,
                    help="multiply the train false rate (1-P) by this: test has ~2x the unmatched S2/S3 records per S1 of train")
    a = ap.parse_args()
    t0 = time.time()
    header, res = read_res(a.inp)
    res = {s: set(m) for s, m in res.items()}
    owner = {}
    multi = 0
    for s, m in res.items():
        for x in m:
            if x in owner:
                multi += 1
            owner[x] = s
    print(f"loaded {len(res)} S1, {sum(map(len, res.values()))} pairs, records under 2+ S1: {multi} ({time.time()-t0:.0f}s)")
    T = pd.read_pickle(os.path.join(HERE, "test_sure_refined.pkl"))
    T = T[~T.multi_sure][["q_rid", "c_rid", "cty", "cat", "kind", "p"]].copy()
    s1 = [f"S1-{q - B}" for q in T.q_rid.values.tolist()]
    cid = [f"S{c // B}-{c % B}" for c in T.c_rid.values.tolist()]
    T["inX"] = [x in res.get(s, ()) for s, x in zip(s1, cid)]
    T["rule"] = T.cat.str.replace("n_", "", regex=False).str.replace(" | a_", "|", regex=False) + \
        np.where(T.kind != "", "|" + T.kind, "")
    g = T.groupby(["cty", "rule"]).agg(n=("inX", "size"), cov=("inX", "mean"), P=("p", "mean")).reset_index()
    g["P"] = (1 - a.false_mult * (1 - g.P)).clip(0, 1)
    g["bound"] = ((g.P - g["cov"]) / (1 - g["cov"]).clip(lower=1e-9)).clip(0, 1)
    ok = g[(g.bound >= a.bound_min) & (g.n >= 200)]
    allow = {(r.cty, r.rule): r.bound for r in ok.itertuples()}
    print("rules eligible (bound >= %.2f):" % a.bound_min)
    print(g.assign(cty=g.cty.map(CN), eligible=[(c, r) in allow for c, r in zip(g.cty, g.rule)])
          .sort_values(["cty", "n"], ascending=[True, False]).round(3).to_string(index=False))
    gain = collections.defaultdict(float)
    cnt = collections.Counter()
    by = collections.defaultdict(float)
    byn = collections.Counter()
    for s, x, k, rule, inx in zip(s1, cid, T.cty.values.tolist(), T.rule.values.tolist(), T.inX.values.tolist()):
        if inx or (k, rule) not in allow or x in owner:
            continue
        p = allow[(k, rule)]
        kk = len(res[s])
        e = p * 0.25 / (1.25 * kk + 0.25) - (1 - p) / (1.25 * kk + 1.0)
        gain[k] += e; cnt[k] += 1; by[(CN[k], rule)] += e; byn[(CN[k], rule)] += 1
        res[s].add(x); owner[x] = s
    for k in (0, 1, 2):
        print(f"  {CN[k]}: adds {cnt[k]} ({cnt[k]/NS1[k]:.4f}/S1) | worst-case dF {gain[k]/NS1[k]*100:+.3f} pt | dLB {gain[k]/TOT*100:+.3f} pt")
    print(f"  total worst-case dLB {sum(gain.values())/TOT*100:+.3f} pt")
    for key, e in sorted(by.items(), key=lambda z: -z[1])[:12]:
        print(f"    {key[0]:7} {key[1]:34} n={byn[key]:6d}  dLB {e/TOT*100:+.4f}")
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(header if header.endswith("\n") else header + "\n")
        for s, m in res.items():
            f.write(f"{s}\t{','.join(sorted(m))}\n")
    print(f"wrote {a.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
