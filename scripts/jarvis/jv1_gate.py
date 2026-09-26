"""QUEUE 1 density-matched gate for a jarvis train pass (HQ 26 Sep 16:00).

From <out>/oof_pairs.tsv.gz (s1, cand, y, p) + <out>/report.json (chosen rule) + the store (S1 order, GT):
 (a) OOF F0.5 on all sampled train S1, per country + overall;
 (b) the same on v2's 150K S1 (per country q_all.sample(n_v2, random_state=seed): a prefix of the same permutation as
     the run's own sample, so a subset of it); decide + one-to-one re-run on the subset's pairs only;
 (c) per country: pair recall of the kept candidates and cands/S1;
 (d) per country: predicted matches/S1 and empty rate vs the GT.
S1 without any kept candidate are absent from the OOF dump; they count here with an empty prediction.
Usage: python scripts/jarvis/jv1_gate.py --out /home/out_jv/jv1 --store /home/cache_jv/store [--md runs/jarvis/jv1_train.md]
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.metric import f05  # noqa: E402
from src.model import decide  # noqa: E402

V2_N = {"us": 89_969, "india": 60_031}
V2_REF = {"us": 0.9722, "india": 0.9476, "ALL": 0.9623}  # runs/day/v2_oof_by_country.md (ALL = v2's logged OOF)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--md", default=None)
    a = ap.parse_args()

    rep = json.load(open(f"{a.out}/report.json"))
    best = rep["oof"]["chosen"]
    rule, thr, o2o = best["rule"], float(best["thr"]) if best["rule"] == "thr" else 0.0, bool(best["one2one"])
    oof = pd.read_csv(f"{a.out}/oof_pairs.tsv.gz", sep="\t",
                      dtype={"s1": str, "cand": str, "y": np.int8, "p": np.float32})
    in_oof = set(oof["s1"].unique())
    meta = json.load(open(f"{a.store}/train_meta.json"))
    gtp = pd.read_parquet(meta["gt"]["path"])

    samples = {}  # country -> (run sample ids, v2 subset ids)
    notes = []
    for c in sorted(meta["countries"]):
        rec = pd.read_parquet(meta["countries"][c]["path"], columns=["rid", "src"])
        q_all = rec[rec["src"] == 1].reset_index(drop=True)
        ids_oof = q_all["rid"][q_all["rid"].isin(in_oof)]
        # the run's n_c is not in report.json for every version: find it from the OOF ids (largest sampled position)
        perm = np.random.RandomState(a.seed).permutation(len(q_all))
        rank = np.empty(len(q_all), np.int64)
        rank[perm] = np.arange(len(q_all))
        n_run = int(rank[ids_oof.index.to_numpy()].max()) + 1
        run_ids = q_all["rid"].to_numpy()[perm[:n_run]]
        chk = q_all.sample(n_run, random_state=a.seed)["rid"].to_numpy()
        same = bool(np.array_equal(np.sort(chk), np.sort(run_ids)))
        n_v2 = V2_N.get(c, 0)
        v2_ids = q_all.sample(n_v2, random_state=a.seed)["rid"].to_numpy() if n_v2 else np.array([], object)
        contained = bool(np.isin(v2_ids, run_ids).all())
        notes.append(f"{c}: run sample n={n_run:,} (pandas-sample reproduces it: {same}); v2 subset n={n_v2:,} "
                     f"contained in the run sample: {contained}; S1 in OOF dump {len(ids_oof):,}")
        if not contained:
            rs = np.random.RandomState(a.seed)
            v2_ids = rs.choice(run_ids, n_v2, replace=False)
            notes.append(f"  {c}: containment failed -> random subset of the run sample with n={n_v2:,}")
        samples[c] = (run_ids, v2_ids)
    all_ids = np.concatenate([s[0] for s in samples.values()])
    gtp = gtp[gtp["s1"].isin(set(all_ids))]
    gt = gtp.groupby("s1")["m"].agg(frozenset).to_dict()

    def evaluate(ids_by_c):
        ids = np.concatenate(list(ids_by_c.values()))
        idset = set(ids)
        sub = oof[oof["s1"].isin(idset)]
        q_rid, qc = np.unique(sub["s1"].to_numpy(), return_inverse=True)
        d_rid, dc = np.unique(sub["cand"].to_numpy(), return_inverse=True)
        pairs = pd.DataFrame({"q": qc, "c": dc, "p": sub["p"].to_numpy()})
        pred = decide(pairs, q_rid, d_rid, rule, thr, o2o)
        ncand = sub.groupby("s1").size()
        npos = sub.groupby("s1")["y"].sum()
        rows = []
        for c, cids in ids_by_c.items():
            for s in cids:
                t, p = gt.get(s, frozenset()), pred.get(s, frozenset())
                rows.append((c, f05(t, p), len(t), len(p), len(t & p), ncand.get(s, 0), npos.get(s, 0)))
        return pd.DataFrame(rows, columns=["country", "f", "n_true", "n_pred", "tp", "ncand", "npos"])

    out = []
    out.append(f"# jv1 train pass — density-matched gate\n\nSource `{a.out}/oof_pairs.tsv.gz` ({len(oof):,} OOF pairs). "
               f"Rule = run's chosen: {rule} thr {thr:.2f} one2one {o2o}. S1 absent from the dump count as empty.\n")
    out += [f"- {n}" for n in notes]
    res = {}
    for lbl, ids_by_c in [("(a) all sampled S1", {c: s[0] for c, s in samples.items()}),
                          ("(b) v2's 150K S1", {c: s[1] for c, s in samples.items() if len(s[1])})]:
        if not ids_by_c:
            continue
        df = evaluate(ids_by_c)
        out.append(f"\n## {lbl}\n")
        out.append("| country | S1 | OOF F0.5 | v2 ref | cand recall (c) | cands/S1 (c) | pred/S1 (d) | GT/S1 | "
                   "pred-empty (d) | GT-empty | TP recall | precision |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for c, g in list(df.groupby("country")) + [("ALL", df)]:
            res[(lbl[:3], c)] = g.f.mean()
            out.append(f"| {c} | {len(g):,} | {g.f.mean():.5f} | {V2_REF.get(c, float('nan')):.4f} | "
                       f"{g.npos.sum() / max(g.n_true.sum(), 1):.4f} | {g.ncand.mean():.2f} | {g.n_pred.mean():.3f} | "
                       f"{g.n_true.mean():.3f} | {(g.n_pred == 0).mean():.2%} | {(g.n_true == 0).mean():.2%} | "
                       f"{g.tp.sum() / max(g.n_true.sum(), 1):.4f} | {g.tp.sum() / max(g.n_pred.sum(), 1):.4f} |")
    b = res.get(("(b)", "ALL"), float("nan"))
    out.append(f"\n**Gate (b) overall {b:.5f} vs v2 0.9623: {'GO' if b >= 0.9623 else 'STOP'}** "
               f"(delta {100 * (b - 0.9623):+.2f} pt)")
    txt = "\n".join(out)
    print(txt)
    if a.md:
        open(a.md, "w").write(txt + "\n")


if __name__ == "__main__":
    main()
