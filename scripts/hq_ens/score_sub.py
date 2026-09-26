"""Label-free scorer: a submission X vs v2 (public LB 0.947). HQ tool for the upload call.

    python score_sub.py <matching_results.tsv | sets.pkl> [--name jv1] [--grid]

Reports per country: empty rate, matches/S1 (train GT: 3.46), identical sets vs v2, adds/drops per S1, records under
2+ S1, coverage of the 'sure' exact-key pairs, and an EXPECTED dLB vs v2:
  every pair of V u X gets a P(true): exact-key join pairs -> train-calibrated cell rate at full density
  (test_pairs_p.pkl from cells.py; France = mean of US/India), other pairs -> p_common (in both files),
  p_add (only in X), p_drop (only in v2). Per record the P over its owners is capped to sum 1 (GT is one-to-one).
  Per S1: plug-in F0.5 = 1.25TP / (1.25TP + 0.25FN + FP) for both sets; empty set -> P(no true match).
  dLB = sum over S1 (F_X - F_v2) / 1,732,544 (public LB = random subset of test; same expectation).
The fuzzy (non-key) part has no labels: p_add / p_drop are assumptions -> --grid prints the sensitivity. Back-test on
files with a known LB: `python score_sub.py --backtest`.

NOTE (back-test, docs/ENSEMBLE_AND_KEYS.md section 10): the dLB estimate does NOT predict the LB when X and v2 come
from different models (Soha: predicted -0.2/-0.4, actual +0.10; Atharv: +0.1/-0.2, actual -0.30). Use the tables as
sanity checks; trust the dLB only for mechanism-based changes (pairs v2 never scored, duplicate owners)."""
import argparse, os, pickle, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from load import read_res, SUB  # noqa: E402

NS1 = {"us": 663_106, "india": 809_986, "france": 259_452}
TOT = 1_732_544
CN = {0: "us", 1: "india", 2: "france"}
B = 10_000_000_000


def pairs_of(d):
    q, c = [], []
    for s, m in d.items():
        r = int(s[3:]) + B
        for x in m:
            q.append(r)
            c.append(int(x[1]) * B + int(x[3:]))
    return pd.DataFrame({"q": np.array(q, np.int64), "c": np.array(c, np.int64)})


def load_sets(path):
    if path.endswith(".pkl"):
        d = pickle.load(open(path, "rb"))
        return {s: tuple(m) for s, m in d.items()}
    return read_res(path)[1]


_BASE = {}


def base():
    if not _BASE:
        v2, _, cty = pickle.load(open(os.path.join(HERE, "res.pkl"), "rb"))
        code = {"us": 0, "india": 1, "france": 2}
        s1 = np.fromiter((int(s[3:]) + B for s in cty), np.int64, len(cty))
        cc = np.fromiter((code[v] for v in cty.values()), np.int8, len(cty))
        _BASE["cty"] = pd.Series(cc, index=s1)
        _BASE["V"] = pairs_of(v2)
        _BASE["nV"] = pd.Series({int(s[3:]) + B: len(m) for s, m in v2.items()})
        del v2
        P = pd.read_pickle(os.path.join(HERE, "test_pairs_p.pkl"))[["q_rid", "c_rid", "cty", "p", "cell"]]
        P = P.rename(columns={"q_rid": "q", "c_rid": "c", "p": "pk"})
        # only high-rate cells are informative for a pair a model chose to include/exclude (selection effect: in a
        # 3%-cell the pairs a model accepts are the rare true ones) -> cells with P < 0.9 are treated as fuzzy.
        P = P[P.pk.values >= 0.9]
        # worst-case precision of the pairs of a cell that v2 EXCLUDED: (P - cov) / (1 - cov) (all v2 pairs true)
        inv = P[["q", "c"]].merge(_BASE["V"].assign(inV=True), on=["q", "c"], how="left").inV.notna().values
        cov = pd.Series(inv).groupby([P.cty.values, P.cell.astype(str).values]).transform("mean").values
        P = P.assign(pb=np.clip((P.pk.values - cov) / np.maximum(1 - cov, 1e-9), 0, 1))
        _BASE["P"] = P.drop(columns=["cty", "cell"]).reset_index(drop=True)
        S = pd.read_pickle(os.path.join(HERE, "test_sure_refined.pkl"))
        S = S[~S.multi_sure][["q_rid", "c_rid", "cty", "cat", "kind", "in_v2"]]
        _BASE["S"] = S.rename(columns={"q_rid": "q", "c_rid": "c"})
    return _BASE


def build_union(X):
    b = base()
    U = b["V"].assign(inV=True).merge(X.assign(inX=True), on=["q", "c"], how="outer")
    U[["inV", "inX"]] = U[["inV", "inX"]].fillna(False).astype(bool)
    U = U.merge(b["P"], on=["q", "c"], how="left")
    U["cty"] = b["cty"].reindex(U.q.values).values
    return U


def expected_dlb(U, p_add, p_drop, p_common=0.95, m=0.0, key_only=False, key_add="bound"):
    """Returns per-country dF sums (sum over S1 of F_X - F_V). key_add: 'bound' = worst-case precision of the
    key pairs X adds (v2 excluded them), 'rate' = the cell rate."""
    both, xo, vo = U.inV.values & U.inX.values, U.inX.values & ~U.inV.values, U.inV.values & ~U.inX.values
    p = U.pk.values.copy()
    if key_add == "bound":
        p = np.where(xo & ~np.isnan(p), U.pb.values, p)
    fuzzy = np.isnan(p)
    p[fuzzy & both] = p_common
    p[fuzzy & xo] = p_add
    p[fuzzy & vo] = p_drop
    keep = np.ones(len(U), bool)
    if key_only:  # neutralise the fuzzy disagreements: X keeps v2's decision on them
        keep = ~(fuzzy & (xo | vo))
    q, c = U.q.values[keep], U.c.values[keep]
    p, both, xo, vo = p[keep], both[keep], xo[keep], vo[keep]
    # one-to-one consistency: P over the owners of a record sums to <= 1
    tot = pd.Series(p).groupby(c).transform("sum").values
    p = p / np.maximum(1.0, tot)
    # only S1 with a disagreement matter
    diff_q = np.unique(q[xo | vo])
    sel = np.isin(q, diff_q)
    q, p, both, xo, vo = q[sel], p[sel], both[sel], xo[sel], vo[sel]
    lg = np.log(np.clip(1 - p, 1e-9, 1))
    df = pd.DataFrame({"q": q, "tpV": p * (both | vo), "fpV": (1 - p) * (both | vo), "fnV": p * xo,
                       "tpX": p * (both | xo), "fpX": (1 - p) * (both | xo), "fnX": p * vo,
                       "nV": (both | vo).astype(np.int32), "nX": (both | xo).astype(np.int32), "lg": lg})
    g = df.groupby("q").sum()
    pe = np.exp(g.lg.values - m)

    def F(tp, fp, fn, n):
        f = 1.25 * tp / np.maximum(1.25 * tp + 0.25 * (fn + m) + fp, 1e-12)
        return np.where(n > 0, f, pe)
    dF = F(g.tpX.values, g.fpX.values, g.fnX.values, g.nX.values) - F(g.tpV.values, g.fpV.values, g.fnV.values, g.nV.values)
    cty = base()["cty"].reindex(g.index.values).values
    return {CN[k]: float(dF[cty == k].sum()) for k in (0, 1, 2)}


def report(path, name, grid=False, p_add=0.80, p_drop=0.72, quiet=False):
    t0 = time.time()
    b = base()
    Xd = load_sets(path)
    miss = TOT - len(Xd)
    X = pairs_of(Xd)
    nX = pd.Series({int(s[3:]) + B: len(m) for s, m in Xd.items()})
    del Xd
    U = build_union(X)
    cty_q = b["cty"]
    out = {"name": name}
    rows = []
    owners = X.groupby("c").q.size()
    multi_ids = owners[owners > 1]
    mcty = b["cty"].reindex(X[X.c.isin(multi_ids.index)].q.values).values
    for k, cn in CN.items():
        idx = cty_q.index[cty_q.values == k]
        nx, nv = nX.reindex(idx).fillna(0).values, b["nV"].reindex(idx).fillna(0).values
        u = U[U.cty == k]
        xo = (u.inX & ~u.inV).sum(); vo = (u.inV & ~u.inX).sum()
        dq = np.unique(u.q.values[(u.inX ^ u.inV).values])
        fz = u.pk.isna()
        rows.append(dict(country=cn, S1=len(idx), empty_X=(nx == 0).mean(), empty_v2=(nv == 0).mean(),
                         matches_X=nx.mean(), matches_v2=nv.mean(), identical=1 - len(dq) / len(idx),
                         adds=xo / len(idx), drops=vo / len(idx),
                         adds_fuzzy=(u.inX & ~u.inV & fz).sum() / len(idx), drops_fuzzy=(u.inV & ~u.inX & fz).sum() / len(idx),
                         multi_owner_slots=int((mcty == k).sum())))
    T = pd.DataFrame(rows).set_index("country")
    if not quiet:
        print(f"\n######## {name}  ({path})  | S1 missing: {miss} | records under 2+ S1: {len(multi_ids)} "
              f"({int(multi_ids.sum())} slots) | load {time.time()-t0:.0f}s")
        print(T.round(4).to_string())
        # sure-pair coverage by rule
        S = b["S"].merge(X.assign(inX=True), on=["q", "c"], how="left")
        S["inX"] = S.inX.fillna(False).astype(bool)
        S["rule"] = S.cat.str.replace("n_", "").str.replace(" | a_", "|") + np.where(S.kind != "", "|" + S.kind, "")
        cov = S.groupby(["cty", "rule"]).agg(n=("inX", "size"), v2=("in_v2", "mean"), X=("inX", "mean"))
        cov = cov[cov.n >= 2000].reset_index()
        cov["cty"] = cov.cty.map(CN)
        cov["n_per_S1"] = cov.n / cov.cty.map(NS1)
        print("\n  sure-pair coverage (v2 -> X), rules with >= 2000 pairs:")
        print(cov.sort_values(["cty", "n"], ascending=[True, False]).round(3).to_string(index=False))
    # expected dLB
    res = {}
    for ka in ("bound", "rate"):
        kd = expected_dlb(U, p_add, p_drop, key_only=True, key_add=ka)
        fd = expected_dlb(U, p_add, p_drop, key_add=ka)
        res[ka] = dict(key_only=kd, full=fd)
        tot_k, tot_f = sum(kd.values()) / TOT, sum(fd.values()) / TOT
        line = " | ".join(f"{c} {kd[c]/NS1[c]*100:+.2f} / {fd[c]/NS1[c]*100:+.2f}" for c in NS1)
        print(f"\n  [key adds at {ka:5}] dF per country, pt (key-only / full at p_add {p_add}, p_drop {p_drop}): {line}")
        print(f"  [key adds at {ka:5}] EXPECTED dLB, pt: key-only {tot_k*100:+.2f} | full {tot_f*100:+.2f} "
              f"-> LB ~ {0.947 + tot_f:.4f}")
    if grid:
        print("  sensitivity of the full dLB (pt) to the fuzzy assumptions (rows p_add, cols p_drop):")
        pa_list, pd_list = [0.6, 0.7, 0.8, 0.9], [0.6, 0.7, 0.8, 0.9]
        G = pd.DataFrame(index=pa_list, columns=pd_list, dtype=float)
        for pa in pa_list:
            for pdp in pd_list:
                G.loc[pa, pdp] = sum(expected_dlb(U, pa, pdp).values()) / TOT * 100
        print(G.round(2).to_string())
        res["grid"] = G
    out.update(table=T, res=res)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--name", default="X")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--p-add", type=float, default=0.80)
    ap.add_argument("--p-drop", type=float, default=0.72)
    ap.add_argument("--backtest", action="store_true")
    a = ap.parse_args()
    if a.backtest:
        known = [("soha", f"{SUB}/soha_matching_results.tsv", 0.948),
                 ("ens1", f"{SUB}/ens1_consensus_matching_results.tsv", 0.9449),
                 ("atharv", f"{SUB}/atharv_matching_results.tsv", 0.944),
                 ("ens2c (held)", os.path.join(HERE, "out/ens2c_matching_results.tsv"), None)]
        for nm, p, lb in known:
            r = report(p, nm, grid=True, p_add=a.p_add, p_drop=a.p_drop)
            if lb is not None:
                print(f"  ACTUAL dLB {nm}: {(lb - 0.947)*100:+.2f} pt (LB {lb})")
    else:
        report(a.path, a.name, grid=a.grid, p_add=a.p_add, p_drop=a.p_drop)
