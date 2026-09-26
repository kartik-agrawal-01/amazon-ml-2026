"""France-gap diagnostic (read-only w.r.t. src/, the model and the v2 outputs).

Two phases (run from the repo root, inside tmux):
  python scripts/france_diag.py score  [--countries france,us,india] [--block-size 100000]
      Re-runs the EXACT v2 test-time path (same vectorisers: same text sample / seed / max_df, same
      output_v2/model.joblib, same block size -> identical within-block context features) on test block 0
      of each country: blocks -> stage A -> cascade -> stage B -> model. Saves, per country, every kept
      pair with its features + cascade prob + model prob, and per-S1 candidate counts before/after the
      cascade, to --work-dir (gitignored). Also saves the addr_w / full_w vocabularies for part B.
  python scripts/france_diag.py report
      Parts A-E -> runs/france_diag/REPORT.md (+ small json/tsv side files).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.store import load_country, load_meta, text_sample  # noqa: E402

T0 = time.time()
TOK = re.compile(r"(?u)\b\w+\b")
COUNTRIES = ["france", "us", "india"]


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# =============================================================== phase 1: score
def cmd_score(a):
    import joblib
    from src import blocking as _blocking
    from src.blocking import fit_vectorizers
    from src.cascade import cascade_keep
    from src.features import stage_b
    from src.model import decide
    from src.pipeline import CountryContext, need_cols, predict_chunked, release_memory, split_tables

    os.makedirs(a.work_dir, exist_ok=True)
    _blocking.TOPK_DEVICE["device"] = a.topk_device
    mdl = joblib.load(a.model)
    v2 = json.load(open(a.v2_report))["args"]
    views = list(mdl["views"])
    model, feats, a_cols, cascade_model = mdl["model"], mdl["feats"], mdl["a_cols"], mdl["cascade_model"]
    best = mdl["best"]
    thr_best = best["thr"] if best["rule"] == "thr" else 0.0
    top_n, floor = mdl["cascade"]["top"], mdl["cascade"]["floor"]
    log(f"model: {len(feats)} feats, views {views}, rule {best}, cascade top {top_n} floor {floor}")
    meta_te = load_meta(a.cache_dir, "test")
    meta_tr = load_meta(a.cache_dir, "train")

    vec_path = os.path.join(a.work_dir, "vecs.joblib")
    if os.path.exists(vec_path):
        vecs = joblib.load(vec_path)
        log("vectorisers loaded from work dir")
    else:
        # identical to src/pipeline.py step 2 (metas in the same order, same n_rows / seed / max_df)
        sample = text_sample([meta_tr, meta_te], v2["vec_sample"], v2["seed"])
        vecs = fit_vectorizers(sample, views, max_df=v2["max_df"], seed=v2["seed"])
        log(f"vectorisers: {[(v, len(vec.vocabulary_)) for v, vec in vecs.items()]}")
        # part B: document frequencies of address / full words on the same sample (addr_w vocab = the
        # vocabulary an addr_w TfidfVectorizer with the v2 min_df=2 / max_df would keep)
        for col, name in (("n_addr", "addr_w"), ("n_full", "full_w")):
            txt = sample[col].fillna("").values if col in sample else (sample["n_core"] + " " + sample["n_addr"]).values
            df = Counter()
            for s in txt:
                df.update(set(TOK.findall(s)))
            n = len(txt)
            joblib.dump(dict(n=n, df=dict(df), max_df=v2["max_df"]), os.path.join(a.work_dir, f"df_{name}.joblib"))
            del df
        del sample
        joblib.dump(vecs, vec_path)
        release_memory()

    cols = need_cols(False)
    for c in a.countries.split(","):
        out_pairs = os.path.join(a.work_dir, f"pairs_{c}.parquet")
        if os.path.exists(out_pairs):
            log(f"{c}: already scored, skipping")
            continue
        t = time.time()
        rec = load_country(meta_te, c, cols)
        q, d = split_tables(rec)
        del rec
        ctx = CountryContext(f"test/{c}", q, d, vecs, views, None, a.cache_dir, a.n_jobs, None)
        q_idx = np.arange(0, min(a.block_size, len(q)))
        cands, qb, _ = ctx.block(q_idx, v2["k"], verbose=True)
        n_before = np.bincount(cands["q"].to_numpy(), minlength=len(qb))
        pa = predict_chunked(cascade_model, cands, a_cols)
        keep = cascade_keep(cands["q"].to_numpy(), pa, top_n, floor)
        cands["pa"] = pa
        cands = cands[keep].reset_index(drop=True)
        n_after = np.bincount(cands["q"].to_numpy(), minlength=len(qb))
        cands = stage_b(cands, qb, d, a.stage_b_jobs)
        cands["p"] = predict_chunked(model, cands, feats)
        qb_rid = qb["rid"].to_numpy(dtype=object)
        sets = decide(cands, qb_rid, ctx.d_rid, best["rule"], thr_best, bool(best["one2one"]))
        cands["s1"] = qb_rid[cands["q"].to_numpy()]
        cands["cand"] = ctx.d_rid[cands["c"].to_numpy()]
        cands["matched"] = np.fromiter((cd in sets[s] for s, cd in zip(cands["s1"], cands["cand"])), bool, len(cands))
        keep_cols = ["q", "c", "s1", "cand", "pa", "p", "matched"] + [f for f in feats if f not in ("pa", "p")]
        cands[keep_cols].to_parquet(out_pairs, index=False)
        pd.DataFrame({"s1": qb_rid, "n_before": n_before, "n_after": n_after,
                      "n_pred": [len(sets[r]) for r in qb_rid]}).to_parquet(os.path.join(a.work_dir, f"s1_{c}.parquet"),
                                                                           index=False)
        log(f"{c}: block 0 ({len(qb)} S1) scored in {time.time() - t:.0f}s: {int(n_before.sum())} cands -> "
            f"{len(cands)} kept, {sum(1 for r in qb_rid if sets[r])} S1 with matches")
        del ctx, q, d, cands, qb, sets
        release_memory()
    log("score: done")


# ============================================================== phase 2: report
def pct(x, qs=(10, 50, 90)):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return [float(np.percentile(x, q)) if len(x) else float("nan") for q in qs]


def rel_diff(f, o):
    if not np.isfinite(f) or not np.isfinite(o):
        return np.nan
    den = max(abs(o), 1e-9)
    return (f - o) / den


def read_raw(files: dict, ids: set) -> pd.DataFrame:
    """Raw name/address of the given ids from the test TSVs (scan in chunks)."""
    out = []
    for f in files.values():
        for ch in pd.read_csv(f, sep="\t", dtype=str, chunksize=1_000_000, keep_default_na=False,
                              usecols=["entity_id", "business_name", "business_address"]):
            m = ch["entity_id"].isin(ids)
            if m.any():
                out.append(ch[m])
    return pd.concat(out, ignore_index=True).set_index("entity_id")


def decide_thr(pairs: pd.DataFrame, thr: float, min_p: float = 0.02, one2one: bool = True) -> pd.Series:
    """Per-S1 predicted-set sizes under v2's rule (min_p -> one-to-one within the block -> threshold)."""
    df = pairs[pairs["p"] >= min_p]
    if one2one:
        df = df[df["p"] >= df.groupby("c")["p"].transform("max")]
    return df[df["p"] >= thr].groupby("s1").size()


def md_table(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for v in r.values:
            if isinstance(v, (float, np.floating)):
                cells.append(floatfmt.format(v) if np.isfinite(v) else "nan")
            else:
                cells.append(str(v).replace("|", "/"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


FR_ADDR_TOKENS = ["rue", "avenue", "av", "bd", "boulevard", "chemin", "impasse", "allee", "route", "place", "cedex",
                  "bp", "bis", "ter", "zac", "zi"]
FR_LEGAL = {"sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "scp", "earl", "gie", "sem", "scop", "selarl", "sca"}


def cmd_report(a):
    import joblib
    from src.normalize import fold

    rd = a.run_dir
    os.makedirs(rd, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    mdl = joblib.load(a.model)
    thr_v2 = mdl["best"]["thr"]
    meta_te = load_meta(a.cache_dir, "test")
    R = []  # report lines
    R.append("# France gap diagnostic (v2)\n")
    R.append(f"Generated by `scripts/france_diag.py` ({time.strftime('%Y-%m-%d %H:%M')}). Model: `{a.model}` "
             f"(rule {mdl['best']}). Nothing under src/ changed.\n")
    R.append("**Method.** Parts A/C/D re-score test **block 0** (first 100,000 S1 of each country in store order) "
             "through the exact v2 test path (same vectorisers refitted on the identical sample, same block size so "
             "within-block rank/reverse features are identical, cascade -> stage B -> LightGBM -> thr 0.70 + "
             "one-to-one). Distributions in A are over a random 20,000-S1 subset of that block; D uses the full block. "
             "Reproduction check against output_v2/matching_results.tsv below.\n")

    # ------------------------------------------------------------------ load scored blocks
    P, S = {}, {}
    for c in COUNTRIES:
        P[c] = pd.read_parquet(os.path.join(a.work_dir, f"pairs_{c}.parquet"))
        S[c] = pd.read_parquet(os.path.join(a.work_dir, f"s1_{c}.parquet"))

    # reproduction check vs v2 output
    v2m = {}
    wanted = set().union(*[set(S[c]["s1"]) for c in COUNTRIES])
    with open(a.v2_matching) as fh:
        next(fh)
        for line in fh:
            r, _, m = line.rstrip("\n").partition("\t")
            if r in wanted:
                v2m[r] = frozenset(x for x in m.split(",") if x)
    R.append("## 0. Reproduction check (block 0 vs v2 submission)\n")
    rows = []
    for c in COUNTRIES:
        pred = P[c][P[c]["matched"]].groupby("s1")["cand"].apply(frozenset).to_dict()
        s1 = S[c]["s1"].tolist()
        same = np.mean([pred.get(r, frozenset()) == v2m.get(r, frozenset()) for r in s1])
        rows.append(dict(country=c, n_s1=len(s1), exact_set_agreement_with_v2=same))
    R.append(md_table(pd.DataFrame(rows), "{:.4f}") + "\n")

    # ------------------------------------------------------------------ A
    R.append("## A. Re-scored sample: per-country distributions (20,000 S1 each)\n")
    samp = {}
    stats = {}
    for c in COUNTRIES:
        s1s = S[c]["s1"].to_numpy()
        pick = set(rng.choice(s1s, min(a.n_sample, len(s1s)), replace=False))
        samp[c] = pick
        pc = P[c][P[c]["s1"].isin(pick)]
        sc = S[c][S[c]["s1"].isin(pick)]
        top = pc.sort_values(["s1", "p"], ascending=[True, False]).drop_duplicates("s1")
        maxp = sc["s1"].map(top.set_index("s1")["p"]).fillna(0.0).to_numpy()  # S1 with no kept cand -> 0
        n05 = sc["s1"].map(pc[pc["p"] >= 0.5].groupby("s1").size()).fillna(0).to_numpy()
        st = {}
        st["max p per S1"] = pct(maxp)
        st["# cands with p>=0.5"] = pct(n05)
        st["p of top candidate"] = pct(top["p"])
        st["stage-A score of top cand"] = pct(top["score"])
        st["cascade p of top cand"] = pct(top["pa"])
        st["jw_core of top cand"] = pct(top["jw_core"])
        st["jw_addr of top cand"] = pct(top["jw_addr"])
        st["cands per S1 before cascade"] = pct(sc["n_before"])
        st["cands per S1 after cascade"] = pct(sc["n_after"])
        scal = {}
        scal["mean cands before cascade"] = sc["n_before"].mean()
        scal["mean cands after cascade"] = sc["n_after"].mean()
        scal["predicted-empty share"] = (sc["n_pred"] == 0).mean()
        scal["mean matches per S1"] = sc["n_pred"].mean()
        scal["share max p < 0.5"] = (maxp < 0.5).mean()
        scal["share max p in [0.5,0.7)"] = ((maxp >= 0.5) & (maxp < 0.7)).mean()
        for f in ["zip_eq", "zip_conflict", "num_first_eq", "num_both", "addr_empty_c", "legal_eq", "legal_conflict",
                  "legal_missing", "indic_q", "indic_c", "core_eq", "vague_q"]:
            if f in pc:
                scal[f"{f} rate (top cand)"] = float((top[f] > 0).mean())
                scal[f"{f} rate (all kept pairs)"] = float((pc[f] > 0).mean())
        stats[c] = (st, scal, top, pc)
    # distributions table
    rows = []
    for k in stats["france"][0]:
        r = dict(stat=k)
        for c in COUNTRIES:
            p10, p50, p90 = stats[c][0][k]
            r[c] = f"{p10:.3f} / {p50:.3f} / {p90:.3f}"
        rows.append(r)
    R.append("p10 / p50 / p90:\n")
    R.append(md_table(pd.DataFrame(rows)) + "\n")

    rows, flags = [], []
    for k in stats["france"][1]:
        f, u, i = (stats[c][1][k] for c in COUNTRIES)
        du, di = rel_diff(f, u), rel_diff(f, i)
        flag = bool(np.isfinite(du) and np.isfinite(di) and abs(du) > 0.2 and abs(di) > 0.2)
        rows.append(dict(stat=k, france=f, us=u, india=i, rel_vs_us=du, rel_vs_india=di, flag="**FLAG**" if flag else ""))
        if flag:
            flags.append(k)
    # medians of the percentile stats as scalar comparisons too
    for k in stats["france"][0]:
        f, u, i = (stats[c][0][k][1] for c in COUNTRIES)
        du, di = rel_diff(f, u), rel_diff(f, i)
        flag = bool(np.isfinite(du) and np.isfinite(di) and abs(du) > 0.2 and abs(di) > 0.2)
        rows.append(dict(stat=f"median {k}", france=f, us=u, india=i, rel_vs_us=du, rel_vs_india=di,
                         flag="**FLAG**" if flag else ""))
        if flag:
            flags.append(f"median {k}")
    R.append("Rates / means (flag = France differs from BOTH US and India by > 20% relative):\n")
    R.append(md_table(pd.DataFrame(rows)) + "\n")

    # every model feature: mean on the top candidate and on all kept pairs
    feats = [f for f in mdl["feats"] if f in stats["france"][3]]
    rows = []
    for f in feats:
        for lvl, idx in (("top", 2), ("all", 3)):
            v = [float(stats[c][idx][f].astype(float).mean()) for c in COUNTRIES]
            du, di = rel_diff(v[0], v[1]), rel_diff(v[0], v[2])
            if np.isfinite(du) and np.isfinite(di) and abs(du) > 0.2 and abs(di) > 0.2:
                rows.append(dict(feature=f, level=lvl, france=v[0], us=v[1], india=v[2], rel_vs_us=du, rel_vs_india=di))
    R.append(f"All {len(feats)} model features, mean over the top candidate ('top') or all kept pairs ('all'): "
             f"those where France differs from both by > 20% relative ({len(rows)} rows):\n")
    R.append(md_table(pd.DataFrame(rows)) + "\n" if rows else "(none)\n")
    imp = pd.Series(mdl["model"].feature_importances_, index=mdl["feats"]).sort_values(ascending=False)
    R.append("Model split-importance top 15 (context for the flags): " +
             ", ".join(f"{k} {int(v)}" for k, v in imp.head(15).items()) + "\n")
    # accepted-match risk profile: what kind of evidence the ACCEPTED pairs (p >= thr, matched) rest on
    rows = []
    for c in COUNTRIES:
        pc = stats[c][3]
        m = pc[pc["matched"]]
        r = dict(country=c, accepted_pairs=len(m))
        r["house nums both present but disjoint"] = float(((m["num_both"] > 0) & (m["num_any"] == 0)).mean())
        r["first house num differs (both present)"] = float(((m["num_both"] > 0) & (m["num_first_eq"] == 0)).mean())
        r["cand address empty"] = float((m["addr_empty_c"] > 0).mean())
        r["no house num on cand (addr non-empty)"] = float(((m["num_both"] == 0) & (m["addr_empty_c"] == 0)).mean())
        r["legal_conflict"] = float((m["legal_conflict"] > 0).mean())
        r["extra content words on cand"] = float((m["extra_c_content"] > 0).mean())
        r["core_eq"] = float((m["core_eq"] > 0).mean())
        um = pc[~pc["matched"]]
        r["unmatched pairs with p in [0.3,0.7)"] = float(((um["p"] >= 0.3) & (um["p"] < 0.7)).sum() / max(len(stats[c][2]), 1))
        r["unmatched same core name & same first house num"] = float(((um["core_eq"] > 0) & (um["num_first_eq"] > 0)).sum() / max(len(stats[c][2]), 1))
        rows.append(r)
    tb = pd.DataFrame(rows).set_index("country").T.reset_index().rename(columns={"index": "stat"})
    R.append("Evidence behind ACCEPTED pairs (share of accepted pairs; last two rows = count per S1 among unmatched pairs):\n")
    R.append(md_table(tb) + "\n")
    R.append("**Flagged (A):** " + (", ".join(flags) if flags else "none") + "\n")

    # ------------------------------------------------------------------ B
    R.append("## B. Normalisation coverage (2,000 records per country, S1+S2+S3 mixed)\n")
    dfw = joblib.load(os.path.join(a.work_dir, "df_addr_w.joblib"))
    n_s, max_df = dfw["n"], dfw["max_df"]
    dfa = dfw["df"]
    nb = {}
    for c in COUNTRIES:
        rec = load_country(meta_te, c, ["rid", "src", "n_name", "n_core", "legal", "n_addr", "a_nums"])
        idx = np.sort(rng.choice(len(rec), 2000, replace=False))
        nb[c] = rec.iloc[idx].reset_index(drop=True)
        del rec
    ids_b = set().union(*[set(nb[c]["rid"]) for c in COUNTRIES])
    # part C ids: 100 French S1 from the A sample + their matches + top-5 unmatched candidates
    pf = stats["france"][3]
    c_s1 = sorted(rng.choice(sorted(samp["france"]), 100, replace=False))
    c_pairs = pf[pf["s1"].isin(c_s1)]
    ids_c = set(c_s1) | set(c_pairs["cand"])
    log("reading raw text from the test TSVs ...")
    raw = read_raw(meta_te["files"], ids_b | ids_c)
    log(f"raw rows found: {len(raw)}")

    rows = []
    for c in COUNTRIES:
        b = nb[c]
        rn = b["rid"].map(raw["business_name"]).fillna("")
        ra = b["rid"].map(raw["business_address"]).fillna("")
        name_toks = b["n_name"].str.split()
        addr_toks = [TOK.findall(x) for x in b["n_addr"]]
        tot = sum(len(t) for t in addr_toks)
        oov = sum(1 for t in addr_toks for w in t if not (2 <= dfa.get(w, 0) <= max_df * n_s))
        rare = sum(1 for t in addr_toks for w in t if dfa.get(w, 0) < 2)
        freq = sum(1 for t in addr_toks for w in t if dfa.get(w, 0) > max_df * n_s)
        folded_raw_addr = [set(fold(x).split()) for x in ra]
        r = {"country": c,
             "legal detected (legal != '')": float((b["legal"] != "").mean()),
             "FR legal token anywhere in n_name": float(np.mean([bool(FR_LEGAL & set(t)) for t in name_toks])),
             "FR legal token in name but legal==''": float(np.mean([bool(FR_LEGAL & set(t)) and lg == ""
                                                                  for t, lg in zip(name_toks, b["legal"])])),
             "FR legal token left inside n_core": float(np.mean([bool(FR_LEGAL & set(t.split())) for t in b["n_core"]])),
             "addr_w tokens OOV (any reason)": oov / max(tot, 1),
             "  of which too frequent (df > max_df)": freq / max(tot, 1),
             "  of which rare (df < 2)": rare / max(tot, 1),
             "addr has a 5-digit number token": float(np.mean([any(len(x) == 5 for x in s.split()) for s in b["a_nums"]])),
             "addr has a 6-digit number token": float(np.mean([any(len(x) == 6 for x in s.split()) for s in b["a_nums"]])),
             "empty n_addr": float((b["n_addr"] == "").mean()),
             "'et' token in raw name": float(np.mean([" et " in f" {fold(x)} " for x in rn])),
             "'and' in folded name (incl. & -> and)": float(np.mean([" and " in f" {x} " for x in b["n_name"]])),
             "'&' in raw name": float(np.mean(["&" in x for x in rn])),
             "'de'/'du'/'des'/'la'/'le' token in name": float(np.mean([bool({"de", "du", "des", "la", "le"} & set(t))
                                                                    for t in name_toks])),
             }
        for tkn in FR_ADDR_TOKENS:
            r[f"raw addr token '{tkn}'"] = float(np.mean([tkn in s for s in folded_raw_addr]))
        rows.append(r)
    tb = pd.DataFrame(rows).set_index("country").T.reset_index().rename(columns={"index": "stat"})
    R.append(f"addr_w vocabulary = tokens with 2 <= df <= {max_df} x {n_s:,} on the v2 vectoriser text sample "
             f"(addr_w is NOT one of v2's views; v2 uses addr_c3 + full_w for addresses).\n")
    R.append(md_table(tb) + "\n")
    # top OOV / too-frequent French address tokens
    fr_toks = Counter(w for x in nb["france"]["n_addr"] for w in TOK.findall(x))
    top_freq = [(w, n, dfa.get(w, 0) / n_s) for w, n in fr_toks.most_common(60) if dfa.get(w, 0) > max_df * n_s][:25]
    R.append("Most common French n_addr tokens dropped as too frequent (token, count in 2,000 FR records, df share in "
             "sample): " + ", ".join(f"{w} {n} ({s:.3f})" for w, n, s in top_freq) + "\n")
    # examples
    R.append("40 French examples (raw -> normalised):\n")
    ex = nb["france"].head(40)
    lines = ["| src | raw name | n_core | legal | raw address | n_addr |", "|---|---|---|---|---|---|"]
    for _, r in ex.iterrows():
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            r["src"], str(raw["business_name"].get(r["rid"], "?")).replace("|", "/"), r["n_core"], r["legal"],
            str(raw["business_address"].get(r["rid"], "?")).replace("|", "/"), r["n_addr"]))
    R.append("\n".join(lines) + "\n")

    # ------------------------------------------------------------------ C
    R.append("## C. Plausibility: 100 random French S1 (predicted matches; top-5 unmatched for the first 50)\n")
    nm = lambda i: str(raw["business_name"].get(i, "?")).replace("|", "/")  # noqa: E731
    ad = lambda i: str(raw["business_address"].get(i, "?")).replace("|", "/")  # noqa: E731
    for k, s in enumerate(c_s1):
        g = c_pairs[c_pairs["s1"] == s].sort_values("p", ascending=False)
        R.append(f"**{k + 1}. {s}** — {nm(s)} · {ad(s)}  ")
        mt = g[g["matched"]]
        if len(mt) == 0:
            R.append("  - matches: (none)  ")
        for _, r in mt.iterrows():
            R.append(f"  - ✔ p={r['p']:.3f} {r['cand']} — {nm(r['cand'])} · {ad(r['cand'])}  ")
        if k < 50:
            for _, r in g[~g["matched"]].head(5).iterrows():
                R.append(f"  - ✘ p={r['p']:.3f} {r['cand']} — {nm(r['cand'])} · {ad(r['cand'])}  ")
        R.append("")

    # ------------------------------------------------------------------ D
    R.append("## D. Threshold sensitivity (full block 0, v2 rule: min_p 0.02 -> one-to-one -> p >= thr)\n")
    thrs = np.round(np.arange(0.5, 0.951, 0.05), 2)
    prof = {}
    rows = []
    for t in thrs:
        r = dict(thr=t)
        for c in COUNTRIES:
            sz = S[c]["s1"].map(decide_thr(P[c], t)).fillna(0).to_numpy()
            prof[(c, t)] = ((sz == 0).mean(), sz.mean())
            r[f"{c} empty"] = prof[(c, t)][0]
            r[f"{c} mean matches"] = prof[(c, t)][1]
        rows.append(r)
    R.append(md_table(pd.DataFrame(rows)) + "\n")
    t70 = 0.7
    ref_e = np.mean([prof[("us", t70)][0], prof[("india", t70)][0]])
    ref_m = np.mean([prof[("us", t70)][1], prof[("india", t70)][1]])
    dist = {t: np.hypot((prof[("france", t)][0] - ref_e) / max(ref_e, 1e-9), (prof[("france", t)][1] - ref_m) / ref_m)
            for t in thrs}
    tb_ = min(dist, key=dist.get)
    R.append(f"US/India mean profile at thr=0.70: empty {ref_e:.4f}, mean matches {ref_m:.3f}. France thr whose "
             f"(empty, mean matches) is closest (relative Euclidean): **{tb_:.2f}** "
             f"(France there: empty {prof[('france', tb_)][0]:.4f}, mean {prof[('france', tb_)][1]:.3f}; distance "
             f"{dist[tb_]:.3f}; at 0.70 the distance is {dist[0.7]:.3f}).\n")

    # ------------------------------------------------------------------ E
    R.append("## E. Agreement with submissions/soha_matching_results.tsv\n")
    sp_ = os.path.join("submissions", "soha_matching_results.tsv")
    if not os.path.exists(sp_):
        R.append("File not present in the repo -> skipped.\n")
    else:
        from src.metric import read_matches_tsv
        ours = read_matches_tsv(a.v2_matching)
        hers = read_matches_tsv(sp_)
        rec = load_meta(a.cache_dir, "test")
        rows = []
        for c in COUNTRIES:
            ids = load_country(rec, c, ["rid", "src"])
            ids = ids.loc[ids["src"] == 1, "rid"].tolist()
            ex_, jac, one = [], [], []
            for r in ids:
                x, y = ours.get(r, frozenset()), hers.get(r, frozenset())
                ex_.append(x == y)
                jac.append(1.0 if not x and not y else len(x & y) / len(x | y))
                one.append(bool(x) != bool(y))
            rows.append(dict(country=c, n_s1=len(ids), exact_set_agreement=np.mean(ex_), mean_jaccard=np.mean(jac),
                             one_empty_other_not=np.mean(one)))
        R.append(md_table(pd.DataFrame(rows), "{:.4f}") + "\n")

    with open(os.path.join(rd, "REPORT.md"), "w") as fh:
        fh.write("\n".join(R) + "\n")
    log(f"wrote {rd}/REPORT.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["score", "report"])
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--model", default="output_v2/model.joblib")
    ap.add_argument("--v2-report", default="output_v2/report.json")
    ap.add_argument("--v2-matching", default="output_v2/matching_results.tsv")
    ap.add_argument("--work-dir", default="output_france_diag")
    ap.add_argument("--run-dir", default="runs/france_diag")
    ap.add_argument("--countries", default="france,us,india")
    ap.add_argument("--block-size", type=int, default=100_000)
    ap.add_argument("--n-sample", type=int, default=20_000)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--stage-b-jobs", type=int, default=4)
    ap.add_argument("--topk-device", default="cuda")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    cmd_score(a) if a.phase == "score" else cmd_report(a)


if __name__ == "__main__":
    main()
