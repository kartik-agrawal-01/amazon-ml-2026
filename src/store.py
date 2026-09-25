"""Per-country on-disk store of the normalised records — the memory backbone for the 15 GB shared box.

Why: the old single `cache/<split>_norm.pkl` had to be fully in RAM (12M rows -> 6-10 GB). Every stage of the
pipeline is per country anyway (blocking partitions on country, uniqueness counts are keyed by country, all GT
matches are same-country), so we normalise and store each country separately and load ONE country at a time,
only the columns a stage needs.

Layout (all under --cache-dir):
  <split>__<country>.parquet   normalised rows of one country, original row order (S1 rows first, then S2, S3),
                               Arrow-backed strings; columns = KEEP_COLS (+ name/addr when built with keep_raw)
  <split>_meta.json            {"countries": {country: {"path", "n_rows", "n_s1", "n_s2", "n_s3"}}, "n_rows",
                                "gt": {...} , "files": {...}, "schema": {...}, "cols": [...]}
  <split>_gt_pairs.parquet     ground truth as a 2-column pair table (s1, m) — train only
The modelling logic (normalisation, uniqueness counts, features) is untouched: this file only changes WHERE the
tables live and HOW MUCH of them is in memory at once.
"""
from __future__ import annotations

import json
import os
import pickle
import time
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .data import describe, find_files, load_source
from .normalize import add_normalized, fold

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    pd.options.mode.string_storage = "pyarrow"
except ImportError:  # pragma: no cover - the box has pyarrow; laptops may not
    pa = pq = None

KEEP_COLS: List[str] = ["rid", "src", "country", "pos", "n_name", "n_core", "legal", "n_ph", "n_nospace", "n_addr",
                        "n_full", "a_nums", "a_zip", "vague", "indic", "city", "zip", "state",  # city/zip/state: stage B
                        # emits <fld>_fld_eq for every structured field present -> keep them so the feature set is unchanged
                        "cnt_core_s1", "cnt_core_all", "cnt_ph_s1", "cnt_ph_all", "cnt_nsp_s1", "cnt_nsp_all",
                        "cnt_addr_all", "cnt_addr_s1"]
RAW_COLS = ["name", "addr"]          # only kept when the dense view needs the raw text
TEXT_COLS = ["n_core", "n_addr", "n_full", "n_ph"]   # what the TF-IDF vectorisers are fitted on


def _log(msg: str) -> None:
    print(f"[store] {msg}", flush=True)


# ------------------------------------------------------------------ uniqueness
def add_uniqueness(rec: pd.DataFrame) -> pd.DataFrame:
    """Global (per split, per country) duplicate counts: how many S1 records share this core name / phonetic
    key / no-space name, and how many records of any source share this address. Resolves empty-address and
    trade-name candidates: a name that matches exactly ONE S1 entity is a near-certain match.
    (Moved verbatim from pipeline.py; keys include the country, so computing it per country is identical.)"""
    key_ctry = rec["country"].fillna("") if "country" in rec else pd.Series("", index=rec.index)
    s1 = rec["src"] == 1
    for col, name in (("n_core", "core"), ("n_ph", "ph"), ("n_nospace", "nsp")):
        k = key_ctry + "|" + rec[col].fillna("")
        cnt_s1 = k[s1].value_counts()
        rec[f"cnt_{name}_s1"] = k.map(cnt_s1).fillna(0).astype(np.int32)
        rec[f"cnt_{name}_all"] = k.map(k.value_counts()).astype(np.int32)
    ka = key_ctry + "|" + rec["n_addr"].fillna("")
    rec["cnt_addr_all"] = ka.map(ka.value_counts()).astype(np.int32)
    rec.loc[rec["n_addr"].fillna("") == "", "cnt_addr_all"] = 0
    rec["cnt_addr_s1"] = ka.map(ka[s1].value_counts()).fillna(0).astype(np.int32)
    return rec


# ----------------------------------------------------------------- ground truth
def read_gt_pairs(path: str) -> pd.DataFrame:
    """Ground truth TSV -> pair table with columns s1, m (one row per true match; singletons have no row)."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    s1 = df.iloc[:, 0].astype(str).str.strip()
    if s1.duplicated().any():
        raise ValueError(f"{path}: duplicate S1 ids")
    m = df.iloc[:, 1].astype(str).str.strip().str.strip("[]")
    pairs = pd.DataFrame({"s1": s1, "m": m.str.split(",")}).explode("m")
    pairs["m"] = pairs["m"].fillna("").astype(str).str.strip().str.strip("'\"")
    pairs = pairs[(pairs["m"] != "") & (~pairs["m"].str.lower().isin(["nan", "none", "null"]))].reset_index(drop=True)
    pairs.attrs["n_s1"] = int(len(df))
    pairs.attrs["header"] = [str(c) for c in df.columns[:2]]
    return pairs


def gt_dict(pairs: pd.DataFrame, s1_ids: Iterable[str]) -> Dict[str, frozenset]:
    """{s1: frozenset(matches)} for the given S1 ids only (empty set for singletons)."""
    ids = list(s1_ids)
    sub = pairs[pairs["s1"].isin(ids)]
    out = {r: frozenset() for r in ids}
    for s, grp in sub.groupby("s1")["m"]:
        out[s] = frozenset(grp.tolist())
    return out


# ------------------------------------------------------------------- building
def _country_key(rec: pd.DataFrame) -> pd.Series:
    """Partition key = the normalised country (same `fold` the normaliser applies), '' when absent."""
    if "country" not in rec:
        return pd.Series("", index=rec.index)
    return rec["country"].fillna("").map(fold)


def _write_table(df: pd.DataFrame, path: str) -> str:
    if pq is not None:
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, compression="zstd")
        return path
    path = path[: -len(".parquet")] + ".pkl"
    with open(path, "wb") as fh:
        pickle.dump(df, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def _read_table(path: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    if path.endswith(".parquet"):
        t = pq.read_table(path, columns=columns)
        return t.to_pandas(self_destruct=True)
    with open(path, "rb") as fh:
        df = pickle.load(fh)
    return df[columns] if columns else df


def meta_path(cache_dir: str, split: str) -> str:
    return os.path.join(cache_dir, f"{split}_meta.json")


def load_meta(cache_dir: str, split: str) -> Optional[dict]:
    p = meta_path(cache_dir, split)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        meta = json.load(fh)
    if not all(os.path.exists(c["path"]) for c in meta["countries"].values()):
        return None
    return meta


def build_store(data_dir: str, split: str, cache_dir: str, n_jobs: int, keep_raw: bool = False) -> dict:
    """Load a split, split it by country, normalise + count each country separately, write one parquet per
    country and the meta json. Returns meta. Peak RAM ~ raw split + one normalised country."""
    t0 = time.time()
    os.makedirs(cache_dir, exist_ok=True)
    files = find_files(data_dir)[split]
    missing = [s for s in (1, 2, 3) if s not in files]
    if missing:
        raise FileNotFoundError(f"[{split}] could not find source files {missing} under {data_dir}")
    recs, info = [], {"files": files, "schema": {}}
    for s in (1, 2, 3):
        df, sch = load_source(files[s], s)
        recs.append(df)
        info["schema"][s] = sch
    rec = pd.concat(recs, ignore_index=True)
    del recs
    dup = rec["rid"].duplicated()
    if dup.any():
        raise ValueError(f"[{split}] {dup.sum()} record ids repeat across/within sources, e.g. {rec.rid[dup].head(3).tolist()}")
    describe(rec, None, info, split)
    meta = {"split": split, "n_rows": int(len(rec)), "files": files, "schema": info["schema"], "countries": {},
            "cols": KEEP_COLS + (RAW_COLS if keep_raw else []), "keep_raw": keep_raw}
    if "gt" in files:
        pairs = read_gt_pairs(files["gt"])
        n_true = pairs.groupby("s1").size()
        n_s1 = pairs.attrs["n_s1"]
        dist = n_true.clip(upper=4).value_counts().sort_index().to_dict()
        dist[0] = n_s1 - len(n_true)
        s1_ids = set(rec.rid[rec.src == 1])
        gt_ids = set(pd.read_csv(files["gt"], sep="\t", dtype=str, keep_default_na=False, usecols=[0]).iloc[:, 0].str.strip())
        print(f"  GT rows={n_s1} | singletons={(n_s1 - len(n_true)) / max(n_s1, 1):.1%} | matches/S1: "
              f"mean={len(pairs) / max(n_s1, 1):.2f} max={int(n_true.max()) if len(n_true) else 0} | dist={dict(sorted(dist.items()))}")
        print(f"  GT S1 ids not in source1: {len(gt_ids - s1_ids)} | source1 ids not in GT: {len(s1_ids - gt_ids)}")
        gp = _write_table(pairs, os.path.join(cache_dir, f"{split}_gt_pairs.parquet"))
        meta["gt"] = {"path": gp, "n_s1": n_s1, "n_pairs": int(len(pairs)), "header": pairs.attrs["header"],
                      "n_multi": int((pairs["m"].value_counts() > 1).sum()),
                      "singleton_rate": float((n_s1 - len(n_true)) / max(n_s1, 1))}
        del pairs, s1_ids, gt_ids, n_true
    # split the RAW table by country first, then free it, so peak = raw split + one normalised country
    key = _country_key(rec)
    pos = np.arange(len(rec))
    parts: Dict[str, pd.DataFrame] = {}
    for c in sorted(key.unique()):
        mask = (key == c).to_numpy()
        sub = rec[mask].reset_index(drop=True)
        sub["pos"] = pos[mask]
        parts[c] = sub
    del rec, key, pos, mask
    import gc
    gc.collect()
    _log(f"{split}: {len(parts)} countries {[(c, len(p)) for c, p in parts.items()]} ({time.time() - t0:.0f}s)")
    for c in list(parts):
        t = time.time()
        sub = parts.pop(c)
        sub = add_normalized(sub, n_jobs=n_jobs)
        sub["n_full"] = (sub["n_core"] + " " + sub["n_addr"]).str.strip()
        sub = add_uniqueness(sub)
        if not keep_raw:
            sub = sub.drop(columns=[col for col in RAW_COLS if col in sub])
        sub["src"] = sub["src"].astype(np.int8)
        sub["pos"] = sub["pos"].astype(np.int64)
        cols = [col for col in meta["cols"] if col in sub]
        sub = sub[cols]
        fname = "".join(ch if ch.isalnum() else "_" for ch in (c or "none"))
        path = _write_table(sub, os.path.join(cache_dir, f"{split}__{fname}.parquet"))
        src = sub["src"].to_numpy()
        meta["countries"][c] = {"path": path, "n_rows": int(len(sub)), "n_s1": int((src == 1).sum()),
                                "n_s2": int((src == 2).sum()), "n_s3": int((src == 3).sum())}
        _log(f"{split}/{c!r}: normalised {len(sub)} rows -> {os.path.basename(path)} "
             f"({os.path.getsize(path) / 1e6:.0f} MB, {time.time() - t:.0f}s)")
        del sub
        gc.collect()
    with open(meta_path(cache_dir, split), "w") as fh:
        json.dump(meta, fh, indent=1, default=str)
    _log(f"{split}: store complete in {time.time() - t0:.0f}s")
    return meta


def ensure_store(data_dir: str, split: str, cache_dir: str, n_jobs: int, keep_raw: bool = False) -> dict:
    meta = load_meta(cache_dir, split)
    if meta is not None and (not keep_raw or meta.get("keep_raw")):
        _log(f"{split}: using store {meta_path(cache_dir, split)} "
             f"({meta['n_rows']} rows, countries {list(meta['countries'])})")
        return meta
    if meta is not None:
        _log(f"{split}: store lacks raw name/addr needed for --dense -> rebuilding")
    return build_store(data_dir, split, cache_dir, n_jobs, keep_raw=keep_raw)


# -------------------------------------------------------------------- loading
def load_country(meta: dict, country: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """One country's normalised records (original order: S1 rows first, then S2, S3), selected columns only."""
    info = meta["countries"][country]
    cols = None
    if columns is not None:
        avail = set(meta["cols"])
        cols = [c for c in columns if c in avail]
    df = _read_table(info["path"], cols)
    return df


def load_gt_pairs(meta: dict) -> pd.DataFrame:
    return _read_table(meta["gt"]["path"])


def text_sample(metas: List[dict], n_rows: int, seed: int) -> pd.DataFrame:
    """~n_rows rows of the 4 vectoriser text columns, drawn across all splits/countries proportionally
    to their size (all sources; no labels)."""
    total = sum(meta["n_rows"] for meta in metas)
    rng = np.random.default_rng(seed)
    out = []
    for meta in metas:
        for c, info in meta["countries"].items():
            n = int(round(n_rows * info["n_rows"] / max(total, 1)))
            if n <= 0:
                continue
            df = load_country(meta, c, TEXT_COLS)
            idx = np.sort(rng.choice(len(df), min(n, len(df)), replace=False))
            out.append(df.iloc[idx].reset_index(drop=True))
            del df
    return pd.concat(out, ignore_index=True)


def allocate(counts: Dict[str, int], total: int) -> Dict[str, int]:
    """Split `total` across keys proportionally to counts (largest-remainder), capped at each count.
    total <= 0 -> everything."""
    if total <= 0 or total >= sum(counts.values()):
        return dict(counts)
    s = sum(counts.values())
    raw = {k: total * v / s for k, v in counts.items()}
    alloc = {k: int(np.floor(v)) for k, v in raw.items()}
    left = total - sum(alloc.values())
    for k in sorted(counts, key=lambda k: raw[k] - alloc[k], reverse=True)[:left]:
        alloc[k] += 1
    return {k: min(v, counts[k]) for k, v in alloc.items()}
