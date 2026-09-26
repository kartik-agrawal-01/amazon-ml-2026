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

from .data import count_lines, find_files, raw_frame, read_source_chunks
from .normalize import _normalize_frame, fold

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
    # paths are stored relative to the cwd of the build; resolve them against cache_dir so a store can be read
    # from another working directory (e.g. a git worktree) instead of silently being rebuilt
    for info in list(meta["countries"].values()) + ([meta["gt"]] if isinstance(meta.get("gt"), dict) else []):
        alt = os.path.join(cache_dir, os.path.basename(info.get("path", "")))
        if not os.path.exists(info.get("path", "")) and os.path.exists(alt):
            info["path"] = alt
    if not all(os.path.exists(c["path"]) for c in meta["countries"].values()):
        return None
    return meta


RAW_SCHEMA_COLS = ["rid", "src", "name", "addr", "city", "zip", "state", "country", "pos"]


def _raw_schema():
    return pa.schema([(c, pa.int8() if c == "src" else pa.int64() if c == "pos" else pa.string()) for c in RAW_SCHEMA_COLS])


def _stage_raw_by_country(files: dict, cache_dir: str, split: str, chunksize: int = 500_000, skip=()):
    """Stream the three source TSVs in chunks and append every row to cache/<split>__raw__<country>.parquet.
    Only one chunk is in RAM at a time. Returns (paths by country, schema info, n_rows)."""
    writers, paths, schema_info = {}, {}, {}
    offset = 0
    for s in (1, 2, 3):
        path = files[s]
        n = 0
        for chunk, spec in read_source_chunks(path, chunksize):
            out = raw_frame(chunk, s, spec)
            out["pos"] = offset + n + np.arange(len(out), dtype=np.int64)
            key = _country_key(out).to_numpy()
            for c in np.unique(key):
                if c in skip:
                    continue
                sub = out[key == c]
                if c not in writers:
                    fname = "".join(ch if ch.isalnum() else "_" for ch in (c or "none"))
                    paths[c] = os.path.join(cache_dir, f"{split}__raw__{fname}.parquet")
                    writers[c] = pq.ParquetWriter(paths[c], _raw_schema(), compression="zstd")
                writers[c].write_table(pa.Table.from_pandas(sub[RAW_SCHEMA_COLS], schema=_raw_schema(), preserve_index=False))
            n += len(out)
            del out, chunk
            if spec is not None and s not in schema_info:
                schema_info[s] = dict(file=path, id=spec["id"], name=spec["name"], addr=spec["addr"], all_columns=spec["all_columns"])
        schema_info[s]["n"] = n
        schema_info[s]["n_lines"] = count_lines(path)
        flag = "" if n == schema_info[s]["n_lines"] else f"  !! parsed {n} rows but file has {schema_info[s]['n_lines']} lines (quoting?)"
        print(f"  S{s}: {n:>8} rows | id={spec['id']!r} name={spec['name']!r} addr={spec['addr']} | {os.path.basename(path)}{flag}", flush=True)
        offset += n
    for w in writers.values():
        w.close()
    return paths, schema_info, offset


CNT_COLS = ["cnt_core_s1", "cnt_core_all", "cnt_ph_s1", "cnt_ph_all", "cnt_nsp_s1", "cnt_nsp_all", "cnt_addr_all", "cnt_addr_s1"]
UNIQ_KEY_COLS = ["src", "country", "n_core", "n_ph", "n_nospace", "n_addr"]


def _normalise_country(raw_path: str, tmp_path: str, n_jobs: int, keep_raw: bool, chunk: int = 100_000) -> int:
    """Normalise one country's raw rows (file order: S1, S2, S3) with spawned workers, streaming every finished
    100K-row chunk into tmp_path as its own parquet row group. Parent RAM ~ the raw table + one chunk."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    raw = pq.read_table(raw_path).to_pandas(self_destruct=True)
    dup = raw["rid"].duplicated()
    if dup.any():
        raise ValueError(f"[{raw_path}] {int(dup.sum())} record ids repeat, e.g. {raw.rid[dup].head(3).tolist()}")
    raw = raw.sort_values("pos", kind="stable").reset_index(drop=True)
    n = len(raw)
    cols_in = [c for c in ("name", "addr", "city", "zip", "state", "country") if c in raw]
    base_cols = ["rid", "src", "pos"] + (RAW_COLS if keep_raw else [])
    bounds = [(a, min(a + chunk, n)) for a in range(0, n, chunk)]
    writer = None

    def emit(a, b, part):
        nonlocal writer
        frame = raw.iloc[a:b][base_cols].reset_index(drop=True)
        part = part.reset_index(drop=True)
        for c in part.columns:
            frame[c] = part[c]
        frame["n_full"] = (frame["n_core"] + " " + frame["n_addr"]).str.strip()
        frame["src"] = frame["src"].astype(np.int8)
        frame["pos"] = frame["pos"].astype(np.int64)
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(tmp_path, table.schema, compression="zstd")
        writer.write_table(table.cast(writer.schema))

    n_jobs = n_jobs or (os.cpu_count() or 1)
    if n_jobs > 1 and len(bounds) > 1:
        with ProcessPoolExecutor(min(n_jobs, len(bounds)), mp_context=mp.get_context("spawn")) as ex:
            for (a, b), part in zip(bounds, ex.map(_normalize_frame, (raw.iloc[a:b][cols_in] for a, b in bounds))):
                emit(a, b, part)
                del part
    else:
        emit(0, n, _normalize_frame(raw[cols_in]))
    writer.close()
    del raw
    return n


def _finalise_country(tmp_path: str, final_path: str, keep_cols: List[str]) -> dict:
    """Uniqueness counts from the key columns only, then stream tmp -> final row group by row group with the
    count columns attached and the columns in KEEP_COLS order."""
    key = pq.read_table(tmp_path, columns=UNIQ_KEY_COLS).to_pandas(self_destruct=True)
    key = add_uniqueness(key)
    cnt = {c: key[c].to_numpy() for c in CNT_COLS}
    src = key["src"].to_numpy()
    info = dict(n_rows=int(len(key)), n_s1=int((src == 1).sum()), n_s2=int((src == 2).sum()), n_s3=int((src == 3).sum()))
    del key, src
    pf = pq.ParquetFile(tmp_path)
    writer, off = None, 0
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i)
        m = t.num_rows
        for c in CNT_COLS:
            t = t.append_column(c, pa.array(cnt[c][off:off + m], type=pa.int32()))
        t = t.select([c for c in keep_cols if c in t.column_names])
        if writer is None:
            writer = pq.ParquetWriter(final_path, t.schema, compression="zstd")
        writer.write_table(t)
        off += m
    writer.close()
    os.remove(tmp_path)
    info["path"] = final_path
    return info


def _done_path(final_path: str) -> str:
    return final_path[: -len(".parquet")] + ".done.json"


def _load_done(cache_dir: str, split: str) -> Dict[str, dict]:
    """Countries whose final parquet was completed by an earlier (interrupted) build: {country: info}."""
    import glob
    done = {}
    for p in glob.glob(os.path.join(cache_dir, f"{split}__*.done.json")):
        with open(p) as fh:
            info = json.load(fh)
        if os.path.exists(info.get("path", "")) and info.get("keep_raw") is not None:
            done[info["country"]] = info
    return done


def build_store(data_dir: str, split: str, cache_dir: str, n_jobs: int, keep_raw: bool = False) -> dict:
    """Stream the split into per-country RAW parquet files (one chunk in RAM at a time), then normalise + count
    one country at a time (streamed, see _normalise_country/_finalise_country) and write the meta json.
    Resumable: countries with a .done.json marker from an interrupted build are skipped. Returns meta."""
    if pq is None:
        raise ImportError("pyarrow is required for the per-country store (pip install pyarrow)")
    t0 = time.time()
    os.makedirs(cache_dir, exist_ok=True)
    files = find_files(data_dir)[split]
    missing = [s for s in (1, 2, 3) if s not in files]
    if missing:
        raise FileNotFoundError(f"[{split}] could not find source files {missing} under {data_dir}")
    done = {c: i for c, i in _load_done(cache_dir, split).items() if bool(i["keep_raw"]) == bool(keep_raw)}
    if done:
        _log(f"{split}: resuming, countries already built: {sorted(done)}")
    print(f"== {split}", flush=True)
    raw_paths, schema_info, n_rows = _stage_raw_by_country(files, cache_dir, split, skip=set(done))
    _log(f"{split}: staged {n_rows} raw rows into {len(raw_paths)} country files ({time.time() - t0:.0f}s)")
    meta = {"split": split, "n_rows": int(n_rows), "files": files, "schema": schema_info, "countries": {},
            "cols": KEEP_COLS + (RAW_COLS if keep_raw else []), "keep_raw": keep_raw}
    if "gt" in files:
        pairs = read_gt_pairs(files["gt"])
        n_true = pairs.groupby("s1").size()
        n_s1 = pairs.attrs["n_s1"]
        dist = n_true.clip(upper=4).value_counts().sort_index().to_dict()
        dist[0] = n_s1 - len(n_true)
        s1_ids = set()
        for rp in list(raw_paths.values()) + [i["path"] for i in done.values()]:
            t = pq.read_table(rp, columns=["rid", "src"], filters=[("src", "==", 1)])
            s1_ids |= set(t.column("rid").to_pylist())
            del t
        gt_ids = set(pd.read_csv(files["gt"], sep="\t", dtype=str, keep_default_na=False, usecols=[0]).iloc[:, 0].str.strip())
        print(f"  GT rows={n_s1} | singletons={(n_s1 - len(n_true)) / max(n_s1, 1):.1%} | matches/S1: "
              f"mean={len(pairs) / max(n_s1, 1):.2f} max={int(n_true.max()) if len(n_true) else 0} | dist={dict(sorted(dist.items()))}")
        print(f"  GT S1 ids not in source1: {len(gt_ids - s1_ids)} | source1 ids not in GT: {len(s1_ids - gt_ids)}", flush=True)
        gp = _write_table(pairs, os.path.join(cache_dir, f"{split}_gt_pairs.parquet"))
        meta["gt"] = {"path": gp, "n_s1": n_s1, "n_pairs": int(len(pairs)), "header": pairs.attrs["header"],
                      "n_multi": int((pairs["m"].value_counts() > 1).sum()),
                      "singleton_rate": float((n_s1 - len(n_true)) / max(n_s1, 1))}
        del pairs, s1_ids, gt_ids, n_true
    import gc
    for c in sorted(set(raw_paths) | set(done)):
        if c in done:
            meta["countries"][c] = {k: done[c][k] for k in ("path", "n_rows", "n_s1", "n_s2", "n_s3")}
            continue
        t = time.time()
        fname = "".join(ch if ch.isalnum() else "_" for ch in (c or "none"))
        tmp_path = os.path.join(cache_dir, f"{split}__tmp__{fname}.parquet")
        final_path = os.path.join(cache_dir, f"{split}__{fname}.parquet")
        n = _normalise_country(raw_paths[c], tmp_path, n_jobs, keep_raw)
        gc.collect()
        info = _finalise_country(tmp_path, final_path, meta["cols"])
        meta["countries"][c] = info
        with open(_done_path(final_path), "w") as fh:
            json.dump({**info, "country": c, "keep_raw": keep_raw}, fh)
        _log(f"{split}/{c!r}: normalised {n} rows -> {os.path.basename(final_path)} "
             f"({os.path.getsize(final_path) / 1e6:.0f} MB, {time.time() - t:.0f}s)")
        gc.collect()
        os.remove(raw_paths[c])
    meta["n_rows"] = int(sum(i["n_rows"] for i in meta["countries"].values()))
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
