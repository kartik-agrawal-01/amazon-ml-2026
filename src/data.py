"""Dataset discovery + loading.

Schema-agnostic on purpose: we write this before seeing the real files. It finds the
source/ground-truth files by name and guesses id / name / address columns. After the
first look at the real data, pin the names in COLUMN_OVERRIDES instead of relying on
guesses.

Unified record table (one row per record, all three sources stacked):
  rid   original id (string, used verbatim in outputs)
  src   1, 2 or 3
  name  raw business name
  addr  raw address (all address-like columns joined with ", ")
  city/zip/state/country  structured fields when the source provides them, else ""
"""
from __future__ import annotations

import glob
import os
import re
from typing import Dict, Optional, Tuple

import pandas as pd

from .metric import Matches, read_matches_tsv

# Official schema (README, 25 Sep): entity_id, business_name, business_address, country.
# `country` is kept as a structured field (partition key), NOT folded into the address text.
_OFFICIAL = {"id": "entity_id", "name": "business_name", "addr": ["business_address"], "struct": {"country": "country"}}
COLUMN_OVERRIDES: Dict[int, dict] = {1: _OFFICIAL, 2: _OFFICIAL, 3: _OFFICIAL}

_SRC_RE = re.compile(r"(?:source|src|s)[ _-]?([123])(?!\d)", re.I)
_ADDR_KEYS = ("addr", "street", "line", "city", "town", "locality", "district", "state", "province",
              "region", "zip", "postal", "postcode", "post_code", "pin", "country")
_STRUCT = {"city": ("city", "town", "locality"), "zip": ("zip", "postal", "postcode", "post_code", "pincode", "pin"),
           "state": ("state", "province"), "country": ("country",)}


def find_files(data_dir: str) -> Dict[str, Dict]:
    """Return {'train': {1: path, 2: path, 3: path, 'gt': path}, 'test': {...}} found under data_dir."""
    out: Dict[str, Dict] = {"train": {}, "test": {}}
    paths = [p for p in glob.glob(os.path.join(data_dir, "**", "*"), recursive=True)
             if os.path.isfile(p) and p.lower().endswith((".tsv", ".csv", ".txt"))]
    for p in sorted(paths):
        rel = os.path.relpath(p, data_dir).replace("\\", "/").lower()
        base = os.path.basename(rel)
        split = "train" if "train" in rel else ("test" if "test" in rel else None)
        if split is None or "hidden" in base:
            continue
        if re.search(r"ground[_ -]?truth|label|(^|[_-])gt([_.-]|$)", base):
            out[split]["gt"] = p
            continue
        if re.search(r"match|candidate|submission|sample|result|pred", base):
            continue
        m = _SRC_RE.search(base)
        if m:
            out[split][int(m.group(1))] = p
    return out


def _pick(cols, keys, exclude=()) -> Optional[str]:
    for c in cols:
        lc = c.lower()
        if any(k in lc for k in keys) and not any(e in lc for e in exclude):
            return c
    return None


def _read_table(path: str) -> pd.DataFrame:
    sep = "," if path.lower().endswith(".csv") else "\t"
    # PS says: read with sep="\t" (default quoting). describe() cross-checks row counts vs raw line counts.
    return pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)


def load_source(path: str, src: int) -> Tuple[pd.DataFrame, dict]:
    df = _read_table(path)
    cols = list(df.columns)
    ov = COLUMN_OVERRIDES.get(src, {})
    if ov and not all(c in cols for c in [ov["id"], ov["name"], *ov["addr"]]):
        ov = {}  # not the official schema (e.g. synthetic data) -> fall back to guessing
    idc = ov.get("id") or _pick(cols, ("id",)) or cols[0]
    namec = ov.get("name") or _pick([c for c in cols if c != idc], ("name",)) or [c for c in cols if c != idc][0]
    addr_cols = ov.get("addr") or [c for c in cols if c not in (idc, namec) and any(k in c.lower() for k in _ADDR_KEYS)]
    if not addr_cols:
        addr_cols = [c for c in cols if c not in (idc, namec)]
    if idc not in cols or namec not in cols or any(c not in cols for c in addr_cols):
        raise ValueError(f"{path}: expected columns {idc!r}, {namec!r}, {addr_cols} but file has {cols}")
    out = pd.DataFrame({"rid": df[idc].astype(str).str.strip(), "src": src, "name": df[namec].fillna("")})
    parts = df[addr_cols].fillna("").astype(str).apply(lambda s: s.str.strip())
    out["addr"] = parts.apply(lambda r: ", ".join(v for v in r if v), axis=1) if len(addr_cols) > 1 else parts.iloc[:, 0]
    struct = ov.get("struct", {})
    for key, keys in _STRUCT.items():
        c = struct.get(key) or (_pick(addr_cols, keys) if len(addr_cols) > 1 else None)
        out[key] = df[c].fillna("").astype(str).str.strip() if c and c in cols else ""
    with open(path, "rb") as fh:
        n_lines = sum(1 for _ in fh) - 1  # minus header
    schema = dict(file=path, id=idc, name=namec, addr=addr_cols, n=len(df), n_lines=n_lines, all_columns=cols)
    return out, schema


def load_split(data_dir: str, split: str) -> Tuple[pd.DataFrame, Optional[Matches], Dict]:
    """Load one split. Returns (records, ground_truth or None, info)."""
    files = find_files(data_dir)[split]
    missing = [s for s in (1, 2, 3) if s not in files]
    if missing:
        listing = "\n  ".join(sorted(glob.glob(os.path.join(data_dir, "**", "*"), recursive=True))[:50])
        raise FileNotFoundError(f"[{split}] could not find source files {missing} under {data_dir}. Files seen:\n  {listing}")
    recs, info = [], {"files": files, "schema": {}}
    for s in (1, 2, 3):
        df, sch = load_source(files[s], s)
        recs.append(df)
        info["schema"][s] = sch
    rec = pd.concat(recs, ignore_index=True)
    dup = rec["rid"].duplicated()
    if dup.any():
        raise ValueError(f"[{split}] {dup.sum()} record ids repeat across/within sources, e.g. {rec.rid[dup].head(3).tolist()}")
    gt = read_matches_tsv(files["gt"]) if "gt" in files else None
    return rec, gt, info


def describe(rec: pd.DataFrame, gt: Optional[Matches], info: Dict, split: str) -> None:
    print(f"== {split}")
    for s, sch in info["schema"].items():
        flag = "" if sch["n"] == sch["n_lines"] else f"  !! parsed {sch['n']} rows but file has {sch['n_lines']} lines (quoting?)"
        print(f"  S{s}: {sch['n']:>8} rows | id={sch['id']!r} name={sch['name']!r} addr={sch['addr']} | {os.path.basename(sch['file'])}{flag}")
    if gt is not None:
        n_true = pd.Series([len(v) for v in gt.values()])
        print(f"  GT rows={len(gt)} | singletons={(n_true == 0).mean():.1%} | matches/S1: mean={n_true.mean():.2f} "
              f"max={n_true.max()} | dist={n_true.clip(upper=4).value_counts().sort_index().to_dict()}")
        s1_ids = set(rec.rid[rec.src == 1])
        print(f"  GT S1 ids not in source1: {len(set(gt) - s1_ids)} | source1 ids not in GT: {len(s1_ids - set(gt))}")
