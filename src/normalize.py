"""Text normalisation for business names and addresses.

First-pass rules written BEFORE seeing the data (multi-region guesses: US/IN/UK/EU).
After EDA, extend LEGAL_CANON / ADDR_CANON / NAME_PREFIXES with what actually shows up —
"region-specific patterns in names and addresses" is an explicit PS hint.

Adds to the record table:
  n_name   folded name (lowercase, accents/punct stripped, spaced single letters joined)
  n_core   n_name without legal-form suffixes / honorific prefixes  -> main name signal
  legal    canonical legal form(s) found, e.g. "inc", "pvt ltd"
  n_addr   folded address with canonical street-type / direction abbreviations
  a_nums   digit tokens in the address (house numbers, zips) as a space-joined string
  a_zip    long digit tokens (>=5 digits; zip / pin code) as a space-joined string
  vague    1 if the address is landmark-style ("near ...", "opp ...")
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

from .translit import is_indic, phonetic_key, to_latin

# legal forms -> canonical token (checked from the END of the name, repeatedly)
LEGAL_CANON = {
    "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp", "co": "co", "company": "co",
    "cos": "co", "llc": "llc", "ltd": "ltd", "limited": "ltd", "ltda": "ltd", "pvt": "pvt", "private": "pvt",
    "plc": "plc", "llp": "llp", "lp": "lp", "pllc": "llc", "gmbh": "gmbh", "ag": "ag", "kg": "kg", "ohg": "ohg",
    "ug": "ug", "sa": "sa", "sas": "sas", "sarl": "sarl", "srl": "srl", "spa": "spa", "bv": "bv", "nv": "nv",
    "oy": "oy", "ab": "ab", "as": "as", "aps": "aps", "kk": "kk", "pte": "pte", "pty": "pty", "bhd": "bhd",
    "sdn": "sdn", "opc": "opc", "cv": "cv",
    # France (test-only country) — SARL/SAS/SA/EURL/SCI/SNC/SCP/SASU/SEM/EARL/GIE
    "sarl": "sarl", "sas": "sas", "sasu": "sas", "eurl": "eurl", "sci": "sci", "snc": "snc", "scp": "scp",
    "earl": "earl", "gie": "gie", "sem": "sem", "scop": "scop", "selarl": "selarl", "sca": "sca",
    # India extras (incl. transliterated forms: प्राइवेट लिमिटेड -> praivet limited)
    "pvtltd": "pvt ltd", "pvt.ltd": "pvt ltd", "opcpvt": "opc pvt", "ltd.": "ltd",
    "praivet": "pvt", "praiveta": "pvt", "praivete": "pvt", "privet": "pvt", "praiveet": "pvt",
    "limiteda": "ltd", "limitad": "ltd", "limitet": "ltd", "limiteet": "ltd", "lim": "ltd", "li": "ltd", "pra": "pvt",
    "prai": "pvt", "limitid": "ltd", "limtd": "ltd", "limted": "ltd", "privat": "pvt", "priv": "pvt",
    "de": None, "and": None, "et": None,  # None = drop only when trailing after a legal token
}
NAME_PREFIXES = {"the", "ms", "messrs", "m s", "sarl", "sas", "sa", "eurl", "sci"}  # FR legal forms often lead

ADDR_CANON = {
    "st": "street", "str": "strasse", "strase": "strasse", "rd": "road", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court", "crt": "court", "pl": "place", "sq": "square",
    "hwy": "highway", "pkwy": "parkway", "cir": "circle", "ter": "terrace", "fl": "floor", "flr": "floor",
    "ste": "suite", "apt": "apartment", "bldg": "building", "blk": "block", "twr": "tower", "mkt": "market",
    "ngr": "nagar", "sec": "sector", "ph": "phase", "nr": "near", "opp": "opposite", "bhnd": "behind",
    "n": "north", "s": "south", "e": "east", "w": "west", "no": "number", "pl.": "place",
    # France
    "av": "avenue", "bd": "boulevard", "bvd": "boulevard", "ch": "chemin", "che": "chemin", "imp": "impasse",
    "all": "allee", "rte": "route", "sq": "square", "pl": "place", "fbg": "faubourg", "r": "rue",
    "st": "street", "ste": "suite",
    # India
    "rd.": "road", "nagar": "nagar", "clny": "colony", "soc": "society", "apts": "apartment", "apt": "apartment",
    "bldg": "building", "flr": "floor", "gr": "ground", "nr": "near", "opp": "opposite", "bh": "behind",
    "b/h": "behind", "dist": "district", "tal": "taluka", "vill": "village", "po": "post", "ps": "police station",
    "h.no": "house number", "hno": "house number", "kh": "khasra", "sec": "sector", "ph": "phase", "ext": "extension",
    "mkt": "market", "cplx": "complex", "ind": "industrial", "indl": "industrial", "est": "estate",
    "gali": "gali", "chowk": "chowk", "marg": "marg", "pkt": "pocket", "blk": "block", "fl": "floor",
    "rly": "railway", "stn": "station", "hosp": "hospital", "sch": "school",
    # US
    "ste": "suite", "hwy": "highway", "pkwy": "parkway", "cir": "circle", "trl": "trail", "ter": "terrace",
    "ct": "court", "ln": "lane", "dr": "drive", "ave": "avenue", "blvd": "boulevard", "rd": "road",
    "unit": "unit", "fl": "floor", "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast",
    "nw": "northwest", "se": "southeast", "sw": "southwest",
}
# region-specific: state / UT names -> codes (US + India), applied to the trailing 1-2 address tokens
US_STATES = {"alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
             "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
             "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
             "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
             "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
             "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
             "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
             "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx",
             "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
             "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc", "puerto rico": "pr"}
IN_STATES = {"andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br", "chhattisgarh": "cg",
             "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp", "jharkhand": "jh",
             "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn",
             "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
             "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "tamilnadu": "tn", "telangana": "tg", "tripura": "tr",
             "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk", "west bengal": "wb", "delhi": "dl",
             "new delhi": "dl", "chandigarh": "ch", "puducherry": "py", "pondicherry": "py", "jammu and kashmir": "jk",
             "jammu kashmir": "jk", "ladakh": "la", "andaman and nicobar islands": "an", "dadra and nagar haveli": "dn",
             "daman and diu": "dd", "lakshadweep": "ld", "maharashtr": "mh", "karnatak": "ka", "gujrat": "gj",
             "telengana": "tg", "chattisgarh": "cg", "uttarpradesh": "up", "westbengal": "wb", "madhyapradesh": "mp"}
STATE_MAP = {**US_STATES, **IN_STATES}
_STATE_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, STATE_MAP), key=len, reverse=True)) + r")\b")


def canon_states(a: str) -> str:
    """Replace full state names anywhere in a folded address by their code (illinois -> il, maharashtra -> mh)."""
    return _STATE_RE.sub(lambda m: STATE_MAP[m.group(1)], a)


VAGUE_TOKENS = {"near", "opposite", "behind", "beside", "next", "adjacent", "landmark", "nr", "opp", "pres", "face"}

_ws = re.compile(r"\s+")
_punct = re.compile(r"[^\w\s]")
_apos = re.compile(r"[’'`´]")
_digits = re.compile(r"\d+")


_domain = re.compile(r"^(?:https?://)?(?:www\.)?|\.(?:com|co\.in|in|net|org|biz|info)\b", re.I)
_null = re.compile(r"\b(?:null|none|nan|n/?a)\b", re.I)


_LEET = str.maketrans("0153478", "oisеatb".replace("е", "e"))
_leet_tok = re.compile(r"\b(?=[a-z0-9]*[a-z])(?=[a-z0-9]*[0-9])[a-z0-9]+\b")


def unleet(s: str) -> str:
    """'precisi0n' -> 'precision', '5ervices' -> 'services'; leaves '3m', '12a', '1st' alone."""
    def fix(m):
        t = m.group(0)
        letters = sum(ch.isalpha() for ch in t)
        digits = len(t) - letters
        return t.translate(_LEET) if letters >= 3 and digits <= 2 else t
    return _leet_tok.sub(fix, s)


def fold(s: str) -> str:
    """Transliterate Indic scripts, lowercase, strip accents, punctuation -> space, join spaced initials,
    drop 'null'-type tokens and web-domain decorations (hightowerarray.com -> hightowerarray)."""
    s = to_latin(str(s))
    s = _null.sub(" ", _domain.sub(" ", s))
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).casefold()
    s = s.replace("&", " and ").replace("@", " at ").replace("+", " and ")
    s = _apos.sub("", s)
    s = _punct.sub(" ", s).replace("_", " ")
    toks = s.split()
    # join runs of single alphabetic characters: "l l c" -> "llc", "i b m" -> "ibm"
    out, run = [], []
    for t in toks:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.append("".join(run))
            run = []
        out.append(t)
    if run:
        out.append("".join(run))
    return " ".join(out)


def split_legal(n_name: str):
    """'acme robotics pvt ltd' -> ('acme robotics', 'pvt ltd')."""
    toks = n_name.split()
    for p in sorted(NAME_PREFIXES, key=len, reverse=True):
        pt = p.split()
        if len(toks) > len(pt) and toks[:len(pt)] == pt:
            toks = toks[len(pt):]
            break
    legal = []
    while len(toks) > 1:
        t = toks[-1]
        c = LEGAL_CANON.get(t, "")
        if c:
            legal.append(c)
            toks.pop()
        elif c is None and legal:        # 'and' / 'de' directly before an already-stripped legal token
            toks.pop()
        else:
            break
    return " ".join(toks), " ".join(reversed(legal))


_hno = re.compile(r"\b(?:h\s*no|hno|h\s*number|house\s*number|door\s*no|door\s*number|plot\s*no|plot\s*number|"
                  r"shop\s*no|shop\s*number|flat\s*no|flat\s*number|office\s*no|office\s*number|unit\s*no|no|number)\b\.?\s*")


def norm_addr(addr: str) -> str:
    a = fold(addr)
    a = canon_states(a)
    toks = a.split()
    # never expand the final token: it is usually the state code (fl = Florida, not floor; ct = Connecticut)
    a = " ".join([ADDR_CANON.get(t, t) for t in toks[:-1]] + toks[-1:])
    a = _hno.sub("", a)  # drop house-number labels: 'h no 1565', 'door number 7c', 'plot no 26' -> bare numbers
    return re.sub(r"\s+", " ", a).strip()


def _normalize_frame(rec: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=rec.index)
    out["indic"] = rec["name"].map(is_indic).astype("int8")
    out["n_name"] = rec["name"].map(fold).map(unleet)
    core_legal = out["n_name"].map(split_legal)
    out["n_core"] = core_legal.str[0]
    out["legal"] = core_legal.str[1]
    out["n_ph"] = out["n_core"].map(phonetic_key)
    out["n_nospace"] = out["n_core"].str.replace(" ", "", regex=False)
    out["n_addr"] = rec["addr"].map(norm_addr)
    nums = out["n_addr"].map(_digits.findall)
    out["a_nums"] = nums.map(" ".join)
    out["a_zip"] = nums.map(lambda ns: " ".join(n for n in ns if len(n) >= 5))
    out["vague"] = out["n_addr"].map(lambda a: int(bool(VAGUE_TOKENS.intersection(a.split())))).astype("int8")
    for c in ("city", "zip", "state", "country"):
        if c in rec:
            out[c] = rec[c].map(fold)
    return out


_NR = {"rec": None}


def _norm_chunk(bounds):
    a, b = bounds
    return _normalize_frame(_NR["rec"].iloc[a:b])


def add_normalized(rec: pd.DataFrame, n_jobs: int = 0, chunk: int = 100_000) -> pd.DataFrame:
    """Add normalised columns. Parallel over row chunks (fork) when the table is large."""
    import multiprocessing as mp
    import os
    n_jobs = n_jobs or (os.cpu_count() or 1)
    bounds = [(a, min(a + chunk, len(rec))) for a in range(0, len(rec), chunk)]
    if n_jobs > 1 and len(bounds) > 1 and "fork" in mp.get_all_start_methods():
        _NR["rec"] = rec
        with mp.get_context("fork").Pool(min(n_jobs, len(bounds))) as pool:
            parts = pool.map(_norm_chunk, bounds)
        _NR["rec"] = None
        extra = pd.concat(parts)
    else:
        extra = _normalize_frame(rec)
    rec = rec.drop(columns=[c for c in extra.columns if c in rec.columns])
    return pd.concat([rec, extra], axis=1)


if __name__ == "__main__":
    for s in ["Acme Robotics Inc.", "ACME ROBOTICS INCORPORATED", "M/s. Sharma Traders Pvt. Ltd.",
              "Müller GmbH & Co. KG", "The I.B.M. Corp", "L L C Holdings LLC", "Zen Traders"]:
        print(f"{s!r:36} -> {split_legal(fold(s))}")
    for a in ["500 Market St, San Jose, CA", "Nr. City Hall, San Jose", "12 Hauptstraße, Berlin 10115",
              "Flat 3, Opp. Bus Stand, MG Rd, Pune 411001"]:
        print(f"{a!r:44} -> {norm_addr(a)!r}")
