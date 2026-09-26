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

import os
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
    "praibhet": "pvt", "praibheta": "pvt", "praibhete": "pvt", "praiwet": "pvt", "praiweta": "pvt", "praivat": "pvt",
    "elelpi": "llp", "elelsi": "llc", "inka": "inc", "ink": "inc", "korp": "corp", "korporeshan": "corp",
    "limiteda": "ltd", "limitad": "ltd", "limitet": "ltd", "limiteet": "ltd", "limitedu": "ltd", "limiteda": "ltd",
    "limiteda": "ltd", "limitad": "ltd", "limitet": "ltd", "limiteet": "ltd", "lim": "ltd", "li": "ltd", "pra": "pvt",
    "prai": "pvt", "limitid": "ltd", "limtd": "ltd", "limted": "ltd", "privat": "pvt", "priv": "pvt",
    # France, from the test data (26 Sep): EI (entreprise individuelle) is the 7th most common S1 legal form and was
    # kept inside the core name; SASU/SNC/SELAS/SARLU variants; "as" removed (Norwegian AS never occurs, but truncated
    # French words did: "Saint-Nazaire As" -> legal "as").
    "ei": "ei", "eirl": "ei", "selas": "selas", "sarlu": "sarl", "scm": "scm", "scea": "scea", "sasu": "sas",
    "de": None, "and": None, "et": None,  # None = drop only when trailing after a legal token
}
LEGAL_CANON.pop("as", None)
# AML_PH=2 (day loop s7, opt-in): Tamil/Malayalam/Telugu transliterations of "private limited" that were kept in the core
# name (runs/day/india_blocked_sample.md: "piraivet limitet", "praivarr limirrad").
if os.environ.get("AML_PH", "") == "2":
    LEGAL_CANON.update({t: "pvt" for t in ["piraivet", "piraivett", "pirayvet", "praivarr", "praivar",
                                          "praivett", "praivettu", "piraivettu", "praivetu", "prayvet"]})
    LEGAL_CANON.update({t: "ltd" for t in ["limirrad", "limirrat", "limitettu", "limitedu", "limited", "limittet",
                                          "limitat", "limitedd", "limirred", "limitetu", "limiteddu"]})
# legal forms that lead the name (5.7% of French S2/S3 names, 0.01% of S1: "SARL JEUNE PHARMACIE") + honorifics
NAME_PREFIXES = {"the", "ms", "messrs", "m s", "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "scp", "selarl"}
# name-token canon applied inside fold(): French "et" is the "&"/"and" variant (1.1% of FR S2/S3 names vs 0.01% of S1);
# "Cie" <-> "Compagnie" abbreviation swaps.
NAME_TOKEN_CANON = {"et": "and", "cie": "compagnie"}
# legal forms glued to the end of web-domain names ("folieclubsas.com" <-> "Folie Club SAS"): 35% of French domain
# records embed the legal form (US/India: ~0%). Applied only to domain/handle-derived single-token names.
_DOMAIN_LEGAL = re.compile(r"^(.{4,}?)(sarl|sasu|sas|eurl|sci|snc|ei|inc|llc|ltd|llp|corp|pvtltd|pvt)$")

ADDR_CANON = {
    "st": "street", "str": "strasse", "strase": "strasse", "rd": "road", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court", "crt": "court", "pl": "place", "sq": "square",
    "hwy": "highway", "pkwy": "parkway", "cir": "circle", "ter": "terrace", "fl": "floor", "flr": "floor",
    "ste": "suite", "apt": "apartment", "bldg": "building", "blk": "block", "twr": "tower", "mkt": "market",
    "ngr": "nagar", "sec": "sector", "ph": "phase", "nr": "near", "opp": "opposite", "bhnd": "behind",
    "n": "north", "s": "south", "e": "east", "w": "west", "no": "number", "pl.": "place",
    # France (abbreviations measured on the test S2/S3: r 12.7%, av 3.6%, all 1.6%, bd 1.5%, ave 1.4%, imp/pl/rte 0.8%,
    # ch 0.5%, blvd 0.4%, crs 0.3%, q 0.3%)
    "av": "avenue", "bd": "boulevard", "bvd": "boulevard", "boulevrd": "boulevard", "ch": "chemin",
    "che": "chemin", "chem": "chemin", "imp": "impasse", "all": "allee", "rte": "route", "sq": "square", "pl": "place",
    "fbg": "faubourg", "fg": "faubourg", "r": "rue", "crs": "cours", "q": "quai", "rle": "ruelle", "prom": "promenade",
    "esp": "esplanade", "res": "residence", "st": "street", "ste": "suite",
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

# France: the test S1 addresses ALWAYS end with the region ("Nantes, Pays de la Loire"); S2/S3 carry the region (32%),
# the DEPARTMENT instead (32%: "Nantes, Loire-Atlantique") or neither (36%). Region and department are both mapped to
# one region code (the US analogue of "Illinois" / "IL"), so the pair agrees on that component whenever both have one.
# Department names are matched only as a WHOLE comma component (nord, var, lot, cher ... are ordinary words too);
# multi-word region names are also matched inside the string (commas may be missing).
FR_REGIONS = {
    "hdf": ["hauts de france", "nord pas de calais", "picardie", "aisne", "nord", "oise", "pas de calais", "somme"],
    "naq": ["nouvelle aquitaine", "aquitaine", "limousin", "poitou charentes", "charente", "charente maritime",
            "correze", "creuse", "dordogne", "gironde", "landes", "lot et garonne", "pyrenees atlantiques",
            "deux sevres", "vienne", "haute vienne"],
    "pdl": ["pays de la loire", "loire atlantique", "maine et loire", "mayenne", "sarthe", "vendee"],
    "idf": ["ile de france", "region parisienne", "seine et marne", "yvelines", "essonne", "hauts de seine",
            "seine saint denis", "val de marne", "val d'oise"],
    "ara": ["auvergne rhone alpes", "rhone alpes", "auvergne", "ain", "allier", "ardeche", "cantal", "drome", "isere",
            "loire", "haute loire", "puy de dome", "rhone", "savoie", "haute savoie"],
    "bfc": ["bourgogne franche comte", "bourgogne", "franche comte", "cote d'or", "doubs", "jura", "nievre",
            "haute saone", "saone et loire", "yonne", "territoire de belfort"],
    "bre": ["bretagne", "cotes d'armor", "finistere", "ille et vilaine", "morbihan"],
    "cvl": ["centre val de loire", "cher", "eure et loir", "indre", "indre et loire", "loir et cher", "loiret"],
    "cor": ["corse", "corse du sud", "haute corse"],
    "ges": ["grand est", "alsace", "lorraine", "champagne ardenne", "ardennes", "aube", "marne", "haute marne",
            "meurthe et moselle", "meuse", "moselle", "bas rhin", "haut rhin", "vosges"],
    "nor": ["normandie", "basse normandie", "haute normandie", "calvados", "eure", "manche", "orne", "seine maritime"],
    "occ": ["occitanie", "languedoc roussillon", "midi pyrenees", "ariege", "aude", "aveyron", "gard", "haute garonne",
            "gers", "herault", "lot", "lozere", "hautes pyrenees", "pyrenees orientales", "tarn", "tarn et garonne"],
    "pac": ["provence alpes cote d'azur", "paca", "alpes de haute provence", "hautes alpes", "alpes maritimes",
            "bouches du rhone", "var", "vaucluse"],
}
_FR_CANON = {}  # folded name -> code (filled after fold() is defined)


def canon_states(a: str) -> str:
    """Replace full state names anywhere in a folded address by their code (illinois -> il, maharashtra -> mh)."""
    return _STATE_RE.sub(lambda m: STATE_MAP[m.group(1)], a)


VAGUE_TOKENS = {"near", "opposite", "behind", "beside", "next", "adjacent", "landmark", "nr", "opp", "pres", "face"}

_ws = re.compile(r"\s+")
_punct = re.compile(r"[^\w\s]")
_apos = re.compile(r"[’'`´]")
_digits = re.compile(r"\d+")


_domain = re.compile(r"^(?:https?://)?(?:www\.)?|\.(?:com|co\.in|in|net|org|biz|info|fr|eu|io|de|uk)\b", re.I)
_handle = re.compile(r"^\s*[@#](?=\w)")          # social handles used as names: "@asseclub", "#priveamis"
_is_compact = re.compile(r"^\s*(?:[@#]\w|(?:https?://|www\.)|\S+\.(?:com|co\.in|in|net|org|biz|info|fr|eu|io|de|uk)\b)", re.I)
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
    s = _null.sub(" ", _domain.sub(" ", _handle.sub("", s)))
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
        out.append(NAME_TOKEN_CANON.get(t, t))
    if run:
        out.append("".join(run))
    return " ".join(out)


for _code, _names in FR_REGIONS.items():
    for _n in _names:
        _FR_CANON[fold(_n)] = _code
_FR_MULTI_RE = re.compile(r"\b(" + "|".join(sorted((re.escape(k) for k in _FR_CANON if " " in k), key=len, reverse=True)) + r")\b")


def split_legal(n_name: str, compact: bool = False):
    """'acme robotics pvt ltd' -> ('acme robotics', 'pvt ltd'); 'sarl jeune pharmacie' -> ('jeune pharmacie', 'sarl').
    compact=True (name came from a web domain / handle, one glued token): 'folieclubsas' -> ('folieclub', 'sas')."""
    toks = n_name.split()
    legal = []
    for p in sorted(NAME_PREFIXES, key=len, reverse=True):
        pt = p.split()
        if len(toks) > len(pt) and toks[:len(pt)] == pt:
            toks = toks[len(pt):]
            if LEGAL_CANON.get(p):
                legal.append(LEGAL_CANON[p])
            break
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
    if compact and len(toks) == 1 and not legal:
        m = _DOMAIN_LEGAL.match(toks[0])
        if m:
            toks = [m.group(1)]
            legal.append(LEGAL_CANON.get(m.group(2), m.group(2)))
    return " ".join(toks), " ".join(reversed(legal))


_hno = re.compile(r"\b(?:h\s*no|hno|h\s*number|house\s*number|door\s*no|door\s*number|plot\s*no|plot\s*number|"
                  r"shop\s*no|shop\s*number|flat\s*no|flat\s*number|office\s*no|office\s*number|unit\s*no|no|number)\b\.?\s*")
# leading house-number labels of a comma component: "N° 32", "Nº 32", "No 32", "no. 32", "nr 32", "numero 32" -> "32".
# A bare "n" is a label only before a plain integer ("n 108"), so "N 5th St" keeps its direction (-> north 5th street).
_numlabel = re.compile(r"^(?:no|nr|num|numero|number)\s+(?=\d)|^n\s+(?=\d+(?:\s|$))")
_bister = re.compile(r"\b(\d+)(bis|ter|quater)\b")   # "5bis" -> "5 bis"
_lead0 = re.compile(r"\b0\d+\b")


def _strip_lead0(m) -> str:
    """0332 -> 332, 00262 -> 262, 01 -> 1 (French vendors zero-pad house numbers); a 5-digit US ZIP with one leading
    zero (01234) is kept."""
    t = m.group(0)
    s = t.lstrip("0") or "0"
    return s if (len(t) - len(s) >= 2 or len(t) <= 4) else t


def norm_addr(addr: str) -> str:
    comps = []
    for raw in str(addr).split(","):
        c = fold(raw)
        if not c:
            continue
        c = _numlabel.sub("", c)
        c = _bister.sub(r"\1 \2", _lead0.sub(_strip_lead0, c))
        if c in _FR_CANON:                       # whole component = French region / department -> region code
            comps.append(_FR_CANON[c])
            continue
        toks = c.split()
        if len(toks) > 1 and toks[0] in ("st", "ste") and not any(ch.isdigit() for ch in c):
            toks[0] = "saint" if toks[0] == "st" else "sainte"   # "St-Nazaire" (city component) vs "Main St"
        comps.append(" ".join(toks))
    a = _FR_MULTI_RE.sub(lambda m: _FR_CANON[m.group(1)], " ".join(comps))
    a = canon_states(a)
    toks = a.split()
    out = []
    for i, t in enumerate(toks):
        # never expand the final token: it is usually the state code (fl = Florida, not floor; ct = Connecticut)
        if i == len(toks) - 1 or (t == "ter" and i > 0 and toks[i - 1].isdigit()):   # "22 ter rue" is not a terrace
            out.append(t)
        else:
            out.append(ADDR_CANON.get(t, t))
    a = _hno.sub("", " ".join(out))  # drop house-number labels: 'h no 1565', 'door number 7c', 'plot no 26' -> bare numbers
    return re.sub(r"\s+", " ", a).strip()


def _normalize_frame(rec: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=rec.index)
    out["indic"] = rec["name"].map(is_indic).astype("int8")
    out["n_name"] = rec["name"].map(fold).map(unleet)
    compact = rec["name"].astype(str).str.contains(_is_compact, regex=True).to_numpy()
    core_legal = [split_legal(n, bool(k)) for n, k in zip(out["n_name"].to_numpy(), compact)]
    # build as string arrays (not object tuples): same values, ~3x less RAM on 12M rows (pandas 3 -> pyarrow str)
    out["n_core"] = pd.Series([t[0] for t in core_legal], index=rec.index, dtype="str")
    out["legal"] = pd.Series([t[1] for t in core_legal], index=rec.index, dtype="str")
    del core_legal
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


def add_normalized(rec: pd.DataFrame, n_jobs: int = 0, chunk: int = 100_000) -> pd.DataFrame:
    """Add normalised columns. Parallel over row chunks (spawned workers, explicit payloads) when large."""
    import multiprocessing as mp
    import os
    n_jobs = n_jobs or (os.cpu_count() or 1)
    bounds = [(a, min(a + chunk, len(rec))) for a in range(0, len(rec), chunk)]
    if n_jobs > 1 and len(bounds) > 1:
        from concurrent.futures import ProcessPoolExecutor
        cols = [c for c in ("name", "addr", "city", "zip", "state", "country") if c in rec]
        with ProcessPoolExecutor(min(n_jobs, len(bounds)), mp_context=mp.get_context("spawn")) as ex:
            parts = list(ex.map(_normalize_frame, (rec.iloc[a:b][cols] for a, b in bounds)))
    else:
        parts = [_normalize_frame(rec)]
    # Assemble column by column (freeing each chunk column as we go) instead of two wide concats:
    # same result, but peak RAM ~ rec + parts instead of rec + parts + extra + concat copy (box has ~10 GB).
    new_cols = list(parts[0].columns)
    rec = rec.drop(columns=[c for c in new_cols if c in rec.columns])
    for c in new_cols:
        col = pd.concat([p[c] for p in parts]) if len(parts) > 1 else parts[0][c]
        assert len(col) == len(rec)
        rec[c] = col
        for p in parts:
            del p[c]
    return rec


if __name__ == "__main__":
    for s in ["Acme Robotics Inc.", "ACME ROBOTICS INCORPORATED", "M/s. Sharma Traders Pvt. Ltd.",
              "Müller GmbH & Co. KG", "The I.B.M. Corp", "L L C Holdings LLC", "Zen Traders"]:
        print(f"{s!r:36} -> {split_legal(fold(s))}")
    for a in ["500 Market St, San Jose, CA", "Nr. City Hall, San Jose", "12 Hauptstraße, Berlin 10115",
              "Flat 3, Opp. Bus Stand, MG Rd, Pune 411001"]:
        print(f"{a!r:44} -> {norm_addr(a)!r}")
