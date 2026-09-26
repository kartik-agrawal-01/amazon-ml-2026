"""Rule-based Indic-script -> Latin transliteration + a phonetic key for cross-script name matching.

Why: ~24% of Indian Source-2 records carry the business name (and often the state) in an Indic
script — a phonetic transliteration of the English name ("Digital Constructions Private Limited"
-> Kannada). Char n-grams cannot bridge scripts, so we map every Indic block to Latin with ONE
table: the Unicode blocks Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada
and Malayalam share the same layout (same offset = same phoneme). Pure Python, no data files
(complies with the no-external-data rule).

    to_latin("डिजिटल कंस्ट्रक्शंस प्राइवेट लिमिटेड") -> "dijital kanstrakshans praivet limited"
    phonetic_key("Digital Constructions Private Limited") -> "djtl knstrktns prvt lmtd"
    phonetic_key(to_latin(...))                            -> "djtl knstrksns prvt lmtd"
"""
from __future__ import annotations

import os
import re
import unicodedata

BLOCKS = [0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00]  # block bases

# offset within block -> Latin (independent vowels 0x05..0x14, consonants 0x15..0x39)
_VOWELS = {0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri", 0x0C: "li",
           0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au"}
_CONS = {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng", 0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh",
         0x1E: "ny", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d",
         0x27: "dh", 0x28: "n", 0x29: "n", 0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m", 0x2F: "y",
         0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l", 0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s",
         0x39: "h"}
# vowel signs 0x3E..0x4C (same order as independent vowels 0x06..0x14)
_SIGNS = {0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri", 0x45: "e", 0x46: "e",
          0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au", 0x55: "e", 0x56: "ai", 0x57: "au",
          0x62: "li", 0x63: "li"}
_NUKTA_MAP = {"k": "q", "kh": "kh", "g": "g", "j": "z", "d": "r", "dh": "rh", "ph": "f", "y": "y"}
_MALAYALAM_CHILLU = {0x0D7A: "n", 0x0D7B: "n", 0x0D7C: "r", 0x0D7D: "l", 0x0D7E: "l", 0x0D7F: "k"}
_EXTRA = {0x0950: "om", 0x0964: ".", 0x0965: ".", 0x200C: "", 0x200D: "", 0x0BB7: "sh"}  # danda, ZWJ/ZWNJ


def _block(cp: int):
    for b in BLOCKS:
        if b <= cp < b + 0x80:
            return b
    return None


def is_indic(s: str) -> bool:
    return any(_block(ord(ch)) is not None for ch in s)


def to_latin(s: str) -> str:
    """Transliterate Indic-script text to plain Latin; non-Indic characters pass through unchanged."""
    if not is_indic(s):
        return s
    out = []
    cons_open = False  # last emitted token is a consonant awaiting an inherent vowel
    for ch in s:
        cp = ord(ch)
        if cp in _MALAYALAM_CHILLU:
            out.append(_MALAYALAM_CHILLU[cp])
            cons_open = False
            continue
        if cp in _EXTRA:
            out.append(_EXTRA[cp])
            cons_open = False
            continue
        b = _block(cp)
        if b is None:
            if cons_open:
                out.append("a")
            out.append(ch)
            cons_open = False
            continue
        off = cp - b
        if off in _CONS:
            if cons_open:
                out.append("a")
            out.append(_CONS[off])
            cons_open = True
        elif off in _VOWELS:
            if cons_open:
                out.append("a")
            out.append(_VOWELS[off])
            cons_open = False
        elif off in _SIGNS:
            out.append(_SIGNS[off])
            cons_open = False
        elif off == 0x4D:  # virama: kills the inherent vowel
            cons_open = False
        elif off == 0x3C:  # nukta modifies the previous consonant
            if out and out[-1] in _NUKTA_MAP:
                out[-1] = _NUKTA_MAP[out[-1]]
        elif off in (0x01, 0x02):  # candrabindu / anusvara
            if cons_open:
                out.append("a")
            out.append("n")
            cons_open = False
        elif off == 0x03:  # visarga
            if cons_open:
                out.append("a")
            out.append("h")
            cons_open = False
        elif 0x66 <= off <= 0x6F:  # digits
            if cons_open:
                out.append("a")
            out.append(str(off - 0x66))
            cons_open = False
        else:  # rare marks: drop
            pass
    # inherent vowel at word end is deleted (schwa deletion) -> nothing appended when cons_open
    txt = "".join(out)
    return txt


_ph_rules = [(r"ph", "f"), (r"chh", "ch"), (r"sh", "s"), (r"ck", "k"), (r"c(?=[eiy])", "s"), (r"c", "k"), (r"q", "k"),
             (r"x", "ks"), (r"w", "v"), (r"z", "j"), (r"th", "t"), (r"dh", "d"), (r"bh", "b"), (r"gh", "g"),
             (r"kh", "k"), (r"jh", "j"), (r"ny", "n"), (r"ng", "n"), (r"y", "i")]
# AML_PH=2 (day loop s7, opt-in): Tamil script has no voiced/voiceless or f contrast ("kulopal pilak" = "global black",
# "hpavunteshn" = "foundation"), so fold hp -> f and g/b/d -> k/p/t in the key.
if os.environ.get("AML_PH", "") == "2":
    _ph_rules = [(r"hp", "f")] + _ph_rules + [(r"g", "k"), (r"b", "p"), (r"d", "t")]
_ph_compiled = [(re.compile(p), r) for p, r in _ph_rules]


def phonetic_key(s: str) -> str:
    """Consonant skeleton per word: transliterate, fold, apply phonetic merges, drop vowels except a leading one,
    collapse repeats. Robust to transliteration spelling ('praivet' vs 'private' -> both 'prvt')."""
    s = to_latin(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    words = re.findall(r"[a-z0-9]+", s)
    keys = []
    for w in words:
        for pat, rep in _ph_compiled:
            w = pat.sub(rep, w)
        lead = w[0] if w and w[0] in "aeiou" else ""
        w = lead + re.sub(r"[aeiou]", "", w[1:] if lead else w)
        w = re.sub(r"(.)\1+", r"\1", w)
        if w:
            keys.append(w)
    return " ".join(keys)


if __name__ == "__main__":
    tests = ["डिजिटल कंस्ट्रक्शंस प्राइवेट लिमिटेड", "ಡಿಜಿಟಲ್ ಕನ್‌ಸ್ಟ್ರಕ್ಷನ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್",
             "बेस्ट प्रॉपर्टीज प्रा. लि.", "स्काई एग्रो प्राइवेट लिमिटेड", "અર્બન પ્રોજેક્ટ્સ લિમિટેડ",
             "ઓમ વેન્ચર્સ પ્રાઇવેટ લિમિટેડ", "பிரைட் இன்ஃப்ராஸ்ட்ரக்சர் பிரைவேட் லிமிடெட்", "ಕರ್ನಾಟಕ", "Uttar Pradesh"]
    for t in tests:
        print(f"{t[:40]:<40} -> {to_latin(t)!r:<45} key={phonetic_key(t)!r}")
    for t in ["Digital Constructions Private Limited", "Best Properties Pvt Ltd", "Sky Agro Private Limited",
              "Urban Projects Limited", "Om Ventures Private Limited", "Bright Infrastructure Private Limited", "Karnataka"]:
        print(f"{t:<40} key={phonetic_key(t)!r}")
