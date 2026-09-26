# --thr-adapt floor: evidence (loop session 9, 26 Sep ~17:15). NEXT 3 of session 8.

Question: xc_us picks t = 0.03 on the slice. Is a low t safe at FULL density (~12x more same-name decoys)?
Method (label-free w.r.t. the test): in-distribution OOF curves F0.5(t) − F0.5(t_ref) per country, full-density v2 train
pass (`output_v2_train/oof_pairs.tsv.gz`, 150K S1, t_ref 0.70) vs the slice (n3k slice_mix OOF, 40K S1, t_ref 0.85).
Script: `scripts/day/floor_evidence.py`; raw: `runs/day/floor_v2.txt`, `runs/day/floor_mix_n3k.txt`.

| t | full density, ALL (IN / US) | slice, ALL (IN / US) | ratio |
|---|---|---|---|
| 0.03 | −0.1046 (−0.093 / −0.112) | −0.0473 (−0.047 / −0.048) | 2.2x |
| 0.10 | −0.0478 | −0.0261 | 1.8x |
| 0.15 | −0.0337 | −0.0196 | 1.7x |
| 0.30 | −0.0144 | −0.0100 | 1.4x |
| 0.50 | −0.0035 | −0.0045 | ~1x |

In distribution, a low t costs 1.4–2.2x more at full density than on the slice. Crude transfer to xc_us (gain = under-
confidence benefit − precision cost; add the extra full-density cost to the slice xc_us gains vs t 0.85):
0.03: +0.016 − 0.057 ≈ **−0.04**; 0.15: +0.019 − 0.014 ≈ +0.005; 0.30: +0.0176 − 0.0044 ≈ **+0.013**; 0.50: +0.014 − 0 ≈ +0.014.
Unfloored R2 at full density could be badly wrong on an under-confident country.

Slice xc_us (n3k, check_adapt.py with floor): floor 0.02 → t 0.03 / 0.9480; 0.15 → 0.9511; **0.30 → 0.9497**; 0.50 → 0.9459.
xc_in / slice_mix: adapt picks t ≥ 0.67 there, so a floor ≤ 0.50 changes nothing.

**Recommendation: `--thr-adapt --thr-adapt-floor 0.30` for any full-data file** (costs 0.0014 on the slice xc_us vs the
slice-best floor, and keeps the full-density precision cost small). New flag `--thr-adapt-floor` (default 0.02 =
unchanged behaviour; the n3ka screen runs unfloored). Caveats: the cost ratio mixes t_ref 0.70 vs 0.85 and v2 vs n3k
features; the transfer arithmetic assumes the precision cost adds linearly. For France, v2's counts gave no under-
confidence signature, so the floor matters only if the v3/jv1 France counts do.
