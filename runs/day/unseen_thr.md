# QUEUE 2d: threshold for an unseen country (loop, 26 Sep ~16:50)

Scripts: `scripts/day/unseen_thr.py <data> <out>` (sweep + rules), `scripts/day/check_adapt.py` (checks the pipeline
implementation offline). Raw outputs: `runs/day/unseen_thr_{xc_us_n3k,xc_in_n3,mix_n3k}.txt`.
Decision approximation = pipeline's: p >= 0.02, one-to-one per candidate record (max-p S1), then p >= t. Key-rule forcing
is ignored (n3k was run with --key-rules 0.96, so n3k numbers are without forcing; at the OOF threshold that gives
exactly the reported 0.9321).

## (i) Hidden F0.5 vs threshold, target country

| t | xc_us (US -> India), n3k | xc_in (India -> US), n3 | slice_mix (US+IN), n3k |
|---|---|---|---|
| 0.02 | 0.9461 | – | – |
| 0.05 | 0.9496 | – | – |
| 0.10 | 0.9509 | – | – |
| 0.15 | **0.9511** | – | – |
| 0.30 | 0.9497 | 0.9729 | – |
| 0.50 | 0.9459 | 0.9774 | – |
| 0.70 | 0.9404 | 0.9807 | 0.9822 |
| 0.74 | | | **0.9823** |
| 0.85 (OOF-chosen, all three) | 0.9321 | **0.9819** (best 0.86) | 0.9815 |
| 0.95 | 0.9128 | 0.9795 | 0.9772 |

The US-trained model is under-confident on India: the curve rises monotonically down to t ≈ 0.15, **+0.019** over the
OOF-chosen 0.85. The India-trained model on the US is best at its OOF threshold. So "the unseen country needs a lower
threshold" holds in one direction only.

## (ii) Label-free rules (per target country)

| rule | xc_us t / F (Δ vs OOF-chosen) | xc_in t / F | slice_mix t / F |
|---|---|---|---|
| OOF-chosen | 0.85 / 0.9321 | 0.85 / 0.9819 | 0.85 / 0.9815 |
| hidden best | 0.15 / 0.9511 | 0.86 / 0.9819 | 0.74 / 0.9823 |
| R1 count matching (GT/S1 3.46) | 0.02 / 0.9461 (+0.0139) | 0.97 / 0.9766 (−0.0053) | IN 0.48, US 0.93 / 0.9804 (−0.0011) |
| R2 empty matching | 0.03 / 0.9480 (+0.0158) | 0.99 / 0.9644 (−0.0175) | IN 0.67, US 0.98 / 0.9758 (−0.0058) |
| **R1 lower-only** = min(t_oof, R1) | 0.02 / 0.9461 (+0.0139) | 0.85 / 0.9819 (0) | IN 0.48, US 0.85 / 0.9817 (+0.0002) |
| **R2 lower-only** = min(t_oof, R2) | 0.03 / 0.9480 (+0.0158) | 0.85 / 0.9819 (0) | IN 0.67, US 0.85 / 0.9820 (+0.0005) |

- No plain rule lands within 0.001 of the hidden best on both screens: R1/R2 would raise the US threshold far too high on
  xc_in. The **lower-only** versions never hurt (xc_in 0, mix +0.0002/+0.0005) and recover 73–83% of the xc_us gap
  (−0.0031 / −0.0050 from the best). Est. ΔQ for R2 lower-only ≈ 0.5·0.0005 + 0.3·0.0158 ≈ **+0.005**.
- R3 (sure-key recall matching) is NOT done yet. It needs the key tables of the source OOF; next session if still useful.

## What it means for France (v2 numbers, no new run)
v2 France test: empty 5.2%, 3.30 pred/S1. v2 OOF at t 0.70: empty 6.08%, 3.20 pred/S1 (GT 3.46). France is already at or
above the source levels, so **both lower-only rules keep France at v2's threshold**: France shows no under-confidence
signature in the counts. Either v2 is fine on France, or its French errors are false positives (key pairs are
`disjoint|a_eq|invented`-type), which a lower-only rule cannot fix. So the rule is a safe default (it only fires when a
country looks under-confident) but not a France fix in v2. Recheck on the v3/jv1 France counts: the France fixes change
France's score distribution.

## Implementation (opt-in, default unchanged)
`src/pipeline.py --thr-adapt`: training stores the OOF predicted-empty rate in `best["src_empty"]`. At test time each
country is decided at the p floor, then after global one-to-one `adapt_threshold` picks t ≤ t_oof (0.01 grid) matching
that empty rate and drops pairs below it. The per-country t is logged and stored in `test_pred_by_country[].thr`.
Offline check on the saved probs (check_adapt.py): xc_us 0.9321 → 0.9480 (t 0.03), xc_in 0.9819 → 0.9819 (t 0.85).
Screen running: tmux `chain6` = n3ka (n3k + --thr-adapt) on xc_us, slice_mix and xc_in. chain4 is paused (SIGSTOP on its
bash) until chain6 is done, then resumes with n3r.
Caveat: t = 0.03 is extreme. At full density (12x more decoys) a very low threshold may cost more precision than on the
slice. A floor (e.g. 0.10–0.15; 0.15 is the xc_us best) would be safer, but choosing it from this one screen would be tuning on the test.

## (iii) R3 sure-key matching (loop, 26 Sep 19:20) — `scripts/day/unseen_r3.py <data> <cache_n3> <out>`
Raw: `runs/day/unseen_r3_{xc_us_n3k,xc_in_n3,mix_n3k}.txt`. Sure pairs = hq_keys key_pairs + apply_rules (p_min 0.96),
rules calibrated on the source's OOF S1 (context all S1; unseen target -> min over train countries). R3 = t where the
target's sure-pair recall = source OOF recall at t_oof; R3n = same on recall relative to the p-floor (0.02) recall.

| screen | source recall @t_oof (rel. floor) | target sure prec (hidden) | R3 t / ΔF | R3n t / ΔF | R3n lower-only ΔF | hidden best |
|---|---|---|---|---|---|---|
| xc_us (US->IN) | 0.9987 (0.99+) | 0.994 | 0.02 / +0.0139 | 0.03 / +0.0158 | +0.0158 | 0.15 / +0.0190 |
| xc_in (IN->US) | 0.9823 (0.9975) | 0.999 | 0.99 / −0.0175 | 0.98 / −0.0088 | 0 | 0.86 / ~0 |
| slice_mix | 0.9940 (0.9981) | IN 0.994, US 0.999 | IN .83 US .99 / −0.0102 | IN .81 US .99 / −0.0101 | +0.0002 | 0.74 / +0.0008 |

Verdict: R3 is NOT usable two-sided: sure-pair recall saturates at 0.98-0.999, so tiny recall differences move t to
the grid ends (0.02 or 0.99). Lower-only R3n reproduces R2 lower-only (xc_us +0.0158, xc_in 0, mix +0.0002) and adds
nothing; no rule is within 0.001 of the hidden best on xc_us (0.9511 at t 0.15). Keep --thr-adapt (R2 lower-only) as is.
Side fact: the sure pairs are 99.4% (India) / 99.9% (US) true on the hidden GT even when the country is unseen.
