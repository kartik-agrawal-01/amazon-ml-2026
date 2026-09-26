# India recall decomposition (QUEUE 5a) — v2 full-density train pass

60,031 India train S1, 207,851 GT pairs; union 3,588,823 pairs (59.8/S1), post-cascade 552,891 (9.2/S1). Rule thr 0.7 + one-to-one (v2). FPs: 1,739.

| stage | pairs | share of GT |
|---|---|---|
| blocked | 13,909 | 6.69% |
| cascade cut | 2,919 | 1.40% |
| model FN | 6,892 | 3.32% |
| TP | 184,131 | 88.59% |
| (model FP) | 1,739 | 0.84% |

## by script

| script | GT | share of GT | blocked | cascade cut | model FN | TP rate | FP | lost pairs (share of all lost) |
|---|---|---|---|---|---|---|---|---|
| both ASCII | 170,572 | 82.1% | 3.7% | 1.0% | 3.7% | 91.6% | 1,582 | 14,290 (60.2%) |
| cand Indic | 37,279 | 17.9% | 20.5% | 3.3% | 1.5% | 74.7% | 157 | 9,430 (39.8%) |

## by cand_addr

| cand_addr | GT | share of GT | blocked | cascade cut | model FN | TP rate | FP | lost pairs (share of all lost) |
|---|---|---|---|---|---|---|---|---|
| present | 199,662 | 96.1% | 5.9% | 1.1% | 2.5% | 90.5% | 1,565 | 19,066 (80.4%) |
| empty | 8,189 | 3.9% | 25.1% | 8.5% | 23.2% | 43.2% | 174 | 4,654 (19.6%) |

## by name_jac

| name_jac | GT | share of GT | blocked | cascade cut | model FN | TP rate | FP | lost pairs (share of all lost) |
|---|---|---|---|---|---|---|---|---|
| .5-.75 | 68,005 | 32.7% | 3.4% | 0.8% | 3.6% | 92.2% | 418 | 5,280 (22.3%) |
| 1 | 53,725 | 25.8% | 1.8% | 0.6% | 2.0% | 95.6% | 137 | 2,368 (10.0%) |
| .25-.5 | 31,612 | 15.2% | 7.3% | 1.2% | 4.8% | 86.7% | 451 | 4,193 (17.7%) |
| 0 | 24,554 | 11.8% | 14.8% | 3.6% | 4.8% | 76.8% | 490 | 5,690 (24.0%) |
| 0-.25 | 22,066 | 10.6% | 20.5% | 3.4% | 1.5% | 74.6% | 178 | 5,606 (23.6%) |
| .75-1 | 7,889 | 3.8% | 2.8% | 0.7% | 3.9% | 92.6% | 65 | 583 (2.5%) |

## by addr_jac

| addr_jac | GT | share of GT | blocked | cascade cut | model FN | TP rate | FP | lost pairs (share of all lost) |
|---|---|---|---|---|---|---|---|---|
| .75-1 | 130,735 | 62.9% | 0.3% | 0.4% | 1.3% | 98.0% | 706 | 2,624 (11.1%) |
| .5-.75 | 34,455 | 16.6% | 3.6% | 1.4% | 4.3% | 90.7% | 283 | 3,190 (13.4%) |
| .25-.5 | 27,203 | 13.1% | 25.7% | 3.4% | 4.6% | 66.4% | 200 | 9,151 (38.6%) |
| 0 | 8,190 | 3.9% | 25.1% | 8.5% | 23.2% | 43.2% | 399 | 4,654 (19.6%) |
| 0-.25 | 7,268 | 3.5% | 44.6% | 4.3% | 7.5% | 43.6% | 151 | 4,101 (17.3%) |

## by legal

| legal | GT | share of GT | blocked | cascade cut | model FN | TP rate | FP | lost pairs (share of all lost) |
|---|---|---|---|---|---|---|---|---|
| same | 96,457 | 46.4% | 8.4% | 1.5% | 1.6% | 88.4% | 390 | 11,173 (47.1%) |
| one side | 55,113 | 26.5% | 5.7% | 1.5% | 5.6% | 87.2% | 857 | 7,059 (29.8%) |
| differ | 31,017 | 14.9% | 4.0% | 0.9% | 4.5% | 90.6% | 262 | 2,912 (12.3%) |
| none | 25,264 | 12.2% | 5.5% | 1.4% | 3.3% | 89.8% | 230 | 2,576 (10.9%) |

Blocked GT pairs per S1-with-a-blocked-pair: 1.38; S1 with >= 1 blocked pair: 10,045
Cascade-cut GT pairs: 2,919. Model FN p quantiles (10/50/90%): [0.053, 0.411, 0.656]

## Reading (loop, 26 Sep 15:50)
- India loses 11.4% of GT pairs; **59% of the loss is blocking** (6.7% of GT), 12% cascade (1.4%), 29% model FN (3.3%).
  Precision is fine (0.84% FP/GT). So India recall = candidate generation first.
- Blocking misses sit where BOTH name and address differ: Indic-script candidate names (20.5% blocked vs 3.7% ASCII;
  40% of all lost pairs), name-token Jaccard ≤ 0.25 (15–21% blocked), address Jaccard < 0.5 (26–45% blocked).
  -> a view that survives script/transliteration and word reordering (name_ph, char n-grams on a transliterated
  skeleton, or the reverse pass) is the lever; a pure address view won't catch them.
- Empty candidate address (3.9% of GT): TP rate only 43% — 25% blocked, 8.5% cascade cut, **23% model FN**. The
  model gives no-address candidates low p; a separate threshold / features for no-address candidates is worth a screen.
- Model FNs are mostly near-threshold (median p 0.41, 90th pct 0.66 at thr 0.70).
