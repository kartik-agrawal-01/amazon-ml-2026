# v2 vs Soha, consensus result, one-to-one bug, exact-key rules (HQ, 26 Sep ~12:00 IST)

Scripts: `scripts/hq_ens/` (prototype, how ens2c was built) and `src/hq_keys.py` (pipeline version; reproduces the
prototype's France 'sure' set exactly: 727,508 / 727,508 pairs).

## 1. v2 (LB 0.947) vs Soha (LB 0.948) on test

| country | S1 | empty v2 / Soha | matches/S1 v2 / Soha | identical sets | v2-only pairs/S1 | Soha-only pairs/S1 | F(v2 vs Soha) |
|---|---|---|---|---|---|---|---|
| US | 663,106 | 5.5% / 6.1% | 3.35 / 3.29 | 78.7% | 0.164 | 0.101 | 0.948 |
| India | 809,986 | 6.7% / 7.8% | 3.12 / 3.08 | 65.7% | 0.264 | 0.225 | 0.900 |
| France | 259,452 | 5.2% / 5.9% | 3.30 / 3.28 | 56.3% | 0.335 | 0.309 | 0.883 |

Almost all disputes are presence disputes (one side assigns the record, the other leaves it unassigned). Conflicts
where the two give the same record to different S1s are only ~0.01/S1. Pattern in France: v2 leans on the NAME (same
name, different house number / different address) and Soha leans on the ADDRESS (same address, one word swapped or a
different name).

**Consensus (ens1 = v2 ∩ Soha, keep v2 where empty/disjoint): LB 0.9449, −0.2 vs v2.** The dropped v2-only pairs
(0.204/S1) are ~71–75% true (break-even ≈ 70%), so the disagreement set is not where the loss is. Intersection/consensus
goes under tried & failed. A union is expected to be about neutral.

## 2. Bug: one-to-one is block-local in v2

`decide(..., one2one=True)` runs per 100K-S1 block, so a record can be kept for S1s in different blocks. In the v2
submission, **13,806 S2/S3 ids are assigned to 2–5 S1s** (28,302 pair slots; France 6,672 ids, India 4,899, US 2,235;
21,250 S1 affected). The GT is strictly one-to-one, so at least 14,496 of those pairs are false. France is hit
hardest because of its dense same-name decoys: the box diagnostic shows grp_n_close 16 vs 8 in the US. Soha's file has 0
such ids. The OOF never showed this because the train pass is a single block per country.
Fix: `hq_keys.global_one_to_one(q_rid, c_rid, p)` after all blocks of a country (keep the max-p S1 per record).

## 3. Exact-key rules calibrated on the FULL train

Every (S1, S2/S3) pair of the full US/India train that shares the exact core name, the exact no-space name or the exact
normalised address (sorted tokens) was enumerated with joins (12.0M pairs, 4.86M of the 7.64M GT pairs). Each pair got
a name relation (core equal / no-space equal / reorder / subset / one-token swap / partial / disjoint), an address
relation (exact / same street text, different number / same street, a number missing / same first number, street text
differs / candidate empty / other) and a **context**: `core_other` = another S1 of the country has the candidate's
exact core name, `addr_other` = another S1 has the candidate's exact address ('ff' = neither). Different-name pairs
are split into **invented** names (at least one token not used by ≥ 5 S1 names of the country: trade names like
"Deltavantagepyra", domain names like "comitdesfonds.com", heavy typos) and **realword** names (only S1 vocabulary).

| rule (`hq_keys.RULES`) | P(match) US (n) | P(match) India (n) |
|---|---|---|
| core_eq, exact address | 0.9999 (951K) | 0.9999 (293K) |
| subset / reorder, exact address | 0.999 / 1.000 | 0.999 / 1.000 |
| nsp_eq, exact address, name not another S1's | 1.000 (63K) | 1.000 (24K) |
| core_eq, same first number, street text differs, not (core_other & addr_other) | 0.997 (615K) | 0.958 (461K) |
| nsp_eq, same first number, ff | 0.999 | 0.981 |
| core_eq, same street, number missing | 1.000 (105K) | **0.755** (6K) |
| core_eq, candidate address empty, name not another S1's | 0.972 | 0.989 |
| one-token swap, exact address, ff, invented / realword | 0.970 / 0.900 | 0.981 / 0.979 |
| partial overlap, exact address, ff, invented / realword | 0.961 / 0.882 | 0.987 / 0.996 |
| **disjoint name, exact address, ff, invented** | **0.963** (85K) | **0.982** (147K) |
| disjoint name, exact address, ff, realword | 0.491 | 0.789 |
| (for contrast) core_eq, same street, different house number | 0.54 | 0.79 |
| (for contrast) core_eq, other address, name also another S1's | 0.04 | — |

Context is decisive. For example, a disjoint name at the exact address is 96% a match when neither the name nor the
address belongs to another S1, and 0.1% when both do. **Calibrate at full density:** on the 8% slice the ff-rule rates
fall to 0.86, because the decoy S1s that set the context flags are mostly missing from the slice. The same density
effect makes the slice Q blind to decoy-driven problems.

## 4. Coverage on test: which 'sure' pairs do the models predict?

| rule | US v2 / Soha | India v2 / Soha | France v2 / Soha |
|---|---|---|---|
| core_eq, exact address | 99.9 / 100.0 | 99.8 / 99.0 | 99.5 / 99.4 |
| core_eq, same number, street text differs | 99.1 / 99.4 | 93.1 / 93.0 | 94.2 / 95.1 |
| **disjoint invented name, exact address, ff** | **92.1 / 95.7** | 94.2 / 87.9 | **33.8 / 37.7** |
| disjoint realword name, exact address, ff | 47.4 / 55.2 | 81.3 / 69.3 | 12.5 / 18.5 |
| swap1 exact address ff | 91.9 / 92.8 | 97.2 / 92.7 | 85.9 / 81.8 |
| partial exact address ff | 91.7 / 92.7 | 98.1 / 92.4 | 75.5 / 71.1 |
| nsp_eq, same number, ff | 94.6 / 98.8 | 86.4 / 85.3 | 73.1 / 76.4 |

**France recall hole.** Different trade/domain names at the same exact address are true 96–98% of the time in
train. v2 finds 92% of them in the US but only 34% in France (Soha 38%). France has 37.5K such pairs (0.14/S1), and
both models miss about 24K (0.10/S1). Likely mechanism: France S1s have ~16 same-name decoys and every French S1 hits
the cascade's top-10 cap (p10 = p50 = p90 = 10 in the box diagnostic). The old normaliser also made the addresses
differ ("N° 32" → "north 32", département vs région, zero-padded numbers), so the address views often fail to retrieve
the right record: the box diagnostic's addr_c3 rank field for the top candidate averages 46 in France vs 9 in the
US. Then the record is cut before the model ever sees it.

**Why the box diagnostic saw a "normal" France:** it profiles *accepted* pairs. The missed trade-name records (−0.10/S1)
and the extra look-alike / duplicate-owner pairs (the one-to-one bug alone accounts for 0.033/S1 in France) roughly
cancel in the counts.

**Match-count consistency** (train GT: 3.46 matches per S1, identical for US and India, so the generator uses one
distribution): v2 predicts 3.35 in the US, **3.12 in India** and 3.30 in France. ens2c brings France to 3.43. India's
deficit, ~0.35 true pairs/S1, is the largest recall loss on the board (47% of test) and is **not** fixable with exact
keys (India's key-rule coverage is already 93–99%). It's fuzzy: Indic scripts, landmark addresses, and blocking recall
0.933 at k=10 on full-density docs.

## 5. ens2c (LOG #3): what it changes and why the estimate is conservative

- Global one-to-one on v2: for every multi-owner record keep the owner with a 'sure' key match (4,266 ids), else
  Soha's owner (6,004), else none (3,536). 18,032 pairs removed.
- Add sure pairs that v2 missed, only where the **worst-case** precision of the missed pairs,
  (P − coverage)/(1 − coverage) (assume every pair v2 accepted is true), is ≥ 0.80. The F0.5 break-even for typical set
  sizes is 0.73–0.77. That leaves 50,723 pairs (France 41,752: mostly invented-name/same-address, same name + same
  number, no-space names; US 1,784; India 7,187). Rules where v2's coverage ≈ the train rate (most US/India rules) are
  **not** forced, because the misses there may be the true negatives.
- Expected LB: adds +0.20 pt with the worst-case bounds (+0.6 at train rates), one-to-one +0.1 to +0.2.
  Validator PASS (`--check-ids`); 0 multi-owner records left.

## 6. Recommendations (QUEUE)

1. Global one-to-one in the pipeline (correctness fix, keep regardless of slice Q; it cannot show on 1–2-block slices).
2. France fixes (normalisation) → v3. Under the new normaliser the trade-name pairs become exact-address pairs, so the
   model can see them. Check v3's coverage of `disjoint|a_eq|invented` in France (target ≥ 85%, v2 34%).
3. `src/hq_keys.py` wiring: (a) union the sure key pairs into the post-cascade candidates (bypassing the top-N cap)
   in both the train and test passes, so the model learns and scores them; (b) optionally `force_mask` (bound ≥ 0.80)
   after `decide`. Calibrate on the full-density train pass with `kq_ctx` = all S1 of the country.
4. India recall: the largest remaining loss (≈0.35 pairs/S1). Blocking k and cascade cap for India, name_ph view.
   Evaluate at full density, where the train pass's per-country pair recall after the cascade is the metric.

## 7. Result of the box's check (QUEUE 2b, runs/day/keys_check.md, 12:20)

For the 130,678 sure key pairs v2 missed: was the pair in v2's candidate set (the model rejected it) or cut earlier
by blocking/the cascade?

| country | missed | cut before the model | model-rejected |
|---|---|---|---|
| France | 56,600 | **71%** | 29% |
| India | 49,283 | 48% | 52% |
| US | 24,795 | 16% | **84%** |

By rule: France `disjoint|a_eq|invented` 76% cut, `core_eq|num_eq` 84% cut, `nsp_eq|num_eq` 99.9% cut; India
`nsp_eq|num_eq` 99% cut, `core_eq|c_empty` 94% cut, `core_eq|num_eq` 44% cut (90% cut when v2 gave the record to
another S1). US misses are almost all model rejections (`swap1|a_eq`: 98–99% rejected).

Reading: France's and much of India's misses are **candidate-generation** losses. The model never saw those pairs,
so their precision is the calibrated rule rate (~0.96), not lower. The US misses are the model's own judgement and
may well be the rule's ~3–5% negatives. Consequences (26 Sep 13:40): key pairs and reverse blocking enter as
post-cascade candidates (Jarvis lane item 1b; box QUEUE item 4 moved there). No forcing for US rules. An ens2c
variant restricted to the CUT pairs would add France ~40K + India ~24K + US ~4K pairs, estimated +0.45–0.5 pt with
the one-to-one fix, still below the 1-pt upload bar.

## 8. Atharv's file (LB 0.944) and a 2-of-3 vote (26 Sep 13:30)

| country | Atharv = Soha (identical sets) | Atharv = v2 | Soha = v2 | vote = Soha | vote vs v2: pairs removed / added per S1 |
|---|---|---|---|---|---|
| US | 91.5% | 78.6% | 78.7% | 95.6% | −0.141 / +0.077 |
| India | 84.9% | 65.6% | 65.7% | 91.9% | −0.211 / +0.186 |
| France | 77.9% | 56.2% | 56.3% | 88.2% | −0.267 / +0.243 |
| all | 86.4% | 69.2% | 69.2% | 92.8% | −0.193 / +0.153 |

Atharv's pipeline is close to Soha's (pseudo-F0.5 of Atharv against Soha 0.968, against v2 0.919), so it's a weak
third voter. A 2-of-3 vote is Soha's file with small v2 tie-breaks (vs Soha −0.037 / +0.044 pairs per S1). Expected LB
≈ Soha's 0.948, and the ens1 result says the v2-only pairs it would drop are ~73% true. Not uploaded (1-pt bar). Voting
between these three files is closed. Gains have to come from the shared misses (candidate generation, India recall,
France), which all three pipelines have.

## 9. Exact phonetic-key join (follow-up to box QUEUE 5a, 26 Sep 14:00)

5a found that 32% of India's full-density blocking misses have an identical phonetic key (`src.translit.phonetic_key`:
'shakti kansaltantsa' and 'shakti consultants' both give 'skt knsltnts'). Full-train calibration of pairs with the
SAME phonetic key but a different core / no-space name (2.37M pairs). Context: ph_other = another S1 has the same
phonetic key; addr_other = another S1 has the candidate's exact address.

| address relation | US P(match) | India P(match) |
|---|---|---|
| exact address | 0.91–0.92 | 0.96 (ph unique) / 0.90 (shared) |
| same first number, street text differs | 0.90–0.91 | 0.89 (ph unique) / 0.40 (shared) |
| same street, number missing | 1.00 | 0.44 |
| same street, different number | 0.34 | 0.41–0.54 |
| other address, ph unique | 0.13 | 0.21 |
| other address, ph shared | ≤ 0.005 | ≤ 0.003 |
| candidate address empty, ph unique / shared | 0.70 / 0.04 | 0.54 / 0.01 |

Verdict: not precise enough for 'sure' rules (≥ 0.96). As extra **model candidates** it is cheap: excluding
(other address | empty address) × shared phonetic key leaves 0.07–0.12 pairs/S1, ~50–60% of them true on train,
≈ 0.049 true pairs/S1 in India. v2 predicts only 46% of the India same-number bucket and 12% of the French one.
Expected value after name_ph comes back as a view (Jarvis item 1): small, ≈ +0.1–0.2 pt. It is parked until Jarvis
reports how many India misses with identical phonetic keys remain after item 1. Wiring it would need a separate
injection threshold in the box's `--key-rules` code (inject at rule P ≥ ~0.4 with key_p as the feature, keep
key_sure at 0.96). Data: HQ sandbox ens/ph_train_pairs.pkl, ens/ph_test_pairs.pkl.
