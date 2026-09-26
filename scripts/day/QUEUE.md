# Experiment queue — top = next. Humans/HQ may edit (push from the laptop). The loop marks items DONE/FAILED.

HQ note (26 Sep 12:00): read docs/ENSEMBLE_AND_KEYS.md first. Two facts change how to evaluate. (1) The 8% slices hold
~12x fewer same-name decoys than the full data. Blocking, cascade, one-to-one and key-rule changes can look neutral on
the slice Q and still matter at full density. For those items the metric is the FULL-DENSITY train pass (150K train S1
against all docs of the country, OOF F0.5 per country + pair recall after the cascade), plus test statistics per
country. (2) Train GT has 3.46 true matches per S1 in both US and India. v2 predicts 3.35 (US), **3.12 (India)**,
3.30 (France).

1. [P0] FAST LANE + correctness gate (see DAY_TASK.md). Nothing else until it passes.
2. [P0.5, correctness — KEEP without the Q rule] GLOBAL one-to-one. v2 applies one-to-one per 100K-S1 block only. In
   the v2 submission, 13,806 S2/S3 ids are assigned to 2–5 S1s (28,302 pair slots, ≥ 14.5K guaranteed false positives;
   France 48%). The OOF never showed it because the train pass is one block per country. Fix: in the test pass, buffer
   each country's decided pairs WITH p across all blocks, then `src/hq_keys.py: global_one_to_one(q_rid, c_rid, p)`
   (keep the max-p S1 per record) before writing matching_results.tsv (candidate_pairs.tsv unchanged). Verify on the
   fast lane (v2 rescored → 0 multi-owner ids; report pairs removed per country) and use it in every later file.
2b. [Evidence, cheap, ungated, ~10 min of light work; do it while a heavy job runs] Why does v2 miss the 'sure' key
   pairs? HQ exported them (computed with the NEW normaliser over all test records):
   `scripts/hq_ens/sure_pairs_missed_by_v2.tsv.gz` (130,678 pairs: source1_entity_id, entity_id, country, rule, flags).
   For each pair, look up output_v2/candidate_pairs.tsv and tag it: in v2's candidate set (the model rejected it) or
   not (blocking/cascade cut it). Write counts by rule and country to `runs/day/keys_check.md`. HQ decides from this
   which rules to force: cut pairs are ~96% true at the train rates, model-rejected ones may be the real negatives.
   The key rows: India `core_eq|num_eq` (29.8K pairs, train rate 0.958) and France `disjoint|a_eq|invented` (24.8K).
3. [France, HQ-DONE diagnosis → APPLY + EVALUATE] HQ's France fixes are in `src/normalize.py` + `src/features.py`
   (commit 7defba7 "france: normalisation fixes"; write-up `docs/FRANCE_FIXES.md`). Bugs fixed: "N° 32" normalised
   to "north 32" (5% of French addresses); zero-padded house numbers (also 3% of US / 5% of India S2/S3); région vs
   département never agreeing; St-Nazaire → "street nazaire"; legal form lost when it leads the name; EI not a legal
   form; French filler words (groupe/fils/associés/développement/…) counted as content words in extra_*_content. On
   French test pseudo-pairs, same-name pairs with identical normalised address go from 14% to 33%. Steps:
   a) NORMALISATION CHANGED ⇒ rebuild the per-country stores (new cache dirs; never delete v2's) for the slices and
      the full data. Candidate sets built before stay valid.
   b) Slice Q (expect ≥ Q0; the leading-zero fix may help US/India a little) AND the full-density train pass (OOF per
      country must not drop).
   c) Full data via the fast lane + global one-to-one (item 2) → validator → `submissions/v3_matching_results.tsv` →
      `runs/v3/NOTES.md`. Include a France table: empty rate (v2 5.2%), mean matches (v2 3.30), and **coverage of the
      French 'sure' key pairs by rule** (`src/hq_keys.py`: key_pairs + apply_rules, p_min 0.96; v2 covers only 34% of
      `disjoint|a_eq|invented` in France vs 92% in US; target ≥ 85%). SUBMIT-READY if the train-pass OOF holds; HQ
      decides the upload.
   d) Save per-pair test probabilities (P0-c).
4. [Keys — GATED: HQ writes GO or NO-GO on this line after the LB of submissions LOG #3 (ens2c), ~13:00] Wire
   `src/hq_keys.py` (self-contained; `python -m src.hq_keys` self-test). Per country: `record_keys` for ALL S1 (kq_ctx)
   and all docs; per block `key_pairs(kq_block, kd, kq_ctx=kq_all)`.
   a) TRAIN pass: `calibrate(P, y)` per country → store the rule table in model.joblib. Union the pairs of rules with
      P ≥ 0.96 (`apply_rules`) into the post-cascade candidates. They bypass `--cascade-top` (France S1s all sit at the
      cap of 10) and are deduplicated with the existing ones. Then stage B and the model train on them as usual.
   b) TEST pass: same union (an unseen country gets the MINIMUM rate over the train countries: `rules_for_country`), so
      they reach stage B, the model and candidate_pairs.tsv. After `decide`: `force_mask(..., bound_min=0.80)` adds sure
      pairs the model still rejects, but only in rules where the model's coverage is far below the calibrated rate.
      Then run global one-to-one.
   c) Evaluate: full-density train-pass OOF per country, test coverage of sure pairs per rule and country, mean
      matches/S1 per country, candidates/S1 (expect +0.1 to +0.3/S1). Full-data file → SUBMIT-READY.
5. [India recall — moved up; the largest measured loss] India: 3.12 predicted matches/S1 vs 3.465 in the train GT (US
   3.35 vs 3.459). That is ≈ 0.3 true pairs/S1 missing on 47% of test. Exact keys don't fix it: India's key-rule
   coverage is already 93–99%, so the loss is fuzzy (Indic scripts, landmark addresses, heavy reordering). Full-density
   blocking recall is 0.933 at k=10 (US 0.984), and the cascade loses more.
   a) Decompose India's missed GT pairs on the full-density train pass: blocking miss / cascade cut / model FN / model
      FP, by category (Indic-script name, empty candidate address, name-token Jaccard bins, address Jaccard bins,
      transliterated legal forms). Write `runs/day/india_recall.md`.
   b) Attack the biggest bucket: India-only k 10 → 20 on the name views (GPU top-k is cheap), the `name_ph` view back
      for India only, and/or India cascade_top 12. Metric: India pair recall after the cascade + India OOF F0.5 on the
      full-density train pass. Promote at India OOF +0.5 pt. Keep final cands/S1 ≤ 12 and report it.
6. [France] Country-neutral model: adversarial validation (classifier France-vs-US/India pairs on the pair features),
   drop or re-normalise the most country-shifted features (e.g. rank/percentile within country instead of raw
   counts); evaluate Q. Also: 36% of French S2/S3 addresses have no region/département while S1 always has one
   (US 4.5%, India 14%) — consider imputing the region code from the city using S1 itself (no external data).
7. [Accuracy, cheap] Model capacity on the fast lane: more trees/leaves, train S1 150K → 300K, 3-seed average.
8. [Efficiency] cascade_top 10 → 8 only if the full-density pair recall after the cascade holds PER COUNTRY. Every
   French S1 already sits at the cap, so consider a per-country cap before any global cut.
9. [Speed] Profile the full-data run by stage; vectorise the slowest stage-B features.
