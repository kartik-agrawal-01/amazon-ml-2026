# Jarvis queue — top = next. HQ (modelling chat) owns this file; the Jarvis loop only reads it (from main).
# HQ version 26 Sep 17:45 — NEW PLAN "S2F" = Soha-2 + France fixes, overnight on this A30 (24 GB GPU, 16 vCPU, ~503 GB RAM).
# Everything below replaces the 16:30 version (jv1 / cross-encoder / judge items are superseded).

## Situation (read first, 5 min)
- Public LB: **Soha-2 = 0.980795**, the team's best (submissions/LOG.md #4; file submissions/soha_matching_results_2.tsv,
  md5 7b184487). Our own pipeline (v2 0.947, jv1) is superseded. jv1 was STOPPED ~16:45 (instance paused): do NOT
  resume the jv1 / chain / ce tmux sessions or their caches' jobs.
- Soha-2's method (from Soha): (1) blocking + TF-IDF retrieval + a fine-tuned multilingual-e5-small bi-encoder (its
  nearest neighbours add 34M train / 27M test candidate pairs; cosine + rank are features); (2) a fine-tuned e5-small
  cross-encoder (1M pairs, trained on a disjoint half of the S1 entities) whose probability feeds a stage-2 LightGBM
  with competition-context features (validation F0.5 0.968 → 0.986); (3) a stage-3 "sibling" model (similarity of a
  pair to the S1's other confident matches; validation 0.9865); then threshold 0.80 and one-to-one.
- Where the remaining loss is: LB 0.981 vs validation 0.9865 on US/India ⇒ **France ≈ 0.95**. Soha-2 finds only 30% of
  the French "different invented name at the SAME exact address, no other S1 with that name or that address" pairs
  (95% in US/India; 96–98% true on the full train), predicts 3.21 matches/S1 in France (US 3.36, India 3.34) and 6.1%
  empty (US/India 5.9%). Her 0.80 threshold was tuned on US/India, and a country the model never saw is less
  confident (box finding, runs/day/LOG.md 15:13).
- **Target: France ≥ 0.97 → LB ≈ 0.984+.** A validated file by 27 Sep ~08:00 IST (Day 3 has 5 uploads; lanes freeze
  27 Sep 20:00).

## Inputs (Gaurav / infra provide them; item 0 waits for them)
- Soha's code in the repo under `soha/` (with a README: how to run each stage). Keep her files unchanged: put every
  change in NEW files (`soha/s2f_*.py`) or on the jarvis branch.
- Her trained artifacts copied to `/home/soha/` on this A30: bi-encoder, cross-encoder, stage-2 and stage-3 models,
  plus every cached table she has (candidate pairs per split, stage-1 scores, cross-encoder probabilities for train and
  test, stage-2 feature tables). Instance-to-instance copy or the JarvisLabs shared filesystem; NOT via git.

## Items
0. [Reproduce first — gate for everything below] Write `runs/jarvis/soha_map.md`: per stage the script, inputs,
   outputs, what is cached, and the time to recompute on this A30. Then reproduce Soha-2's test output from her
   artifacts WITHOUT changes: ≥ 99.9% identical S1 sets vs submissions/soha_matching_results_2.tsv and her validation
   F0.5 (0.9865). If a stage must be recomputed (e.g. cross-encoder test scores), estimate the time first; if it's
   > 5 h, report it in LOG.md and work from the cached parts only.
1. [S2F-1: exact-key rule features in stage 2 — the main item] `src/hq_keys.py` on main (self-test:
   `python -m src.hq_keys`):
   a) keys for ALL records of each split: `record_keys` on the normalised core name and address (src/normalize.py:
      `split_legal(unleet(fold(name)), compact)` for the core name, `norm_addr` for the address — exactly as in
      `src/hq_keyfill.py: _keys_chunk`); `key_pairs` per country with `kq_ctx` = ALL S1 of the country (train: all 2.2M
      train S1, never a sample — the context flags need full density);
   b) `calibrate` P(match) per (rule, country) on the train S1 half NOT used to fit stage 2 (her cross-encoder half),
      so there is no leakage; France = minimum over US/India (`rules_for_country`);
   c) features per (s1, cand) pair, joined onto her stage-2 train / validation / test tables: key_rule (categorical
      code, 0 = none), key_p (calibrated P, 0 if no rule), ctx_core_other, ctx_addr_other, name-relation code (ncat),
      address-relation code (acat). Pairs without an exact key get 0;
   d) retrain stage 2 with her params and split, then stage 3 on the new stage-2 outputs. **Gate: validation F0.5 ≥
      0.9865 − 0.0005 overall AND per country (US, India).** Report per-country validation F before/after and the
      importance of the new features;
   e) test: rescore → per-country matches/S1, empty rate, coverage of the 'sure' key pairs by rule (France
      `disjoint|a_eq|invented` target ≥ 85%, Soha-2 30%), records under 2+ S1 = 0.
   Evidence it helps: on the box's unseen-country screen (US-trained → India) the same features gave +0.48 pt (n3k).
2. [S2F-2: France threshold — cheap, right after item 1] Pick t_FR label-free. Report France matches/S1, empty rate
   and sure-key coverage for t = 0.50, 0.55, …, 0.80. Rule: if `runs/day/unseen_thr.md` (box QUEUE 2d) has landed, use
   the rule it validates; otherwise count matching: t_FR = the threshold at which France's predicted matches/S1 equals
   the US/India average on the same run (≈ 3.35 for Soha-2), bounded to [0.50, 0.80]. US/India keep 0.80. One-to-one
   after. Write both files (item 1 alone, item 1 + t_FR).
3. [S2F-3: France normalisation fixes — only if items 1–2 are done and ≥ 4 h remain before 06:00] If her pipeline
   normalises names/addresses with its own code for TF-IDF / exact-match features, apply src/normalize.py (France
   fixes: "N° 32" → "32", zero-padded numbers, région / département → one code, St → saint, leading legal forms, French
   filler words) to the FRENCH records only (US/India inputs unchanged ⇒ no retraining), recompute France's lexical
   features and rescore France. The neural parts (bi-encoder, cross-encoder) stay as they are.
4. [Output, for every file] Validator: `python data/student_resource/utils/validate_submission.py --matching …
   --candidate … --test-dir data/student_resource/dataset/test` → `submissions/s2f_<tag>_matching_results.tsv` plus its
   candidate_pairs.tsv (exactly the pairs fed to the final model; report mean candidates/S1 — smaller ranks higher) →
   `runs/jarvis/S2F_NOTES.md` (validation F per country before/after, France stats, what changed) → SCOREBOARD row
   "SUBMIT-READY". HQ decides uploads.
- Billing: the A30 bills per hour. If nothing runs for > 20 min, write it in LOG.md so the humans can pause it.
