# CONTEXT.md — team ground truth (update after every session / check-in)

## Problem (PS video + official README in the zip, 25 Sep)
- Task: **Business entity resolution** across 3 sources, no shared ID. S1 = deduplicated
  reference; S2, S3 = vendor sources. For every S1 entity output ALL matching S2/S3 ids.
- Columns (all sources): `entity_id` (prefix S1-/S2-/S3-), `business_name`,
  `business_address`, `country`. **Train countries: US, India. Test adds France (unseen,
  259K of 1.73M test S1 = 15%)** -> nothing may be hard-coded to {US, India}.
- Sizes: train S1 2,206,821 (US 1.32M, India 0.88M); test S1 1,732,544 (India 810K, US 663K,
  France 259K). S2/S3 files ~500 MB each (~5M rows each, TBC) -> ~24M records total.
- **Ground truth (train): singletons only 5.6%; mean 3.46 matches per S1 (max 11);
  every S2/S3 id appears under at most ONE S1 (1-to-1 holds).** S2 per S1: 0:13% 1:36% 2:30%
  3:15% 4+:6%; S3 similar. -> S2/S3 hold several records per entity; recall matters much more
  than the video suggested (all-empty scores 0.056).
- Metric: **macro F0.5** per S1 entity, averaged over all S1 (singletons: empty = 1.0, any
  prediction = 0). Public LB = a subset of test; **private LB = the remaining part of test;
  final ranking = private LB**.
- Files: TSV `sep="\t"` (addresses contain commas). Submission `matching_results.tsv`:
  columns `source1_entity_id`, `matched_entity_ids` (comma list, no quoting, empty for
  singletons, one row per test S1, S2/S3 ids only, no duplicates).
- Final zip: `output/matching_results.tsv` + `output/candidate_pairs.tsv` (columns
  `source1_entity_id`, `candidate_entity_ids`; = the EXACT set fed to the model at
  inference, matches must be a subset) + `code/business_entity_resolution/{src,README.md,
  requirements.txt}` + filled `Documentation_template.md`.
- Validate before every upload: `python utils/validate_submission.py --matching ... --candidate ... --test-dir dataset/test`.
- Rules: no external data / APIs / geocoding (disqualification). **Model must be MIT/Apache-2.0
  licensed and <= 8B params** (MiniLM = Apache-2.0, bge-small = MIT: OK).
- Noise (README): abbreviations, legal suffixes, DBA names, & vs and, word-order swaps, typos,
  transliterations; addresses: abbreviations, missing PIN/state, landmark refs, municipal
  numbering, **component reordering** (seen: region listed first in France/US rows).
- Data location: `data/data_extracted/student_resource/` (zip 1.09 GB, gitignored). Box copy:
  `unzip -o <zip> -d ~/amazon-ml-2026/data/`. Files >400 MB are split with
  `scripts/split_tsv.py` only for the laptop->Claude transfer.
- Only the TEAM LEADER's Unstop account can open the round: dataset + every upload go through
  the leader. **Leader: Soha Chand** (has dataset + upload access). Box account: Kartik Agrawal.
- **Official update (25 Sep evening):** `candidate_pairs.tsv` is reviewed in the final evaluation —
  a SMALLER candidate set per S1 ranks higher (beyond the public/private LB); blocking must scale
  (no all-pairs); the code that produces it is reviewed. Every run report must show mean/median
  candidates per S1; the dense view must re-rank/replace candidates, never add to them.
- Submission flow: log row in `submissions/LOG.md` -> push file to `submissions/` ->
  leader pulls + validates + uploads -> leader reports LB score -> fill LOG.md.

## Current state (26 Sep ~10:40 IST)
- **Public LB: v2 = 0.947** (26 Sep 09:45, uploaded by Soha). Soha's own pipeline 0.948. Target 98.4+.
  v2 = commit 48bc4e7, full data: 4 TF-IDF views (name_c3,name_w,addr_c3,full_w), GPU top-k, cascade 9.2 cands/S1,
  LightGBM on 150K train S1, thr 0.70 + 1-to-1; OOF 0.9623 -> LB gap -1.5. Runtime 4.9 h, peak RSS 7.3 GB.
  Test: predicted-empty 6.0% (FR 5.2 / IN 6.7 / US 5.5), mean matches 3.23. Files: output_v2/, runs/v2/NOTES.md.
- Cross-country proxy (slice, hidden holdout): train US+IN 0.9805 (IN 0.9717 / US 0.9863); train US only -> India
  0.9339 (-3.8); train India only -> US 0.9810. India full-data blocking recall 0.933 (US 0.984) -> India = 47% of
  test is the biggest known loss; France (15%, unseen) the biggest unknown.
- Best local CV on the 8% slice: 0.9768 OOF / 0.9826 holdout (runs/slice_v3, 6 views, HistGB, 40K S1).
- **Box now runs Claude Code unattended ("day loop", scripts/day/DAY_TASK.md + QUEUE.md, until 27 Sep 20:00 IST)**:
  objective Q = 0.5*slice_mix + 0.3*xc_us + 0.2*xc_in (Q0 = 0.9666); P0 = fast lane (--reuse-candidates: full-data
  model in ~1 h). It never submits; SUBMIT-READY files appear in runs/day/SCOREBOARD.md. HQ steers via QUEUE.md.
- **France fixes (HQ, 26 Sep, docs/FRANCE_FIXES.md)** in src/normalize.py + src/features.py, queued as QUEUE item 2:
  "N° 32" was normalised to "north 32", zero-padded house numbers (also 3% US / 5% IN), region vs departement,
  St-Nazaire -> "street", bis/ter, legal form leading the name, EI, et/&, domain names with glued legal forms,
  @handles, French filler words in the generic-token set. French pseudo-pair address agreement 14% -> 33%.
- Submissions used: Day1 0/5 · **Day2 1/5** · Day3 0/5 (see submissions/LOG.md)

## Decisions
- 25 Sep — Pipeline-first: blocking -> candidate_pairs -> pairwise model -> per-entity
  selection -> matching_results; refine only after a validated submission exists.
- 25 Sep — AWS free tier not used for now (free SageMaker hours are CPU-only).
- 25 Sep — Box GPU is UP (admin fixed the driver). Plan: lexical TF-IDF blocking per country
  + exact dense top-k on GPU (MiniLM) as a second view; GBDT matcher; no LLMs.

## Tried & failed (don't repeat these)
- Post-hoc decision machinery on OOF probabilities: expected-F0.5 set selection, isotonic calibration, per-country
  thresholds, 2nd-stage group model, 2nd-stage candidate-candidate model -> no gain (scripts/tune_rules.py).
- Key blocking + dense MiniLM as a REPLACEMENT for TF-IDF (src/blocking_b.py): India pair recall 0.855 vs 0.966,
  US 0.948 vs 0.994; encoder alone ~2 h for 24M records. Exact-key views as an ADDITION: +1.0 pt India / +0.1 US
  at +4 cands/S1 (not yet applied).
- 6 TF-IDF views on the full data: too slow (~9 h); name_ph + addr_w dropped (base+full_w keeps 99.3%/99.9% of
  the 6-view recall). k=15 India: +0.4 pt recall for +55% candidates (parked).
- Concurrent heavy jobs on the box (v2 killed once by the memory guard; pandas-3 Arrow `rid` indexing stalled the
  test pass -> .to_numpy(dtype=object) fix, commit ea7375a).

## Open / next up
1. Box P0 fast lane -> QUEUE 2 France fixes (rebuild stores, Q check, full-data v3 via fast lane, France tables in
   runs/v3/NOTES.md) -> HQ decides upload v3 vs v2 (log in submissions/LOG.md first).
2. Ensemble with Soha's pipeline: put her matching_results.tsv in submissions/soha_matching_results.tsv; per-country
   agreement; intersection/union/vote evaluated on a train-derived holdout.
3. India blocking recall 0.93 -> 0.97 keeping <= 10 cands/S1 after the cascade (exact-key views, name_ph, k=15 India).
4. Country-neutral features / adversarial validation (QUEUE 3); region imputation from city for France.
5. Day 3: Documentation_template.md numbers, scripts/make_package.py, final robust upload; runs/day/FINAL.md.
- Real-data facts (slice EDA): ~24% of Indian S2 names are in Indic scripts (handled: rule-based
  transliteration + phonetic key, src/translit.py); ALL-CAPS, "null" tokens, leet typos (Precisi0n),
  domain names as names, DBA names, state names vs codes (IL/Illinois, MH/Maharashtra), house-number
  labels (H No / Door No / Plot No), 3.8% empty addresses in S2/S3, zero-padded house numbers, @handles.
  All GT matches are same-country. "Different name at the same exact address" is a MATCH 98.6% of the time.

## Who's on what
- Gaurav: git bridge laptop<->GitHub<->box, leader liaison, HQ (Claude) steering.
- Soha Chand (leader): uploads, own pipeline (LB 0.948).
- Kartik Agrawal: box account.
- Member 2:
- Member 3:
- Member 4:
