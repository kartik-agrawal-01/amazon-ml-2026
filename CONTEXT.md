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

## Current state (26 Sep ~12:00 IST)
- **Public LB: v2 = 0.947** (09:45) · ens1 consensus v2∩Soha = **0.9449** (11:07) · Soha's own pipeline 0.948.
  Target 98.4+. Uploads used: Day2 2/5. **Pending: LOG #3 ens2c** (v2 + global one-to-one + exact-key 'sure' pairs,
  expected +0.2 to +0.5 pt; submissions/ens2c.zip.part000-002).
- v2 = commit 48bc4e7: 4 TF-IDF views, GPU top-k, cascade top-10 (9.2 cands/S1), LightGBM on 150K train S1, thr 0.70
  + one-to-one; OOF 0.9623 (full-density train pass) -> LB gap -1.5. Test: empty 6.0% (FR 5.2 / IN 6.7 / US 5.5).
- **Where the loss is (docs/ENSEMBLE_AND_KEYS.md):** (1) the disagreement set with Soha is ~73% true pairs, so it isn't the
  problem (ens1 fell). (2) v2's one-to-one is block-local: 13,806 test records sit under 2–5 S1s (≥ 14.5K guaranteed
  FPs, 48% French). (3) France recall hole: different trade/domain names at the same exact address are 96–98% matches
  on the full train; v2 finds 92% of them in the US but 34% in France. They are cut before the model among France's
  dense same-name decoys, and the old normaliser also broke French addresses. (4) **India recall**: v2 predicts 3.12
  matches/S1 vs 3.465 in the train GT (US 3.35 vs 3.459), about 0.3 true pairs/S1 missing on 47% of test, and it is
  fuzzy (not exact-key). The box France diagnostic (runs/france_diag) found France's accepted-pair profile US-like;
  that is consistent, since its FNs (trade names) and FPs (look-alikes, duplicate owners) cancel in the counts.
- Cross-country proxy (slice, hidden holdout): train US+IN 0.9805 (IN 0.9717 / US 0.9863); train US only -> India
  0.9339; train India only -> US 0.9810. **Slices hold ~12x fewer decoys than the full data**: blocking/cascade/
  one-to-one/key-rule changes must be judged on the full-density train pass, not the slice Q alone.
- Best local CV on the 8% slice: 0.9768 OOF / 0.9826 holdout (runs/slice_v3); France fixes on the slice (US/India
  regression check): 0.9777 / 0.9825 (runs/slice_frnorm).
- **Box = Claude Code day loop** (scripts/day/DAY_TASK.md, QUEUE.md, runs/day/) until 27 Sep 20:00 IST; it never
  submits. Order now: P0 fast lane → global one-to-one → France fixes → v3 → hq_keys wiring (gated on the ens2c LB) →
  India recall.

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
- **Consensus/intersection with Soha's file (ens1, LB 0.9449 vs 0.947):** the pairs only v2 predicts are ~71–75% true
  (F0.5 break-even ≈ 70%). Stricter thresholds / intersections won't help; a plain union is expected to be ~neutral.
- Calibrating exact-key/context rules on the 8% slice: ff-rule rates drop to 0.86 vs 0.96 on the full train (decoy S1s
  missing from the slice). Always calibrate context rules at full density.

## Open / next up
1. Upload LOG #3 ens2c → fill its LB. If ≥ +0.15: write GO on QUEUE item 4 (hq_keys wiring); else NO-GO.
2. Box: P0 fast lane → global one-to-one (QUEUE 2) → France fixes + v3 (QUEUE 3) → hq_keys (QUEUE 4) → India recall
   (QUEUE 5, the largest measured loss).
3. Day 3: final choice by robustness (private LB = the rest of test), Documentation_template.md numbers,
   scripts/make_package.py (candidate_pairs.tsv must contain every submitted match; the key pairs are part of the
   candidate set when wired in the pipeline), final upload.
- Real-data facts: ~24% of Indian S2 names in Indic scripts (rule-based transliteration + phonetic key); ALL-CAPS,
  "null" tokens, leet typos, domain names, @handles, DBA names, state names vs codes, house-number labels, zero-padded
  numbers, 3.8% empty addresses in S2/S3. All GT matches are same-country; GT strictly one-to-one. France: 13
  régions/96 départements (S1 always région), N°/Nº labels, bis/ter, dense same-name decoys (~16 per S1).

## Who's on what (one writer per file)
- HQ (modelling chat, Claude): QUEUE.md, CONTEXT.md, submissions/LOG.md, docs/, CLAUDE.md, project doc
  claude/competition-state.md, every upload decision. New code only in NEW files (src/hq_*.py, scripts/hq_*/,
  patches/) + a QUEUE item while the loop runs.
- Infra chat: scripts/day/day_loop.sh + DAY_TASK.md, scripts/night/, guards, crontab, transfers, claude/infra-status.md.
- Day loop (box Claude Code): src/, runs/day/; in QUEUE.md only DONE/FAILED marks.
- Gaurav: git bridge laptop<->GitHub<->box, leader liaison. Soha Chand (leader): uploads, own pipeline (LB 0.948).
  Kartik Agrawal: box account.
