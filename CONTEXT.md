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

## Current state (26 Sep ~16:20 IST)
- **Public LB: Soha's new model (soha_matching_results_2.tsv) = 0.980795 — NEW BEST (+3.4 vs v2)** (LOG #4, ~16:00).
  Earlier: v2 = 0.947 · ens1 consensus v2∩Soha = 0.9449 · Atharv = 0.944 · Soha's first pipeline 0.948. Target 98.4+.
  Uploads: **Day2 4/5** (1 left, confirmed; expires at midnight). HQ proposes `soha2k2` for it: Soha-2 + 34.9K French
  exact-key pairs she misses (worst case +0.14, likely +0.2 to +0.3; LOG proposed row). The ≥ +1 pt bar was set when the
  best was 0.947; jv1 (our pipeline) can no longer beat the best on its own — its value is now as a blend partner.
- **Soha-2 vs v2 (label-free, scripts/hq_ens/dissect.py):** India +0.22 matches/S1 (3.34 vs 3.12), almost all fuzzy
  non-key pairs; its per-source counts match the train GT (S2=0 14.0% vs GT 13.0%, v2 17.9%). Stricter on same name +
  same street + different house number. France: 3.21 matches/S1, empty 6.1%, and still only 30% coverage of the
  different-trade-name / same-exact-address pairs (v2 34%) → soha2k2. Asked Soha for: method summary, per-pair test
  probabilities, train OOF probabilities (for measured blends, a France threshold, and packaging).
- Test has ~2x the unmatched S2/S3 records per S1 of train (S2/S1 2.8 vs 2.3 in every country): more decoys at test
  time, one reason for OOF → LB gaps; exact-key rule rates are discounted for it (augment_keys --false-mult 2).
- v2 = commit 48bc4e7: 4 TF-IDF views, GPU top-k, cascade top-10 (9.2 cands/S1), LightGBM on 150K train S1 (6.8% of
  train S1), thr 0.70 + one-to-one; OOF 0.9623 overall (60% US / 40% India; no per-country split logged) -> LB gap
  -1.5. Test: empty 6.0% (FR 5.2 / IN 6.7 / US 5.5), matches/S1 US 3.35 / IN 3.12 / FR 3.30 vs train GT 3.46.
- **Where the loss is (docs/ENSEMBLE_AND_KEYS.md):** (1) the disagreement set with Soha is ~73% true pairs (ens1 fell).
  (2) v2's one-to-one is block-local: 13,806 test records under 2–5 S1s (≥ 14.5K guaranteed FPs, 48% French); global
  one-to-one is now the default on main (+~0.13 pt, test-only: v2's OOF never had the bug). (3) France recall hole:
  different trade/domain names at the same exact address (96–98% matches on train) — v2 finds 92% in the US, 34% in
  France; cut before the model + the old normaliser broke French addresses. (4) **India recall**: 3.12 matches/S1 vs
  3.465 GT, fuzzy (Indic scripts); blocking recall 0.933 at k 10.
- **Jarvis lane (A30, branch jarvis): jv1** = main's pipeline, new normaliser (France fixes), 6 views k 15, 600K train
  S1, `--key-rules 0.96 --reverse-k 3 --reverse-bypass 2`, global one-to-one. India train phase A: pre-cascade GT
  recall **0.960** (v2 0.933): reverse top-3 +1.0 pt, name_ph +0.15, exact keys ~0 in India. Reverse-'sure' pairs
  bypass the cascade cap: 5.35 cands/S1 at 54% precision -> watch final cands/S1 (official: smaller candidate sets
  rank higher). Train pass ETA ~17:30, test pass ~21:00–22:30. **Gate changed 16:00 (density-matched):** jv1's OOF is
  also scored on v2's 150K S1 subset (one-to-one recomputed) — 600K vs 150K OOFs are not comparable. Cross-encoder
  pilot (MiniLM-L12, 2 folds) ranks well (recall@10 0.919 vs 0.863 for a crude proxy); jv2 = CE features, next.
- **Box day loop** (scripts/day/QUEUE.md): base (HEAD code, v2 stores) Q 0.9669. **n3 = France fixes: HOLD** — slice_mix
  +0.04, but the unseen-country screen xc_us (US-trained -> India) −0.71; ablation n3f (v2 generic set) −1.39 vs base,
  so the generic set is not the cause. xc_us swings ±0.7 pt under small changes, largely through the threshold picked
  on the train country's OOF (an unseen country is less confident -> needs a lower threshold). This matters for France.
  Box 2c (v2 OOF per country) and 2d (label-free threshold rules for an unseen country, scored on xc_us / xc_in) queued.
- HQ sandbox tools (scripts/hq_ens/ prototypes): `score_sub.py` (label-free sanity per country: counts vs GT, records
  under 2+ S1, coverage of the 'sure' key pairs); `augment_keys.py` (adds 'sure' key pairs a file misses where the
  worst-case precision of its misses is ≥ 0.80; on v2: +0.17 pt worst case, mostly French trade names).

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
- **Label-free LB estimate for model-vs-model files** (HQ score_sub.py: calibrated exact-key cells + assumed precision
  of the other disagreements): back-test predicted Soha −0.2/−0.4, ens1 −0.3, Atharv +0.1/−0.2 vs actual +0.10, −0.21,
  −0.30 (wrong ranking). When two models disagree on a pair from a 96–100% key cell, the pair is far less likely true
  than the cell rate. Only mechanism-based changes (pairs one file never scored, duplicate owners) are estimable
  label-free; model changes need the OOF.

## Open / next up
1. Tonight: soha2k2 in the last Day-2 slot (Gaurav/Soha decide) → its LB tells whether the France key pairs help on top
   of Soha-2 (then they go into the final).
2. Soha's method + probabilities → re-plan both lanes around her model (blend with jv1 / the cross-encoder measured on
   train OOF; France threshold via box 2d; key pairs inside her pipeline for the package).
3. jv1 runs on automatically (train gate → test pass, file ~21:00–22:00): blend partner, not an upload on its own.
4. Day 3: final choice by robustness (private LB = rest of test); Documentation_template.md; scripts/make_package.py
   (candidate_pairs.tsv = exactly the pairs fed to the model incl. any rule-added pairs; keep cands/S1 low); final upload
   before the 27 Sep 20:00 freeze.
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
