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

## Current state
- Best local CV: **0.9768 OOF macro-F0.5** (40K-S1 train slice, 3-fold, sklearn HistGB, thr=0.80 + 1-to-1);
  **0.9826 on a disjoint 30K-S1 train-derived holdout** (runs/slice_v3; v1 was 0.966/0.974, v2 0.970/0.977).
  Gains came from: word views over name+address (blocking recall 97.1% -> 98.6%), global name/address
  uniqueness counts (how many S1 share a candidate's name -> resolves empty-address / trade-name records),
  extra-token and house-number-detail features. Post-hoc rules (calibration, 2nd-stage group model,
  per-country thresholds) gave NOTHING (tried, see scripts/tune_rules.py). Full-data box run pending.
- Best public LB:  — (nothing yet)
- Submissions used: Day1 0/5 · Day2 0/5 · Day3 0/5

## Decisions
- 25 Sep — Pipeline-first: blocking -> candidate_pairs -> pairwise model -> per-entity
  selection -> matching_results; refine only after a validated submission exists.
- 25 Sep — AWS free tier not used for now (free SageMaker hours are CPU-only).
- 25 Sep — Box GPU is UP (admin fixed the driver). Plan: lexical TF-IDF blocking per country
  + exact dense top-k on GPU (MiniLM) as a second view; GBDT matcher; no LLMs.

## Tried & failed (don't repeat these)
- 

## Open / next up
- **Box run v1** (someone with RDP access, see docs/BOX_RUNBOOK.md): `git pull` -> `bash scripts/gpu_check.sh`
  -> unzip data -> `bash scripts/run_pipeline.sh v1 --max-df 0.01 --train-s1 300000 --block-size 100000`
  (add `--dense` once the GPU works) -> validate -> `submissions/v1_matching_results.tsv` -> LOG.md -> leader uploads.
- Real-data facts (slice EDA): ~24% of Indian S2 names are in Indic scripts (handled: rule-based
  transliteration + phonetic key, src/translit.py); ALL-CAPS, "null" tokens, leet typos (Precisi0n),
  domain names as names, DBA names, state names vs codes (IL/Illinois, MH/Maharashtra), house-number
  labels (H No / Door No / Plot No), 3.8% empty addresses in S2/S3. All GT matches are same-country.
- Blocking (lexical, 4 views, k=10, max_df=0.01): pair recall 97.1% (US 98.3%, India 95.3%) at ~62 cands/S1
  -> ~52 after pruning. Recall ceiling for India is the first thing to raise (dense view / better translit).
- Then: dense GPU view, bigger train sample + LightGBM, France sanity check (unseen country), 2nd-stage.

## Who's on what
- Gaurav:
- Member 2:
- Member 3:
- Member 4:
