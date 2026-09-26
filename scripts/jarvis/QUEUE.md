# Jarvis queue — top = next. HQ (modelling chat) owns this file; the Jarvis loop only reads it (from main).
# HQ version 26 Sep 13:55 (13:40 version + the box's main features; the infra draft of 13:15 is folded in). Machine: A30 (24 GB GPU, 16 vCPU, ~112 GB RAM);
# it can be resumed on an RTX PRO 6000 (96 GB) — items marked [RTX] only after that switch.

## Read first (5 min)
- docs/ENSEMBLE_AND_KEYS.md, runs/day/keys_check.md, CONTEXT.md. Where the loss is:
  (1) CANDIDATE GENERATION at full density. v2 pair recall is 0.953 after the cascade (India blocking 0.933, US
      0.984). Of the high-precision exact-key pairs v2 missed, the share that blocking/cascade cut before the model
      ever saw them is France 71%, India 48%, US 16% (runs/day/keys_check.md). France S1s all sit at the cascade cap
      of 10 among ~16 same-name decoys.
  (2) India recall. v2 predicts 3.12 matches/S1 vs 3.465 in the train GT (US 3.35 vs 3.46), about 0.3 true pairs/S1
      missing on 47% of test.
  (3) v2's one-to-one was block-local (13.8K records under 2–5 S1s). `src/hq_keys.global_one_to_one` fixes it.
- Upload policy: HQ uploads only files expected to beat the best LB (0.947) by ≥ 1 pt. So build ONE strong bundle
  per item, not many small variants.

## Division of labour (updated 26 Sep 13:55 — READ THIS)
- While git was broken on the box (12:35–13:45), the box implemented the item-1 candidate features on MAIN in
  src/pipeline.py: exact-key candidates `--key-rules 0.96` (5b8720f; per-rule coverage table + key_coverage_<c>.csv),
  reverse blocking `--reverse-k 3 --reverse-bypass 2` (628591f; rev_rank/rev_n/rev_best/rev_sure features, rev_sure
  bypasses the cascade cap; exact vs brute force in scripts/day/test_reverse.py), post-cascade candidate augmentation
  (f4fd6f1), global one-to-one ON by default (`--no-global-o2o` = v2 behaviour), fast-lane caches (`--cand-cache`,
  `--vec-cache`, `--feat-cache`, `--save-probs`) and GPU power-safe env vars (AML_GPU_HOST_DENSIFY, AML_GPU_DUTY; the box
  needs them, you don't).
- So: `git merge origin/main` and USE THOSE FLAGS. Do NOT reimplement keys/reverse/augmentation/global o2o. If
  session 1 started doing that in src/pipeline_jv.py, drop it (keep the file only for things main lacks).
- The box OWNS src/pipeline.py and the files it edits (features/normalize/store). It only slice-screens these
  features. Jarvis owns the FULL-DENSITY runs and the full-data files. If a main feature breaks at full density, make
  the smallest fix on the jarvis branch (one commit per fix, message "fix(main): ..."), describe it in
  `runs/jarvis/ISSUES.md` (file, symptom, diff summary) and carry on. The box ports the fix to main.
- New evidence for India (box QUEUE 5a, runs/day/india_recall.md): India's full-density blocking recall is 0.933.
  55% of India's misses are Indic-transliterated candidate names (recall 0.795; 'kansaltantsa' = consultants,
  'bildarsa' = builders), and 32% of misses have an IDENTICAL phonetic key. v2 had dropped the name_ph view, so
  name_ph is essential here.
0. [Smoke] runs/jarvis/env.txt shows the GPU. Pipeline smoke (60K test S1) with --n-jobs nproc-2,
   --stage-b-jobs nproc/2, --topk-device cuda. Record runtime per stage and project the full-data runtime for
   4 vs 6 views and k = 10 vs 15.
1. [A, full-strength candidates + more data → jv1] main's src/pipeline.py at full density, for example:
   `python -m src.pipeline --data-dir data --cache-dir cache_j1 --out-dir output_jv1 --run-name jv1
    --views name_c3,name_w,addr_c3,name_ph,full_w,addr_w --k 15 --max-df 0.01 --train-s1 600000 --block-size 100000
    --key-rules 0.96 --reverse-k 3 --reverse-bypass 2 --cand-cache /home/pools/j1 --vec-cache /home/pools/j1/vecs.joblib
    --feat-cache /home/pools/j1_feats --save-probs --n-jobs $(( $(nproc) - 2 )) --stage-b-jobs $(( $(nproc) / 2 )) --topk-device cuda`
   Pass --max-df 0.01 explicitly: the default is 0.05, v2 used 0.01. If the smoke projects more than ~5 h, use k 10.
   Keep all 6 views and name_ph (India). Run the train pass first (`--skip-test`) and check the numbers below before
   the test pass. Everything (vectorisers, candidate cache, features) is cached, so the test pass or a model-only
   change reuses it.
   Report on the full-density train pass, per country:
   - pair recall after the union, after the cascade, and after the key/reverse bypass (v2: 0.953 after the cascade;
     India blocking 0.933, US 0.984);
   - GT pairs recovered by each source (keys / reverse / name_ph vs v2's 4 views);
   - OOF F0.5 (v2 0.9623 overall);
   - final cands/S1 (v2 9.2; keep ≤ 12).
   Test-side, label-free: key_coverage_<country>.csv (target France `disjoint|a_eq|invented` ≥ 90%, v2 34%), mean
   matches/S1 per country vs the train GT 3.46 (v2: US 3.35 / India 3.12 / France 3.30), 0 records under two S1s.
   The --cand-cache union is the pool for item 2 (top-40 per S1 by the cascade score).
   Gate: OOF ≥ v2 + 0.3 pt with no country down → full test → validator → `submissions/jv1_matching_results.tsv`,
   SUBMIT-READY with the tables above and an expected LB gain. Record the jarvis-branch commit.
2. [B, GPU pair model on the pool → jv2] Fine-tune a cross-encoder: start from paraphrase-multilingual-MiniLM-L12-v2
   (Apache-2.0; a bi-encoder checkpoint, so train it with a pair-classification head on (text_a, text_b));
   xlm-roberta-base or intfloat/multilingual-e5-base (MIT) only if throughput allows. Text per side
   "name | address". Positives = GT pairs, hard negatives = pool non-matches. Cap at ~3M pairs per fold, 1 epoch,
   2 folds by S1, so every train pair gets an out-of-fold score; test = mean of the 2 fold models.
   (a) Re-rank: keep the top 10 of the 40-pool by the cross-encoder (or a blend with the cascade score), plus the
       item-1 augmentation pairs → recall per country vs item 1.
   (b) Feature: ce_prob and its rank within the S1 go into the matcher → OOF.
   Time box: if fold 1 isn't trained ~3 h after starting, stop and report. KEEP if OOF +0.3 pt over jv1, or India
   recall +1 pt at ≤ 12 cands/S1 → full test → jv2 SUBMIT-READY.
3. [RTX, only if 2 KEEPs AND ≥ 8 h remain before the 27 Sep 20:00 freeze] Judge for borderline pairs:
   Qwen2.5-7B-Instruct (Apache-2.0, 7.6B params) + LoRA on pairs with matcher p in [0.3, 0.9], same 2-fold protocol.
   KEEP if OOF +0.2 pt.
