# Jarvis queue — top = next. HQ (modelling chat) owns this file; the Jarvis loop only reads it (from main).
# HQ version 26 Sep 13:40 (the infra draft of 13:15 is folded in). Machine: A30 (24 GB GPU, 16 vCPU, ~112 GB RAM);
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

## Division of labour (don't duplicate)
- BOX (day loop, ~10 GB RAM): fast lane, global one-to-one, France normalisation → v3 on v2's candidate union,
  India miss decomposition (analysis only).
- JARVIS (this lane): everything that needs RAM/GPU at full density: wider candidate generation, key and reverse
  candidates, more training data, pair models.
- Merge hygiene: the box edits src/pipeline.py, src/features.py, src/normalize.py, src/store.py. Don't edit those
  here, or `git merge origin/main` will conflict. Put orchestration in `src/pipeline_jv.py` (start it as a copy of
  main's src/pipeline.py) and new logic in `src/jv_*.py`. Re-copy pipeline.py changes you need by hand, and log it.

0. [Smoke] runs/jarvis/env.txt shows the GPU. Pipeline smoke (60K test S1) with --n-jobs nproc-2,
   --stage-b-jobs nproc/2, --topk-device cuda. Record runtime per stage and project the full-data runtime for
   4 vs 6 views and k = 10 vs 15.
1. [A, full-strength candidates + more data → jv1] HEAD of main (France fixes included) via src/pipeline_jv.py.
   a) Views: all 6 (name_c3, name_w, addr_c3, name_ph, full_w, addr_w) and k = 15 if the smoke projects the full
      run (train pass + test) at ≤ 5 h; otherwise v2's 4 views with k = 15. name_ph matters for India
      (transliterations).
   b) Candidate augmentation AFTER the cascade. These pairs bypass the top-10 cap and go to stage B + the model:
      (i) exact-key 'sure' pairs from `src/hq_keys.py`: record_keys → key_pairs(kq_block, kd, kq_ctx = ALL S1 of the
          country) → calibrate on the train pass → apply_rules(p_min 0.96; France gets the minimum over train
          countries);
      (ii) REVERSE blocking: every S2/S3 record queries its top-3 S1 under full_w and name_c3 (GPU top-k with the
          roles swapped), and those pairs join their S1's candidates.
      Report for each source: pairs added per S1 and GT pairs recovered on the train pass, per country.
   c) Training: 600K train S1 (v2: 150K), 5-fold OOF by S1, same threshold sweep. Then
      `hq_keys.global_one_to_one` on the decided pairs after ALL blocks of a country.
   d) Save the pre-cascade pool (up to 40 per S1 with view scores/ranks) for train and test under /home/pools/
      for item 2.
   Metrics (full-density train pass, per country): pair recall after the union / after the cascade / after the
   augmentation (v2: 0.953 after the cascade), OOF F0.5 (v2 0.9623 overall), final cands/S1 (v2 9.2; keep ≤ 12).
   Test-side, label-free: coverage of the sure pairs by rule (v2 France `disjoint|a_eq|invented` 34%, target
   ≥ 90%), mean matches/S1 per country vs the train GT 3.46 (v2: US 3.35 / India 3.12 / France 3.30), 0 records under
   two S1s.
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
