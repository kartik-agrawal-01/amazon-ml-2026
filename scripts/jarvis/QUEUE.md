# Jarvis queue — top = next. HQ (modelling chat) owns this file; the Jarvis loop only reads it (from main).
# Initial version drafted by the infra chat on 26 Sep ~13:15 as a proposal: HQ, edit freely.
# Machine: A30 (24 GB GPU, 16 vCPU, ~112 GB RAM). It can be resumed on an RTX PRO 6000 (96 GB, 28 vCPU) if a GPU
# step is too slow; items marked [RTX] only make sense after that switch.

0. [Smoke] runs/jarvis/env.txt shows the GPU. Run the pipeline smoke (60K test S1) with --n-jobs nproc-2,
   --stage-b-jobs nproc/2, --topk-device cuda; record runtime per stage and estimate the full-data runtime.
1. [A, full strength within RAM] HEAD of main (France fixes included), v2's 4 views, train S1 600K instead of
   150K, otherwise as v2 (--max-df 0.01, cascade cap 10). Add name_ph + addr_w (6 views) only if the smoke says the
   full run fits in ~3 h. Report the full-density train-pass OOF per country and pair recall after the cascade vs
   v2. Also save the PRE-cascade pool (union of the views' top-k, up to 40 per S1, with the view scores) for train
   and test as parquet under /home/pools/ for item 2. If OOF ≥ v2 + 0.3 pt → full test → jv1 SUBMIT-READY.
2. [B, GPU pair model] Fine-tune a multilingual cross-encoder on train pool pairs: on the A30 start with
   paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0; ~1 h to score 70M pairs); xlm-roberta-base or
   intfloat/multilingual-e5-base (MIT) only if throughput allows. Positives = GT pairs, hard negatives = pool
   non-matches, text "name | address | country" for each side, 2 folds by S1 so every train pair gets an
   out-of-fold score; test = mean of the 2 fold models.
   (a) Re-rank: keep the top 10 of the 40-pool by the cross-encoder (or a blend with the cascade score) →
       recall at 10 vs item 1 (targets: India recall, French 'sure' pairs lost to same-name decoys).
   (b) Feature: add ce_prob and its rank within the S1 to the matcher → OOF.
   KEEP if OOF +0.3 pt, or India recall +1 pt at ≤ 10 cands/S1 → full test → jv2 SUBMIT-READY.
3. [RTX, only if 2 KEEPs] Judge for borderline pairs: Qwen2.5-7B-Instruct (Apache-2.0) + LoRA on pairs with
   matcher p in [0.3, 0.9], same 2-fold protocol. KEEP if OOF +0.2 pt.
