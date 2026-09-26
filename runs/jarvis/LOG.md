# Jarvis LOG (memory across sessions)

## Session 1 — 26 Sep 07:30 (machine time)
- Machine: A30 24 GB, 16 vCPU, free reports 503 GB RAM (482 avail), /home 242 GB free. venv /home/venv (torch 2.11 cu130, lightgbm 4.7, pandas 2.3.3).
- Data at data/student_resource/dataset/{train,test} (not data_extracted/; pipeline finds by name under --data-dir data).
- git merge origin/main: already up to date (b68bd46).
- QUEUE 0 (smoke) PLAN: src.pipeline HEAD, all 6 views, k=15, 60K train S1, 60K test S1, block 60000,
  --n-jobs 14 --stage-b-jobs 8 --topk-device cuda, store under /home/cache_jv/store (first run builds the full store = normalisation of 24M rows),
  caches /home/cache_jv/smoke_*. Log -> runs/jarvis/smoke.log. Then project full-run time for 4 vs 6 views, k 10 vs 15.
- 07:40 Store built under /home/cache_jv/store (test 122s stage + ~4 min normalise; train 477s). Vectorisers (6 views, max_df 0.01) cached at /home/cache_jv/vec6_s42.joblib.
- Wrote src/pipeline_jv.py (copy of main's src/pipeline.py @ b68bd46) + src/jv_aug.py:
  * exact-key rule pairs (hq_keys; ctx = ALL S1 of the country) and reverse top-3 pairs (every S2/S3 -> S1 of the whole
    country, views full_w + name_c3) are merged into the pre-cascade union; features key_rule, rev_full_w_rank,
    rev_name_c3_rank, src_fwd; 'sure' key pairs (calibrated P >= --key-pmin 0.96, France = min over train countries)
    are force-kept after the cascade; --rev-force none|r0|p<x> optionally force-keeps reverse rank-0 pairs.
  * train pass prints a policy table per country (recall fwd union / union+aug / cascade / +sure / +r0 ...; cands/S1).
  * per-country OOF F0.5 (all sampled S1), mean matches vs GT; test: sure-pair coverage by rule, records under 2+ S1.
  * --pool-dir: top-40 pre-cascade pool per S1 (stage-A feats, pa, y, kept, sure) as parquet (for QUEUE 2).
  * Synthetic run (/home/synth) end-to-end OK.
- FINDING: GPU dense-block top-k (blocking.topk_sparse_gpu) is SLOW on the A30 for the reverse direction:
  4.13M India docs x 883K S1, full_w, k=3 -> 784 s. CPU sparse_dot_topn: 200K docs in 17.4 s on 6 threads
  (=> ~160 s for all docs on 14 threads, ~5x faster). Forward CPU 20K S1 x 200K docs 0.3 s (6 thr).
  -> jv_aug.reverse_topk now uses CPU sparse_dot_topn. Forward top-k: compare GPU vs CPU on the smoke next.
- 08:17 smoke RESTARTED (tmux 'smoke', log runs/jarvis/smoke.log):
  python -m src.pipeline_jv --data-dir data --out-dir /home/out_jv/smoke --cache-dir /home/cache_jv/store
    --views name_c3,name_w,addr_c3,name_ph,full_w,addr_w --k 15 --max-df 0.01 --train-s1 60000 --test-limit 60000
    --block-size 60000 --folds 3 --n-jobs 14 --stage-b-jobs 16 --topk-device cuda --vec-cache /home/cache_jv/vec6_s42.joblib
    --pool-dir /home/pools/smoke
  NEXT SESSION: if it died (pause), rerun that exact command. Then: read per-view top-k times (block 0 verbose lines
  "S1->S2 <view> ... (Xs)"), stage-B time, peak RSS; project full run; decide GPU vs CPU forward top-k
  (a 20K-S1 CPU forward per view/source ~0.3 s x (docs/200K) on 6 threads); write QUEUE 0 DONE; start jv1.
- 08:28 smoke timings so far (India train, 24K S1 sample, 4.13M docs): doc matrices 43 s (6 views), exact keys 57 s,
  reverse CPU full_w 266 s / name_c3 182 s (14 thr), forward GPU ~10 s per view per source for 24K S1
  (-> ~340 s per view-source for India test 810K S1 on GPU; CPU sdt est. ~70-140 s) => forward on CPU for jv1.
  Train key calibration (India, smoke sample): core_eq|num_eq P=0.83 (HQ full-train 0.958): with a sampled kq,
  hq_keys.key_pairs' group cap (n_q*n_d <= 2000) admits big chain groups the full-S1 enumeration skips -> lower P
  (conservative). Sure rules India smoke: core_eq|a_eq, core_eq|c_empty, disjoint|a_eq|invented, nsp_eq|a_eq,
  partial|a_eq|*, reorder, subset, swap1|a_eq|*.
- 08:28 jv1 LAUNCHED in parallel (tmux 'jv1', log runs/jarvis/jv1.log, commit e38f8fa):
  python -m src.pipeline_jv --data-dir data --out-dir /home/out_jv/jv1 --cache-dir /home/cache_jv/store
    --views name_c3,name_w,addr_c3,name_ph,full_w,addr_w --k 15 --max-df 0.01 --train-s1 600000 --block-size 500000
    --folds 5 --n-jobs 14 --stage-b-jobs 16 --topk-device cpu --vec-cache /home/cache_jv/vec6_s42.joblib
    --cand-cache /home/cache_jv/cand_jv1 --feat-cache /home/cache_jv/feat_jv1 --pool-dir /home/pools/jv1
  (rev-force none: reverse pairs only enter the pre-cascade union; sure key pairs force-kept.)
  If killed by a pause: rerun the same command (cand-cache skips finished forward top-k blocks; model.joblib in
  /home/out_jv/jv1 -> add --load-model /home/out_jv/jv1/model.joblib to resume at the test pass).
- 08:38 smoke India train (24K S1): union+aug 3.19M cands (133/S1), GT found 79738/83062 = 0.960 (v2 blocking k=10
  India 0.933); phase A 724 s, peak RSS 11.1 GB. GPU forward: 10-11 s per view per source for 24K S1 (12 calls).
  Both smoke and jv1 are in reverse top-k now (sharing 16 CPUs, so both are slower than solo).
- SESSION 1 END (08:40). Running: tmux 'smoke' (QUEUE 0) and tmux 'jv1' (QUEUE 1). NEXT SESSION:
  1. check both logs (Traceback? `dmesg | tail`); smoke: per-country policy table (after "cascade fitted"), OOF per
     country, stage-B time, test timings, validator is not needed for the smoke. Write QUEUE 0 DONE + projection.
  2. choose --rev-force from the smoke/jv1 policy table (keep cands/S1 <= 12; v2 9.17).
  3. jv1: when training finishes, compare OOF per country vs v2 0.9623 (gate +0.3 pt, no country down); the test pass
     follows automatically -> validator -> submissions/jv1_matching_results.tsv -> SCOREBOARD SUBMIT-READY.
     Validator: python data/student_resource/utils/validate_submission.py --matching /home/out_jv/jv1/matching_results.tsv
       --candidate /home/out_jv/jv1/candidate_pairs.tsv --test-dir data/student_resource/dataset/test
- 08:46 reverse top-k under contention (smoke + jv1 together): jv1 India full_w 529 s (solo 266 s), smoke US full_w 637 s.
  TODO next session: cache reverse pairs per (split, country, views, k) under /home/cache_jv/rev_* so reruns skip
  them (edit src/jv_aug.py; safe while jobs run: modules already imported). Reverse is the largest new cost
  (~1-1.5 h of the full run); consider rev views = full_w only if the policy table shows name_c3 adds little.

## Session 2 — 26 Sep 08:47 UTC
- QUEUE (HQ 13:55 IST) now says: use main's --key-rules/--reverse-k/global o2o, drop pipeline_jv reimplementation.
  Moved untracked env.txt aside (identical to main's), `git merge origin/main` clean (91e04b6).
- Killed tmux smoke + jv1 (both src.pipeline_jv; smoke was at US train reverse, jv1 at India forward). src/pipeline_jv.py and
  src/jv_aug.py stay in the tree but are NOT used any more.
- QUEUE 0 DONE (partial, from the pipeline_jv smoke; same top-k/stage-A code as main): India train 24K S1 phase A 724 s,
  peak RSS 11.1 GB; forward top-k GPU 10-11 s per view per source per 24K S1 (-> CPU sdt for full runs); reverse top-3 on
  CPU 14 thr: 180-640 s per view per country (depends on contention); doc matrices 45-50 s per country.
  Projection jv1 (6 views, k 15, main reverse on ALL 6 views): reverse ~1 h per split, forward ~15 min per split,
  stage B + model TBD -> train pass ~3-4 h, test pass ~3-4 h. k 15 kept (projection < 5 h per pass).
- 08:52 jv1 RELAUNCHED on MAIN's src.pipeline (commit 91e04b6 code; tmux 'jv1', log runs/jarvis/jv1.log, command in
  runs/jarvis/jv1.cmd): 6 views k 15, 600K train S1, block 100000, --key-rules 0.96 --reverse-k 3 --reverse-bypass 2,
  --topk-device cpu (GPU reverse 5x slower), n-jobs 14, stage-b-jobs 8, caches /home/cache_jv/j1_cand, j1_feats, --skip-test.
  If paused: rerun `bash -c "$(cat runs/jarvis/jv1.cmd)"` (cand-cache skips forward top-k).
- e025b62 fix(main): reverse pairs cached under --cand-cache (ISSUES.md #1). The running jv1 process has the old code, so
  its train reverse is NOT cached; the test pass (separate invocation, --load-model) will cache.
- 08:52 jv1 RESTARTED (commit 8792474) with `--pool-dir /home/pools/jv1` (new flag: top-40 pre-cascade pool per S1 by the
  cascade score, OOF on train, + keep + y; test pool per block) and the reverse cache. Command in runs/jarvis/jv1.cmd.
  Synthetic check: matching_results byte-identical with/without --pool-dir.
- QUEUE 2 prep: src/jv_ce.py (train/test/report). Fake pool (4K India S1, GT pos + random neg): end-to-end OK.
  A30 throughput under jv1's CPU load: fine-tune 2.2K pairs/s (bs 256, bf16, max_len 96), predict 21K pairs/s
  (pre-tokenised unique texts; tokenizer-per-pair was 5K/s). => 3M pairs/fold ~23 min, OOF 24M pool pairs ~20 min,
  test 1.73M S1 x 40 x 2 folds ~1.8 h (--test-top 20: ~55 min).
  Run after jv1 train pass: python -m src.jv_ce train --pool /home/pools/jv1 --store /home/cache_jv/store --out /home/pools/jv1_ce --pred-bs 2048
- 2b prep: src/jv_ce_feats.py (ce_p, ce_rank, ce_gap over the S1's pool) writes an augmented feat-cache so main's
  `python -m src.rescore` retrains the matcher with ce features (OOF) and rescores the cached test blocks. Not yet run.
- NEXT SESSION PLAN (in order):
  1. jv1 train pass (tmux jv1): if dead w/o "done" -> rerun `bash -c "$(cat runs/jarvis/jv1.cmd)"` (reverse + forward cached).
     When done: record per-country recall (union / cascade / +bypass), OOF per country + overall, cands/S1 in SCOREBOARD.
  2. Gate OOF >= 0.9653 & no country down -> test pass: same command minus --skip-test plus
     `--load-model /home/out_jv/jv1/model.joblib` (keeps --pool-dir so the test pool is written). Then validator -> SUBMIT-READY.
  3. As soon as /home/pools/jv1/train__*.parquet exist (after the cascade, before stage B finishes) the GPU is free:
     tmux ce: python -m src.jv_ce train --pool /home/pools/jv1 --store /home/cache_jv/store --out /home/pools/jv1_ce --pred-bs 2048
     (host RAM: pool 24M rows + texts ~ 6-8 GB; check memory.current + jv1 peak < 56 GB first).
  4. After the test pool exists: jv_ce test (--test-top 40 or 20), jv_ce_feats, src.rescore --feat-cache /home/cache_jv/j2_feats.

## Session 3 — 26 Sep 09:11 UTC
- jv1 (tmux jv1, pid 31319, started 08:52, commit 8792474) alive: train/india reverse top-k (6 views) in progress at 09:11,
  RSS 8.4 GB, memory.current 20 GB. GPU idle (waiting for /home/pools/jv1 train pool for QUEUE 2).
- Merged origin/main (conflict only in CountryContext.reverse: took main's port b28824c of ISSUES #1). NOTE: running jv1
  (old code) writes reverse caches as `*__rev3_<views joined by '-'>_N.parquet`; the merged code looks for views joined
  by '+'. Before any rerun/test pass: `cd /home/cache_jv/j1_cand && for f in *__rev3_*-*.parquet; do mv "$f" "${f//-/+}"; done`
  (check the rename touches only the view list).

## Session 4 — 26 Sep 09:23 UTC
- QUEUE unchanged (HQ 13:55 IST). origin/main has nothing new to merge. Removed stale conflict markers from LOG.md.
- jv1 train pass (tmux jv1, pid 31319, 08:52 start) alive at 09:23: India reverse top-3 over 6 views took 1665 s
  (12.8M pairs), key calibration 71 s (sure rules = 10, same set as smoke), now India forward top-k. RSS 7.4 GB,
  memory.current 19 GB. Projection: US reverse ~50 min -> pool (/home/pools/jv1/train__*.parquet) ~11:00 UTC,
  train pass done ~12:00 UTC.
- 09:40 scripts/jarvis/source_recall.py (GT pairs per candidate source, from --cand-cache). India train block 0 (100K S1,
  345,808 GT pairs): v2 4 views k10 0.9345 (= v2's 0.933) | 6 views k15 forward 0.9496 (120 cands/S1) | name_ph finds
  0.300 of GT but only 0.15 pt that no other view finds | reverse top-3 alone 0.9366, +1.00 pt new -> pre-cascade union
  0.9596 (154/S1) | exact keys: 92 new pairs of 92,667 sure (~0) | reverse-sure (rev_best>=2, bypasses the cap):
  5.35 pairs/S1, precision 0.54 -> WATCH final cands/S1 (budget <= 12); if too many, --reverse-bypass 3 is a
  model-only rerun from the cand cache.
- 09:43 jv1 India train phase A done in 2977 s (peak RSS 14.4 GB); blocks: 331896/345808, 331950/345834, 133974/139479
  GT found pre-cascade (0.960). US phase A started 09:43 (359,875 S1 sampled, 6.19M docs; reverse first).
- 09:45 QUEUE 2 PILOT (runs/jarvis/cepilot.log): pseudo-pool from the India block-0 cand cache, first 30K S1, top-40 per
  S1 by a crude rank proxy (pool holds 95,424 of 103,789 GT = 0.919), jv_ce train 2 folds by S1, 600K pairs/fold,
  MiniLM-L12 1 epoch: fine-tune 2.17K pairs/s (277 s/fold), predict 25.7K pairs/s. OOF: AUC proxy 0.952 vs CE 0.9994;
  recall@5 / @10: proxy 0.766 / 0.863, CE 0.880 / 0.919 (= the pool ceiling). The real pool uses the cascade score
  (much stronger than the proxy), so the gain vs the cascade is still open, but the CE clearly ranks well.
  Code works end to end on real store texts.
- 09:56 tmux 'ce' (runs/jarvis/ce_train.cmd, log runs/jarvis/ce_train.log): waits for "pool: train/us" in jv1.log, then
  `python -m src.jv_ce train --pool /home/pools/jv1 --store /home/cache_jv/store --out /home/pools/jv1_ce --pred-bs 2048`
  (projection: 2 x (3M pairs ~23 min + 12M OOF pairs ~8 min) ~ 1.1 h). If paused: rerun `bash runs/jarvis/ce_train.cmd`.
