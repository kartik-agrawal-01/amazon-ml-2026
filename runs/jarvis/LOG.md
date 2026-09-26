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
