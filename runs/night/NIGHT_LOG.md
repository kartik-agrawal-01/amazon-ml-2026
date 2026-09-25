- 2026-09-25 22:34:53 [Gaurav] A1 DONE: GPU top-k fix is commit 333fb28 (India block 312 s vs 1251 s CPU). Smoke attempt 8 is running in tmux 'aml'. DECISION for A3: launch v2 with 4 views, not 6: keep the views that together find >=98% of the pairs all 6 find in the smoke train blocks. Target: v2 finished by 06:30. If 4 views still project past 08:00, use 3.

## Session 1 (Claude, from 22:45)
- 22:48 A0: smoke (tmux aml, commit d9c768c + 333fb28 working tree, old src/pipeline.py without cascade) at the OOF stage.
  Blocking k=10: pair recall 0.9676 (India 0.940 = 78092/83062, US 0.986 = 122472/124219), 79.2 cands/S1;
  score-prune -> 0.9657. GPU top-k speed in the pipeline: India 0.85-1.6 ms/query/pass (4.1M docs), US 1.2-2.1 (6.2M docs).
  PLAN: kill the smoke after "model fitted" (its 60K-per-country TEST pass would cost ~1 h of GPU); v2 is the real test.
- 22:50 B1: slice build launched, tmux `slice`: make_slice.py --frac 0.08 (train + 8% test) then --frac 0.04 --lo 0.5
  --train-as-test (overwrites data_slice/test with a disjoint train-derived holdout + test_ground_truth_HIDDEN.tsv).
  Log: runs/night/make_slice.txt.
- v2 runtime projection (6 views, 12 passes x ~1.1 ms/query): test blocking alone 1.73M S1 -> ~5.5 h, + train + stage B
  -> ~9 h => later than 08:00. Decision: 4 views (per Gaurav's note). scripts/view_ablation.py (new) measures per-view
  recall on 5K S1 per country to choose which 2 to drop (base name_c3,name_w,addr_c3 kept).

## Session 2 (Claude, from 22:53)
- 22:58 A2 DONE: `git mv src/pipeline_next.py src/pipeline.py` (cascade pipeline), `python -m src.pipeline --help` OK,
  commit c6c969d pushed (also adds scripts/view_ablation.py). Smoke (old pipeline) still in OOF fold 2; plan: read
  OOF F0.5, then kill it (its 60K x 3 country test pass would cost ~1 h GPU that v2 needs).
- 22:59 A0 smoke RESULT (old pipeline, 60K train S1, 3 folds, 6 views, GPU top-k): blocking k=10 pair recall 0.9676
  (India 0.940, US 0.986), 79.2 cands/S1, after prune 0.9657; OOF (lightgbm) AUC=0.99976 AP=0.99640;
  OOF macro F0.5 (chosen) = 0.96226 (thr 0.75, one2one). Runtime to model: 2575 s. Killed before the test pass
  (GPU needed for v2) -> no smoke France table; v2 will give it. Odd: India blocking recall 0.940 vs HQ slice 0.986.
  Stopped the old scripts/memguard.sh (pkill -f src.pipeline would also kill claude sessions); driver guard covers v2.
- 23:01 view ablation launched (tmux ablate, runs/night/view_ablation.txt): 5K S1 x {india,us}, cuda, 8 jobs.
- 23:08 SESSION 1 STANDS DOWN: the driver was restarted at 22:53 (second night_loop, PID 217857) so two Claude sessions
  ran concurrently. Session 2 owns the queue from here (A2 done, ablation running). Session 1 leaves: data_slice/ is
  BUILT (train 176,603 S1 / 675K S2 / 713K S3; holdout-as-test 87,964 S1 with data_slice/test_ground_truth_HIDDEN.tsv,
  frac 0.04 lo 0.5; log runs/night/make_slice.txt) -> B1 done, use --data-dir data_slice --cache-dir cache_slice.
  Smoke killed at 23:04 by session 1 (after the OOF line); memguard/rss_watch for smoke stopped. Only ONE driver should run.
- 23:10 VIEW ABLATION (runs/night/view_ablation.txt, 5K S1/country, k=10): 6-view pair recall India 0.9386 / US 0.9863.
  full_w is by far the strongest single view (India 0.896, US 0.970); the 3 "base" views alone give only 0.885 / 0.948.
  base+full_w = 0.9322 / 0.9858 = 99.3% / 99.9% of the 6-view recall (>=98% rule) at ~61 cands/S1 before cascade.
  base+name_ph 0.889/0.950, base+addr_w 0.912/0.967 -> dropped name_ph and addr_w.
  DEVIATION from Gaurav's "keep base + drop 2": the >=98% criterion cannot be met without full_w. If 3 views are ever
  needed: full_w + addr_c3 + name_w (not measured; "only full_w" alone = 95.4% / 98.3% of ALL).
  Per-pass cost (ms/query): India ~1.0, US ~1.4 (full_w the costliest pass).
- 23:09 A3 LAUNCHED v2 (tmux run_v2, pid 230743, commit c6c969d): runs/v2/cmd.txt =
  bash scripts/run_pipeline.sh v2 --n-jobs 8 --max-df 0.01 --train-s1 150000 --block-size 100000 --views name_c3,name_w,addr_c3,full_w --topk-device cuda --stage-b-jobs 4
  Pre-launch projection: test blocking US 663K x ~11.8 ms + India 810K x ~8.6 ms + France ~0.4 h = ~4.5 h, + stage A/B
  per block + train (~1 h) => ~6.5-7 h => ~06:00. Will re-project from the first train block and first test block.
- 23:13 B1/B2: slice store building (tmux slice, cache_slice/, runs/night/slice_store.txt, n_jobs 2). New code:
  src/blocking_b.py (KEY blocking: k_core/k_nsp/k_ph/k_tok_zip/k_tok_city with bucket cap; DENSE MiniLM parts) and
  scripts/track_b_block_eval.py (recall/cands/time of keys, dense, tfidf-4-views and unions on 20K S1 per country).
  Eval chained after the store (tmux trackb -> runs/night/track_b_block_eval.txt).
- 23:30 A3 CHECK v2: train/india block 0 (60031 S1, 4.13M docs): passes name_c3 50s, name_w 63s, addr_c3 50s, full_w 96-108s
  per source => 8.6 ms/S1 for 8 passes; block 3.59M cands (59.8/S1), GT pairs found 193942/207851 = 0.933 (k=10);
  phase A 704s, peak RSS 4.7 GB. US phase A started at 940s (23:25), expected ~23 min.
  PROJECTION: train ~1.1 h total -> test starts ~00:15; test India 8.1 blocks x ~1100s (2.5 h) + US 6.6 x ~1430s (2.6 h)
  + France ~0.4 h = ~5.5 h => v2 done ~05:45 (deadline 08:00, slack ~2 h). DECISION: keep 4 views, no relaunch.
  Re-check after the first TEST block (India, 100K S1: expect ~1100 s; if > 1700 s => projection past 08:00 => relaunch lighter).
- 23:34 B2/B3 RESULT (runs/night/track_b_block_eval.txt; slice train, 20K S1/country, docs India 557K / US 831K):
    India: KEYS 0.522 (8.9/S1) | DENSE MiniLM k=10 0.787 (20/S1) | KEYS+DENSE 0.855 (26/S1) | TFIDF 4 views 0.9655 (61/S1)
           TFIDF+KEYS 0.9757 (65/S1) | TFIDF+DENSE 0.9678 | TFIDF+KEYS+DENSE 0.9772 (79/S1)
    US:    KEYS 0.583 | DENSE 0.9355 | KEYS+DENSE 0.948 | TFIDF 0.9944 (62/S1) | TFIDF+KEYS 0.9956 | TFIDF+KEYS+DENSE 0.9966
    Encoder throughput 3.0K (India) / 4.8K (US) rec/s fp16 => 24M records = ~1.7-2.2 h of encoding alone.
    Dense top-k itself is cheap (0.08-0.09 ms/S1 vs 1.5-2.1 ms/S1 for 4 TF-IDF views); key blocking ~1-1.4 ms/S1.
  CONCLUSION: key+dense blocking CANNOT replace TF-IDF (India recall 0.855 vs 0.966; US 0.948 vs 0.994) and the
  encoder alone busts the 2 h budget => B5 criteria not met, no B_READY. Cheap gain for HQ: exact-key views add
  +1.0 pt (India) / +0.1 pt (US) pair recall at +4 cands/S1 (modelling change -> not applied to v2).
  India zip/city fields are ~empty in the store (a_zip non-empty 0.4%, city 0%) so k_tok_zip/k_tok_city are dead there.
- 23:35 B4 launched (tmux trackb2, scripts/night/track_b_chain.sh, log runs/night/xc_slices.txt): cross-country slices
  data_xc_us (train US 8% -> test = India 4% holdout) and data_xc_in (train India -> US holdout), then 3 pipeline runs
  one at a time with --n-jobs 2: slice_mix (data_slice), xc_us, xc_in; each --train-s1 40000 --test-gt <hidden GT>,
  4 views, cascade. Reports land in runs/{slice_mix,xc_us,xc_in}/ (report.json has test_f05_hidden).
- 23:36 !! v2 attempt 1 KILLED by the driver guard at 23:25:50 (MemAvailable 1272 MB) while it was building the US
  train doc matrices (6.2M docs x 4 views, 8 spawned transform workers) AND my Track B eval (MiniLM encoding of the
  slice, ~2.5 GB) ran at the same time. The eval is not a src.pipeline process, so the guard's "newest src.pipeline"
  was v2 itself. Lesson: NO other heavy job (pipeline or not) during v2's phase-A/doc-matrix stages; the box has only
  ~8.5 GB available for us in total. Log of the killed attempt: runs/v2/stdout_attempt1_killed.txt.
  RELAUNCHED v2 at 23:35:40 (same cmd, pid 245598, tmux run_v2). Track B chain (trackb2) stopped before it started
  its slices; rerun it only when v2 is in a stable low-memory stage (scripts/night/track_b_chain.sh skips done runs).
  Memory profile logger: tmux memlog -> runs/night/mem_v2.txt (MemAvailable, v2 RSS, GPU, stage; every 60 s).
  New projection: train done ~00:45, test ~5.5 h => v2 done ~06:15 (slack ~1.7 h to 08:00).
- 23:55 v2 attempt 2 passed the US doc-matrix stage alone: MemAvailable dipped to ~3.7 GB (60 s samples; v2 RSS 5.8 GB)
  => the box has NO room for a concurrent slice pipeline (~3 GB) during v2's doc-matrix/stage-A spikes. India phase A
  675 s; US blocking started at 1005 s (23:52), US block expected ~00:12, then cascade/phase B/OOF, test from ~00:45.
  SESSION 3 TODO: (1) check v2 (runs/v2/stdout.txt, runs/night/mem_v2.txt, guard.log); on the first TEST block
  (India, 100K S1) re-project: ~1100 s/block => done ~06:15; > 1700 s/block => past 08:00 => relaunch lighter per A3.
  (2) If v2 died: crash procedure (--stage-b-jobs 2 -> --n-jobs 4 -> --block-size 50000) and relaunch at once.
  (3) B4 chain (`bash scripts/night/track_b_chain.sh` in tmux trackb2) ONLY if v2 is in test blocking with
  MemAvailable > 6 GB; the guard kills the slice job first (it IS a src.pipeline process), but do not risk v2 for it.
  (4) No B_READY: Track B dense blocking fails the recall bar (see 23:34 entry); v2 is the deliverable.
- 00:12 v2 train phase A complete: India 60031 S1 -> 3.59M cands, recall 0.933; US 89969 S1 -> 5.31M cands (59/S1),
  recall 305685/310675 = 0.984 (k=10, 4 views); US passes 1.2/1.4/1.2/1.9 ms per S1 (11.5 ms per S1 for 8 passes);
  phase A 1258 s, peak RSS 6.2 GB, MemAvailable ~6.8 GB. Cascade fit now (2149 s = 00:11); test expected from ~00:45.
  Session 2 ends 00:12; v2 alive (pid 245598, tmux run_v2); memlog running (tmux memlog).

## Session 3 (Claude, from 00:42)
- 00:44 A4 prep: scripts/night/post_v2.sh (tmux post_v2) waits for the v2 pid, then runs the official validator,
  `cp -r output output_v2`, submissions/v2_matching_results.tsv (if < 100 MB), auto-writes runs/v2/NOTES.md from
  report.json, commits + pushes, then starts the B4 chain (`NJ=6 SKIP_WAIT=1 scripts/night/track_b_chain.sh`).
  scripts/night/salvage_partial.py = emergency fallback (partial output/ -> valid submission, missing S1 empty).
- 01:05 !! v2 attempt 2 STALLED in test/france block 0: the 8 top-k passes took 423 s (00:43) and then nothing for
  25+ min: main thread 100% CPU, no stage-B workers, no I/O, RSS flat 3.9 GB. Root cause (verified with a
  micro-benchmark on cache/test__france.parquet): pandas 3.0 stores `rid` as an Arrow-backed str column, and
  `d["rid"].values` is an ArrowStringArray; `d_rid[grp.values]` costs 20.8 ms per call vs 0.013 ms on a numpy
  object array. The test path calls it ~200K times per 100K block (decide() + candidate-row loop) = ~70 min per
  block => v2 could never finish (18 blocks). The train/OOF path uses numpy str arrays (np.unique(...).astype(str)),
  so training was unaffected (OOF macro F0.5 0.96231, thr 0.70 one2one, AUC 0.99876; see
  runs/v2/stdout_attempt2_train_only.txt). Train/test blocks in the smoke run were 60K S1 -> would have shown the
  same stall on its test pass (never reached; killed before).
- 01:13 FIX (commit ea7375a): `.to_numpy(dtype=object)` for d_rid / qb_rid / gt_pairs_block ids; new
  `--load-model PATH` skips training and reuses model.joblib (model, cascade, decision rule); the vectorisers are
  refit from the seeded sample (deterministic). Verified on data_slice (runs/tiny): train path OK end to end
  (4K train S1, 1.5K holdout S1, hidden F0.5 0.977), load path output byte-identical to the train path, test blocks
  of 600-900 S1 in ~40 s including doc matrices. Killed the stalled v2 (pid 245598) at 01:13; its model saved to
  output_v2_train/model.joblib (+ oof_pairs).
- 01:27 A3 RELAUNCHED v2 (attempt 3, tmux run_v2, pid 301447, commit ea7375a), training skipped:
  runs/v2/cmd.txt = bash scripts/run_pipeline.sh v2 --n-jobs 8 --max-df 0.01 --train-s1 150000 --block-size 100000
  --views name_c3,name_w,addr_c3,full_w --topk-device cuda --stage-b-jobs 4 --load-model output_v2_train/model.joblib
  Watcher restarted (tmux post_v2, pid 301447). Projection: vecs ~3 min, France 3 blocks x ~9 min, India 8.1 x ~19 min,
  US 6.6 x ~24 min => ~5.5-6 h => done ~07:00-07:30; then the watcher validates/pushes (~10 min) and starts B4.
  Note: another user's 10-thread job (pid 1536) has been running all night; load ~11 on 20 threads.
- B4 not started concurrently: v2 needs ~6.5 GB at its India/US doc-matrix spikes, the slice pipeline ~3-4 GB and
  make_slice on full train more; the box has ~11 GB for us. The chain runs automatically after v2 (post_v2.sh).
- 01:41 A3 CHECK v2 attempt 3: vectoriser vocab identical to the trained run (19294/106274/22727/287718);
  test/france block 0 (100K S1) = 606 s total: passes 423 s + stage A/cascade/stage B/decide/write ~180 s
  (was > 70 min before the fix). 6.15M cands -> 979K kept (9.8/S1); 94668/100000 S1 with matches => France
  predicted-empty 5.3% in block 0 (US/India train-set OOF empty rate ~5.6%) -> no France anomaly so far.
  GPU memory 14.4 GB of 16.3 (torch cache; was 11.9 GB in attempt 2) - watch for CUDA OOM.
  PROJECTION: France done ~01:57; India 8.1 blocks x ~1060 s = 2.4 h -> ~04:20; US 6.6 x ~1350 s = 2.5 h -> ~06:50;
  watcher validate + push ~07:00; B4 chain ~07:00-08:15. DECISION: keep running (fits 08:00 with ~1 h slack).
  SESSION 4+ TODO: (1) check runs/v2/stdout.txt block times vs projection (India block ~1060 s, US ~1350 s; if a
  block is > 1.5x that, re-project; relaunch is NOT an option any more - use salvage_partial.py at 07:45 at the
  latest if v2 is still running, then validate + push the salvage as submissions/v2_partial_matching_results.tsv);
  (2) runs/night/post_v2.log + guard.log for kills; (3) if v2 died: relaunch the SAME cmd (runs/v2/cmd.txt,
  training is skipped so a restart costs only the blocks done so far - no resume; consider `--test-limit`? no:
  it caps per-country pos, not a resume); (4) when the watcher has pushed runs/v2/NOTES.md, add the OOF numbers
  from runs/v2/stdout_attempt2_train_only.txt to NOTES.md (report.json of the load-model run has no OOF block);
  (5) B4 results land in runs/{slice_mix,xc_us,xc_in}/report.json (test_f05_hidden) -> morning summary.
- 01:42 session 3 ends; v2 alive (pid 301447, tmux run_v2), watcher alive (tmux post_v2), memlog alive.
