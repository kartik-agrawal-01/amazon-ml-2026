# Day loop log

## memory note — 26 Sep (from runs/france_diag)
- Daytime (other users active; mariadbd of another user ~2.3 GB RSS), GPU top-k, --n-jobs 8:
  test/india block 0 was guard-killed (MemAvailable 1270 MB < 1500 MB guard) with **4 stage-B workers**, in a process
  that had already scored France + US (heap not fully returned between countries).
- Rerun of India ALONE in a fresh process with **--stage-b-jobs 3** passed (1242 s for one 100K block).
- Peak RSS: the diag script did not log per-process peak RSS (my omission). Observed system-wide during the US block:
  ~10.1 GB used / 5.5 GB available of 15.6 GB. Reference: v2's own log shows test/india peak RSS 6846 MB (night run, 4 workers).
- => Daytime full-data runs: start with --stage-b-jobs 3 (or 2), and run heavy countries (India) in a fresh process
  when possible; keep the MemAvailable guard at 1.5 GB.

## Session 1 — 26 Sep 11:37
Plan (QUEUE 1, P0 fast lane):
- Stage-A features include per-view top-k ranks and group/rank context over the FULL pre-cascade union
  (grp_n, c_nq, rq_score, r_*), so rescoring only v2's kept candidate_pairs cannot reproduce v2. Fast lane =
  cache the pre-cascade union (q, c, <view>_rank, n_views) per (split, country, block) as parquet
  (`--cand-cache DIR`); everything after top-k is recomputed from it (stage A cosines are cheap row-wise dots).
  `--cands-only` builds the cache (train 150K sample + test) and stops. `--vec-cache FILE` persists vectorisers.
  `--save-probs` writes per-pair test probabilities (s1, cand, p, score) per country.
- Top-k GPU speedup: densify the query chunk on the card (was: CPU toarray 0.5 GB + PCIe copy per chunk).
  Bench (slice India, 821K docs, 20K queries): full_w 11.0 s -> 6.4 s, name_c3 6.7 -> 6.0 s; results identical (1.0).
- v2 never saved its union -> one full top-k pass is unavoidable (gate run A: --cands-only with v2 code, ~3-4 h GPU),
  then gate run B: rescore from cache with v2's model -> compare to output_v2/matching_results.tsv.
- Smoke (xc_us, 10K train / 20K test, /tmp): run 1 builds the cache, run 2 reads it -> OOF identical, hidden F identical
  (0.92600), matching rows 99.985% identical; kept candidates differ for 3.6% of S1 because the RETRAINED cascade
  LightGBM is not bit-deterministic. Rescoring from the cache with run 1's saved model: 100.000% identical
  matching_results AND candidate_pairs -> the cache path itself is exact.
- INCIDENT 11:54: first gate launch from ~/aml_gate started REBUILDING v2's test store in cache/ (meta paths are
  relative to the repo cwd -> load_meta returned None). Killed after ~1 min, before any store file was touched
  (only my new partial test__raw__*.parquet staging files existed; removed). Fix: load_meta resolves paths against
  cache_dir (commit 802ee11); gate.sh aborts if the store does not resolve.
- 12:05 gate restarted (tmux `gate`, log runs/day/gate.log): A = --cands-only -> cands_v2/ (train 150K + full test,
  v2 code), then B = rescore test with output_v2_train/model.joblib -> ~/aml_gate_out -> compare_matches.py.
- LightGBM (model + cascade): deterministic=True, force_row_wise=True -> refits are bit-reproducible (tested:
  identical probabilities on 2 fits). Numerics differ slightly from the Q0 runs; it doesn't change v2 gate B (loaded model).

### NEXT (for the following sessions)
1. Watch tmux `gate` / runs/day/gate.log. When "GATE PASS": keep the v2 per-pair probs:
   `mkdir output_v2_probs && mv ~/aml_gate_out/test_probs_*.parquet output_v2_probs/` (HQ wants v2 vs v3 France
   score distributions), record per-stage runtimes of gate B in LOG, then `rm -rf ~/aml_gate_out` and
   `git worktree remove --force ~/aml_gate`; mark QUEUE 1 DONE. If FAIL: debug (fast-lane bug).
2. QUEUE 2 (France fixes, HEAD): `bash scripts/day/qeval.sh n3 _n3 "xc_us slice_mix"` (builds cache_*_n3 stores
   first), then xc_in if promising -> `python scripts/day/q_table.py n3`. KEEP/REVERT.
3. v3 full data (if KEEP): store cache_n3 (new; never touch cache/), fast lane:
   `bash scripts/run_pipeline.sh v3 --cache-dir cache_n3 --out-dir output_v3 --n-jobs 8 --stage-b-jobs 3 --topk-device cuda
    --max-df 0.01 --train-s1 150000 --block-size 100000 --views name_c3,name_w,addr_c3,full_w --cand-cache cands_v2
    --vec-cache cache_n3/vecs.joblib --save-probs`  (NB run_pipeline.sh hardcodes --out-dir output: pass the
   python command directly instead). Candidates = v2's union (ranks from v2 blocking, cosines recomputed).
- `--feat-cache DIR` (pipeline) keeps train_X.f32 + train_meta.npz/json and one parquet per test block (s1, cand,
  final features, p) + the block's S1 ids; `python -m src.rescore --feat-cache DIR --cache-dir <store> --out-dir X
  [--drop-feats ..] [--load-model ..] [--save-probs]` retrains/re-predicts from it. Smoke (xc_us small, deterministic
  LightGBM): rescore vs pipeline = 100% identical matching_results + candidate_pairs, per-pair p identical.
- Gate A progress: train/india union 3,588,823 pairs = exactly v2's count; top-k per view with the GPU densify:
  name_c3 49->47 s, name_w 63->46 s, addr_c3 50->46 s, full_w 94->48 s (train/india phase 575 s -> ~390 s).
- NEXT addition for step 3 (v3 full data): add `--feat-cache feats_v3` to the v3 command -> later model/rule
  experiments on full data via src.rescore in minutes.

## Session 2 — 26 Sep 12:44 (driver restarted after a REBOOT)
- Box rebooted ~12:15-12:42 (uptime 2 min at 12:44; power). Gate A died in train/us top-k (gate log -> runs/day/gate_try1.log).
  cands_v2/ kept train_india b000 + vecs.joblib -> 12:45 gate restarted (tmux `gate`, same script); it reuses the cached
  vectorisers and the train/india union, recomputes the rest.
- GIT AUTH LOST with the reboot: the credential cache (credential.helper cache) is empty, no ssh key / gh on the box ->
  `git pull`/`git push` fail ("could not read Username"). Commits stay LOCAL until a human re-authenticates
  (e.g. `git push` once interactively in tmux). HQ cannot see this session's results until then.
- Found uncommitted WIP from the previous session in src/pipeline.py: QUEUE 4a key rules (`--key-rules 0.96`:
  calibrate hq_keys rules per train country on ALL S1 context, inject sure pairs into the union, bypass the cascade cap,
  features key_p/key_sure; test pass: unseen country gets the min over train countries). Reviewed; smoke-testing it.
