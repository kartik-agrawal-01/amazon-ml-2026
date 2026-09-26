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

## Session 3 — 26 Sep 12:53 (second REBOOT ~12:51; gate A died again right after train/india)
- Git auth still missing (pull/push fail) -> commits stay local.
- Committed the key-rules WIP (5b8720f, flag default off). Added per-rule coverage of the sure key pairs in the test
  pass (after global o2o): log table + <out>/key_coverage_<country>.csv, report per_country key_sure/_decided (QUEUE 3c).
- Heavy work now runs as ONE resumable chain: `scripts/day/chain1.sh` (tmux `chain`, log runs/day/chain1.log;
  finished steps leave runs/day/chain/<step>.done, so after a reboot just relaunch the same command):
  gateA (cands_v2) -> gateB (rescore v2 + compare, runs/day/gateB.log) -> qeval base (HEAD, v2 stores; new baseline
  because deterministic LightGBM changed numerics vs Q0) -> qeval n3 (France fixes, _n3 stores) -> qeval n3k (+ --key-rules 0.96).
  Relaunch: `~/miniforge3/envs/aml/bin/tmux new -d -s chain 'bash scripts/day/chain1.sh > runs/day/chain1.log 2>&1'`
  (check first with pgrep that nothing runs).

## Session 4 — 26 Sep 13:03 (THIRD reboot, ~13:00)
- Reboot pattern: all 3 hard power-offs (12:15:47, 12:47:40, 12:58:49; no clean shutdown in the journal) happened
  within seconds of the START of the GPU top-k on train/us (gate A timings match to ~10 s every time). v2 ran 5 h of GPU
  top-k at night without trouble using the host-densify path; session 1's "densify on the card" keeps the GPU at 100%
  duty (sparse.mm back to back) -> suspected PSU/power trip under sustained full load (power limit 180 W; no sudo for
  `nvidia-smi -pl`). Mitigation (commit aed5cf2): env AML_GPU_HOST_DENSIFY=1 (v2's path) + AML_GPU_DUTY=0.6 (sleep
  after each chunk so the card is busy ~60%); chain1/chain2 export both. Same top-k results (scripts/day/topk_check.py:
  cpu = card = host, overlap 1.0). GPU power/util logged every 5 s to runs/day/gpu_power.log (tmux `gpulog`).
  If the box still reboots at train/us: run gate A with `--topk-device cpu` (edit gate.sh C=...) or AML_GPU_DUTY=0.3.
- 13:04 chain1 relaunched (tmux `chain`): gateA resumes (train/india cached) -> gateB -> base -> n3 -> n3k.
- QUEUE 5a DONE (previous session's analysis, committed now): runs/day/india_recall.md. India full-density blocking
  recall 0.933; 55% of misses are Indic-transliterated candidate names (recall 0.795: 'kansaltantsa' = consultants,
  'bildarsa' = builders); 32% of misses have IDENTICAL phonetic keys (src.translit.phonetic_key maps both spellings to
  'knsltnts', 'bldrs'). v2 dropped the name_ph view -> QUEUE 5b first try = name_ph back (all countries, one model).
- 13:10 chain2 (tmux `chain2`, waits for runs/day/chain/n3k.done): n3ph = n3 + `--views ...,name_ph` Q screen.
  Compare n3ph vs n3 (and cands/S1). If it KEEPs, next: full-density India recall via the candidate-augmentation path
  (name_ph pass only on cands_v2 train/india, then pair recall after the cascade).

### NEXT
- After a reboot: check `pgrep -af '^python[0-9.]* (-m src[.]|scripts/)'`, then relaunch BOTH (chain2 waits for chain1):
  `T=~/miniforge3/envs/aml/bin/tmux; $T new -d -s chain 'bash scripts/day/chain1.sh > runs/day/chain1.log 2>&1';
   $T new -d -s chain2 'bash scripts/day/chain2.sh > runs/day/chain2.log 2>&1';
   $T new -d -s gpulog 'nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,temperature.gpu,clocks.sm --format=csv,noheader -l 5 >> runs/day/gpu_power.log'`
- GATE PASS handling: see Session 1 NEXT 1.  Q tables: `python scripts/day/q_table.py <tag>` (base, n3, n3k, n3ph).

## Session 5 — 26 Sep 13:21 (FOURTH reboot, 13:18:17; driver restarted, "Session 1" in its prompt)
- Gate A died again in train/us top-k (name_c3/name_w/addr_c3 done, inside full_w). This time the GPU was throttled
  (host densify + 60% duty): gpu_power.log max 64 W, 44 W / 0% util at the last sample 13:18:15 -> NOT a GPU power
  trip. journalctl: boots -4/-3 at 12:48:44 and 12:49:24 lasted ~1 s each (power flapping), no kernel error before any
  power-off -> hard power loss.
- SECURITY/LOAD NOTE for humans: at boot a process `postgres: user dbname 127.0.0.1 idle` (user `user`, pid 1542)
  burns ~975% CPU (10 cores, 23 CPU-min in the first 2.5 min) with 2.3 GB RSS. An *idle* postgres backend should not
  do that; together with the sshd brute-force attempts in the journal (invalid users from 188.168.86.6) it looks like
  a possible cryptominer disguised as postgres. Not our process -> not touched. It also doubles the box's CPU/power
  load while our jobs run (possible reason for the power trips). Please have the machine owner check it.
- Committed the previous session's uncommitted candidate-augmentation WIP (f4fd6f1; only active with --cand-cache /
  --vec-cache when views are missing from the cache; py_compile OK, not yet exercised).
- New order (scripts/day/chain3.sh, tmux `chain`, log runs/day/chain3.log): slice screens base -> n3 -> n3k -> n3ph
  first (they never triggered a reboot and give the Q decisions HQ needs), then gate A (AML_GPU_DUTY=0.3) -> gate B.
  Slice screens use --n-jobs 6 (QE_NJOBS) to lower the box load next to the 10-core postgres process.
  tmux `sysmon` logs MemAvailable/load/max CPU temp/top process every 5 s (fsync'd) to runs/day/sysmon.log.
- 13:23 chain3 started (base).
- 13:26 QUEUE 5b reverse blocking implemented (628591f): `--reverse-k 3 --reverse-bypass 2`. Per country, every S2/S3
  doc queries its top-3 S1 among ALL S1 of the country (train: all train S1, then mapped to the sample, so the
  competition matches the test pass) per view; pairs are unioned pre-cascade with features rev_rank (best rank over
  views), rev_n (#views), rev_best (#views where the S1 is the doc's #1) and rev_sure = rev_best >= 2, which bypasses
  the cascade cap (like key_sure). Stored in model.joblib (`reverse`), restored by --load-model. Unit test vs brute force:
  scripts/day/test_reverse.py (exact). Not yet run end-to-end -> first real run is the chain step n3r.
- Chain restarted 13:26 with n3r inserted after n3ph: base -> n3 -> n3k -> n3ph -> n3r -> gateA -> gateB.
- 14:07 base DONE: Q=0.9669 (mix 0.9808, xc_us 0.9344, xc_in not run -> Q0 value). Within noise of Q0 (+0.0003); this
  is the reference for n3/n3k/n3ph/n3r (compare to it, not to Q0). Runtimes with --n-jobs 6: xc_us 15.5 min,
  slice_mix 25.1 min; peak RSS 3.3 GB. No reboot during base (sysmon: CPU max ~72 C, MemAvailable ~8.6 GB).
- 14:07 n3 started (builds the cache_*_n3 stores first).

### NEXT (session 6+)
1. Chain3 (tmux `chain`) runs n3 -> n3k -> n3ph -> n3r -> gateA -> gateB unattended. After each step:
   `python scripts/day/q_table.py <tag>`, compare to base (0.9669), SCOREBOARD row, KEEP/REVERT decision.
   n3 (France fixes) is KEEP-without-Q-rule per QUEUE 3 unless OOF drops. n3k / n3ph / n3r are measured against n3.
2. If n3r crashes (first end-to-end run of --reverse-k): runs/day_n3r_*/stdout.txt, fix, rerun n3r alone.
3. After a reboot: pgrep check, then relaunch sysmon, gpulog and chain (see Session 4 NEXT; chain3.sh instead of
   chain1/chain2 - it is resumable via runs/day/chain/*.done). Check runs/day/sysmon.log tail for the moments before.
4. Git auth still missing -> everything is committed locally only (HQ can't see it). Humans: please `git push` once.
