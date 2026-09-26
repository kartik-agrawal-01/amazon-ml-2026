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
- 14:12 GIT AUTH WORKS AGAIN: pulled HQ a7b2489 (QUEUE 4/5b: box only slice-screens keys/reverse/name_ph; full-density
  runs of them belong to the Jarvis lane; port fixes from runs/jarvis/ISSUES.md — file does not exist yet) and pushed
  everything (e05b148). The gate A/B at the end of chain3 is still needed for the box's fallback v3 (France fixes +
  global o2o only, per the new UPLOAD POLICY).
- 14:25 n3 xc_us DONE: hidden F 0.9273 vs base 0.9344 (-0.0071) although the train OOF rose (0.98395 vs 0.98365).
  India test: empty 8.0% (base 7.6%), mean matches 2.93 (2.98), losses in every GT bucket (1-true 0.822 vs 0.838,
  3+ 0.937 vs 0.944); cands/S1 5.3 vs 5.1, train cascade recall same (0.990). So the US-trained model transfers WORSE
  to India with HQ's France fixes (normalize.py + features.py of 7defba7) - exactly the unseen-country situation of
  France. slice_mix running (n3), then n3k/n3ph/n3r (all on the _n3 stores).
### NEXT (adds to the list above)
5. If n3 slice_mix also does not beat base (0.9808): n3 is NOT a clear KEEP -> write runs/day/n3_NOTES.md for HQ with
   both tables, and run the ablation once the chain is idle: (a) `n3f` = n3 stores + v2's features.py
   (`git show 48bc4e7:src/features.py`) to split normalisation vs feature effects; (b) xc_in for n3 (train India ->
   test US) for the full Q. n3k/n3ph/n3r then need to be compared against n3 AND re-run on base stores if n3 is reverted.
- 14:30 Ported Jarvis ISSUES 1 to main: CountryContext.reverse(..., cache=CandCache) saves/loads the raw reverse pairs
  as <cand-cache>/<tag>__rev<r>_<views>_<nS1>.parquet (only with --cand-cache and no doc remap). Unit test still OK.
  Jarvis note: reverse top-k on --topk-device cuda is ~5x slower than CPU sparse_dot_topn at full density (docs as
  queries); slices are small enough that n3r keeps cuda.
- Session 5 ends 14:31: chain3 running n3 slice_mix; nothing else of ours running besides sysmon/gpulog.

## Session 6 — 26 Sep 14:27 (Session 2 of this driver)
- chain3 running n3 slice_mix (started 14:25; store build first).
- n3 xc_us drop analysis (scripts/day/diff_runs.py; output runs/day/n3_vs_base_xc_us.txt): n3 vs base on India test
  loses 2965 TP and gains 1443. Only 62 of the lost TPs were NOT n3 candidates, so blocking is not the cause (n3's
  candidate recall is actually higher, 0.9534 vs 0.9516). The model now rejects them: 67% of lost TPs have an
  Indic-script candidate name (1998 lost vs 290 gained). TP count on Indic-name GT pairs drops 19478 -> 17770 (-9%);
  ASCII names are net +148. scripts/day/norm_diff.py: for 759 of 800 lost-pair records the v2 and HEAD normalisers
  give IDENTICAL name/core/legal/address (the other 41 are only leading-zero or "tg" -> "telangan"). So the change
  comes from the model trained on US with the new features, not from the Indic records' own features. The only
  features.py diff vs v2 is the larger `generic` token set (dba/fka/aka/shri/sri/dr/mr + French words).
- Plan: ablation n3f = n3 stores + v2's generic set (new env switch AML_GENERIC=v2 in src/features.py; default = HEAD
  behaviour, so the running job is not affected). If n3f recovers xc_us -> the generic set is the cause (US-trained
  extra_*_content shifts the decision surface for India transliterations); if not -> the normalisation (US training
  pairs look different) is. Then n3 xc_in (never run) for a full Q.
- scripts/day/chain4.sh (tmux `chain4`) waits for chain3 to exit. chain3's remaining steps n3k/n3ph/n3r/gateA/gateB are
  skipped via placeholder .done markers (listed in runs/day/chain/.chain4_placeholders; chain4 removes them at start).
  chain3 will print "ALL DONE" even though it skipped them; ignore that. chain4 order: n3f (xc_us, slice_mix) ->
  n3xin -> n3k -> n3ph -> n3r -> gateA -> gateB.
### NEXT (session 7+)
1. After a reboot: relaunch sysmon + gpulog and **chain4** (not chain3). If .chain4_placeholders still exists and n3.done
   exists, just start chain4. If n3.done is missing: remove the placeholders listed there, then start chain3.
2. n3f vs n3 on xc_us (n3 0.9273, base 0.9344), then on slice_mix. Decide the generic set. If AML_GENERIC=v2 wins on
   xc_us with no mix loss: make v2's set the default (only for India/US? the France words matter only for France,
   which the slices can't measure. Option: generic = v2 set + French words only, dropping dba/fka/aka/shri/sri/dr/mr/
   ta/as/www). Write the result for HQ in runs/day/n3_NOTES.md.
- Where GT pairs are lost, by candidate-name script (diff_runs.py section "A: where GT pairs are lost"):
  | run (base) | tag | GT | not candidate | cand. but rejected | TP rate |
  |---|---|---|---|---|---|
  | xc_us (US-trained -> India) | Indic-script name | 28491 | 3720 (13.1%) | 5293 (18.6%) | 68.4% |
  | xc_us | ASCII name | 88662 | 1258 (1.4%) | 4981 (5.6%) | 93.0% |
  | slice_mix (US+IN trained) | Indic-script name | 42167 | 3181 (7.5%) | 819 (1.9%) | 90.5% |
  | slice_mix | ASCII name | 249939 | 1643 (0.7%) | 4609 (1.8%) | 97.5% |
  | slice_mix | empty cand. address | 12299 | 1126 (9.2%) | 3423 (27.8%) | 63.0% |
  With India in training, the Indic-name loss is mostly BLOCKING (10x the ASCII miss rate, at 8% density -> worse at
  full density) -> n3ph (name_ph view) is the right screen. Without a country in training (xc_us, the France
  situation) the model rejects 19% of that country's script-variant pairs. Empty-address candidates are the other
  big hole (37% lost, mostly model rejections): candidate for a per-feature look later (QUEUE 6-ish).
- 14:55 n3 slice_mix DONE: 0.9812 (base 0.9808), OOF 0.97644 (base 0.97653) -> n3 Q=0.9650 (xc_in not run). SCOREBOARD
  row 2 = HOLD. runs/day/n3_NOTES.md written for HQ.
- 15:13 n3f xc_us DONE: 0.9205, WORSE than n3 (0.9273) and base (0.9344), although its US OOF is the best of the three
  (0.98406 vs 0.98395 / 0.98365). So the generic set does not explain the n3 drop. xc_us swings by ±0.007 under small
  feature changes that leave OOF flat. Part of the cause: the threshold is picked on the US OOF, and n3f picked
  thr 0.90 where base/n3 picked 0.85. A US-trained model is less confident on India, so a higher threshold costs recall
  there. This is the France situation, and it suggests the decision rule matters more for unseen countries than the
  features do.
  -> qeval.sh now passes --save-probs (from chain4's n3xin step on; the file was replaced atomically, so the qeval
  process already running is unaffected). New scripts/day/thr_sweep.py computes hidden F0.5 at each threshold offline.
### NEXT (session 7+), replaces the session-6 list item 2
2. When n3xin/n3k/... finish, run `PYTHONPATH=. python scripts/day/thr_sweep.py data_xc_us output_day_n3k_xc_us` (and
   on slice_mix and xc_in) to get the F-vs-threshold curve for xc_us and slice_mix. If xc_us peaks well below the
   OOF-chosen threshold while mix is flat, candidate change = "cap the OOF-chosen threshold" or "choose the lower
   threshold within 0.0005 OOF of the best" (a robustness tie-break). That is cheap and France-relevant. First check
   that thr_sweep at the chosen threshold reproduces report.json's test_f05_hidden (±0.001; key rules off).
3. n3 vs base is not settled: the xc_us noise (±0.007) is larger than the n3 effect. Consider 2 seeds (or bagging) for
   the xc_us screens before any KEEP/REVERT that hinges on xc_us.
4. AML_GENERIC=v2fr mode exists (v2 set + French words) but has NOT been run. Low priority now.
- 15:36 n3f slice_mix DONE: 0.9810, OOF 0.97637 -> n3f Q=0.9629: REVERT (AML_GENERIC=v2 stays an opt-in env switch
  only; default behaviour = HEAD). chain4 continues: n3xin (first run with --save-probs) -> n3k -> n3ph -> n3r -> gateA
  -> gateB. Champion unchanged (cycle 0/base). n3 status: HOLD, pending xc_in and the threshold sweep (NEXT 2).
- Session 6 ends ~15:37. Running: chain4 (tmux `chain4`), sysmon, gpulog. chain3 has exited.

## Session 7 — 26 Sep 15:37 (Session 3 of this driver)
- chain4 running n3xin at start. git auth OK.
- 15:45 QUEUE 2c DONE -> runs/day/v2_oof_by_country.md (scripts/day/v2_oof_by_country.py on output_v2_train/oof_pairs.tsv.gz):
  v2 OOF India 0.9476 / US 0.9722 (all 0.96234 = logged 0.9623). Pred matches/S1 India 3.10 vs GT 3.46; US 3.27 vs 3.45.
  India GT pairs reaching the model: 0.919 (US 0.976).
- 15:50 QUEUE 5a DONE -> runs/day/india_recall.md (scripts/day/india_recall.py: v2's FULL-DENSITY India union from
  cands_v2/train_india__b000 (gate A cache) + OOF dump). India loses 11.4% of GT: blocked 6.7%, cascade 1.4%, model FN
  3.3% (FP 0.8%). Indic-script candidate names: 20.5% blocked (ASCII 3.7%). Empty candidate address: TP rate 43%.
- 15:52 offline OOF sweep (scripts/day/oof_rule_sweep.py): a separate lower threshold for empty-address candidates
  LOSES (0.30: −0.0043; 0.60: −0.0003). TRIED & FAILED — don't retry. (Script's per-country columns are broken; ALL valid.)
- 15:56 runs/day/india_blocked_sample.md (40 random blocked India GT pairs). Two patterns:
  (a) candidate = same name + legal words, EMPTY address ("Sai Solutions Pvt", "United Consultancy Private Limited"):
      the ph key is IDENTICAL, but at full density the top-10 of each view is full of same-name decoys. Key rules
      (n3k) / name_ph can help.
  (b) Indic-script names: Devanagari/Gujarati already give identical ph keys ("sautha pavara" -> "st pvr" = "south
      power"), but Tamil/Malayalam do not: their transliterated legal words stay in the core name ("piraivet",
      "praivarr limirrad"), and Tamil has no g/k, b/p, d/t or f contrast ("kulopal pilak tek" = "global black tech").
- NEW opt-in switch AML_PH=2 (src/normalize.py + src/translit.py; default = HEAD behaviour, the running chain and the
  _n3 stores are unaffected): extra Tamil/Malayalam/Telugu legal-form spellings + hp->f + voicing fold g/b/d -> k/p/t in
  phonetic_key. Only the store build uses these, so a run needs AML_PH=2 only while its stores are built (new
  suffix _n4). Check: "குளோபல் பிளாக் டெக் பிரைவேட் லிமிடெட்" -> core "kulopal pilak tek", ph "klpl plk tk" =
  "Global Black Tech Private Limited".
- 15:55 n3 xc_in DONE: 0.9819 (Q0 value 0.9810), OOF 0.96962, thr 0.85 -> n3 Q = 0.9652 (base 0.9669). n3 stays HOLD
  (the xc_us drop dominates). thr_sweep on n3 xc_in reproduces 0.9819 at the chosen 0.85, and 0.85 is the peak
  (0.80: 0.9818, 0.90: 0.9816, 0.70: 0.9807). For India->US the OOF-chosen threshold is right.
- tmux chain5 (scripts/day/chain5.sh) waits for chain4 to exit, then runs n4ph = AML_PH=2, _n4 stores, views +name_ph,
  xc_us + slice_mix. Compare it with n3ph.
- ph_check (runs/day/ph_check.md): AML_PH=2 exact ph agreement on Indic-candidate GT pairs 39.4% -> 43.3%, decoy
  merge +2.9%. Modest; n4ph decides.
- 16:11 n3k xc_us DONE (key rules 0.96 on _n3): 0.9321 vs n3 0.9273 (+0.0048; ≈ +0.0014 on Q), cands/S1 5.31 (n3 5.26).
  India test: 49,183 sure key pairs, only 104 new to the union.
- 16:38 n3k slice_mix DONE: 0.9815 (n3 0.9812), OOF 0.97727, cands/S1 6.04 (n3 6.33, −4.6%). n3k Q = 0.9668 with n3's
  xc_in -> ΔQ +0.0016 vs n3 = KEEP on the n3 line (SCOREBOARD row 4). It only ties base (0.9669), and base lacks the
  France fixes (which the slices cannot measure). Recommendation for v3: n3 + key rules 0.96. An xc_in run of n3k is
  still needed for a clean Q (queue it after chain5 if nothing better).
- Session 7 ends 16:40. Running: chain4 (n3ph next, then n3r, gateA, gateB), chain5 (waits for chain4, then n4ph), sysmon, gpulog.
### NEXT (session 8+)
1. After a reboot: relaunch sysmon, gpulog, chain4 and chain5 (both skip steps that have a .done marker).
2. n3k / n3ph / n3r: after each one, run q_table + `PYTHONPATH=. python scripts/day/thr_sweep.py data_xc_us output_day_<tag>_xc_us`
   (not valid for n3k, which uses key rules). The xc_us threshold curve is the France-relevant question (session 6 NEXT 2).
3. n4ph vs n3ph when chain5 finishes. If n4ph wins on xc_us with mix flat -> run its xc_in; KEEP -> AML_PH=2 behaviour
   becomes the default (remove the switch, and tell HQ in LOG: it changes normalize.py = store rebuild).
4. Gate A/B are at the end of chain4 (hours away). If HQ needs v3 sooner, consider running the gate before n3r.

## Session 8 — 26 Sep 16:41 (Session 4 of this driver)
- Start: chain4 running n3ph (xc_us), chain5 waiting. Plan: QUEUE 2d (unseen-country threshold; HQ needs it by 21:30).
- 16:50 QUEUE 2d (i)+(ii) DONE -> runs/day/unseen_thr.md (scripts/day/unseen_thr.py). xc_us (US->India): hidden F rises
  monotonically as t drops, 0.85 -> 0.15: 0.9321 -> 0.9511 (+0.019). xc_in: the OOF-chosen 0.85 is already best. R1
  (count) and R2 (empty-rate) matching overshoot on xc_in (0.97/0.99: −0.005/−0.018), but their LOWER-ONLY versions
  min(t_oof, R) never hurt: R2lo xc_us +0.0158, xc_in 0, mix +0.0005 -> est ΔQ ≈ +0.005. v2 France counts (empty 5.2%,
  3.30/S1) are already at the source level, so R2lo would NOT change v2's France threshold.
- Implemented opt-in `--thr-adapt` (R2 lower-only per test country) in src/pipeline.py; the default is unchanged.
  Offline check (scripts/day/check_adapt.py) reproduces 0.9480 / 0.9819.
- tmux chain6: SIGSTOPs the chain4 bash (pid 54443; its running n3ph python is unaffected), waits for n3ph to exit,
  runs n3ka = --key-rules 0.96 --thr-adapt on _n3 (xc_us, slice_mix, xc_in), then SIGCONTs chain4 (trap on EXIT).
  If a reboot/kill leaves chain4 stopped: `kill -CONT $(pgrep -xf 'bash scripts/day/chain4.sh')`.
- 16:58 n3ph xc_us DONE: 0.9310 (n3 0.9273, n3k 0.9321), OOF 0.98394, cands/S1 5.65 (n3 5.26, +7%), 18.5 min, 2.9 GB.
  check_adapt on n3ph xc_us: 0.9310 -> 0.9501 (t 0.03), the same +0.019 as on n3k. So the threshold effect is ~4x any
  feature/blocking effect measured on xc_us so far. n3ph slice_mix is running (chain4); n3ka starts after it.
- Session 8 ends ~17:05. Running: chain4 (n3ph slice_mix; its bash is SIGSTOPped by chain6), chain6 (n3ka after n3ph),
  chain5 (n4ph after chain4), sysmon, gpulog.
### NEXT (session 9+)
1. n3ka done (runs/day/qeval_n3ka.log, q_table): SCOREBOARD row. The KEEP rule is vs n3k (Q 0.9668 with n3 xc_in). Expected
   xc_us ≈ 0.948, mix ≈ 0.9820, xc_in ≈ n3k's. If KEEP, --thr-adapt joins the v3 recipe. Tell HQ in LOG and in unseen_thr.md
   (jv1 France threshold; the Jarvis lane uses the same pipeline flag).
2. Check chain4 resumed after chain6 (ps STAT must not be T). n3ph slice_mix -> SCOREBOARD row (n3ph vs n3; name_ph view).
3. Floor question: t=0.03 at full density. Evidence to collect without tuning on the test: the full-density train pass
   (Jarvis) OOF curve at low t for the unseen-like 1-true bucket. Or a leave-one-country-out OOF: train US, predict India
   OOF at full density. Don't pick a floor from the xc_us curve alone.
4. QUEUE 2d (iii) R3 (sure-key recall matching) still open; low value now that lower-only R2 works.

## Session 9 — 26 Sep 16:59 (Session 5 of this driver)
- Start: chain4 running n3ph slice_mix (its bash SIGSTOPped by chain6), chain6 waits (n3ka next), chain5 waits (n4ph).
  Jarvis ISSUES 1 was already ported (14:30). Plan: NEXT 3 (thr-adapt floor evidence) while n3ph runs.
- 17:03 runs/day/floor_evidence.md: in-distribution OOF, a low t costs 1.4–2.2x more at FULL density (v2 train pass)
  than on the slice (t 0.03: −0.105 vs −0.047). Transferred to xc_us, the unfloored t 0.03 could be ≈ −0.04 at full
  density; floor 0.30 ≈ +0.013. New flag `--thr-adapt-floor` (default 0.02 = unchanged). Slice xc_us with floor 0.30:
  0.9497 (unfloored 0.9480, best 0.9511). RECOMMEND `--thr-adapt --thr-adapt-floor 0.30` for full-data files (HQ / jv1).
