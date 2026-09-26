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
