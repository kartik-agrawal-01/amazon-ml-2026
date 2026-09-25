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
