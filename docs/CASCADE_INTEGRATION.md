# Cascade filter — integration spec (HQ, 25 Sep evening)

Organisers' update: `candidate_pairs.tsv` is reviewed in the final evaluation and a SMALLER candidate set
per S1 ranks higher; blocking must scale. HQ measured on the 8% slice (`scripts/cascade_study.py`,
`runs/slice_v3`): the current score-threshold pruning leaves ~66 candidates per S1; a cascade (cheap
GBDT on the 36 stage-A features, keep top-10 per S1 with a 0.002 floor) leaves **~8 per S1 at 98.0% pair
recall with no measurable F0.5 loss** (0.9762 vs 0.9765). A plain score threshold or top-N by score cannot
do this (top-20 by score → recall 0.78).

`src/cascade.py` (new, self-contained) provides:
- `stage_a_columns(cands)` → the 36 stage-A feature names (+ `c_src`)
- `fit_cascade(X, y, groups, n_folds=3, seed)` → (OOF probabilities for train pairs, final fitted model)
- `cascade_keep(q, pa, top_n=10, floor=0.002)` → boolean mask (top_n per query by pa, pa ≥ floor)
- `candidate_stats(q, n_queries, y=None, n_true_pairs=None)` → mean/median/p90/max/zero_share (+ pair_recall)

## Changes to `src/pipeline.py` (after the memory restructure)
1. New flags: `--cascade-top 10`, `--cascade-floor 0.002`, `--no-cascade` (falls back to the old score prune).
2. TRAIN pass: collect stage-A candidates of all countries/blocks (as today). Then:
   ```python
   cols_a = stage_a_columns(cand_tr)               # before stage B is run
   Xa = cand_tr[cols_a].to_numpy(np.float32)
   pa, cascade_model = fit_cascade(Xa, cand_tr.y.values, cand_tr.q_global.values, 3, seed)
   report["cands_before_cascade"] = candidate_stats(cand_tr.q_global.values, n_train_s1, cand_tr.y.values, n_true_pairs)
   keep = cascade_keep(cand_tr.q_global.values, pa, a.cascade_top, a.cascade_floor)
   cand_tr = cand_tr[keep]; report["cands_after_cascade"] = candidate_stats(...)   # + pair recall
   ```
   (`q_global` = a query id unique across countries/blocks; per-block `q` is fine if stats are computed
   per block and aggregated.) Keep the old `calibrate_prune` only when `--no-cascade`.
   Then stage B + the full model exactly as before on the kept pairs.
3. TEST pass, per block: stage A → `pa = cascade_model.predict_proba(Xa)[:, 1]` → `cascade_keep` →
   stage B → full model → decide. `candidate_pairs.tsv` = the kept pairs (unchanged writer).
   Accumulate `candidate_stats` over all test S1 (and per country) into `report["test_candidates"]`.
4. Save `cascade_model` next to the full model in `model.joblib`. Print in the log:
   `cascade: before mean X / median Y per S1, after mean X / median Y, train pair recall R`.
5. Dense view: unchanged in blocking (it may add pairs to the union), but since the cascade caps every
   S1 at `--cascade-top`, dense candidates can only REPLACE lexical ones, never grow the set. `cos_dense`
   is a stage-A feature and therefore part of the cascade score.
6. Runbook/CLAUDE.md: report `cands_after_cascade` mean/median in every NOTES.md.

Expected on the full data: candidate_pairs.tsv ≈ 8 ids per S1 (≈14M pairs for 1.73M test S1), stage B
and the matcher ~8x cheaper, same or better F0.5.
