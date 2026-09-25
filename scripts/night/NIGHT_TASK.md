# Overnight task — Claude Code headless on the box

You are the on-box operator described in `CLAUDE.md` (its hard rules apply), running UNATTENDED
overnight. Nobody will answer questions — never ask; decide, act, log. A driver
(`scripts/night/night_loop.sh`) starts you in sessions of ≤85 min, about every 30 min while a long run
is going. Launch long runs DETACHED in tmux, then end the session. Memory between sessions:
`runs/night/NIGHT_LOG.md` (create if missing). Every session: `cd ~/amazon-ml-2026 && git pull --no-edit`
(if git auth fails, note it and work locally), read NIGHT_LOG.md, CONTEXT.md, `runs/night/guard.log`.

## Detecting running pipelines (important)
Use ONLY `pgrep -af '^python[0-9.]* -m src[.]pipeline'` to see whether a pipeline run is active. A plain
`pgrep -f src.pipeline` also matches YOUR OWN claude process (this prompt contains that text) — never use it,
and never kill a process that is not a `python -m src.pipeline` run you or the operator started.

## Goal for 08:30 IST (non-negotiable order)
A. A VALIDATED full-data submission file by 08:30 (Track A). This is the deliverable.
B. The best possible score. Bar to beat: team leaderboard 94.8 (that model scored 98.5 locally →
   ~3.7 pts lost on the leaderboard, most likely on FRANCE: 15% of test S1, unseen in training).
   Local target ≥ 0.99, but a model that generalises to France beats a higher local number.
C. Use the GPU for the heavy lifting. Smaller candidate sets per S1 also rank higher (official rule).

## TRACK A — deliverable (do these first, in order)
A0. Smoke run (tmux `aml`, `runs/smoke/`). If running, go to Track B work. When finished, record in
    NIGHT_LOG.md: pass/fail, blocking recall per country, OOF F0.5, the smoke TEST per-country table
    (France empty rate + mean matches vs US/India). If it died: fix per CLAUDE.md, relaunch once.
A1. GPU top-k speed fix (cuts full blocking from ~24 h to ~4–5 h). Diagnosis: in `src/blocking.py`'s
    GPU path, building the dense query chunk on the CPU (`toarray()` + transpose copy, ~0.8 s/chunk)
    dominates; GPU spmm 0.05 s, transpose 0.29 s, topk 0.06 s → 2.7–5.5 ms/query instead of 0.8.
    Fix: send the query chunk to the GPU as SPARSE and densify/transpose there (or transpose Q once).
    Check real parquet column names first. Verify on ~20K India queries × 2 views: same top-k similarity
    values (ties may reorder), ms/query ≤ 1.2. Commit + push.
A2. Swap `src/pipeline_next.py` → `src/pipeline.py` (cascade: 77.5 → 5.4 cands/S1, OOF 0.9846 vs 0.9849).
    `python -m src.pipeline --help` must work. Commit + push.
A3. Launch v2 detached: `~/miniforge3/envs/aml/bin/tmux new -d -s run_v2 '<cmd>'`,
    `<cmd>` = `bash scripts/run_pipeline.sh v2 --n-jobs 8 --max-df 0.01 --train-s1 150000 --block-size 100000`
    + the GPU top-k flag (see --help) + `--stage-b-jobs 4`. Save it to `runs/v2/cmd.txt`. Watch up to
    25 min for the first block time; project the finish. If later than 08:00: kill and relaunch lighter,
    in order: `--train-s1 100000`; drop the 2 views with the lowest recall contribution (keep name_c3,
    name_w, addr_c3); `--block-size 50000` only if memory-bound. Log projection + decision.
A4. v2 finished → official validator (CLAUDE.md). Valid → `cp -r output output_v2`;
    `submissions/v2_matching_results.tsv` (if < 100 MB); `runs/v2/NOTES.md` (cands/S1, per-country empty
    rate + mean matches, runtime, peak RSS); commit + push. Crashed → CLAUDE.md crash procedure
    (lower `--stage-b-jobs` → `--n-jobs` → `--block-size`), relaunch with the A3 runtime check.

## TRACK B — GPU-native fast approach (work on it whenever v2 is RUNNING or DONE; slice data only)
Purpose: a pipeline that runs the full data in ≤ 2 h, uses the GPU for the heavy work, keeps
candidates ≈ 5/S1, and generalises to an unseen country.
B1. Slice: use/build `data_slice/` via `scripts/make_slice.py` (read --help; 8% train S1 + matches +
    8% other S2/S3; disjoint 4% train-derived holdout via `--train-as-test --lo`).
    Use `--data-dir data_slice --cache-dir cache_slice --out-dir output_slice`.
B2. Blocking = union of (a) KEY blocking: pandas merges within country on exact normalised name,
    phonetic key, first name token + postcode/pin, name token + city — cap bucket size; and (b) DENSE
    GPU blocking: MiniLM (`sentence-transformers/all-MiniLM-L6-v2`, Apache-2.0) fp16 embeddings of
    normalised "name | address", exact top-k by chunked matmul on the GPU per country (`src/dense.py`
    is a starting point). Then the existing cascade → existing stage-B features → LightGBM.
B3. Measure vs the current TF-IDF blocking on the same slice: pair recall per country, mean cands/S1,
    holdout F0.5, wall time per stage; project full-data runtime.
B4. Unseen-country proxy (the France risk): train on US only → evaluate on India holdout, and the
    reverse; report the drop for TF-IDF vs Track B. Prefer features that don't depend on country
    vocabulary; never hard-code US/India.
B5. If Track B has holdout F0.5 ≥ Track A's − 0.001 AND projected full runtime ≤ 2.5 h AND a smaller or
    equal cross-country drop → write `runs/night/B_READY.md` with the exact full-data command.
    Launch it as run `v3` (`--out-dir output_v3`) ONLY when v2 is finished (or dead) AND it can finish
    by 08:30. Validate, `submissions/v3_matching_results.tsv`, `runs/v3/NOTES.md`, commit + push.
    Never delete or overwrite `output_v2/`.

## Resource rules while v2 runs
- Slice/Track-B jobs only when MemAvailable > 4 GB, `--n-jobs 2`, GPU memory use < 5 GB, one at a time.
- The guard kills the NEWEST `src.pipeline` process first when MemAvailable < 1.5 GB — so your slice
  job dies before v2 does. If it does, wait for v2 to leave its heavy stage.

## Finish
At the end (or when all done): put "QUEUE DONE" + a morning summary at the TOP of NIGHT_LOG.md: which
submission file(s) exist and are validated, their OOF/holdout F0.5, cands/S1, per-country table
(France!), cross-country drop, and your recommendation (v2 or v3) with reasons. Commit + push.
Each NIGHT_LOG.md entry: time, track/item, command, outcome, metrics, next step.
