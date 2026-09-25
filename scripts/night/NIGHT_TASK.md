# Overnight autonomous task — Claude Code headless on the box

You are running UNATTENDED overnight on a shared Ubuntu box. Nobody will answer questions.
Never ask — decide, act, log. A driver script (scripts/night/night_loop.sh) restarts you for
each new session; your memory between sessions is `runs/night/NIGHT_LOG.md`.

## Read first, every session
1. `runs/night/NIGHT_LOG.md` — your log from earlier sessions tonight (create it if missing). Continue from it.
2. `CONTEXT.md` — team ground truth: problem, metric, rules, submission format.
3. `docs/BOX_RUNBOOK.md`, `python -m src.pipeline --help`, `runs/night/guard.log` (if a run died silently, look here first).

## Problem in brief
Business entity resolution: for every Source-1 (S1) entity output ALL matching S2/S3 ids.
Metric: macro F0.5 per S1 (singletons: empty prediction = 1.0). Outputs: `matching_results.tsv` +
`candidate_pairs.tsv` (the EXACT candidate set fed to the model; matches must be a subset).
**NEW OFFICIAL RULE (25 Sep): a SMALLER candidate set per S1 ranks HIGHER in the final evaluation,
beyond the leaderboard. Blocking must scale (no all-pairs).**
HQ's best so far (8% slice, 40K train S1, 91 features): holdout F0.5 0.9826, OOF 0.9768,
blocking pair recall 98.6%, ~64 candidates/S1. Decision rule: global threshold ≈0.80 + 1-to-1.
DEAD ENDS (do not retry): expected-F0.5 rule, isotonic calibration, per-country thresholds,
2nd-stage group model on p.

## Hard machine limits
- 15 GB RAM shared with other users → ~9 GB usable. A guard kills any `src.pipeline` process when
  MemAvailable < 1.5 GB. `--n-jobs 8` maximum.
- ONE experiment process at a time. Never start a run while `pgrep -f src.pipeline` finds one.
  Run experiments in the FOREGROUND and let them finish inside your session (stray processes are
  killed when your session ends).
- GPU: RTX 5060 Ti 16 GB, torch CUDA works — use it for embeddings if useful.
- No sudo. Stay inside ~/amazon-ml-2026. `data/data_extracted/` is read-only input.
- Power cuts happen: make long jobs resumable (cache intermediate artifacts to disk).
- Stay on git branch `night-run`. Never touch `main`, never `git push`.

## Directory hygiene (important — don't clobber the evening's results)
- Slice experiments: `--data-dir data_slice --cache-dir cache_slice --out-dir output_slice`.
- Full-data runs: `--out-dir output_night` (NEVER write to `output/` — it holds the evening submission).
- `cache/` belongs to full-data runs; only delete files there you created tonight or proved stale.

## Competition rules (breaking these = disqualification)
- No external data, no APIs, no geocoding, no internet lookups of entities. pip-installing libraries is fine.
- Pretrained models must be MIT or Apache-2.0 licensed and ≤ 8B parameters.
- Never edit `utils/validate_submission.py`. Never submit anything anywhere.

## Goals — strict priority
**P0. A FULL-DATA run that completes on this box within the memory limit** and produces validated
`output_night/matching_results.tsv` + `output_night/candidate_pairs.tsv`.
Known blocker: TRAIN normalisation is killed by the guard — the main process carries ~5 GB of
test data into train normalisation, and freed Python memory is not returned to the OS.
Likely fixes: normalise each split in a separate fresh process that only writes
`cache/<split>_norm.pkl` (or parquet) and exits; compact dtypes (category / int32 / float32);
chunked processing; free big tables before spawning workers. Until P0 works, P0 is the only goal.
Note: `cache/test_norm.pkl` may be stale from older normalize.py code — regenerate if in doubt.
**P1. Improve slice holdout F0.5 while NOT increasing (ideally reducing) mean candidates per S1.**
**P2. Speed and peak memory of the full run.**

## Workflow for each session
1. Read NIGHT_LOG.md; record any result left pending by the previous session.
2. Pick ONE experiment (smallest change, best expected gain). Write the plan in NIGHT_LOG.md BEFORE running.
3. Evaluate on the slice. Build it once with `scripts/make_slice.py` (read its --help; HQ used 8% of
   train S1 + all their matches + 8% of other S2/S3, and a disjoint 4% train-derived holdout via
   `--train-as-test --lo`) into `data_slice/`. In session 1, baseline the CURRENT code on this slice
   so every comparison is apples to apples.
4. KEEP a change only if holdout F0.5 improves by ≥ 0.0005, OR mean candidates/S1 drops ≥ 10% with
   an F0.5 loss ≤ 0.0005. Otherwise revert it (`git checkout -- <files>`).
5. Commit kept changes to `night-run` with a clear message.
6. Keep `runs/night/BEST_CMD.sh` = the exact full-data command for the best kept code
   (with `--n-jobs 8 --out-dir output_night`, followed by the official validator). The driver runs it
   automatically at the deadline. It must be correct bash, runnable from the repo root.
7. Append to NIGHT_LOG.md: time, change, slice holdout F0.5, OOF F0.5, mean candidates/S1, pair recall,
   runtime, peak RAM, KEPT/REVERTED.
8. End the session after about one experiment. Respect the session time cap given in the prompt.

## Ideas with real signal (from HQ)
- Candidate pruning: tighter top-k per view, stage-A score floor, per-S1 cap tuned on OOF → far fewer
  candidates/S1 at ~same F0.5 (mean true matches per S1 is only 3.46).
- Empty-address candidates are ~45% of false negatives → features on how many S1 share the exact name
  within the country × how many candidates compete.
- Dense MiniLM view on GPU (`src/dense.py`, untested on real data) — as a feature / re-ranker or to
  REPLACE weak lexical candidates, never simply to add more.
- LightGBM (auto-picked if installed) + more train S1 (150K–300K) — cheap accuracy.
- S2↔S3 consistency features.
