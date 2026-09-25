# CLAUDE.md — context for Claude Code running ON THE GPU BOX (`~/amazon-ml-2026`)

You are the on-box operator for our Amazon ML Challenge 2026 entry (business entity resolution,
macro-F0.5). The modelling decisions are made in a separate "HQ" chat; your job here is to RUN the
pipeline on the full data, keep it alive, fix crashes in the code when they are environment/memory
bugs, and push small results to GitHub so HQ can read them. Read `CONTEXT.md` and
`docs/BOX_RUNBOOK.md` first — CONTEXT.md is the team's ground truth.

## Hard rules
- Never modify, delete or move anything under `data/` (the dataset). Never `git add` data, caches,
  `output/`, models or anything > 20 MB. `output/`, `cache/`, `data/` are gitignored — keep it so.
- Never touch Unstop / the competition portal from this machine.
- No external data, APIs or geocoding in the pipeline (competition rule → disqualification).
- This is a SHARED machine (other users, ~10 GB of the RAM is usable for us, 16 GB GPU). Do not
  start more than one pipeline run at a time. Always run inside tmux (`~/miniforge3/envs/aml/bin/tmux`).
- Commit only: `runs/<name>/` (report.json, stdout.txt, commit.txt), `submissions/*.tsv` (final
  matching_results files, < 100 MB each), `logs/*.txt`, and code fixes. Push after every run.
- Do not change the modelling logic (features, views, decision rule, model) on your own initiative;
  fix crashes, memory issues, wrong paths, missing packages, GPU/CPU fallbacks, logging. If a
  modelling change seems necessary, write the reason to `runs/<name>/NOTES.md`, commit, and stop.

## Environment
- Ubuntu 24.04, user `kartik`, no sudo. Conda env: `source ~/miniforge3/etc/profile.d/conda.sh && conda activate aml`.
- 20 CPU threads shared with others; **use `--n-jobs 8`**. RAM 15–24 GB total (check `free -g`), keep
  MemAvailable > 1.5 GB (a kill-guard loop is described in the runbook). Swap 4 GB — if swapping,
  reduce `--n-jobs`, `--block-size` (100000 → 50000) or `--train-s1` (150000 → 100000).
- GPU: RTX 5060 Ti 16 GB, torch 2.11+cu128 (`python -c "import torch;print(torch.cuda.is_available())"`).
  The dense view (`--dense`) must keep its similarity chunks small (already handled in `src/dense.py`);
  if you see CUDA OOM, lower `sim_budget_bytes` in `topk_dense` or run without `--dense`.
- Data: `data/data_extracted/student_resource/dataset/{train,test}/*.tsv` (24M records). The pipeline
  finds files by name under `--data-dir data`.
- Office power can drop (box rebooted twice on 25 Sep). Caches in `cache/` are the resume points:
  a rerun with the same `--cache-dir` skips normalisation.

## Commands
```bash
cd ~/amazon-ml-2026 && git pull
bash scripts/gpu_check.sh                                  # once per session; appends to logs/gpu_box_env.txt
bash scripts/run_pipeline.sh smoke --n-jobs 8 --max-df 0.01 --train-s1 60000 --test-limit 60000 --block-size 60000 --folds 3
bash scripts/run_pipeline.sh v2 --n-jobs 8 --max-df 0.01 --train-s1 150000 --block-size 100000 [--dense]
python data/data_extracted/student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv \
    --test-dir data/data_extracted/student_resource/dataset/test
cp output/matching_results.tsv submissions/<run>_matching_results.tsv
git add runs/<run> submissions/<run>_matching_results.tsv logs && git commit -m "run <run>" && git push
```
`runs/<run>/stdout.txt` holds the log. Key lines: blocking recall table, `OOF (…): AUC=`,
`OOF macro F0.5 (chosen)`, the per-country test table, `test: … rows written`, `done`.

## What "good" looks like (from HQ's 8%-slice runs, sklearn model, 40K train S1)
- blocking pair recall ≈ 0.986 at k=10, ~64 candidates per S1 after pruning
- OOF macro-F0.5 ≈ 0.977, holdout ≈ 0.983; on the full data with LightGBM + 150K train S1 we expect higher.
- Test predicted-empty rate ≈ 5–6% for US/India. **France is unseen in training** — if France shows an
  empty rate > 15% or mean matches < 2.5, write it to NOTES.md (do not "fix" it yourself).

## If something crashes
1. Read the traceback in `runs/<run>/stdout.txt` and `dmesg | tail` (OOM kills show there).
2. MemoryError / OOM-killed / BrokenProcessPool → lower `--n-jobs`, then `--block-size`, then `--train-s1`;
   never raise them. Delete stale `cache/` only if the normalisation code changed (git log src/normalize.py).
3. Import errors → `pip install <pkg>` inside the env (allowed: sparse_dot_topn, pyarrow, joblib,
   lightgbm, sentence-transformers). Never install packages that fetch external business data.
4. Fix, commit the fix with a clear message, rerun. If the same crash repeats twice, stop and write
   `runs/<run>/NOTES.md` with the traceback + what you tried, commit, push.

## Reporting
After every finished run, append a short entry to `runs/<run>/NOTES.md`: command used, runtime, the key
lines above, anything odd. Commit + push. HQ reads the repo; it cannot see this machine.
