# v1 (commit 3c0932a) — KILLED, no results

- Command: `bash scripts/run_pipeline.sh v1 --max-df 0.01 --train-s1 300000 --block-size 100000` (no `--n-jobs`, so 20 workers)
- Loaded the test normalisation cache, then started normalising train (12.5M rows). A memory kill-guard stopped it
  at 17:43 IST with MemAvailable=1057 MB (see watchdog.log). Old fork-based workers; code has since moved to spawned workers.
- The `cache/test_norm.pkl` it produced (16:11) predates the normalize.py change in bf65a35 -> deleted as stale before the next run.
