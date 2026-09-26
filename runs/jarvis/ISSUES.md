# Jarvis ISSUES (fixes to main's code made on the jarvis branch; the box ports them)

## 1. src/pipeline.py: reverse blocking pairs are recomputed on every run (perf, not a bug)
- Symptom: at full density `--reverse-k 3` costs ~3-8 min per view per country per split on 14 CPU threads
  (6 views -> ~1 h per split); a rerun after a pause or a model-only change (--load-model) recomputes all of it.
- Fix: `CountryContext.reverse(..., cache=CandCache)` saves the raw all-S1 x docs reverse pairs to
  `<cand-cache>/<tag>__rev<r>_<views>_<n_S1>.parquet` and loads it on later runs (both train and test calls pass `cache`).
  Only active with --cand-cache. Synthetic run twice: identical pair counts and matching_results.
- Also: with `--topk-device cuda` the reverse top-k goes to the GPU dense-block path, ~5x slower than CPU
  sparse_dot_topn on the A30 (4.13M docs x 883K S1, full_w: 784 s GPU vs ~160 s CPU/14 thr). Jarvis runs `--topk-device cpu`.
