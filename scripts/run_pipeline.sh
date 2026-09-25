#!/usr/bin/env bash
# Run the full pipeline on the box (CPU is enough). Start it inside tmux for long runs.
# Usage:  bash scripts/run_pipeline.sh <run_name> [extra args for src.pipeline]
#   e.g.  bash scripts/run_pipeline.sh v1_baseline --k 10
# Small artefacts (report.json, matching_results.tsv, stdout.txt, commit.txt) land in runs/<run_name>/
# -> commit + push that folder so HQ (Claude) can read the results from the laptop clone.
set -euo pipefail
RUN="${1:?usage: bash scripts/run_pipeline.sh <run_name> [args]}"
shift || true
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate aml
mkdir -p "runs/$RUN"
git rev-parse --short HEAD > "runs/$RUN/commit.txt" 2>/dev/null || true
python -m src.pipeline --data-dir data --out-dir output --run-name "$RUN" "$@" 2>&1 | tee "runs/$RUN/stdout.txt"
echo "== done. Push results:  git add runs/$RUN && git commit -m 'run $RUN' && git push"
