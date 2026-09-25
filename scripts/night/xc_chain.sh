#!/usr/bin/env bash
# Sequential guarded B4 runs while v2 is in its India test blocks: waits for the running xc_us, then xc_in,
# then slice_mix (baseline, reuses cache_slice via symlink). Each via xc_guarded.sh (CPU top-k, 2 jobs, killswitch).
set -u
cd ~/amazon-ml-2026 || exit 1
export STOP_AT="${STOP_AT:-04:05}"
while pgrep -f '^python -m src.pipeline --data-dir data_xc_us' >/dev/null; do sleep 20; done
[ -f runs/xc_in/report.json ] || bash scripts/night/xc_guarded.sh xc_in data_xc_in
[ -e cache_slice_mix ] || ln -s cache_slice cache_slice_mix
[ -f runs/slice_mix/report.json ] || bash scripts/night/xc_guarded.sh slice_mix data_slice
echo "$(date '+%F %T') xc_chain done" >> runs/night/xc_guarded.log
