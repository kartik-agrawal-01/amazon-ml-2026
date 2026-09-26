#!/bin/bash
# Waits for the jv1 train pass (tmux jv1) to end, then: rename old-code reverse caches ('-' -> '+'), run the
# density-matched gate (runs/jarvis/jv1_train.md) and, on GO, start the jv1 test pass (runs/jarvis/jv1_test.cmd,
# log runs/jarvis/jv1_test.log). On STOP it only writes the gate file. Log: runs/jarvis/chain_jv1.log
source /home/venv/bin/activate
cd /home/amazon-ml-2026
while ! grep -q "skip-test: done" runs/jarvis/jv1.log; do
  pgrep -f "src.pipeline --data-dir data --cache-dir /home/cache_jv/store --out-dir /home/out_jv/jv1 " >/dev/null \
    || { echo "jv1 train gone without 'skip-test: done' $(date -u)"; exit 1; }
  sleep 60
done
echo "train pass done $(date -u)"
cd /home/cache_jv/j1_cand && for f in *__rev3_*-*.parquet; do [ -e "$f" ] && mv -v "$f" "${f//-/+}"; done
cd /home/amazon-ml-2026
python scripts/jarvis/jv1_gate.py --out /home/out_jv/jv1 --store /home/cache_jv/store --md runs/jarvis/jv1_train.md \
  || { echo "gate script failed $(date -u)"; exit 1; }
if grep -q "GO\*\*\|: GO" runs/jarvis/jv1_train.md && ! grep -q "STOP" runs/jarvis/jv1_train.md; then
  echo "gate GO -> test pass $(date -u)"
  bash -c "$(cat runs/jarvis/jv1_test.cmd)" > runs/jarvis/jv1_test.log 2>&1
  echo "test pass exit $? $(date -u)"
else
  echo "gate STOP $(date -u)"
fi
