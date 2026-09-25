#!/usr/bin/env bash
# Overnight driver for the box. Start it inside tmux, from the repo root:
#   ~/miniforge3/envs/aml/bin/tmux new -d -s night 'bash scripts/night/night_loop.sh'
# Env knobs: FINAL_START=HH:MM (last moment to start the final full run, default 02:30)
#            MAX_SESSIONS (default 10, caps Claude usage)   CLAUDE_BIN (path to claude)
# Resumable: after a reboot it continues with the same deadline/branch; exits if the final run is done.
set -u
REPO="$HOME/amazon-ml-2026"; cd "$REPO" || exit 1
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate aml
CLAUDE_BIN="${CLAUDE_BIN:-$(command -v claude)}"
FINAL_START="${FINAL_START:-02:30}"
MAX_SESSIONS="${MAX_SESSIONS:-10}"
LOG=runs/night; mkdir -p "$LOG"
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG/driver.log"; }

[ -f "$LOG/FINAL_DONE" ] && { say "final run already done - exiting"; exit 0; }
[ -x "$CLAUDE_BIN" ] || { say "claude not found (set CLAUDE_BIN)"; exit 1; }

# fixed deadline, remembered across reboots
if [ ! -f "$LOG/deadline" ]; then
  d=$(date -d "$FINAL_START" +%s)
  [ "$d" -lt "$(date +%s)" ] && d=$(date -d "tomorrow $FINAL_START" +%s)
  echo "$d" > "$LOG/deadline"
fi
DEADLINE=$(cat "$LOG/deadline")
say "driver start; final full run at $(date -d @"$DEADLINE" '+%F %T'); max sessions $MAX_SESSIONS"

# memory guard (kills only src.pipeline processes)
( while true; do
    a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    if [ "$a" -lt 1500 ] && pgrep -u "$USER" -f src.pipeline >/dev/null; then
      pkill -u "$USER" -f src.pipeline; echo "$(date '+%F %T') GUARD killed src.pipeline, MemAvailable=${a}MB" >> "$LOG/guard.log"
    fi; sleep 3; done ) &
GUARD=$!; trap 'kill $GUARD 2>/dev/null' EXIT

# never switch code under a running pipeline (spawn workers re-import src/)
while pgrep -u "$USER" -f src.pipeline >/dev/null; do say "waiting for a running pipeline to finish"; sleep 120; done

git rev-parse --verify night-run >/dev/null 2>&1 && git checkout night-run || git checkout -b night-run
say "on branch $(git branch --show-current) @ $(git rev-parse --short HEAD)"

# fallback final command so there is always something to run
if [ ! -f "$LOG/BEST_CMD.sh" ]; then cat > "$LOG/BEST_CMD.sh" << 'CMD'
bash scripts/run_pipeline.sh night_final --n-jobs 8 --max-df 0.01 --train-s1 150000 --block-size 100000 --out-dir output_night && \
python data/data_extracted/student_resource/utils/validate_submission.py --matching output_night/matching_results.tsv --candidate output_night/candidate_pairs.tsv --test-dir data/data_extracted/student_resource/dataset/test
CMD
fi

n=$(ls "$LOG"/session_*.log 2>/dev/null | wc -l)
while [ "$(date +%s)" -lt "$DEADLINE" ] && [ "$n" -lt "$MAX_SESSIONS" ]; do
  left=$(( DEADLINE - $(date +%s) )); cap=$(( left < 7200 ? left : 7200 ))
  [ "$cap" -lt 900 ] && break
  n=$((n+1)); ts=$(date +%m%d-%H%M)
  say "session $n start (cap $((cap/60)) min)"
  timeout --kill-after=60 "$cap" "$CLAUDE_BIN" -p "$(cat scripts/night/NIGHT_TASK.md)

---
Session $n of tonight. Now: $(date). The final full-data run starts at $(date -d @"$DEADLINE"). This session must end within $((cap/60)) minutes." \
    --dangerously-skip-permissions --max-turns 80 > "$LOG/session_$ts.log" 2>&1
  say "session $n exit $?"
  pkill -u "$USER" -f src.pipeline 2>/dev/null
  if grep -qiE "usage limit|rate limit|limit reached|resets at" "$LOG/session_$ts.log"; then
    say "Claude usage limit hit - sleeping 30 min"; sleep 1800
  fi
  sleep 30
done

say "FINAL full run start"; pkill -u "$USER" -f src.pipeline 2>/dev/null; sleep 5
bash "$LOG/BEST_CMD.sh" > "$LOG/final_run.log" 2>&1; rc=$?
say "FINAL full run exit $rc (log: $LOG/final_run.log)"; date > "$LOG/FINAL_DONE"
