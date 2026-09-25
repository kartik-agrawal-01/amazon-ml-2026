#!/usr/bin/env bash
# Overnight driver for the box (see scripts/night/NIGHT_TASK.md + CLAUDE.md).
#   ~/miniforge3/envs/aml/bin/tmux new -d -s night 'bash ~/amazon-ml-2026/scripts/night/night_loop.sh'
# Env: END_AT=HH:MM (default 08:30)  MAX_SESSIONS (default 14)  CLAUDE_BIN  CLAUDE_MODEL (optional --model)
set -u
REPO="$HOME/amazon-ml-2026"; cd "$REPO" || exit 1
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate aml
export GIT_TERMINAL_PROMPT=0
CLAUDE_BIN="${CLAUDE_BIN:-$(command -v claude)}"
END_AT="${END_AT:-08:30}"; MAX_SESSIONS="${MAX_SESSIONS:-14}"
MODEL_ARGS=(); [ -n "${CLAUDE_MODEL:-}" ] && MODEL_ARGS=(--model "$CLAUDE_MODEL")
LOG=runs/night; mkdir -p "$LOG"
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG/driver.log"; }
[ -x "$CLAUDE_BIN" ] || { say "claude not found (set CLAUDE_BIN)"; exit 1; }
if [ ! -f "$LOG/end_at" ]; then
  e=$(date -d "$END_AT" +%s); [ "$e" -lt "$(date +%s)" ] && e=$(date -d "tomorrow $END_AT" +%s)
  echo "$e" > "$LOG/end_at"
fi
END=$(cat "$LOG/end_at")
say "driver start; stops at $(date -d @"$END" '+%F %T'); max sessions $MAX_SESSIONS; model ${CLAUDE_MODEL:-default}"

# memory guard: when MemAvailable < 1.5 GB, kill the NEWEST src.pipeline (slice jobs die before v2)
( while true; do
    a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    if [ "$a" -lt 1500 ]; then
      p=$(pgrep -n -u "$USER" -f src.pipeline)
      if [ -n "$p" ]; then kill "$p"; pkill -u "$USER" -P "$p" 2>/dev/null
        echo "$(date '+%F %T') GUARD killed newest src.pipeline pid $p ($(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-120)), MemAvailable=${a}MB" >> "$LOG/guard.log"; sleep 10; fi
    fi; sleep 3; done ) &
GUARD=$!; trap 'kill $GUARD 2>/dev/null' EXIT

n=$(ls "$LOG"/session_*.log 2>/dev/null | wc -l)
while [ "$(date +%s)" -lt "$END" ] && [ "$n" -lt "$MAX_SESSIONS" ]; do
  grep -q "QUEUE DONE" "$LOG/NIGHT_LOG.md" 2>/dev/null && { say "queue done - stopping"; break; }
  n=$((n+1)); ts=$(date +%m%d-%H%M)
  say "session $n start (pipelines running: $(pgrep -u "$USER" -fc src.pipeline))"
  timeout --kill-after=60 5400 "$CLAUDE_BIN" -p "$(cat scripts/night/NIGHT_TASK.md)

---
Session $n. Now: $(date). Driver stops at $(date -d @"$END"). This session must end within 85 minutes." \
    "${MODEL_ARGS[@]}" --dangerously-skip-permissions --max-turns 120 > "$LOG/session_$ts.log" 2>&1
  say "session $n exit $?"
  if grep -qiE "usage limit|rate limit|limit reached|resets at" "$LOG/session_$ts.log"; then
    say "Claude usage limit hit - sleeping 30 min"; sleep 1800
  fi
  sleep 1800   # next session in 30 min (long runs keep going in their own tmux sessions)
done
say "driver end"
