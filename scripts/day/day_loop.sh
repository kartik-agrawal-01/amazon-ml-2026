#!/usr/bin/env bash
# Improvement loop driver for the box (see scripts/day/DAY_TASK.md + QUEUE.md).
#   ~/miniforge3/envs/aml/bin/tmux new -d -s day 'bash ~/amazon-ml-2026/scripts/day/day_loop.sh'
# Env: END_AT (default "2026-09-27 20:00")  MAX_SESSIONS (default 40)  CLAUDE_BIN  CLAUDE_MODEL
# Stop gracefully: touch ~/amazon-ml-2026/runs/day/STOP
set -u
REPO="$HOME/amazon-ml-2026"; cd "$REPO" || exit 1
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate aml
export GIT_TERMINAL_PROMPT=0
CLAUDE_BIN="${CLAUDE_BIN:-$(command -v claude)}"
END_AT="${END_AT:-2026-09-27 20:00}"; MAX_SESSIONS="${MAX_SESSIONS:-40}"
MODEL_ARGS=(); [ -n "${CLAUDE_MODEL:-}" ] && MODEL_ARGS=(--model "$CLAUDE_MODEL")
LOG=runs/day; mkdir -p "$LOG"
PAT='^python[0-9.]* (-m src[.]|scripts/)'
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG/driver.log"; }
[ -x "$CLAUDE_BIN" ] || { say "claude not found (set CLAUDE_BIN)"; exit 1; }
END=$(date -d "$END_AT" +%s)
say "driver start; ends $(date -d @"$END" '+%F %T'); max sessions $MAX_SESSIONS; model ${CLAUDE_MODEL:-default}"

# memory guard: MemAvailable < 1.5 GB -> kill the NEWEST heavy python job (and its children)
( while true; do
    a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    if [ "$a" -lt 1500 ]; then
      p=$(pgrep -n -u "$USER" -f "$PAT")
      if [ -n "$p" ]; then pkill -u "$USER" -P "$p" 2>/dev/null; kill "$p"
        echo "$(date '+%F %T') GUARD killed pid $p ($(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-120)) MemAvailable=${a}MB" >> "$LOG/guard.log"; sleep 10; fi
    fi; sleep 3; done ) &
GUARD=$!; trap 'kill $GUARD 2>/dev/null' EXIT

n=0; quick=0
while [ "$(date +%s)" -lt "$END" ] && [ "$n" -lt "$MAX_SESSIONS" ]; do
  [ -f "$LOG/STOP" ] && { say "STOP file found - stopping"; break; }
  # pull first so DAY_TASK.md edits pushed from the laptop reach THIS session's prompt (it is read below)
  timeout 120 git pull --no-edit -q >> "$LOG/driver.log" 2>&1 || say "git pull failed (the session retries)"
  n=$((n+1)); ts=$(date +%m%d-%H%M); t0=$(date +%s)
  say "session $n start (heavy jobs running: $(pgrep -u "$USER" -fc "$PAT"))"
  timeout --kill-after=60 5100 "$CLAUDE_BIN" -p "$(cat scripts/day/DAY_TASK.md)

---
Session $n. Now: $(date). Loop ends $(date -d @"$END"). End this session within 80 minutes." \
    "${MODEL_ARGS[@]}" --dangerously-skip-permissions --max-turns 150 > "$LOG/session_$ts.log" 2>&1
  rc=$?; dur=$(( $(date +%s) - t0 ))
  say "session $n exit $rc after $((dur/60)) min"
  if grep -qiE "usage limit|rate limit|limit reached|resets at" "$LOG/session_$ts.log"; then
    say "Claude usage limit hit - sleeping 30 min"; sleep 1800; continue
  fi
  if [ "$dur" -lt 120 ]; then quick=$((quick+1)); say "session ended in < 2 min ($quick in a row)"
    [ "$quick" -ge 3 ] && { say "3 instant failures - stopping (check session logs)"; break; }
    sleep 600; continue
  fi
  quick=0; sleep 60
done
say "driver end"
