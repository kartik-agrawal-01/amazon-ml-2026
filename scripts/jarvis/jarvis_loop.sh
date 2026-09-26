#!/usr/bin/env bash
# Driver for the Jarvis lane (JarvisLabs RTX PRO 6000). Mirrors scripts/day/day_loop.sh.
#   tmux new -d -s jarvis 'bash /home/amazon-ml-2026/scripts/jarvis/jarvis_loop.sh'
# Stop gracefully: touch /home/amazon-ml-2026/runs/jarvis/STOP
# The brief is read from origin/main each session, so edits pushed from the laptop apply next session.
set -u
export TZ=Asia/Kolkata GIT_TERMINAL_PROMPT=0
REPO=/home/amazon-ml-2026; cd "$REPO" || exit 1
[ -f /home/venv/bin/activate ] && source /home/venv/bin/activate
CLAUDE_BIN="${CLAUDE_BIN:-$(command -v claude || echo /root/.local/bin/claude)}"
END_AT="${END_AT:-2026-09-27 20:00 +0530}"; MAX_SESSIONS="${MAX_SESSIONS:-40}"
LOG=runs/jarvis; mkdir -p "$LOG"
PAT='^python[0-9.]* (-m src[.]|scripts/)'
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG/driver.log"; }
[ -x "$CLAUDE_BIN" ] || { say "claude not found (set CLAUDE_BIN)"; exit 1; }
END=$(date -d "$END_AT" +%s)
say "driver start; ends $(date -d @"$END" '+%F %T'); max sessions $MAX_SESSIONS"

# memory guard (160 GB box): MemAvailable < 12 GB -> kill the NEWEST heavy python job
( while true; do
    a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    if [ "$a" -lt 12000 ]; then
      p=$(pgrep -n -f "$PAT")
      if [ -n "$p" ]; then pkill -P "$p" 2>/dev/null; kill "$p"
        echo "$(date '+%F %T') GUARD killed pid $p ($(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-120)) MemAvailable=${a}MB" >> "$LOG/guard.log"; sleep 10; fi
    fi; sleep 3; done ) &
GUARD=$!; trap 'kill $GUARD 2>/dev/null' EXIT

n=0; quick=0
while [ "$(date +%s)" -lt "$END" ] && [ "$n" -lt "$MAX_SESSIONS" ]; do
  [ -f "$LOG/STOP" ] && { say "STOP file found - stopping"; break; }
  timeout 120 git fetch -q origin >> "$LOG/driver.log" 2>&1 || say "git fetch failed (the session retries)"
  TASK=$(git show origin/main:scripts/jarvis/JARVIS_TASK.md 2>/dev/null || cat scripts/jarvis/JARVIS_TASK.md)
  n=$((n+1)); ts=$(date +%m%d-%H%M); t0=$(date +%s)
  say "session $n start (heavy jobs running: $(pgrep -fc "$PAT"))"
  # root: --dangerously-skip-permissions is refused, so allow the tools explicitly instead
  timeout --kill-after=60 5100 "$CLAUDE_BIN" -p "$TASK

---
Session $n. Now: $(date). Lane ends $(date -d @"$END"). End this session within 80 minutes." \
    --allowedTools "Bash,Read,Edit,Write,Glob,Grep,TodoWrite" --max-turns 150 > "$LOG/session_$ts.log" 2>&1
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
