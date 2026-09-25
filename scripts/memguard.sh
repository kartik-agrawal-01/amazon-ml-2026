#!/usr/bin/env bash
# Kill-guard for the shared box: stop the pipeline before the kernel OOM-killer picks a victim.
# Usage: bash scripts/memguard.sh <run_name> [min_available_mb=1500]   (run in its own tmux window)
RUN="${1:?usage: bash scripts/memguard.sh <run_name> [min_mb]}"; MIN="${2:-1500}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"; LOG="$REPO/runs/$RUN/watchdog.log"; mkdir -p "$REPO/runs/$RUN"
echo "$(date) memguard started: kill src.pipeline if MemAvailable < ${MIN}MB" >> "$LOG"
PEAK=0
while true; do
  AV=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1024 ))
  USED=$(( $(free -m | awk '/^Mem:/{print $2}') - AV )); [ "$USED" -gt "$PEAK" ] && PEAK=$USED
  if [ "$AV" -lt "$MIN" ] && pgrep -f "src.pipeline" > /dev/null; then
    echo "$(date) KILLED run: MemAvailable=${AV}MB (peak used ${PEAK}MB)" >> "$LOG"
    pkill -f "src.pipeline"; sleep 5; pkill -9 -f "src.pipeline"
    # spawned worker processes survive their parent's death and keep RAM + the tee pipe open -> kill them too
    sleep 2; pkill -u "$USER" -f "spawn_mai[n]"; sleep 3; pkill -9 -u "$USER" -f "spawn_mai[n]"
  fi
  sleep 3
done
