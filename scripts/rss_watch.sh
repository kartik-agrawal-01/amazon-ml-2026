#!/usr/bin/env bash
# Sample the pipeline's memory every 5 s while it runs: main-process RSS, main+workers RSS, MemAvailable.
# Usage: bash scripts/rss_watch.sh <run_name>      -> runs/<run_name>/rss.log (peaks in the last line)
RUN="${1:?usage: bash scripts/rss_watch.sh <run_name>}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"; LOG="$REPO/runs/$RUN/rss.log"; mkdir -p "$REPO/runs/$RUN"
sleep 5; PEAK_MAIN=0; PEAK_ALL=0; MIN_AV=999999
echo "$(date '+%F %T') rss_watch started" >> "$LOG"
while true; do
  MAIN=$(pgrep -f "^python -m src.pipeline .*--run-name $RUN( |$)" | head -1)
  [ -z "$MAIN" ] && { echo "$(date '+%F %T') pipeline gone. PEAK main=${PEAK_MAIN}MB all=${PEAK_ALL}MB min MemAvailable=${MIN_AV}MB" >> "$LOG"; exit 0; }
  R_MAIN=$(( $(ps -o rss= -p "$MAIN" 2>/dev/null | awk '{s+=$1} END{print s+0}') / 1024 ))
  R_ALL=$(( $(ps -o rss= -p "$MAIN" --ppid "$MAIN" 2>/dev/null | awk '{s+=$1} END{print s+0}') / 1024 ))
  AV=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1024 ))
  [ "$R_MAIN" -gt "$PEAK_MAIN" ] && PEAK_MAIN=$R_MAIN; [ "$R_ALL" -gt "$PEAK_ALL" ] && PEAK_ALL=$R_ALL; [ "$AV" -lt "$MIN_AV" ] && MIN_AV=$AV
  echo "$(date '+%T') main=${R_MAIN}MB all=${R_ALL}MB avail=${AV}MB peak_main=${PEAK_MAIN}MB peak_all=${PEAK_ALL}MB" >> "$LOG"
  sleep 5
done
