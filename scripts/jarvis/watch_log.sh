#!/bin/bash
# usage: watch_log.sh <log> <pgrep pattern>: print new log lines, exit when the process is gone
log=$1; pat=$2; n=$(wc -l < "$log")
while true; do
  m=$(wc -l < "$log")
  if [ "$m" -gt "$n" ]; then tail -n $((m-n)) "$log" | cut -c1-300; n=$m; fi
  pgrep -f "$pat" >/dev/null || { echo "PROCESS GONE: $pat"; tail -5 "$log"; exit 0; }
  sleep 20
done
