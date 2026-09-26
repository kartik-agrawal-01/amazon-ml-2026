#!/usr/bin/env bash
# Every 5 s: time, MemAvailable MB, load1, max coretemp (C), top CPU process (for the reboot diagnosis). fsync'd.
L=${1:-$HOME/amazon-ml-2026/runs/day/sysmon.log}
while true; do
  m=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo); l=$(cut -d' ' -f1 /proc/loadavg)
  t=$(cat /sys/class/hwmon/hwmon*/temp*_input 2>/dev/null | sort -n | tail -1); t=$((t/1000))
  p=$(ps -eo pcpu,user,comm --sort=-pcpu --no-headers | head -2 | tr -s ' ' | paste -sd'|')
  echo "$(date '+%F %T') mem=${m} load=${l} tmax=${t} top=${p}" >> "$L"; sync "$L"; sleep 5
done
