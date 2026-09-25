#!/usr/bin/env bash
# Run ONE Track-B cross-country slice job while v2 runs: CPU top-k, 2 jobs, and a private kill-switch that
# stops it when MemAvailable < ${MINAVAIL:-2300} MB (before the driver guard's 1.5 GB) or after ${STOP_AT:-04:30}.
# Usage: bash scripts/night/xc_guarded.sh <name> <data_dir>     e.g.  xc_us data_xc_us
set -u
name=$1; dd=$2; cd_=cache_$name; od=output_$name
cd ~/amazon-ml-2026 || exit 1
source ~/miniforge3/etc/profile.d/conda.sh && conda activate aml
mkdir -p runs/$name
COMMON="--n-jobs 2 --stage-b-jobs 2 --topk-device cpu --max-df 0.01 --train-s1 40000 --block-size 50000 --folds 3 --views name_c3,name_w,addr_c3,full_w"
CMD="python -m src.pipeline --data-dir $dd --cache-dir $cd_ --out-dir $od --run-name $name --test-gt $dd/test_ground_truth_HIDDEN.tsv $COMMON"
echo "$CMD" > runs/$name/cmd.txt
$CMD > runs/$name/stdout.txt 2>&1 &
P=$!
echo "$(date '+%F %T') $name started pid $P" >> runs/night/xc_guarded.log
while kill -0 $P 2>/dev/null; do
  a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  if [ "$a" -lt "${MINAVAIL:-2300}" ] || [[ "$(date +%H:%M)" > "${STOP_AT:-04:30}" ]]; then
    echo "$(date '+%F %T') KILLSWITCH $name pid $P MemAvailable=${a}MB" >> runs/night/xc_guarded.log
    pkill -P $P; kill $P; sleep 5; pkill -9 -P $P 2>/dev/null; kill -9 $P 2>/dev/null; echo "KILLSWITCH avail=${a}MB" >> runs/$name/stdout.txt; break
  fi
  sleep 3
done
wait $P; echo "EXIT=$?" >> runs/$name/stdout.txt
echo "$(date '+%F %T') $name ended: $(tail -1 runs/$name/stdout.txt)" >> runs/night/xc_guarded.log
