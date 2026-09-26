#!/usr/bin/env bash
# Box v3 (Jarvis fallback) = champion recipe (SCOREBOARD row 6) on the FULL data via the fast lane:
#   HQ France fixes (HEAD normalize/features -> new store cache_n3; never touches cache/ or output_v2/)
#   + --key-rules 0.96 + --thr-adapt --thr-adapt-floor 0.30 + global one-to-one (HEAD default),
#   candidates = v2's cached unions (cands_v2, built by gate A; doc ids remapped to the new store), cosines recomputed.
# Run ONLY after gate B passed (runs/day/gateB.log: GATE_B_EXIT=0 and compare_matches >= 99.9%).
# Usage: tmux new -d -s v3 'bash scripts/day/v3.sh > runs/day/v3.log 2>&1'
# If OOM in the key-rule calibration: drop --key-rules (log it) - the rest of the recipe stays.
set -u
R=$HOME/amazon-ml-2026
cd "$R" || exit 1
source ~/miniforge3/etc/profile.d/conda.sh && conda activate aml
export AML_GPU_HOST_DENSIFY=${AML_GPU_HOST_DENSIFY:-1} AML_GPU_DUTY=${AML_GPU_DUTY:-0.3}
mkdir -p runs/v3
cmd="python -m src.pipeline --data-dir data --cache-dir cache_n3 --out-dir output_v3 --run-name v3 --n-jobs 8 --stage-b-jobs 3
 --topk-device cuda --max-df 0.01 --train-s1 150000 --block-size 100000 --views name_c3,name_w,addr_c3,full_w
 --cand-cache cands_v2 --vec-cache cache_n3/vecs.joblib --feat-cache feats_v3 --save-probs
 --key-rules 0.96 --thr-adapt --thr-adapt-floor 0.30"
echo "$cmd" > runs/v3/cmd.txt; git rev-parse --short HEAD > runs/v3/commit.txt
/usr/bin/time -v $cmd > runs/v3/stdout.txt 2>&1; rc=$?; echo "EXIT=$rc" >> runs/v3/stdout.txt
[ $rc -eq 0 ] || exit $rc
python data/data_extracted/student_resource/utils/validate_submission.py --matching output_v3/matching_results.tsv \
  --candidate output_v3/candidate_pairs.tsv --test-dir data/data_extracted/student_resource/dataset/test > runs/v3/validate.txt 2>&1
echo "VALIDATE_EXIT=$?" >> runs/v3/validate.txt
echo V3_DONE
