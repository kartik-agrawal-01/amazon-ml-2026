#!/usr/bin/env bash
# P0 correctness gate for the fast lane (see scripts/day/DAY_TASK.md P0-b).
# Worktree ~/aml_gate = HEAD with v2's normalize.py/features.py; reads v2's store (cache/) read-only.
#  A) --cands-only: build the candidate cache cands_v2/ (v2 blocking: train 150K sample + full test)  [GPU top-k]
#  B) rescore test from the cache with v2's model -> ~/aml_gate_out -> compare with output_v2/matching_results.tsv
set -u
R=$HOME/amazon-ml-2026
G=$HOME/aml_gate
source ~/miniforge3/etc/profile.d/conda.sh && conda activate aml
cd "$G" || exit 1
C="--data-dir $R/data --cache-dir $R/cache --n-jobs 8 --stage-b-jobs 3 --topk-device cuda --max-df 0.01 --train-s1 150000
   --block-size 100000 --views name_c3,name_w,addr_c3,full_w --cand-cache $R/cands_v2 --vec-cache $R/cands_v2/vecs.joblib"
STEP=${1:-all}
if [ "$STEP" = all ] || [ "$STEP" = A ]; then
  python -m src.pipeline $C --out-dir $HOME/aml_gate_out --cands-only; echo "GATE_A_EXIT=$?"
fi
if [ "$STEP" = all ] || [ "$STEP" = B ]; then
  python -m src.pipeline $C --out-dir $HOME/aml_gate_out --load-model $R/output_v2_train/model.joblib --save-probs; echo "GATE_B_EXIT=$?"
  python $R/scripts/day/compare_matches.py $R/output_v2/matching_results.tsv $HOME/aml_gate_out/matching_results.tsv \
     $R/output_v2/candidate_pairs.tsv $HOME/aml_gate_out/candidate_pairs.tsv
fi
