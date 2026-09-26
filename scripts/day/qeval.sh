#!/usr/bin/env bash
# Q evaluation on the hidden-holdout slices (day loop objective).
# Usage: bash scripts/day/qeval.sh <tag> <cache_suffix> "<runs>" [extra src.pipeline args...]
#   runs: subset of "xc_us slice_mix xc_in" (screen: "xc_us slice_mix"); cache_suffix: "" = the original stores
#   (v2 normalisation), "_n3" = stores rebuilt with the France-fixed normalisation (built on first use).
# Outputs: runs/day_<tag>_<run>/{report.json,stdout.txt,cmd.txt}, output_day_<tag>_<run>/ ; then python scripts/day/q_table.py <tag>
set -u
TAG=${1:?tag}; SUF=${2-}; RUNS=${3:-"xc_us slice_mix"}; shift 3 || shift $#
cd ~/amazon-ml-2026 || exit 1
source ~/miniforge3/etc/profile.d/conda.sh && conda activate aml
COMMON="--n-jobs ${QE_NJOBS:-8} --stage-b-jobs 4 --topk-device cuda --max-df 0.01 --train-s1 40000 --block-size 50000 --folds 3 --views name_c3,name_w,addr_c3,full_w"
for run in $RUNS; do
  case $run in xc_us) dd=data_xc_us; cd_=cache_xc_us;; slice_mix) dd=data_slice; cd_=cache_slice;; xc_in) dd=data_xc_in; cd_=cache_xc_in;; esac
  name=day_${TAG}_$run; mkdir -p runs/$name
  cmd="python -m src.pipeline --data-dir $dd --cache-dir ${cd_}${SUF} --out-dir output_$name --run-name $name --test-gt $dd/test_ground_truth_HIDDEN.tsv $COMMON $*"
  echo "$cmd" > runs/$name/cmd.txt; git rev-parse --short HEAD > runs/$name/commit.txt
  $cmd > runs/$name/stdout.txt 2>&1; echo "EXIT=$?" >> runs/$name/stdout.txt
done
python scripts/day/q_table.py $TAG
echo QEVAL_DONE
