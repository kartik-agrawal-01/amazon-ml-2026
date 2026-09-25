#!/usr/bin/env bash
# Track B (night): after the blocking eval, build cross-country slices and run the Track-A pipeline on
#  (1) the mixed slice (train US+India -> holdout US+India), (2) train US only -> India holdout,
#  (3) train India only -> US holdout. One job at a time, --n-jobs 2 (v2 is running on the same box).
set -u
cd ~/amazon-ml-2026 || exit 1
source ~/miniforge3/etc/profile.d/conda.sh && conda activate aml
if [ "${SKIP_WAIT:-0}" != 1 ]; then until grep -q TB_EXIT runs/night/track_b_block_eval.txt 2>/dev/null; do sleep 20; done; fi
L=runs/night/xc_slices.txt
if [ ! -f data_xc_us/test_ground_truth_HIDDEN.tsv ]; then
  python scripts/make_slice.py --data-dir data --out data_xc_us --frac 0.08 --country US > $L 2>&1
  python scripts/make_slice.py --data-dir data --out data_xc_us --frac 0.04 --lo 0.5 --country India --train-as-test >> $L 2>&1
  python scripts/make_slice.py --data-dir data --out data_xc_in --frac 0.08 --country India >> $L 2>&1
  python scripts/make_slice.py --data-dir data --out data_xc_in --frac 0.04 --lo 0.5 --country US --train-as-test >> $L 2>&1
  echo SLICES_OK >> $L
fi
NJ="${NJ:-2}"
COMMON="--n-jobs $NJ --stage-b-jobs $NJ --topk-device cuda --max-df 0.01 --train-s1 40000 --block-size 50000 --folds 3 --views name_c3,name_w,addr_c3,full_w"
for job in "xc_us data_xc_us cache_xc_us output_xc_us" "slice_mix data_slice cache_slice output_slice" "xc_in data_xc_in cache_xc_in output_xc_in"; do
  set -- $job; name=$1; dd=$2; cd_=$3; od=$4
  [ -f runs/$name/report.json ] && continue
  mkdir -p runs/$name
  echo "python -m src.pipeline --data-dir $dd --cache-dir $cd_ --out-dir $od --run-name $name --test-gt $dd/test_ground_truth_HIDDEN.tsv $COMMON" > runs/$name/cmd.txt
  python -m src.pipeline --data-dir $dd --cache-dir $cd_ --out-dir $od --run-name $name --test-gt $dd/test_ground_truth_HIDDEN.tsv $COMMON > runs/$name/stdout.txt 2>&1
  echo "EXIT=$?" >> runs/$name/stdout.txt
done
echo CHAIN_DONE >> $L
