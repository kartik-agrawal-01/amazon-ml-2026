#!/usr/bin/env bash
# Session-14 chain: replaces chain5 + chain7 (their tmux sessions were killed, they were only in wait loops) so that the
# post-v3 fast-lane work runs BEFORE the slice screens (NEXT 4, session 13), without the chain8-exit race that would
# let chain5 start n4ph the moment chain8 exits. Order after chain8 exits:
#   v3 done (runs/day/chain/v3.done):
#     v3rs   rescore sanity: src.rescore --load-model output_v3/model.joblib == output_v3 (compare_matches, >= 99.9%)
#     cap    scripts/day/model_capacity.py feats_v3 --variants base,big,deep,seed3 -> runs/day/model_capacity.md
#     v4     (only if v3rs passed) winner of pick_variant.py (>= +0.0015 OOF in every country) -> src.rescore --variant <w> -> output_v4
#            -> validator -> submissions/v4_matching_results.tsv (a session writes NOTES/SCOREBOARD; humans upload)
#   then the slice screens that were queued in chain5/chain7: n4ph, n3r (placeholder removed first), n3ka_s7.
# Usage: tmux new -d -s chain9 'bash scripts/day/chain9.sh > runs/day/chain9.log 2>&1'
set -u
R=$HOME/amazon-ml-2026
cd "$R" || exit 1
source ~/miniforge3/etc/profile.d/conda.sh && conda activate aml
while pgrep -xf 'bash scripts/day/chain[48].sh' > /dev/null; do sleep 30; done
echo "[chain9] $(date '+%F %T') chain4/chain8 exited"
step() {
  local n=$1; shift
  if [ -f runs/day/chain/$n.done ]; then echo "[chain] $n already done"; return 0; fi
  echo "[chain] $(date '+%F %T') start $n"
  "$@"; local rc=$?
  echo "[chain] $(date '+%F %T') end $n rc=$rc"
  [ $rc -eq 0 ] && touch runs/day/chain/$n.done
  return $rc
}
v3rs() {
  rm -rf output_v3_rs
  python -m src.rescore --feat-cache feats_v3 --cache-dir cache_n3 --out-dir output_v3_rs --load-model output_v3/model.joblib \
    --n-jobs 8 --thr-adapt --thr-adapt-floor 0.30 > runs/day/v3rs.log 2>&1 || return 1
  python scripts/day/compare_matches.py output_v3/matching_results.tsv output_v3_rs/matching_results.tsv >> runs/day/v3rs.log 2>&1
  grep -q "GATE PASS" runs/day/v3rs.log
}
cap() {
  python scripts/day/model_capacity.py feats_v3 --variants base,big,deep,seed3 --n-jobs 8 \
    > runs/day/model_capacity.md 2> runs/day/model_capacity.err
}
v4() {
  local w
  w=$(python scripts/day/pick_variant.py runs/day/model_capacity.md 2> runs/day/pick_variant.txt)
  cat runs/day/pick_variant.txt
  if [ -z "$w" ]; then echo "[chain9] no variant beats base by >= 0.0015 in every country -> no v4"; return 0; fi
  echo "[chain9] winner: $w" | tee -a runs/day/pick_variant.txt
  mkdir -p runs/v4
  local cmd="python -m src.rescore --feat-cache feats_v3 --cache-dir cache_n3 --out-dir output_v4 --variant $w --n-jobs 8
 --thr-adapt --thr-adapt-floor 0.30 --save-probs"
  echo "$cmd" > runs/v4/cmd.txt; git rev-parse --short HEAD > runs/v4/commit.txt
  /usr/bin/time -v $cmd > runs/v4/stdout.txt 2>&1 || return 1
  python data/data_extracted/student_resource/utils/validate_submission.py --matching output_v4/matching_results.tsv \
    --candidate output_v4/candidate_pairs.tsv --test-dir data/data_extracted/student_resource/dataset/test > runs/v4/validate.txt 2>&1
  local rc=$?; echo "VALIDATE_EXIT=$rc" >> runs/v4/validate.txt
  cmp -s output_v3/candidate_pairs.tsv output_v4/candidate_pairs.tsv && echo "candidate_pairs identical to v3" >> runs/v4/validate.txt
  python scripts/day/compare_matches.py output_v3/matching_results.tsv output_v4/matching_results.tsv > runs/v4/vs_v3.txt 2>&1
  [ $rc -eq 0 ] && cp output_v4/matching_results.tsv submissions/v4_matching_results.tsv \
    && echo "[chain9] submissions/v4_matching_results.tsv written"
  return $rc
}
if [ -f runs/day/chain/v3.done ]; then
  step v3rs v3rs  # model_capacity does not use src.rescore, so it runs even if the sanity check fails
  step cap cap && [ -f runs/day/chain/v3rs.done ] && step v4 v4
else
  echo "[chain9] no v3.done -> post-v3 steps skipped"
fi
# slice screens formerly in chain5 / chain7 (same commands)
export AML_GPU_HOST_DENSIFY=${AML_GPU_HOST_DENSIFY:-1} AML_GPU_DUTY=${AML_GPU_DUTY:-0.6} QE_NJOBS=${QE_NJOBS:-6}
qeval() { bash scripts/day/qeval.sh "$@" > runs/day/qeval_$1.log 2>&1; grep -q QEVAL_DONE runs/day/qeval_$1.log; }
AML_PH=2 step n4ph qeval n4ph _n4 "xc_us slice_mix" --views name_c3,name_w,addr_c3,full_w,name_ph
if [ -f runs/day/chain/.chain7_placeholders ]; then
  while read -r f; do rm -f "runs/day/chain/$f"; done < runs/day/chain/.chain7_placeholders
  rm -f runs/day/chain/.chain7_placeholders
fi
step n3r qeval n3r _n3 "xc_us slice_mix" --key-rules 0.96 --thr-adapt --reverse-k 3 --reverse-bypass 2
step n3ka_s7 qeval n3ka_s7 _n3 "xc_us" --key-rules 0.96 --thr-adapt --seed 7
echo "[chain9] ALL DONE"
