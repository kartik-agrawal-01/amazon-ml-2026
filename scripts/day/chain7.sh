#!/usr/bin/env bash
# Session-10 chain: gate A/B moved ahead of n3r. A placeholder runs/day/chain/n3r.done (listed in
# runs/day/chain/.chain7_placeholders) makes chain4 skip n3r and go straight to gateA -> gateB after chain6 resumes it.
# This script waits for chain4 AND chain5 (n4ph) to exit, removes the placeholder, then runs n3r (the champion recipe
# n3ka + reverse blocking, so it is comparable with the row-6 champion).
# Usage: tmux new -d -s chain7 'bash scripts/day/chain7.sh > runs/day/chain7.log 2>&1'
set -u
R=$HOME/amazon-ml-2026
cd "$R" || exit 1
while pgrep -f 'bash scripts/day/chain[456].sh' > /dev/null; do sleep 30; done
if [ -f runs/day/chain/.chain7_placeholders ]; then
  while read -r f; do rm -f "runs/day/chain/$f"; done < runs/day/chain/.chain7_placeholders
  rm -f runs/day/chain/.chain7_placeholders
fi
export AML_GPU_HOST_DENSIFY=${AML_GPU_HOST_DENSIFY:-1} AML_GPU_DUTY=${AML_GPU_DUTY:-0.6} QE_NJOBS=${QE_NJOBS:-6}
step() {
  local n=$1; shift
  if [ -f runs/day/chain/$n.done ]; then echo "[chain] $n already done"; return 0; fi
  echo "[chain] $(date '+%F %T') start $n"
  "$@"; local rc=$?
  echo "[chain] $(date '+%F %T') end $n rc=$rc"
  [ $rc -eq 0 ] && touch runs/day/chain/$n.done
  return $rc
}
qeval() { bash scripts/day/qeval.sh "$@" > runs/day/qeval_$1.log 2>&1; grep -q QEVAL_DONE runs/day/qeval_$1.log; }
step n3r qeval n3r _n3 "xc_us slice_mix" --key-rules 0.96 --thr-adapt --reverse-k 3 --reverse-bypass 2
echo "[chain7] ALL DONE"
# appended s10: 2nd-seed n3ka xc_us (NEXT 4: is the +0.019 xc_us gain outside the ±0.007 seed noise?)
step n3ka_s7 qeval n3ka_s7 _n3 "xc_us" --key-rules 0.96 --thr-adapt --seed 7
echo "[chain7] seed step done"
