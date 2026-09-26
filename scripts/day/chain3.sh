#!/usr/bin/env bash
# Session-5 chain (replaces chain1+chain2 order after the 4th reboot, 13:18, again inside the full-data train/us top-k
# of gate A, this time with the GPU at ~45-60 W -> not a GPU power trip). Slice Q screens first (never rebooted), the
# full-data gate last. Same resumable .done markers as chain1/chain2 (runs/day/chain/<step>.done).
# Usage: tmux new -d -s chain 'bash scripts/day/chain3.sh > runs/day/chain3.log 2>&1'
set -u
R=$HOME/amazon-ml-2026
cd "$R" || exit 1
mkdir -p runs/day/chain
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
gate_a() { bash scripts/day/gate.sh A > runs/day/gate.log 2>&1; grep -q "GATE_A_EXIT=0" runs/day/gate.log; }
gate_b() { bash scripts/day/gate.sh B > runs/day/gateB.log 2>&1; grep -q "GATE_B_EXIT=0" runs/day/gateB.log; }
qeval() { bash scripts/day/qeval.sh "$@" > runs/day/qeval_$1.log 2>&1; grep -q QEVAL_DONE runs/day/qeval_$1.log; }
step base qeval base "" "xc_us slice_mix"
step n3 qeval n3 _n3 "xc_us slice_mix"
step n3k qeval n3k _n3 "xc_us slice_mix" --key-rules 0.96
step n3ph qeval n3ph _n3 "xc_us slice_mix" --views name_c3,name_w,addr_c3,full_w,name_ph
step n3r qeval n3r _n3 "xc_us slice_mix" --reverse-k 3 --reverse-bypass 2   # QUEUE 5b reverse blocking
AML_GPU_DUTY=0.3 step gateA gate_a || exit 1
step gateB gate_b || exit 1
echo "[chain] ALL DONE"
