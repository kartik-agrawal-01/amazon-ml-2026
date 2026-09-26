#!/usr/bin/env bash
# Session-1 heavy-job chain (one job at a time, resumable after a reboot: finished steps leave runs/day/chain/<step>.done).
#   gateA  -> fast-lane candidate cache cands_v2/ (v2 code, worktree ~/aml_gate)          [scripts/day/gate.sh A]
#   gateB  -> rescore test with v2's model + compare with output_v2                       [scripts/day/gate.sh B]
#   base   -> HEAD baseline Q screen on v2 stores (deterministic LightGBM changed numerics vs Q0)
#   n3     -> QUEUE 3 France fixes: Q screen on rebuilt _n3 stores
#   n3k    -> QUEUE 4a key rules on top of n3 (--key-rules 0.96)
# Usage: tmux new -d -s chain 'bash scripts/day/chain1.sh > runs/day/chain1.log 2>&1'
set -u
R=$HOME/amazon-ml-2026
cd "$R" || exit 1
mkdir -p runs/day/chain
# 26 Sep: 3 hard reboots at the start of full-speed GPU top-k on train/us -> v2's host-densify path + 60% GPU duty
export AML_GPU_HOST_DENSIFY=${AML_GPU_HOST_DENSIFY:-1} AML_GPU_DUTY=${AML_GPU_DUTY:-0.6}
step() {  # step <name> <cmd...>; success = exit 0 and (for gate steps) the marker line
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
step gateA gate_a || exit 1
step gateB gate_b || exit 1
step base qeval base "" "xc_us slice_mix"
step n3 qeval n3 _n3 "xc_us slice_mix"
step n3k qeval n3k _n3 "xc_us slice_mix" --key-rules 0.96
echo "[chain] ALL DONE"
