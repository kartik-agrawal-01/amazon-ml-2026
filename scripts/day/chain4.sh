#!/usr/bin/env bash
# Session-6 chain: takes over from chain3 after its n3 step. chain3 is made to skip its remaining steps via placeholder
# .done markers (listed in runs/day/chain/.chain4_placeholders); this script waits for chain3 to exit, removes them,
# then runs n3f (n3 stores + v2's generic set, AML_GENERIC=v2: splits the n3 xc_us drop into normalize vs features),
# n3 xc_in, then chain3's remaining steps.
# Usage: tmux new -d -s chain4 'bash scripts/day/chain4.sh > runs/day/chain4.log 2>&1'
set -u
R=$HOME/amazon-ml-2026
cd "$R" || exit 1
while pgrep -f 'bash scripts/day/chain3.sh' > /dev/null; do sleep 30; done
if [ -f runs/day/chain/.chain4_placeholders ]; then
  while read -r f; do rm -f "runs/day/chain/$f"; done < runs/day/chain/.chain4_placeholders
  rm -f runs/day/chain/.chain4_placeholders
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
gate_a() { bash scripts/day/gate.sh A > runs/day/gate.log 2>&1; grep -q "GATE_A_EXIT=0" runs/day/gate.log; }
gate_b() { bash scripts/day/gate.sh B > runs/day/gateB.log 2>&1; grep -q "GATE_B_EXIT=0" runs/day/gateB.log; }
qeval() { bash scripts/day/qeval.sh "$@" > runs/day/qeval_$1.log 2>&1; grep -q QEVAL_DONE runs/day/qeval_$1.log; }
[ -f runs/day/chain/n3.done ] || { echo "[chain4] n3 not done - abort"; exit 1; }
AML_GENERIC=v2 step n3f qeval n3f _n3 "xc_us slice_mix"
step n3xin qeval n3 _n3 "xc_in"
step n3k qeval n3k _n3 "xc_us slice_mix" --key-rules 0.96
step n3ph qeval n3ph _n3 "xc_us slice_mix" --views name_c3,name_w,addr_c3,full_w,name_ph
step n3r qeval n3r _n3 "xc_us slice_mix" --reverse-k 3 --reverse-bypass 2
AML_GPU_DUTY=0.3 step gateA gate_a || exit 1
step gateB gate_b || exit 1
echo "[chain] ALL DONE"
