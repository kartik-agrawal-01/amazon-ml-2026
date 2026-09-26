#!/usr/bin/env bash
# Session-7 chain: waits for chain4 (n3xin -> n3k -> n3ph -> n3r -> gateA -> gateB) to exit, then screens
# n4ph = n3ph + AML_PH=2 (Indic transliterated legal forms + voicing-folded phonetic key; stores _n4, built on first use).
# Compare with n3ph (same views, _n3 stores). Usage: tmux new -d -s chain5 'bash scripts/day/chain5.sh > runs/day/chain5.log 2>&1'
set -u
R=$HOME/amazon-ml-2026
cd "$R" || exit 1
while pgrep -f 'bash scripts/day/chain4.sh' > /dev/null; do sleep 30; done
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
AML_PH=2 step n4ph qeval n4ph _n4 "xc_us slice_mix" --views name_c3,name_w,addr_c3,full_w,name_ph
echo "[chain5] ALL DONE"
