#!/usr/bin/env bash
# Session-8 chain: pauses the chain4 driver (SIGSTOP on its bash only) so it does not start n3r when n3ph ends,
# runs n3ka = n3k + --thr-adapt (QUEUE 2d rule R2 lower-only) on xc_us, slice_mix, xc_in, then resumes chain4.
set -u
R=$HOME/amazon-ml-2026
cd "$R" || exit 1
C4=$(pgrep -xf 'bash scripts/day/chain4.sh' | head -1)
[ -n "$C4" ] && kill -STOP "$C4" && echo "[chain6] paused chain4 bash $C4"
trap '[ -n "$C4" ] && kill -CONT "$C4" && echo "[chain6] resumed chain4"' EXIT
while pgrep -f '^python[0-9.]* -m src[.]pipeline' > /dev/null; do sleep 30; done
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
step n3ka qeval n3ka _n3 "xc_us slice_mix xc_in" --key-rules 0.96 --thr-adapt
echo "[chain6] ALL DONE"
