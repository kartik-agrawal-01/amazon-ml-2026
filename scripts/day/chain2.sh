#!/usr/bin/env bash
# Session-4 chain (runs after chain1; resumable like chain1: runs/day/chain/<step>.done).
#   n3ph -> QUEUE 5b: + name_ph view (phonetic consonant skeleton, char 3-grams) on top of n3. runs/day/india_recall.md:
#           55% of India's blocking misses are Indic-transliterated names ('bildarsa'/'kansaltantsa') and 4.5K misses
#           (32%) have IDENTICAL phonetic keys -> a name_ph top-k pass should recover most of them.
# Usage: tmux new -d -s chain2 'bash scripts/day/chain2.sh > runs/day/chain2.log 2>&1'
set -u
R=$HOME/amazon-ml-2026
cd "$R" || exit 1
mkdir -p runs/day/chain
export AML_GPU_HOST_DENSIFY=${AML_GPU_HOST_DENSIFY:-1} AML_GPU_DUTY=${AML_GPU_DUTY:-0.6}
while [ ! -f runs/day/chain/n3k.done ]; do   # wait for chain1 (one heavy job at a time)
  pgrep -f 'scripts/day/chain1.sh' > /dev/null || { echo "[chain2] chain1 not running and n3k not done -> exit"; exit 1; }
  sleep 60
done
step() {
  local n=$1; shift
  if [ -f runs/day/chain/$n.done ]; then echo "[chain2] $n already done"; return 0; fi
  echo "[chain2] $(date '+%F %T') start $n"
  "$@"; local rc=$?
  echo "[chain2] $(date '+%F %T') end $n rc=$rc"
  [ $rc -eq 0 ] && touch runs/day/chain/$n.done
  return $rc
}
qeval() { bash scripts/day/qeval.sh "$@" > runs/day/qeval_$1.log 2>&1; grep -q QEVAL_DONE runs/day/qeval_$1.log; }
step n3ph qeval n3ph _n3 "xc_us slice_mix" --views name_c3,name_w,addr_c3,full_w,name_ph
echo "[chain2] ALL DONE"
