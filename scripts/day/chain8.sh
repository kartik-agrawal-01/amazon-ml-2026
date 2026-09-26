#!/usr/bin/env bash
# Session-11 chain: runs v3 (scripts/day/v3.sh) right after gate B PASSES, before the slice screens of chain5/chain7.
# SIGSTOPs the chain5 bash (it is only in its wait loop; chain7 waits for chain[456] so it stays queued too), waits for
# chain4 (gateA -> gateB) to exit, then:
#   gate PASS (runs/day/gateB.log has "GATE PASS" and GATE_B_EXIT=0): keep v2's per-pair test probs in output_v2_probs/,
#     rm -rf ~/aml_gate_out, remove the ~/aml_gate worktree, run v3, copy the validated file to submissions/;
#   otherwise: log and do nothing (a session decides).
# SIGCONTs chain5 on exit (trap). If a reboot/kill leaves chain5 stopped: kill -CONT $(pgrep -xf 'bash scripts/day/chain5.sh')
# Usage: tmux new -d -s chain8 'bash scripts/day/chain8.sh > runs/day/chain8.log 2>&1'
set -u
R=$HOME/amazon-ml-2026
cd "$R" || exit 1
P5=$(pgrep -xf 'bash scripts/day/chain5.sh' | head -1)
[ -n "$P5" ] && kill -STOP "$P5" && echo "[chain8] $(date '+%F %T') chain5 ($P5) stopped"
trap '[ -n "$P5" ] && kill -CONT "$P5" 2>/dev/null && echo "[chain8] chain5 resumed"' EXIT
while pgrep -xf 'bash scripts/day/chain4.sh' > /dev/null; do sleep 30; done
echo "[chain8] $(date '+%F %T') chain4 exited"
if ! { [ -f runs/day/chain/gateB.done ] && grep -q "GATE PASS" runs/day/gateB.log; }; then
  echo "[chain8] gate B not passed -> no v3 (see runs/day/gateB.log)"; exit 1
fi
grep -E "^(matching_results|candidate_pairs):" runs/day/gateB.log
mkdir -p output_v2_probs && mv "$HOME"/aml_gate_out/test_probs_*.parquet output_v2_probs/ && rm -rf "$HOME/aml_gate_out"
git worktree remove --force "$HOME/aml_gate" && echo "[chain8] gate worktree removed"
touch runs/day/chain/gate_cleanup.done
echo "[chain8] $(date '+%F %T') start v3"
bash scripts/day/v3.sh > runs/day/v3.log 2>&1
echo "[chain8] $(date '+%F %T') end v3: $(tail -n 1 runs/day/v3.log)"
if grep -q V3_DONE runs/day/v3.log && grep -q "VALIDATE_EXIT=0" runs/v3/validate.txt; then
  mkdir -p submissions && cp output_v3/matching_results.tsv submissions/v3_matching_results.tsv && touch runs/day/chain/v3.done
  echo "[chain8] submissions/v3_matching_results.tsv written ($(du -m submissions/v3_matching_results.tsv | cut -f1) MB)"
fi
