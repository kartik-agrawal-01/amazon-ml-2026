#!/usr/bin/env bash
# Night watcher: waits for the v2 pipeline to exit, then does Track A item A4 unattended:
#   validator -> output_v2 archive -> submissions/v2_matching_results.tsv -> runs/v2/NOTES.md -> commit + push.
# On success it starts the Track B cross-country chain (B4) with more workers (box is free by then).
# Usage: RUN=v2 PID=<pipeline pid> bash scripts/night/post_v2.sh   (run detached in tmux)
set -u
RUN="${RUN:-v2}"; PID="${PID:?pipeline pid}"
cd ~/amazon-ml-2026 || exit 1
source ~/miniforge3/etc/profile.d/conda.sh && conda activate aml
export GIT_TERMINAL_PROMPT=0
NL=runs/night/NIGHT_LOG.md
say() { echo "$(date '+%F %T') post_$RUN: $*" | tee -a runs/night/post_${RUN}.log; }
say "waiting for pid $PID"
while kill -0 "$PID" 2>/dev/null; do sleep 30; done
sleep 20   # let tee flush / run_pipeline.sh finish
SO=runs/$RUN/stdout.txt
if ! grep -q 'done | peak RSS' "$SO"; then
  say "FAILED: no 'done' line in $SO (last lines below)"; tail -25 "$SO" | tee -a runs/night/post_${RUN}.log
  printf -- '- %s !! %s DIED (post watcher). Last lines:\n```\n%s\n```\n  NEXT: crash procedure (CLAUDE.md) or salvage partial output (scripts/night/salvage_partial.py) if past ~02:30.\n' \
    "$(date '+%H:%M')" "$RUN" "$(tail -8 "$SO")" >> "$NL"
  git add "$NL" runs/$RUN/stdout.txt runs/$RUN/cmd.txt 2>/dev/null; git commit -qm "night log: $RUN died (post watcher)" && git push -q; exit 1
fi
say "run finished; validating"
python data/data_extracted/student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv \
  --test-dir data/data_extracted/student_resource/dataset/test > runs/$RUN/validate.txt 2>&1
VEXIT=$?
say "validator exit $VEXIT"; tail -15 runs/$RUN/validate.txt | tee -a runs/night/post_${RUN}.log
[ -d output_$RUN ] || cp -r output output_$RUN
rm -f runs/$RUN/matching_results.tsv   # never commit the big tsv from runs/
SZ=$(stat -c %s output/matching_results.tsv)
SUB=""
if [ "$SZ" -lt 100000000 ]; then mkdir -p submissions; cp output/matching_results.tsv submissions/${RUN}_matching_results.tsv; SUB=submissions/${RUN}_matching_results.tsv; fi
python - "$RUN" "$VEXIT" "$SZ" <<'PY' > runs/$RUN/NOTES.md
import json, sys, subprocess
run, vexit, sz = sys.argv[1], sys.argv[2], int(sys.argv[3])
r = json.load(open(f"runs/{run}/report.json"))
cmd = open(f"runs/{run}/cmd.txt").read().strip() if __import__("os").path.exists(f"runs/{run}/cmd.txt") else "?"
print(f"# {run} NOTES (auto-written by scripts/night/post_v2.sh)\n")
print(f"- command: `{cmd}`")
print(f"- commit: {open(f'runs/{run}/commit.txt').read().strip()}")
print(f"- runtime: {r.get('runtime_s', 0)/3600:.2f} h | peak RSS: {r.get('peak_rss_mb')}")
print(f"- validator exit code: {vexit} (see runs/{run}/validate.txt) | matching_results.tsv {sz/1e6:.1f} MB")
print(f"- OOF: {r.get('oof', {}).get('chosen')} | AUC {r.get('oof', {}).get('auc')}")
print(f"- cascade: before {r.get('cands_before_cascade')} | after {r.get('cands_after_cascade')}")
print(f"- blocking recall (train): {r.get('blocking_recall')}")
print(f"- test candidates: {r.get('test_candidates')}")
print(f"- test pred overall: {r.get('test_pred')}")
print("\n## Test per-country\n")
print("| country | n_s1 | empty_rate | mean_matches | cand_mean | cand_median |")
print("|---|---|---|---|---|---|")
for p in r.get("test_pred_by_country", []):
    print(f"| {p['country']} | {p['n_s1']} | {p['empty_rate']:.3f} | {p['mean_matches']:.2f} | {p['cand_mean']:.1f} | {p['cand_median']:.0f} |")
PY
say "NOTES.md written"
{
  printf -- '- %s A4 %s FINISHED (post watcher): validator exit %s; output_%s/ archived; %s\n' "$(date '+%H:%M')" "$RUN" "$VEXIT" "$RUN" "${SUB:-matching_results.tsv >= 100 MB, NOT copied to submissions/}"
  grep -E 'rows written|done \| peak' "$SO" | sed 's/^/    /'
  sed -n '/^## Test per-country/,$p' runs/$RUN/NOTES.md | sed 's/^/    /'
} >> "$NL"
git add "$NL" runs/$RUN/report.json runs/$RUN/stdout.txt runs/$RUN/commit.txt runs/$RUN/cmd.txt runs/$RUN/NOTES.md runs/$RUN/validate.txt $SUB 2>/dev/null
git commit -qm "run $RUN: validated full-data submission (post watcher)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" && git push -q && say "pushed" || say "git commit/push FAILED"
if [ "${CHAIN:-1}" = 1 ]; then
  say "starting Track B chain (B4) with NJ=${NJ:-6}"
  NJ="${NJ:-6}" SKIP_WAIT=1 bash scripts/night/track_b_chain.sh > runs/night/track_b_chain.log 2>&1
  say "chain finished"
fi
