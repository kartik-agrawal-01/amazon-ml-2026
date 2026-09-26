#!/usr/bin/env bash
# Copy the Jarvis lane's small results to main so HQ can read them from the laptop:
# runs/jarvis/*.{md,json,txt} and submissions/jv*_matching_results.tsv. Never copies code.
set -euo pipefail
SRC=/home/amazon-ml-2026; MAIN=/home/aml-main
cd "$MAIN"; git pull -q --no-edit
mkdir -p runs/jarvis
( cd "$SRC/runs/jarvis" && find . -maxdepth 2 -type f \( -name '*.md' -o -name '*.json' -o -name '*.txt' \) -size -5M -print0 ) \
  | while IFS= read -r -d '' f; do mkdir -p "runs/jarvis/$(dirname "$f")"; cp "$SRC/runs/jarvis/$f" "runs/jarvis/$f"; done
for f in "$SRC"/submissions/jv*_matching_results.tsv; do
  [ -e "$f" ] || continue
  if [ "$(stat -c %s "$f")" -lt 99000000 ]; then cp "$f" submissions/ && git add "submissions/$(basename "$f")"
  else gzip -c "$f" > "submissions/$(basename "$f").gz" && git add -f "submissions/$(basename "$f").gz"; fi   # GitHub caps files at 100 MB
done
git add runs/jarvis
if git diff --cached --quiet; then echo "publish: nothing new"; else git commit -qm "jarvis: publish results"; git push -q; echo "publish: pushed"; fi
