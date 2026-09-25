# Amazon ML Challenge 2026 — Team Repo

**Window:** 25 Sep 2026 00:00 IST → 27 Sep 2026 23:59 IST (3 days)
**Submissions:** max 5/day per team. Log EVERY submission in `submissions/LOG.md`.

## Layout
```
data/          # raw + processed data (gitignored — too large for git)
notebooks/     # EDA and experiments (one owner per notebook, prefix with initials)
src/           # shared pipeline code: features, models, train, predict
submissions/   # every submitted file + LOG.md with scores
docs/          # approach.md (the 1-2 page artefact) — update as we go, not at the end
CONTEXT.md     # running team notes — ground truth for decisions/state
```

## Rules that bite
- 5 submissions/day, then the button disables. Never burn one untested.
- Public AND private leaderboard both count → don't overfit public LB; trust local CV.
- Version history of submissions matters for shortlisting — keep every submission file.
- Artefacts due: 1–2 page approach doc + commented source code.

## Conventions
- Pull before push; small commits; no notebooks with unresolved merge conflicts.
- Every model run records: CV score, LB score (if submitted), config, git commit hash.

## Pipeline (src/) — run from the repo root
```
python scripts/eda.py --data-dir data                                   # first look at the real data
python -m src.pipeline --data-dir data --out-dir output --run-name v1   # block -> features -> 5-fold OOF -> predict
bash scripts/run_pipeline.sh v1 [--k 10]                                # same on the box (conda env aml), logs to runs/v1/
python src/metric.py data/<...>/train_ground_truth.tsv <pred.tsv>       # macro F0.5 of any prediction file
python scripts/make_synthetic.py --out data_synth                       # fake data to smoke-test without the dataset
```
- `src/data.py` finds the source/GT files and guesses id/name/address columns (pin real names in `COLUMN_OVERRIDES`).
- `src/normalize.py` folding, legal-form stripping, address abbreviations (extend the dicts after EDA).
- `src/blocking.py` TF-IDF top-k per view (name char-3, name words, address char-3, name+address) -> candidates + recall curve.
- `src/features.py` cosines + rank/context features (stage A), string/number/legal features (stage B, multi-core).
- `src/model.py` LightGBM (falls back to sklearn HistGB), GroupKFold OOF by S1, decision rules (threshold / expected-F0.5, 1-to-1 filter).
- `src/metric.py` local macro-F0.5 scorer (PS definition) + expected-F0.5 top-k selection.
- Outputs: `output/matching_results.tsv` (upload), `output/candidate_pairs.tsv` (+ `_long`), `output/report.json`. `output/` is gitignored; `runs/<name>/` is not.

