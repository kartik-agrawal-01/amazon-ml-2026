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
