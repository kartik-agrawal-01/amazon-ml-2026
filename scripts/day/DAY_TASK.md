# Improvement loop — Claude Code headless on the box (runs until 27 Sep 20:00 IST)

You run UNATTENDED in back-to-back sessions (≤80 min each) started by `scripts/day/day_loop.sh`.
Nobody answers questions — decide, act, log. `CLAUDE.md` hard rules apply, EXCEPT: in this loop you ARE
allowed (and expected) to change modelling logic (features, blocking, model, decision rule).
Memory between sessions: `runs/day/SCOREBOARD.md` + `runs/day/LOG.md` (create if missing).
Every session: `cd ~/amazon-ml-2026 && git pull --no-edit`, read SCOREBOARD.md, LOG.md, `scripts/day/QUEUE.md`
(humans/HQ may add items there — respect its order), CONTEXT.md, `runs/day/guard.log`.

## Detecting running jobs
Use ONLY `pgrep -af '^python[0-9.]* (-m src[.]|scripts/)'`. Plain `pgrep -f src.pipeline` also matches your own
claude process (this prompt contains the text). Never kill a process you did not start.

## Situation (26 Sep ~10:30)
v2 (commit 48bc4e7) LEADERBOARD 0.947; team's other pipeline 0.948. v2 local OOF 0.9623 → LB gap −1.5.
Prime suspect: FRANCE (15% of test, unseen in training). v2 France counts look normal (empty 5.2%, 3.30
matches/S1) but correctness is unknown. Cross-country proxy (slice, hidden F0.5): train US+IN 0.9805
(India 0.9717, US 0.9863); train US only → India 0.9339 (−3.8); train India only → US 0.9810 (−0.5).
India full-data blocking recall only ~0.93 (US 0.986); India = 47% of test. 9.2 candidates/S1 (Amazon ranks
smaller candidate sets higher). Submissions: 5/day; humans submit — you NEVER submit.

## Objective (the number every cycle must move)
Q = 0.5·F(slice_mix) + 0.3·F(xc_us) + 0.2·F(xc_in)   (hidden-holdout macro F0.5 from the existing runs:
`data_slice`/`cache_slice` = slice_mix; `data_xc_us`/`cache_xc_us` = train US → test India;
`data_xc_in`/`cache_xc_in` = train India → test US; see scripts/night/track_b_chain.sh + b4_table.py).
Baseline (B4 table, 26 Sep 07:45): mix 0.9805, xc_us 0.9339, xc_in 0.9810 → **Q0 = 0.9666**.
Secondary (must not regress without reason): mean final candidates/S1, runtime per stage, peak RSS.
KEEP a change if ΔQ ≥ +0.0015, OR it cuts candidates/S1 ≥ 10% or runtime ≥ 20% with ΔQ ≥ −0.0005.
Otherwise revert (`git checkout -- <files>`), and log it under "tried & failed" so it's never retried.
Screen on xc_us + slice_mix first (~30 min, `--n-jobs 8 --topk-device cuda`); run xc_in only before a KEEP.

## Hard resource rules
16 GB RAM shared (~10 GB usable), RTX 5060 Ti 16 GB. ONE heavy job at a time (slice evals ~3.5 GB, full-data
~7.3 GB). A driver guard kills the NEWEST heavy job at MemAvailable < 1.5 GB. Run heavy jobs detached in tmux
(`~/miniforge3/envs/aml/bin/tmux new -d -s <name> '<cmd> > runs/day/<name>.log 2>&1'`); while one runs, do
only light work (code the next experiment, analyse existing outputs). Never delete `data/`, `output_v2*/`,
`submissions/`, built stores/caches.

## Rules that stay fixed (disqualification otherwise)
No external data/APIs/geocoding. Pretrained models only MIT/Apache-2.0 and ≤ 8B params. Never edit
`utils/validate_submission.py`. `candidate_pairs.tsv` must be EXACTLY the set fed to the model.

## Cycle protocol (target ≤ 60 min per cycle — one "hourly model")
1. Pick the top unfinished item of QUEUE.md (or your best-EV idea if the queue is empty). Write the plan in LOG.md.
2. Implement (small, reviewable diff). 3. Evaluate per the Objective. 4. KEEP (commit "cycle N: <change> Q=…")
   or REVERT. 5. Append a SCOREBOARD row: cycle | time | change | Q | mix | xc_us | xc_in | cands/S1 |
   runtime | peak RSS | KEEP/REVERT | full-data file. Mark the champion. 6. `git push` (if auth fails, log it).

## P0 — FAST LANE (do first; up to ~2 h is acceptable). Makes full-data models ~1 h instead of ~5 h.
a) `--reuse-candidates <dir>` (or a new `src/rescore.py`): take a FIXED full-data candidate set (test:
   `output_v2/candidate_pairs.tsv`; train: re-derive once for the 150K train S1 and cache it), recompute
   per-pair stage-A features WITHOUT top-k search (row-wise sparse dot products of the fitted views + ranks
   within each S1/source group), stage-B features, predict, write outputs. Persist fitted vectorisers and
   cache train/test pair FEATURES as parquet so feature/model/rule changes only recompute what changed.
b) Correctness gate: rescoring v2's candidates with v2's model must reproduce ≥ 99.9% of v2's
   matching_results rows exactly. Record full-data runtime per stage.
c) Save per-pair probabilities for test (needed for the France diagnosis).
Also add a "candidate augmentation" path for blocking changes: run only NEW view passes (per affected
country), union with cached candidates, re-apply the cascade, rescore.

## Full-data promotion (produces submission candidates)
When the champion's Q beats the last full-data model's Q by ≥ 0.002 (or xc_us gains ≥ 0.005 from a
France-targeted change): fast-lane full-data predictions → official validator → `submissions/vN_matching_results.tsv`
(+ keep `output_vN/candidate_pairs.tsv`) → `runs/vN/NOTES.md` → SCOREBOARD row marked **SUBMIT-READY** with the
expected gain and why. Humans decide the upload. Never overwrite `output_v2/`.

## Freeze
At 27 Sep 20:00 IST: stop experimenting. Write `runs/day/FINAL.md`: champion, its full-data file, all
SUBMIT-READY files ranked by robustness (weight xc_us/xc_in heavily — final ranking = PRIVATE leaderboard),
and exact packaging commands (`scripts/make_package.py`). Then stop.
