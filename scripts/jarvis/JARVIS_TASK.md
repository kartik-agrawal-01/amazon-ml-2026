# Jarvis lane — Claude Code headless on a JarvisLabs GPU instance (runs until 27 Sep 20:00 IST)

You run UNATTENDED in back-to-back sessions (≤ 80 min each) started by `scripts/jarvis/jarvis_loop.sh`.
Nobody answers questions: decide, act, log. Same competition as the box (read CLAUDE.md for context), but
in this lane you ARE expected to change modelling logic.

## Machine and repos
Starts on an A30 instance (24 GB GPU, 16 vCPU, ~112 GB RAM, not shared); it may later be resumed on an RTX PRO
6000 (96 GB, 28 vCPU, ~160 GB RAM). Check `nproc`, `free -g`, `nvidia-smi` each session and size jobs to what you
find. Only /home survives a pause (a pause kills running jobs: log what was running so the next session resumes it).
- `/home/amazon-ml-2026` on branch **jarvis**: all your code changes live here. Push with
  `git push -u origin jarvis`. Never push to main from this directory.
- `/home/aml-main` is a worktree of main. Never edit it by hand: `bash scripts/jarvis/publish.sh` copies
  `runs/jarvis/*.{md,json,txt}` and `submissions/jv*_matching_results.tsv` there and pushes, so HQ can read them.

## Every session
`cd /home/amazon-ml-2026 && git fetch -q origin` (after `source /home/venv/bin/activate` if /home/venv exists), then read:
`git show origin/main:scripts/jarvis/QUEUE.md` (HQ's queue, top = next; never edit it — record DONE/FAILED in
your LOG), `runs/jarvis/SCOREBOARD.md` + `runs/jarvis/LOG.md` (your memory; create if missing),
`git show origin/main:CONTEXT.md`, `git show origin/main:docs/ENSEMBLE_AND_KEYS.md`, `runs/jarvis/guard.log`.
Then `git merge --no-edit origin/main` if it merges cleanly (HQ's and the box's fixes arrive that way); on a
conflict run `git merge --abort` and log it.

## Why this lane exists
The box has ~10 GB of RAM, so it can't run the full-strength pipeline or a GPU pair model; its own queue keeps
running there. Don't duplicate the box's queue items: they reach you through the main merge.

## Objective
The FULL-DENSITY train pass: OOF macro F0.5 per country and overall (v2: 0.9623 on 150K train S1, thr 0.70 +
one-to-one), pair recall after the cascade (v2: India 0.933, US 0.984), and mean candidates per S1 (v2 9.17,
cap 10; must not grow — the organisers rank smaller candidate sets higher). Every learned stage, a pair model
included, gets out-of-fold scores on train S1 (fixed seed, split by S1): no leakage into the OOF.

## Rules that stay fixed (disqualification otherwise)
No external data/APIs/geocoding. Pretrained models only MIT/Apache-2.0 and ≤ 8B params (e.g. xlm-roberta-base,
intfloat/multilingual-e5-base, paraphrase-multilingual-MiniLM-L12-v2, Qwen2.5-7B-Instruct). Never edit
`utils/validate_submission.py`. `candidate_pairs.tsv` must be EXACTLY the pairs the final model scores. Never
touch Unstop.

## Resources
Heavy jobs run detached: `tmux new -d -s <name> '<cmd> > runs/jarvis/<name>.log 2>&1'`, started as
`python -m src.…` or `python scripts/…` (the driver's guard only sees those; it kills the newest one when
MemAvailable < 12 GB). Up to 2 heavy CPU jobs at once while MemAvailable stays > 24 GB; a GPU job may run
alongside. Start from `--n-jobs $(( $(nproc) - 2 )) --stage-b-jobs $(( $(nproc) / 2 )) --topk-device cuda`. Check `df -h /home` before writing big
pools and delete your own intermediate caches when done. Never delete `data/`.

## Cycle protocol
1. Take the top unfinished QUEUE item; write the plan in LOG.md. 2. Implement. 3. Evaluate per the Objective.
4. KEEP (commit "jarvis N: <change> OOF=…") or REVERT. 5. SCOREBOARD row: time | change | OOF US/IN/overall |
recall after cascade US/IN | cands/S1 | runtime | KEEP/REVERT | file. 6. Push the jarvis branch, then publish.sh.

## SUBMIT-READY
Full test inference with global one-to-one (`src/hq_keys.py: global_one_to_one`) → official validator PASS →
`submissions/jv<N>_matching_results.tsv` (+ `output_jv<N>/candidate_pairs.tsv`, kept on disk, never in git) →
SCOREBOARD row marked **SUBMIT-READY** with the expected gain vs v2 → publish.sh. Humans decide uploads.

## Freeze
At 27 Sep 20:00 IST stop experimenting. Write `runs/jarvis/FINAL.md`: best file, exact commands to reproduce it
on the jarvis branch, candidate stats per country. Publish, then stop.
