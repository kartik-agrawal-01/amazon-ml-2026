# Running Claude Code on the GPU box (so it operates the pipeline by itself)

Claude Code is Anthropic's terminal agent. Installed on the box it can run the pipeline, watch the
log, fix crashes, and commit/push results — no remote desktop juggling. It uses YOUR Claude account
(Pro/Max subscription) or an API key.

## 1. Install (no sudo needed, ~2 min) — inside the RDP session, in a terminal
```bash
curl -fsSL https://claude.ai/install.sh | bash        # native installer, no Node required
export PATH="$HOME/.local/bin:$PATH"                   # add this line to ~/.bashrc too
claude --version
```
If the installer is blocked by the network policy, the npm route also works inside the conda env:
`conda activate aml && conda install -y -c conda-forge nodejs && npm install -g @anthropic-ai/claude-code`.

## 2. Log in (once)
```bash
cd ~/amazon-ml-2026
claude
```
It opens a browser login (the RDP desktop has a browser) or prints a URL + code; log in with the
same claude.ai account you use for Cowork. If a browser is awkward, use an API key instead:
`export ANTHROPIC_API_KEY=...` before `claude`.

## 3. Run it inside tmux so it survives RDP disconnects
```bash
~/miniforge3/envs/aml/bin/tmux new -s claude
cd ~/amazon-ml-2026 && git pull
claude
```
First message to give it (paste as one line):

> Read CLAUDE.md, CONTEXT.md and docs/BOX_RUNBOOK.md. Then: run scripts/gpu_check.sh, run the smoke
> command from CLAUDE.md, and if it finishes cleanly run the full v2 command with --n-jobs 8. Watch
> runs/<run>/stdout.txt, fix crashes as CLAUDE.md allows, validate the output, copy it to
> submissions/, write runs/<run>/NOTES.md, commit and push after each run. Do not change modelling
> logic. Report the key metrics when done.

Detach with `Ctrl-b d`; re-attach with `tmux a -t claude`. Claude Code asks for permission before
running commands; answer once with "yes, and don't ask again for this session" style approvals, or
start it with `claude --dangerously-skip-permissions` **only** inside the repo folder (it cannot
delete data — `data/` rules are in CLAUDE.md, and the machine has no sudo).

## 4. What HQ (this Cowork chat) then does
HQ keeps designing features/models on the slice, pushes code to GitHub, and reads `runs/*/` that
Claude Code pushes. You stay the git bridge on the laptop only for HQ's edits.
