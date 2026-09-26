#!/usr/bin/env bash
# One-time setup on the JarvisLabs RTX PRO 6000 instance (PyTorch template, runs as root).
# Only /home survives pause/resume, so repo, venv, data and git credentials all live there.
# Before running: dataset zip uploaded to /home, repo cloned to /home/amazon-ml-2026.
#   bash /home/amazon-ml-2026/scripts/jarvis/setup_jarvis.sh
set -euo pipefail
REPO=/home/amazon-ml-2026
cd "$REPO"
git config credential.helper "store --file /home/.git-credentials"   # repo-local: survives pauses
git config user.name "jarvis (Claude Code)"; git config user.email "jarvis@localhost"

# 1. branches: code work on 'jarvis' here; /home/aml-main = main, only for publishing results
git fetch -q origin
git show-ref -q --verify refs/heads/jarvis || git branch jarvis origin/main
git checkout -q jarvis
[ -d /home/aml-main ] || git worktree add -q /home/aml-main main

# 2. data (same layout as the box)
if ! find data -name validate_submission.py 2>/dev/null | grep -q .; then
  ZIP=$(ls /home/*student_resource*.zip 2>/dev/null | head -1 || true)
  [ -n "$ZIP" ] || { echo "!! upload the dataset zip to /home first"; exit 1; }
  mkdir -p data && python3 -m zipfile -e "$ZIP" data/
fi
echo "validator: $(find data -name validate_submission.py | head -1)"

# 3. python env in /home; keep the template's torch if it works on this GPU, else CUDA 12.8 wheels
[ -x /home/venv/bin/python ] || python3 -m venv --system-site-packages /home/venv 2>/dev/null \
  || echo "(no venv support: using the system python; re-run this script after a pause)"
[ -f /home/venv/bin/activate ] && source /home/venv/bin/activate
pip install -q --upgrade pip
python -c "import torch; x=torch.randn(2048,2048,device='cuda'); print('torch ok', torch.__version__, (x@x).sum().item()!=0)" \
  || pip install -q --upgrade torch --index-url https://download.pytorch.org/whl/cu128
pip install -q -r requirements.txt

# 4. tools (outside /home: re-run this script after a resume)
command -v tmux >/dev/null || { apt-get update -qq && apt-get install -y -qq tmux; }
command -v claude >/dev/null || curl -fsSL https://claude.ai/install.sh | bash

# 5. environment report
mkdir -p runs/jarvis
{ echo "--- $(date)"; nvidia-smi; python -c "import torch;print('torch',torch.__version__,torch.cuda.get_device_name(0),torch.cuda.get_device_capability(0))";
  echo "cpus $(nproc)"; free -g; df -h /home; } > runs/jarvis/env.txt 2>&1
cat runs/jarvis/env.txt
echo "== setup done. Next: claude once to log in (if 'claude' is not found: /root/.local/bin/claude), then start the loop (scripts/jarvis/README.md)"
