#!/usr/bin/env bash
# One-time setup for the Ubuntu GPU box.
# Run from the repo root:  bash scripts/setup_gpu_box.sh
# Safe to re-run.

cd "$(dirname "$0")/.." || exit 1
echo "== Repo: $(pwd)"

echo "== GPU info"
nvidia-smi || echo "!! nvidia-smi failed — NVIDIA driver missing or broken"

echo "== System packages (asks for your password)"
sudo apt-get update -qq
sudo apt-get install -y python3-venv tmux

echo "== Disable sleep/suspend for the competition"
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target

echo "== Python venv"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt
# swap CPU faiss for the GPU build (both provide the same module; keep one)
pip uninstall -y faiss-cpu
pip install faiss-gpu-cu12

echo "== Torch / CUDA check"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

echo "== Pre-download embedding models"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); SentenceTransformer('BAAI/bge-small-en-v1.5'); print('models cached OK')"

mkdir -p logs
echo "== Saving environment report to logs/gpu_box_env.txt"
{
  nvidia-smi
  python --version
  pip list 2>/dev/null | grep -iE "torch|faiss|lightgbm|xgboost|catboost|transformers|sentence"
  python -c "import torch; print('cuda', torch.cuda.is_available(), torch.version.cuda)"
  nproc; free -h; df -h .
} > logs/gpu_box_env.txt 2>&1

echo
echo "== DONE. Next: remember 'source .venv/bin/activate' in every new terminal."
echo "   Commit the report so Claude can read it:"
echo "   git add -f logs/gpu_box_env.txt && git commit -m 'GPU box env report' && git push"
