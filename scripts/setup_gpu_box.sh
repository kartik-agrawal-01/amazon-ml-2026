#!/usr/bin/env bash
# Setup for the Ubuntu GPU box — NO sudo required (user-space Miniforge).
# Run from anywhere:  bash scripts/setup_gpu_box.sh
# Safe to re-run. GPU only works once the admin installs the NVIDIA driver.

REPO="$(cd "$(dirname "$0")/.." && pwd)"
echo "== Repo: $REPO"

# 1. Miniforge (user-space Python + conda, no admin rights needed)
if [ ! -d "$HOME/miniforge3" ]; then
  echo "== Installing Miniforge to ~/miniforge3"
  cd /tmp || exit 1
  wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O miniforge.sh
  bash miniforge.sh -b -p "$HOME/miniforge3"
  "$HOME/miniforge3/bin/conda" init bash
fi
source "$HOME/miniforge3/etc/profile.d/conda.sh"

# 2. Env 'aml' with Python 3.11 + tmux (tmux from conda, no apt needed)
conda env list | grep -q "^aml " || conda create -y -n aml python=3.11
conda activate aml
conda install -y -c conda-forge tmux

# 3. Torch with CUDA 12.8 wheels (needed for RTX 50-series / Blackwell),
#    then the rest. faiss stays CPU until the driver is installed.
pip install --upgrade pip -q
pip install torch --index-url https://download.pytorch.org/whl/cu128
cd "$REPO" || exit 1
pip install -r requirements.txt

# 4. Pre-download embedding models
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); SentenceTransformer('BAAI/bge-small-en-v1.5'); print('models cached OK')"

# 5. Environment report for Claude
mkdir -p logs
{
  echo "--- nvidia-smi"; nvidia-smi 2>&1 || echo "NO DRIVER"
  echo "--- torch"; python -c "import torch; print(torch.__version__, 'cuda avail:', torch.cuda.is_available())"
  echo "--- python"; python --version
  echo "--- packages"; pip list 2>/dev/null | grep -iE "torch|faiss|lightgbm|xgboost|catboost|transformers|sentence"
  echo "--- hw"; nproc; free -h; df -h ~
} > logs/gpu_box_env.txt 2>&1
cat logs/gpu_box_env.txt

echo
echo "== DONE. In every NEW terminal run:  conda activate aml"
echo "   Then share the report:  git add -f logs/gpu_box_env.txt && git commit -m 'GPU box env report' && git push"
