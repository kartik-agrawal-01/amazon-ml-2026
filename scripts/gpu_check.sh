#!/usr/bin/env bash
# Verify the box GPU + install the optional accelerators. Run:  bash scripts/gpu_check.sh
# Appends a report to logs/gpu_box_env.txt -> commit + push it so HQ can read it.
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate aml
mkdir -p logs
{
  echo "=== $(date) gpu_check"
  nvidia-smi 2>&1 | head -15
  python - <<'PY'
import time, torch
print("torch", torch.__version__, "cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0), "mem GB:", round(torch.cuda.get_device_properties(0).total_memory/1e9, 1))
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16); torch.cuda.synchronize(); t = time.time()
    for _ in range(20): y = x @ x
    torch.cuda.synchronize(); print("fp16 matmul TFLOPs: %.1f" % (20 * 2 * 4096**3 / (time.time() - t) / 1e12))
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda"); m.max_seq_length = 64
    texts = ["Acme Robotics Private Limited, 12 MG Road, Near Bus Stand, Pune, Maharashtra %d" % i for i in range(20000)]
    m.encode(texts[:512], batch_size=512); torch.cuda.synchronize(); t = time.time()
    m.encode(texts, batch_size=1024, normalize_embeddings=True); torch.cuda.synchronize()
    r = 20000 / (time.time() - t); print("MiniLM encode: %.0f texts/s -> 24M texts ~ %.0f min" % (r, 24e6 / r / 60))
PY
  pip install -q sparse_dot_topn pyarrow joblib 2>&1 | tail -2
  python -c "import sparse_dot_topn, pyarrow, joblib; print('sparse_dot_topn', sparse_dot_topn.__version__, 'pyarrow', pyarrow.__version__)"
  free -h | head -2; nproc
} 2>&1 | tee -a logs/gpu_box_env.txt
echo "== now:  git add -f logs/gpu_box_env.txt && git commit -m 'gpu check' && git push"
