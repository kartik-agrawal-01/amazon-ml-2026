"""Tiny check: GPU top-k paths (card densify / host densify / duty throttle) give the CPU result."""
import os, sys, importlib
import numpy as np, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
rng = np.random.default_rng(0)
Q = sp.random(300, 500, density=0.02, random_state=1, format="csr", dtype=np.float32)
D = sp.random(5000, 500, density=0.02, random_state=2, format="csr", dtype=np.float32)
res = {}
for mode, env in [("cpu", {}), ("card", {}), ("host", {"AML_GPU_HOST_DENSIFY": "1", "AML_GPU_DUTY": "0.5"})]:
    os.environ.update(env)
    import src.blocking as b; b = importlib.reload(b)
    b.TOPK_DEVICE["device"] = "cpu" if mode == "cpu" else "cuda"
    r, c, s, k = b.topk_sparse(Q, D, 5, min_sim=0.0)
    res[mode] = set(zip(r.tolist(), c.tolist()))
    print(mode, len(res[mode]))
for m in ("card", "host"):
    print(m, "overlap with cpu:", len(res[m] & res["cpu"]) / len(res["cpu"]))
