"""Normalise every record (new normalisation) and keep compact join keys. Run with PYTHONHASHSEED=0 (stable hash()).
Output per split/source: ens/keys_<split>_s<k>.pkl with id, country, n_core, legal, k_core, k_nsp, k_addr, k_street,
num1, addr_empty, n_tok."""
import os, sys, time, re, pickle
import numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/aml")
from src.normalize import fold, unleet, split_legal, norm_addr, _is_compact

_dig = re.compile(r"^\d+$")

def keys_chunk(df):
    names = df["business_name"].astype(str).tolist()
    addrs = df["business_address"].astype(str).tolist()
    out = {k: [] for k in ("n_core", "legal", "k_core", "k_nsp", "k_addr", "k_street", "num1", "addr_empty", "n_tok")}
    for nm, ad in zip(names, addrs):
        nn = unleet(fold(nm))
        core, legal = split_legal(nn, bool(_is_compact.match(nm)))
        a = norm_addr(ad)
        toks = a.split()
        nums = [t for t in toks if _dig.match(t)]
        street = sorted(t for t in toks if not _dig.match(t))
        out["n_core"].append(core); out["legal"].append(legal)
        out["k_core"].append(hash(core)); out["k_nsp"].append(hash(core.replace(" ", "")))
        out["k_addr"].append(hash(" ".join(sorted(toks)))); out["k_street"].append(hash(" ".join(street)))
        out["num1"].append(int(nums[0][:15]) if nums else -1)
        out["addr_empty"].append(not toks); out["n_tok"].append(len(core.split()))
    res = pd.DataFrame({"id": df["entity_id"].to_numpy(), "country": df["country"].str.strip().str.lower().to_numpy()})
    for k in ("n_core", "legal"): res[k] = out[k]
    for k in ("k_core", "k_nsp", "k_addr", "k_street"): res[k] = np.array(out[k], dtype=np.int64)
    res["num1"] = np.array(out["num1"], dtype=np.int64)
    res["addr_empty"] = np.array(out["addr_empty"], dtype=bool); res["n_tok"] = np.array(out["n_tok"], dtype=np.int8)
    return res

def main():
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    assert os.environ.get("PYTHONHASHSEED") == "0"
    jobs = [("test", s, f"data/test/test_source{s}.tsv") for s in (1, 2, 3)] + [("train", s, f"data/train/train_source{s}.tsv") for s in (1, 2, 3)]
    t0 = time.time()
    with ProcessPoolExecutor(2, mp_context=mp.get_context("spawn")) as ex:
        for split, s, path in jobs:
            outp = f"ens/keys_{split}_s{s}.pkl"
            if os.path.exists(outp):
                continue
            reader = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3, chunksize=50_000,
                                 usecols=["entity_id", "business_name", "business_address", "country"])
            parts = list(ex.map(keys_chunk, reader))
            df = pd.concat(parts, ignore_index=True)
            df.to_pickle(outp)
            print(f"{split} S{s}: {len(df)} rows ({time.time() - t0:.0f}s)", flush=True)
            del parts, df

if __name__ == "__main__":
    main()
