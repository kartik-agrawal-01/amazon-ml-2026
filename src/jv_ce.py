"""Jarvis QUEUE 2: cross-encoder pair model on the cascade pool (src/pipeline.py --pool-dir).

Pool files: <pool>/train__<country>.parquet (s1, cand, pa, rank, keep, y) and <pool>/test__<country>__bNNN.parquet
(s1, cand, pa, rank, keep). Text per side: "n_name | n_addr" from the normalised store.

  python -m src.jv_ce train --pool /home/pools/jv1 --store /home/cache_jv/store --out /home/pools/jv1_ce
  python -m src.jv_ce test  --pool /home/pools/jv1 --store /home/cache_jv/store --out /home/pools/jv1_ce

train: 2 folds by S1 (seed 42). Fold model f is fine-tuned (1 epoch) on the pairs of the OTHER fold's S1 (all positives +
negatives, hard ones first, capped at --max-train pairs) and scores every pool pair of fold f -> every train pair gets an
out-of-fold ce logit (<out>/train_ce.parquet). test: mean of the fold models' sigmoids (<out>/test_ce__<country>__bNNN.parquet).
"""
import argparse
import glob
import math
import os
import time

import numpy as np
import pandas as pd
import torch

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


def load_text(store: str, split: str, country: str, rids: set) -> pd.Series:
    d = pd.read_parquet(os.path.join(store, f"{split}__{country}.parquet"), columns=["rid", "n_name", "n_addr"])
    d = d[d["rid"].isin(rids)]
    return pd.Series((d["n_name"].fillna("") + " | " + d["n_addr"].fillna("")).to_numpy(), index=d["rid"].to_numpy())


def fold_of(s1: np.ndarray, n_folds: int, seed: int) -> dict:
    u = np.sort(np.unique(s1))
    f = np.random.default_rng(seed).integers(0, n_folds, len(u))
    return dict(zip(u, f))


class Encoder:
    """Pairs are given as indices into a list of pre-tokenised texts (each unique text tokenised once);
    input = <s> A </s></s> B </s> (XLM-R pair format, same ids as tokenizer(a, b) without truncation)."""

    def __init__(self, name: str, max_len: int, device: str):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1).to(device)
        self.max_len, self.device = max_len, device
        self.cls, self.sep, self.pad = self.tok.cls_token_id, self.tok.sep_token_id, self.tok.pad_token_id

    def tokenize(self, texts) -> list:
        half = (self.max_len - 4) // 2
        return self.tok(list(texts), add_special_tokens=False, truncation=True, max_length=half)["input_ids"]

    def batch(self, toks, ia, ib):
        seqs = [[self.cls] + toks[i] + [self.sep, self.sep] + toks[j] + [self.sep] for i, j in zip(ia, ib)]
        L = max(len(x) for x in seqs)
        ids = np.full((len(seqs), L), self.pad, np.int64)
        for r, x in enumerate(seqs):
            ids[r, :len(x)] = x
        ids = torch.from_numpy(ids).to(self.device, non_blocking=True)
        return {"input_ids": ids, "attention_mask": (ids != self.pad).long()}

    def logits(self, toks, ia, ib):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model(**self.batch(toks, ia, ib)).logits.float().squeeze(-1)


def train_model(enc: Encoder, toks, ta: np.ndarray, tb: np.ndarray, y: np.ndarray, bs: int, lr: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    o = rng.permutation(len(y))
    n_steps = math.ceil(len(o) / bs)
    opt = torch.optim.AdamW(enc.model.parameters(), lr=lr, weight_decay=0.01)
    warm = max(1, int(0.05 * n_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (n_steps - s) / (n_steps - warm))))
    pos_w = torch.tensor(1.0, device=enc.device)
    enc.model.train()
    t, run = time.time(), 0.0
    for st in range(n_steps):
        i = o[st * bs:(st + 1) * bs]
        out = enc.logits(toks, ta[i], tb[i])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(out, torch.from_numpy(y[i]).float().to(enc.device),
                                                                    pos_weight=pos_w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.model.parameters(), 1.0)
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if st else loss.item()
        if st % 500 == 0 or st == n_steps - 1:
            log(f"  step {st}/{n_steps} loss {run:.4f} ({(st + 1) * bs / (time.time() - t):.0f} pairs/s)")
    enc.model.eval()


@torch.no_grad()
def predict(enc: Encoder, toks, ta: np.ndarray, tb: np.ndarray, bs: int) -> np.ndarray:
    enc.model.eval()
    # sort by length so padding stays small, then restore the order
    tl = np.fromiter((len(x) for x in toks), np.int32, len(toks))
    ln = tl[ta] + tl[tb]
    o = np.argsort(ln, kind="stable")
    out = np.empty(len(ta), np.float32)
    t = time.time()
    for s in range(0, len(o), bs):
        i = o[s:s + bs]
        out[i] = enc.logits(toks, ta[i], tb[i]).cpu().numpy()
        if (s // bs) % 2000 == 0:
            log(f"  predict {s}/{len(o)} ({(s + len(i)) / (time.time() - t):.0f} pairs/s)")
    return out


def pair_index(pool: pd.DataFrame, txt: pd.Series):
    """Row indices of s1 / cand in txt (-1 = missing -> empty text appended at the end)."""
    idx = pd.Series(np.arange(len(txt)), index=txt.index)
    ia = idx.reindex(pool["s1"]).fillna(len(txt)).to_numpy(np.int64)
    ib = idx.reindex(pool["cand"]).fillna(len(txt)).to_numpy(np.int64)
    return ia, ib


def sample_train(p: pd.DataFrame, cap: int, seed: int) -> np.ndarray:
    """All positives, then negatives by pool rank (hard first; random within a rank) until the cap."""
    pos = np.flatnonzero(p["y"].to_numpy() == 1)
    neg = np.flatnonzero(p["y"].to_numpy() == 0)
    n_neg = max(0, min(len(neg), cap - len(pos)))
    rk = p["rank"].to_numpy()[neg].astype(np.float64) + np.random.default_rng(seed).random(len(neg))
    return np.sort(np.r_[pos, neg[np.argsort(rk)[:n_neg]]])


def cmd_train(a) -> None:
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.pool, "train__*.parquet")))
    parts = []
    for f in files:
        c = os.path.basename(f)[len("train__"):-len(".parquet")]
        p = pd.read_parquet(f)
        if a.limit_s1:
            keep_s1 = np.sort(p["s1"].unique())[:a.limit_s1]
            p = p[p["s1"].isin(keep_s1)]
        p["country"] = c
        parts.append(p)
    pool = pd.concat(parts, ignore_index=True)
    del parts
    fmap = fold_of(pool["s1"].to_numpy(), a.folds, a.seed)
    pool["fold"] = pool["s1"].map(fmap).astype(np.int8)
    log(f"train pool: {len(pool)} pairs, {pool.s1.nunique()} S1, {int(pool.y.sum())} positives; folds {np.bincount(pool.fold)}")
    ta = np.empty(len(pool), np.int64)
    tb = np.empty(len(pool), np.int64)
    enc0 = Encoder(a.model, a.max_len, "cpu")
    toks = []
    for c, g in pool.groupby("country"):
        txt = load_text(a.store, "train", c, set(g["s1"]) | set(g["cand"]))
        ia, ib = pair_index(g, txt)
        ta[g.index], tb[g.index] = ia + len(toks), ib + len(toks)
        toks += enc0.tokenize(txt.to_numpy()) + [[]]
        del txt
    del enc0
    log(f"texts tokenised: {len(toks)}")
    ce = np.full(len(pool), np.nan, np.float32)
    for f in range(a.folds):
        tr = pool.index[pool.fold != f].to_numpy()
        sel = tr[sample_train(pool.loc[tr], a.max_train, a.seed + f)]
        log(f"fold {f}: fine-tune on {len(sel)} pairs ({int(pool.y.to_numpy()[sel].sum())} pos) from {len(tr)}")
        torch.manual_seed(a.seed + f)
        enc = Encoder(a.model, a.max_len, "cuda")
        t = time.time()
        train_model(enc, toks, ta[sel], tb[sel], pool.y.to_numpy()[sel].astype(np.float32), a.bs, a.lr, a.seed + f)
        log(f"fold {f}: trained in {time.time() - t:.0f}s")
        enc.model.save_pretrained(os.path.join(a.out, f"fold{f}"))
        enc.tok.save_pretrained(os.path.join(a.out, f"fold{f}"))
        te = pool.index[pool.fold == f].to_numpy()
        ce[te] = predict(enc, toks, ta[te], tb[te], a.pred_bs)
        log(f"fold {f}: OOF scored {len(te)} pairs")
        del enc
        torch.cuda.empty_cache()
    pool["ce"] = ce
    pool[["s1", "cand", "country", "fold", "pa", "rank", "keep", "y", "ce"]].to_parquet(os.path.join(a.out, "train_ce.parquet"),
                                                                                    index=False)
    report(pool, a.store)


def report(pool: pd.DataFrame, store: str) -> None:
    from sklearn.metrics import roc_auc_score
    gt = pd.read_parquet(os.path.join(store, "train_gt_pairs.parquet"))
    n_gt = gt[gt["s1"].isin(set(pool["s1"]))].groupby("s1").size()
    pool["blend"] = 0.5 * (1 / (1 + np.exp(-pool["ce"]))) + 0.5 * pool["pa"]
    for c, g in pool.groupby("country"):
        tot = int(n_gt.reindex(g["s1"].unique()).fillna(0).sum())
        msg = [f"{c}: AUC pa {roc_auc_score(g.y, g.pa):.4f} ce {roc_auc_score(g.y, g.ce):.4f}; GT pairs {tot}, in pool {int(g.y.sum())}"]
        for col in ("pa", "ce", "blend"):
            r = g.assign(r=g.groupby("s1")[col].rank(ascending=False, method="first"))
            for k in (5, 10, 12):
                msg.append(f"recall@{k} by {col} {r.loc[r.r <= k, 'y'].sum() / max(tot, 1):.4f}")
        log(" | ".join(msg))
        log(f"{c}: recall of the pipeline's kept set {g.loc[g.keep == 1, 'y'].sum() / max(tot, 1):.4f} "
            f"({g.keep.sum() / g.s1.nunique():.2f} cands/S1)")
        ce_p = 1 / (1 + np.exp(-g["ce"].to_numpy()))
        n_s1 = g.s1.nunique()
        for w in (0.3, 0.5, 0.7, 0.85, 1.0):  # pipeline --ce-w sweep: top-10 by blend with blend >= 0.005 (without sure pairs)
            b = w * ce_p + (1 - w) * g["pa"].to_numpy()
            r = pd.Series(b).groupby(g["s1"].to_numpy()).rank(ascending=False, method="first").to_numpy()
            for top in (8, 10):
                k = (r <= top) & (b >= 0.005)
                log(f"{c}: w {w} top {top}: recall {g['y'].to_numpy()[k].sum() / max(tot, 1):.4f} at {k.sum() / n_s1:.2f} cands/S1")


def cmd_test(a) -> None:
    files = sorted(glob.glob(os.path.join(a.pool, "test__*.parquet")))
    encs = [Encoder(os.path.join(a.out, f"fold{f}"), a.max_len, "cuda") for f in range(a.folds)]
    for f in files:
        base = os.path.basename(f)[len("test__"):-len(".parquet")]
        outp = os.path.join(a.out, f"test_ce__{base}.parquet")
        if os.path.exists(outp):
            continue
        c = base.split("__")[0]
        p = pd.read_parquet(f)
        if a.test_top:
            p = p[(p["rank"] < a.test_top) | (p["keep"] == 1)]
        txt = load_text(a.store, "test", c, set(p["s1"]) | set(p["cand"]))
        ta, tb = pair_index(p, txt)
        toks = encs[0].tokenize(txt.to_numpy()) + [[]]
        prob = np.mean([1 / (1 + np.exp(-predict(e, toks, ta, tb, a.pred_bs))) for e in encs], axis=0)
        p.assign(ce_p=prob.astype(np.float32))[["s1", "cand", "pa", "rank", "keep", "ce_p"]].to_parquet(outp, index=False)
        log(f"test {base}: {len(p)} pairs scored")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "test", "report"])
    ap.add_argument("--pool", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-train", type=int, default=3_000_000)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--pred-bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--limit-s1", type=int, default=0, help="dev: first N S1 per country")
    ap.add_argument("--test-top", type=int, default=40, help="score only the top-N pool pairs per test S1 (+ kept)")
    a = ap.parse_args()
    if a.cmd == "train":
        cmd_train(a)
    elif a.cmd == "test":
        cmd_test(a)
    else:
        report(pd.read_parquet(os.path.join(a.out, "train_ce.parquet")), a.store)


if __name__ == "__main__":
    main()
