"""Fine-tune a small bi-encoder (MiniLM / bge-small; Apache-2.0 / MIT) on our train matches.

Supervised contrastive training (MultipleNegativesRankingLoss) on (S1 text, matching S2/S3 text)
pairs, with HARD negatives = the top lexical/dense candidate that is NOT a match. Yields a
dense blocking view tuned to this data (SC-Block style). GPU strongly recommended.

Run on the box (repo root, conda env aml):
  python scripts/finetune_biencoder.py --data-dir data --out models/minilm-er --pairs 300000 --epochs 1
Then:  python -m src.pipeline ... --dense --dense-model models/minilm-er

Hard negatives come from `output/oof_pairs.tsv.gz` (written by a previous pipeline run) when
present: for each S1 we take its highest-probability NON-match candidate. Without it we fall
back to in-batch negatives only.
Status: written before the GPU was reachable from HQ — untested; check the sentence-transformers
version on the box (>=3.0 trainer API) and fix imports if needed.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data import load_split  # noqa: E402
from src.dense import dense_text  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--base", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out", default="models/minilm-er")
    ap.add_argument("--pairs", type=int, default=300_000, help="number of (anchor, positive) pairs")
    ap.add_argument("--hard-neg", default="output/oof_pairs.tsv.gz")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    rec, gt, _ = load_split(a.data_dir, "train")
    text = pd.Series(dense_text(rec), index=rec["rid"].values)
    pairs = [(s1, m) for s1, ms in gt.items() for m in ms]
    rng.shuffle(pairs)
    pairs = pairs[:a.pairs]
    anchors = [text[s1] for s1, _ in pairs]
    positives = [text[m] for _, m in pairs]
    negatives = None
    if os.path.exists(a.hard_neg):
        oof = pd.read_csv(a.hard_neg, sep="\t", dtype={"s1": str, "cand": str})
        hn = oof[oof.y == 0].sort_values("p", ascending=False).drop_duplicates("s1").set_index("s1")["cand"]
        negatives = [text[hn[s1]] if s1 in hn.index and hn[s1] in text.index else None for s1, _ in pairs]
        keep = [i for i, n in enumerate(negatives) if n is not None]
        print(f"hard negatives available for {len(keep)}/{len(pairs)} pairs")
        if len(keep) > len(pairs) // 2:
            anchors = [anchors[i] for i in keep]
            positives = [positives[i] for i in keep]
            negatives = [negatives[i] for i in keep]
        else:
            negatives = None

    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses

    model = SentenceTransformer(a.base)
    model.max_seq_length = 64
    cols = {"anchor": anchors, "positive": positives}
    if negatives is not None:
        cols["negative"] = negatives
    ds = Dataset.from_dict(cols)
    loss = losses.MultipleNegativesRankingLoss(model)
    args = SentenceTransformerTrainingArguments(
        output_dir=a.out + "_ckpt", num_train_epochs=a.epochs, per_device_train_batch_size=a.batch,
        learning_rate=a.lr, warmup_ratio=0.05, fp16=True, logging_steps=200, save_strategy="no",
        seed=a.seed, report_to=[],
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=loss)
    trainer.train()
    model.save(a.out)
    print("saved fine-tuned model to", a.out)


if __name__ == "__main__":
    main()
