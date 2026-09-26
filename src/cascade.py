"""Cascade filter = the LAST candidate-generation stage (what candidate_pairs.tsv contains).

A cheap gradient-boosted model on the stage-A features only (view cosines, ranks, gaps, margins,
reverse ranks, mutual-best — no string features) scores every blocking pair; per S1 entity we keep
the top-N by that score, subject to a small probability floor. On the 8% slice this cut the candidate
set from ~66 to ~10 per S1 at 98.1% pair recall with no measurable F0.5 loss (see
scripts/cascade_study.py). The full matcher (stage B + GBDT) then runs only on the kept pairs.

Training: OOF (GroupKFold by S1) probabilities for the train pairs so the downstream pruning of the
training set is not optimistic; a final model is fitted on all train pairs for the test split.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

STAGE_A_EXCLUDE = {"q", "c", "y", "p", "pa", "s1", "cand", "n_true", "part", "c_src"}


def stage_a_columns(cands: pd.DataFrame) -> List[str]:
    cols = [c for c in cands.columns if c not in STAGE_A_EXCLUDE]
    return cols + ["c_src"] if "c_src" in cands else cols


def make_cascade_model(seed: int = 0, n_jobs: int = -1):
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.08, num_leaves=31, min_child_samples=50,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.9, n_jobs=n_jobs,
                                  random_state=seed, verbose=-1, deterministic=True, force_row_wise=True)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                              min_samples_leaf=50, random_state=seed)


def fit_cascade(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_folds: int = 3, seed: int = 0,
                verbose: bool = True, n_jobs: int = -1) -> Tuple[np.ndarray, object]:
    """OOF cascade probabilities for the training pairs + a final model fitted on all of them.
    n_jobs: LightGBM threads (box: pass --n-jobs; all 20 threads spin-wait under load). X may be a memmap."""
    from sklearn.model_selection import GroupKFold
    oof = np.zeros(len(X), dtype=np.float32)
    for f, (tr, va) in enumerate(GroupKFold(n_splits=n_folds).split(X, y, groups)):
        m = make_cascade_model(seed + f, n_jobs=n_jobs)
        m.fit(np.ascontiguousarray(X[tr]), y[tr])
        oof[va] = m.predict_proba(np.ascontiguousarray(X[va]))[:, 1]
    final = make_cascade_model(seed, n_jobs=n_jobs)
    final.fit(np.ascontiguousarray(X), y)
    if verbose:
        from sklearn.metrics import average_precision_score
        print(f"    cascade model: AP={average_precision_score(y, oof):.5f} on {len(y)} pairs, {X.shape[1]} features", flush=True)
    return oof, final


def cascade_keep(q: np.ndarray, pa: np.ndarray, top_n: int = 10, floor: float = 0.005) -> np.ndarray:
    """Boolean mask: keep the top_n pairs per query by cascade probability, dropping any below `floor`."""
    order = np.lexsort((-pa, q))            # by query, best first
    q_sorted = q[order]
    first = np.r_[0, np.flatnonzero(q_sorted[1:] != q_sorted[:-1]) + 1]
    rank = np.arange(len(order)) - np.repeat(first, np.diff(np.r_[first, len(order)]))
    keep = np.zeros(len(q), dtype=bool)
    keep[order] = (rank < top_n) & (pa[order] >= floor)
    return keep


def candidate_stats(q: np.ndarray, n_queries: int, y: np.ndarray = None, n_true_pairs: int = None) -> Dict:
    """Mean / median / p90 / max candidates per query (queries with zero candidates included) + pair recall."""
    counts = np.bincount(q, minlength=n_queries) if n_queries else np.bincount(q)
    out = dict(pairs=int(len(q)), n_s1=int(n_queries), mean=float(counts.mean()), median=float(np.median(counts)),
               p90=float(np.percentile(counts, 90)), max=int(counts.max()) if len(counts) else 0,
               zero_share=float((counts == 0).mean()))
    if y is not None and n_true_pairs:
        out["pair_recall"] = float(y.sum() / n_true_pairs)
    return out
