"""Pairwise match model + decision rules (probabilities -> per-S1 match sets)."""
from __future__ import annotations

import time
from typing import Dict, FrozenSet, List, Tuple

import numpy as np
import pandas as pd

from .metric import expected_f05_topk, macro_f05


# --------------------------------------------------------------------- model
def make_model(seed: int = 0, n_jobs: int = -1):
    """LightGBM when installed (box/laptop), else sklearn's HistGradientBoosting (sandbox)."""
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=700, learning_rate=0.05, num_leaves=63, min_child_samples=40,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                  n_jobs=n_jobs, random_state=seed, verbose=-1), "lightgbm"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
                                              min_samples_leaf=40, l2_regularization=1.0,
                                              early_stopping=False, random_state=seed), "sklearn-hgb"


def subsample_negatives(y: np.ndarray, neg_rate: float, seed: int = 0):
    """Row indices keeping all positives and a `neg_rate` share of negatives, with weights 1/neg_rate on
    the kept negatives so probabilities stay calibrated to the full pair distribution."""
    if neg_rate >= 1.0:
        return np.arange(len(y)), None
    rng = np.random.default_rng(seed)
    keep = (y == 1) | (rng.random(len(y)) < neg_rate)
    idx = np.flatnonzero(keep)
    w = np.where(y[idx] == 1, 1.0, 1.0 / neg_rate).astype(np.float32)
    return idx, w


def oof_predict(X, y: np.ndarray, groups: np.ndarray, n_folds: int = 5, seed: int = 0,
                verbose: bool = True, neg_rate: float = 1.0, n_jobs: int = -1) -> Tuple[np.ndarray, str]:
    """GroupKFold OOF probabilities for ALL rows; each fold model is fitted on (optionally negative-subsampled,
    weighted) training rows. X may be a DataFrame or a float32 ndarray."""
    from sklearn.model_selection import GroupKFold
    Xv = X.to_numpy(dtype=np.float32) if hasattr(X, "to_numpy") else X
    oof = np.zeros(len(Xv), dtype=np.float32)
    kind = ""
    for f, (tr, va) in enumerate(GroupKFold(n_splits=n_folds).split(Xv, y, groups)):
        m, kind = make_model(seed + f, n_jobs=n_jobs)
        sub, w = subsample_negatives(y[tr], neg_rate, seed + f)
        idx = tr[sub]  # index once: Xv[tr][sub] would materialise a second full-fold copy (X may be a memmap)
        t = time.time()
        m.fit(np.ascontiguousarray(Xv[idx]), y[idx], sample_weight=w)
        oof[va] = m.predict_proba(Xv[va])[:, 1]
        if verbose:
            print(f"    fold {f}: fit={len(sub)} valid={len(va)} pos_rate={y[va].mean():.4f} ({time.time() - t:.0f}s)", flush=True)
    return oof, kind


def feature_importance(model, cols) -> pd.Series:
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return pd.Series(dtype=float)
    return pd.Series(imp, index=cols).sort_values(ascending=False)


# ------------------------------------------------------------ decision rules
def check_one_to_one(gt) -> int:
    """Number of S2/S3 ids that appear under more than one S1 id in the ground truth."""
    cnt = pd.Series([m for ms in gt.values() for m in ms]).value_counts()
    return int((cnt > 1).sum())


def one_to_one_filter(pairs: pd.DataFrame) -> pd.DataFrame:
    """Keep, for every candidate record c, only its most probable S1 (S1 is deduplicated)."""
    best = pairs.groupby("c")["p"].transform("max")
    return pairs[pairs["p"] >= best]


def decide(pairs: pd.DataFrame, q_rid: np.ndarray, d_rid: np.ndarray, rule: str = "thr", thr: float = 0.5,
           one2one: bool = True, min_p: float = 0.02) -> Dict[str, FrozenSet[str]]:
    """pairs: columns q, c, p (row indices into the query block / doc table).

    Returns {s1_rid: frozenset(match rids)} for EVERY query in q_rid (empty set = no match).
    """
    out = {r: frozenset() for r in q_rid}
    df = pairs[["q", "c", "p"]]
    df = df[df["p"] >= min_p]
    if one2one and len(df):
        df = one_to_one_filter(df)
    if rule == "thr":
        sel = df[df["p"] >= thr]
        for q, grp in sel.groupby("q")["c"]:
            out[q_rid[q]] = frozenset(d_rid[grp.values])
    elif rule == "expf":
        df = df.sort_values(["q", "p"], ascending=[True, False])
        qs, cs, ps = df["q"].values, df["c"].values, df["p"].values
        bounds = np.flatnonzero(np.diff(qs)) + 1
        for q_arr, c_arr, p_arr in zip(np.split(qs, bounds), np.split(cs, bounds), np.split(ps, bounds)):
            if not len(q_arr):
                continue
            idx, _ = expected_f05_topk(p_arr, max_k=10)
            if len(idx):
                out[q_rid[q_arr[0]]] = frozenset(d_rid[c_arr[idx]])
    else:
        raise ValueError(rule)
    return out


def sweep_rules(pairs: pd.DataFrame, q_rid: np.ndarray, d_rid: np.ndarray, gt, one2one_ok: bool = True,
                thrs=np.round(np.arange(0.10, 0.951, 0.05), 3), q_subset=None) -> Tuple[pd.DataFrame, dict]:
    """Score every rule on OOF probabilities. q_subset: query row indices to evaluate (default all)."""
    eval_q = q_rid if q_subset is None else q_rid[q_subset]
    gt_sub = {r: gt.get(r, frozenset()) for r in eval_q}
    rows: List[dict] = []
    for o2o in ([False, True] if one2one_ok else [False]):
        for t in thrs:
            rows.append(dict(rule="thr", one2one=o2o, thr=float(t),
                             f05=macro_f05(gt_sub, decide(pairs, q_rid, d_rid, "thr", t, o2o))))
        rows.append(dict(rule="expf", one2one=o2o, thr=np.nan,
                         f05=macro_f05(gt_sub, decide(pairs, q_rid, d_rid, "expf", 0, o2o))))
    tab = pd.DataFrame(rows).sort_values("f05", ascending=False).reset_index(drop=True)
    return tab, tab.iloc[0].to_dict()
