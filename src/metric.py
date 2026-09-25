"""Local scorer + decision helpers for Amazon ML Challenge 2026 (business entity resolution).

Metric, as defined in the problem statement (PS video, slide 5):
  * Score = MACRO F0.5 over every Source-1 (S1) entity in the ground truth.
  * Per entity:  F0.5 = 1.25 * P * R / (0.25 * P + R)
                      = 1.25*TP / (1.25*TP + 0.25*FN + FP)
  * Singleton (S1 entity with no true match): empty prediction -> 1.0, any prediction -> 0.0.
  * Entity with true matches but an empty (or all-wrong) prediction -> 0.0.
  => one wrong link costs 4x a missed link in the denominator: precision first.

File format (ground truth and submission): TSV, one row per S1 id; second column is a
comma-separated list of matching S2/S3 ids (empty = no match).
Re-check this against utils/validate_submission.py once the dataset zip arrives.

Usage:
  python src/metric.py GROUND_TRUTH.tsv PREDICTION.tsv     # prints macro F0.5 + breakdown
  python src/metric.py --selftest                          # sanity checks on toy cases
"""
from __future__ import annotations

import sys
from typing import Dict, FrozenSet, Iterable, Tuple

import numpy as np
import pandas as pd

BETA2 = 0.25  # beta = 0.5 -> beta^2 = 0.25

Matches = Dict[str, FrozenSet[str]]


# --------------------------------------------------------------------------- I/O
def parse_id_list(value) -> FrozenSet[str]:
    """'S2-1, S3-9' -> frozenset({'S2-1', 'S3-9'}); '', NaN, None -> empty set."""
    if value is None:
        return frozenset()
    if isinstance(value, float) and np.isnan(value):
        return frozenset()
    s = str(value).strip().strip("[]")
    if not s or s.lower() in {"nan", "none", "null"}:
        return frozenset()
    return frozenset(t.strip().strip("'\"") for t in s.split(",") if t.strip().strip("'\""))


def read_matches_tsv(path: str) -> Matches:
    """Read a GT or submission TSV into {s1_id: frozenset(match_ids)}.

    Column 0 = S1 id, column 1 = comma-separated matches. Handles a missing header row
    (first cell already looks like an id) and files whose match column is entirely empty.
    """
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    first_col = str(df.columns[0])
    if any(ch.isdigit() for ch in first_col) and "id" not in first_col.lower():
        # no header: the first data row was consumed as column names -> re-read
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, header=None)
    if df.shape[1] < 2:
        df["_matches"] = ""
    ids = df.iloc[:, 0].astype(str).str.strip()
    out: Matches = {}
    for i, m in zip(ids, df.iloc[:, 1]):
        if i in out:
            raise ValueError(f"{path}: duplicate S1 id {i!r}")
        out[i] = parse_id_list(m)
    return out


def write_matches_tsv(pred: Matches, path: str, header=("source1_id", "matches")) -> None:
    """Write {s1_id: set} as TSV. Header names are placeholders until we see the official format."""
    rows = [(k, ",".join(sorted(v))) for k, v in pred.items()]
    pd.DataFrame(rows, columns=list(header)).to_csv(path, sep="\t", index=False)


# ------------------------------------------------------------------------ metric
def f05(true: Iterable[str], pred: Iterable[str]) -> float:
    """Per-entity F0.5 with the PS singleton rule."""
    true, pred = frozenset(true), frozenset(pred)
    if not true:
        return 1.0 if not pred else 0.0
    tp = len(true & pred)
    if tp == 0:
        return 0.0
    fp = len(pred) - tp
    fn = len(true) - tp
    return (1 + BETA2) * tp / ((1 + BETA2) * tp + BETA2 * fn + fp)


def macro_f05(truth: Matches, pred: Matches, verbose: bool = False) -> float:
    """Mean per-entity F0.5 over every S1 id in `truth`. Missing ids in `pred` count as empty."""
    extra = set(pred) - set(truth)
    scores = np.array([f05(t, pred.get(k, frozenset())) for k, t in truth.items()], dtype=float)
    if verbose:
        n_true = np.array([len(t) for t in truth.values()])
        n_pred = np.array([len(pred.get(k, ())) for k in truth])
        print(f"macro F0.5 = {scores.mean():.5f}   over {len(scores)} S1 entities")
        print(f"  S1 ids missing from pred: {sum(k not in pred for k in truth)} | extra ids in pred: {len(extra)}")
        for label, mask in [("singletons (0 true)", n_true == 0),
                            ("1 true match", n_true == 1),
                            ("2 true matches", n_true == 2),
                            ("3+ true matches", n_true >= 3)]:
            if mask.any():
                print(f"  {label:<20} n={mask.sum():>7} ({mask.mean():6.1%})  "
                      f"mean F={scores[mask].mean():.4f}  pred-empty={np.mean(n_pred[mask] == 0):.1%}")
    return float(scores.mean())


# ------------------------------------------------------------- decision helpers
def poisson_binomial(p: np.ndarray) -> np.ndarray:
    """Distribution of the number of successes of independent Bernoulli(p_i)."""
    dist = np.zeros(len(p) + 1)
    dist[0] = 1.0
    for q in p:
        dist[1:] = dist[1:] * (1 - q) + dist[:-1] * q  # RHS uses the old dist
        dist[0] *= 1 - q
    return dist


def expected_f05_topk(probs, max_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Pick the prediction set for ONE S1 entity that maximises expected F0.5.

    probs: calibrated match probabilities of that entity's candidates (any order),
           assumed independent. Under independence the optimal set is a top-k by probability,
           so we evaluate k = 0..max_k exactly (Poisson-binomial over TP-in-set and
           positives-left-out) and keep the best.
    Returns (indices_into_probs_to_predict, expected_score_per_k).
    """
    p = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    order = np.argsort(-p, kind="stable")
    ps = p[order]
    n = len(ps)
    kmax = min(max_k, n)
    exp = np.zeros(kmax + 1)
    exp[0] = float(np.prod(1 - ps))  # empty prediction scores 1 only if nothing is a match
    for k in range(1, kmax + 1):
        dx = poisson_binomial(ps[:k])          # TP inside the chosen set
        dy = poisson_binomial(ps[k:])          # true matches left out (FN)
        x = np.arange(k + 1)[:, None]
        y = np.arange(n - k + 1)[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(x > 0, (1 + BETA2) * x / (BETA2 * x + k + BETA2 * y), 0.0)
        exp[k] = float((dx[:, None] * dy[None, :] * f).sum())
    best = int(np.argmax(exp))
    return order[:best], exp


# ---------------------------------------------------------------------- selftest
def _selftest() -> None:
    a, b, c = "S2-a", "S3-b", "S2-c"
    assert f05({a}, {a}) == 1.0
    assert abs(f05({a}, {a, c}) - 1.25 / 2.25) < 1e-12          # 0.5556
    assert abs(f05({a, b}, {a}) - 1.25 / 1.50) < 1e-12          # 0.8333
    assert abs(f05({a, b}, {a, c}) - 1.25 / 2.50) < 1e-12       # 0.5
    assert f05(set(), set()) == 1.0 and f05(set(), {a}) == 0.0 and f05({a}, set()) == 0.0
    assert f05({a}, {c}) == 0.0
    truth = {"S1-1": frozenset(), "S1-2": frozenset({a}), "S1-3": frozenset({a, b})}
    pred = {"S1-1": frozenset(), "S1-2": frozenset({a, c})}
    assert abs(macro_f05(truth, pred) - (1 + 1.25 / 2.25 + 0) / 3) < 1e-12
    assert parse_id_list("") == frozenset() and parse_id_list(" S2-1 , S3-2 ") == {"S2-1", "S3-2"}
    # expected-F selection: confident single candidate -> predict it; weak one -> predict nothing
    idx, e = expected_f05_topk([0.9]);        assert list(idx) == [0] and abs(e[1] - 0.9) < 1e-12
    idx, e = expected_f05_topk([0.3]);        assert len(idx) == 0 and abs(e[0] - 0.7) < 1e-12
    # brute-force check of the expectation on a small random case
    rng = np.random.default_rng(0)
    p = rng.uniform(size=5)
    _, e = expected_f05_topk(p, max_k=5)
    order = np.argsort(-p, kind="stable")
    for k in range(6):
        chosen = set(order[:k].tolist())
        tot = 0.0
        for mask in range(32):
            truth_set = {i for i in range(5) if mask >> i & 1}
            prob = np.prod([p[i] if i in truth_set else 1 - p[i] for i in range(5)])
            tot += prob * f05(truth_set, chosen)
        assert abs(tot - e[k]) < 1e-10, (k, tot, e[k])
    print("metric.py selftest OK")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        _selftest()
    elif len(sys.argv) == 3:
        macro_f05(read_matches_tsv(sys.argv[1]), read_matches_tsv(sys.argv[2]), verbose=True)
    else:
        print(__doc__)
        sys.exit(1)
