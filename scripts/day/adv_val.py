"""QUEUE 6 (France): adversarial validation France vs US+India on the v2 test pair features.

Input: output_france_diag/pairs_{country}.parquet (v2 model + v2 normaliser, test block-0 sample, 20K S1 each).
For two pair populations (all candidates; pairs with v2 p >= 0.5), trains LightGBM France-vs-rest, reports the
3-fold AUC and the top features by gain, then drops the top feature and repeats (5 rounds) to show how much of
the shift is concentrated in a few features. Also a per-feature univariate AUC (|AUC-0.5|) table.
Light work: ~150K rows per class, 4 threads.  python scripts/day/adv_val.py > runs/day/adv_val.md
"""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

D = "output_france_diag"
DROP = {"q", "c", "s1", "cand", "pa", "p", "matched", "score"}
N = 150_000
rng = np.random.default_rng(0)


def load(c, pmin):
    df = pd.read_parquet(f"{D}/pairs_{c}.parquet")
    df = df[df["p"] >= pmin] if pmin > 0 else df
    feats = [x for x in df.columns if x not in DROP]
    return df[feats].astype(np.float32)


def cv_auc(X, y, rounds=200):
    oof = np.zeros(len(y)); imp = np.zeros(X.shape[1])
    for tr, va in StratifiedKFold(3, shuffle=True, random_state=0).split(X, y):
        m = lgb.LGBMClassifier(n_estimators=rounds, learning_rate=0.1, num_leaves=63, n_jobs=4, verbose=-1,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
        m.fit(X.iloc[tr], y[tr]); oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        imp += m.booster_.feature_importance("gain")
    return roc_auc_score(y, oof), pd.Series(imp / imp.sum(), index=X.columns).sort_values(ascending=False)


def run(pmin, label):
    fr = load("france", pmin); rest = pd.concat([load("us", pmin), load("india", pmin)])
    fr = fr.sample(min(N, len(fr)), random_state=0); rest = rest.sample(min(N, len(rest)), random_state=0)
    X = pd.concat([fr, rest], ignore_index=True); y = np.r_[np.ones(len(fr)), np.zeros(len(rest))]
    print(f"\n## {label} (France {len(fr):,} vs US+India {len(rest):,})\n")
    uni = {}
    for col in X.columns:
        v = X[col].fillna(-999).to_numpy()
        if np.unique(v).size > 1:
            uni[col] = abs(roc_auc_score(y, v) - 0.5)
    uni = pd.Series(uni).sort_values(ascending=False)
    print("Univariate |AUC − 0.5| (top 15; means France / rest):\n")
    print("| feature | abs(AUC−0.5) | mean FR | mean US+IN |\n|---|---|---|---|")
    for col, v in uni.head(15).items():
        print(f"| {col} | {v:.3f} | {fr[col].mean():.3f} | {rest[col].mean():.3f} |")
    print("\nMultivariate (LightGBM 3-fold), dropping the top-gain feature each round:\n")
    print("| round | AUC | top-5 gain features (share) |\n|---|---|---|")
    cols = list(X.columns)
    for r in range(6):
        auc, imp = cv_auc(X[cols], y)
        print(f"| {r} | {auc:.4f} | " + ", ".join(f"{k} {v:.2f}" for k, v in imp.head(5).items()) + " |", flush=True)
        cols.remove(imp.index[0])


if __name__ == "__main__":
    print("# Adversarial validation: France vs US+India pair features (v2 test sample, v2 normaliser)")
    print("\nCaveat: features use the v2 normaliser; HQ's France fixes (n3) change the address/legal/extra features.")
    run(0.0, "All candidate pairs")
    run(0.5, "Likely matches (v2 p >= 0.5)")
