# slice_frnorm — regression check of the France normalisation fixes on US/India (HQ sandbox, 26 Sep)

Same data/config as runs/slice_v3 (8% slice, 40K train S1, 6 views, k=10, max_df 0.01, sklearn HistGB, 3 folds,
30K-S1 train-derived hidden holdout), old HQ pipeline; only `src/normalize.py` + `src/features.py` changed
(docs/FRANCE_FIXES.md).

| | slice_v3 | slice_frnorm |
|---|---|---|
| blocking pair recall k=10 | 0.98659 | 0.98679 |
| OOF AUC / AP | 0.99989 / 0.99830 | 0.99990 / 0.99848 |
| OOF macro F0.5 | 0.97683 (thr 0.80) | **0.97769** (thr 0.85) |
| hidden holdout macro F0.5 | 0.98257 | 0.98249 |
| holdout India / US empty rate | — | 6.2% / 5.5% |

=> no US/India regression (OOF +0.09 pt, holdout −0.01 pt = noise). The France-specific gains cannot be measured
(no labels); pseudo-pair proxies in docs/FRANCE_FIXES.md.
