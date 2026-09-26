# OOF rule sweep: separate threshold for empty-address candidates

`output_v2_train/oof_pairs.tsv.gz`; empty-address share of OOF pairs 13.99%, of positives 3.47%.

| thr (addr present) | thr (empty addr) | ALL | India | US |
|---|---|---|---|---|
| 0.65 | 0.30 | 0.95770 | nan | nan |
| 0.65 | 0.40 | 0.95966 | nan | nan |
| 0.65 | 0.50 | 0.96096 | nan | nan |
| 0.65 | 0.60 | 0.96169 | nan | nan |
| 0.65 | 0.70 | 0.96203 | nan | nan |
| 0.65 | 0.80 | 0.96211 | nan | nan |
| 0.70 | 0.30 | 0.95799 | nan | nan |

(stopped early by the loop, 26 Sep 15:52 — per-country columns are nan: country-label mismatch in the script, ALL is valid.)
Reading: lowering the threshold for empty-address candidates only LOSES (0.30 → −0.0043, 0.60 → −0.0003 vs 0.70 at
thr_present 0.65); 0.80 is flat (+0.0001). Empty-address candidates are 14% of OOF pairs but 3.5% of positives, so the
model's low p there is justified. The 23% model-FN rate on empty-address GT pairs (india_recall.md) is NOT a
decision-rule problem: it needs better evidence (features), not a threshold. Tried & failed: separate empty-address threshold.
