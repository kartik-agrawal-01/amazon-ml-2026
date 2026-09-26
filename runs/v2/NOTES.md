# v2 NOTES (auto-written by scripts/night/post_v2.sh)

- command: `bash scripts/run_pipeline.sh v2 --n-jobs 8 --max-df 0.01 --train-s1 150000 --block-size 100000 --views name_c3,name_w,addr_c3,full_w --topk-device cuda --stage-b-jobs 4 --load-model output_v2_train/model.joblib`
- commit: ea7375a
- runtime: 4.93 h | peak RSS: {'store': 651, 'test': 7263}
- validator exit code: 0 (see runs/v2/validate.txt) | matching_results.tsv 94.7 MB
- OOF: {'rule': 'thr', 'one2one': True, 'thr': 0.7, 'f05': 0.9623078254314535} | AUC None
- cascade: before None | after None
- blocking recall (train): None
- test candidates: {'overall': {'pairs': 15881604, 'n_s1': 1732544, 'mean': 9.166638192161354, 'median': 10.0, 'p90': 10.0, 'max': 10, 'zero_share': 4.6174873480846666e-06}, 'per_country': {'france': {'pairs': 2544878, 'n_s1': 259452, 'mean': 9.808665957479612, 'median': 10.0, 'p90': 10.0, 'max': 10, 'zero_share': 0.0}, 'india': {'pairs': 7437928, 'n_s1': 809986, 'mean': 9.182785875311426, 'median': 10.0, 'p90': 10.0, 'max': 10, 'zero_share': 7.4075354388841286e-06}, 'us': {'pairs': 5898798, 'n_s1': 663106, 'mean': 8.895708981671106, 'median': 10.0, 'p90': 10.0, 'max': 10, 'zero_share': 3.0161090383739555e-06}}}
- test pred overall: {'n_s1': 1732544, 'empty_rate': 0.05989285120608769, 'mean_matches': 3.2341481659340254, 'cand_pairs': 15881604, 'cand_pairs_per_s1': 9.166638192161354}

## Test per-country

| country | n_s1 | empty_rate | mean_matches | cand_mean | cand_median |
|---|---|---|---|---|---|
| france | 259452 | 0.052 | 3.30 | 9.8 | 10 |
| india | 809986 | 0.067 | 3.12 | 9.2 | 10 |
| us | 663106 | 0.055 | 3.35 | 8.9 | 10 |
