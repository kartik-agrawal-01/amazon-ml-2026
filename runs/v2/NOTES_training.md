# v2 training notes (attempt 2 trained the model; attempt 3 = same model via --load-model, test only)

Model file: output_v2_train/model.joblib (LightGBM, 83 features, cascade top 10 / floor 0.002).
Training log: runs/v2/stdout_attempt2_train_only.txt. Train S1 sample 150,000 (India 60,031 / US 89,969),
4 views name_c3,name_w,addr_c3,full_w, k=10, --max-df 0.01, GPU top-k.

- blocking pair recall (k=10, train sample): India 193942/207851 = 0.933, US 305685/310675 = 0.984; 59.3 cands/S1
- cascade: 59.3 -> 9.2 cands/S1 (median 10), train pair recall after cascade 0.9533, zero-cand share 0.0000
- OOF (lightgbm, 3 folds): AUC 0.99876, AP 0.99796
- OOF macro F0.5 (chosen) = 0.96231, rule thr 0.70 + one-to-one (expected-F rule 0.96025)
- OOF breakdown (150,000 S1): singletons 5.6% mean F 0.9398 (pred-empty 94.0%); 1 match 5.5% F 0.9031 (pred-empty 8.2%);
  2 matches 17.0% F 0.9551; 3+ matches 71.9% F 0.9703
- top features: jw_name, jw_addr, num_first_jw, cos_full_w, jw_ph, len_ratio, addr_len_q, cos_addr_c3, addr_len_c, c_cnt_core_all
- train runtime to "model fitted": 3555 s, peak RSS 7.9 GB

Test blocks so far (attempt 3): France 3 blocks, 9.8 cands/S1 kept, S1 with matches 94.7% / 95.0% / 94.8% (empty ~5.2%);
India blocks 0-1: 9.2 cands/S1, S1 with matches 93.4% / 93.3% (empty ~6.6%). Block times: France ~605 s, India ~1050-1180 s.
