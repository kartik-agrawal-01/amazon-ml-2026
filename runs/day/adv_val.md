# Adversarial validation: France vs US+India pair features (v2 test sample, v2 normaliser)

Caveat: features use the v2 normaliser; HQ's France fixes (n3) change the address/legal/extra features.


## Reading (loop, 26 Sep 18:55)
- France is trivially separable from US+India on the pair features: AUC 0.997 on all pairs, 0.997 on likely matches,
  still 0.985–0.994 after dropping the top 5 gain features. The shift is spread over many features, so "drop the most
  shifted features" cannot make the model country-neutral; QUEUE 6's drop/re-normalise experiment is low-EV.
- Two groups drive it: (a) ADDRESS FORMAT (addr_len_q, addr_c3_rank, cos_addr_c3, num_only_q, jw_addr): French
  addresses are similar to each other, so addr_c3_rank of a likely match is 53 vs 15 in US/India (rank ~uninformative);
  partly normaliser-dependent (v2 normaliser here, n3 fixes N°/zero-padding). (b) DECOY DENSITY (c_nq 5.5 vs 2.4,
  grp_n_close 16 vs 9, c_cnt_*_s1 +50–60%): France has ~2x the close same-name competitors per S1, which pushes the
  group-relative features toward "not the best candidate" -> lower p for French true pairs. This is the mechanism the
  box's --thr-adapt (R2 lower-only) addresses: a per-country threshold from the predicted-empty rate, not a feature fix.
- Actionable: keep --thr-adapt(+floor 0.30) for France in any of our files; for a blend with Soha's model, France is
  where our p is least calibrated (rank-average or per-country calibration rather than raw p averaging).
- 7.7 min, peak RSS 2.4 GB (above the 1 GB light-work target; free memory was 6 GB).

## All candidate pairs (France 150,000 vs US+India 150,000)

Univariate |AUC − 0.5| (top 15; means France / rest):

| feature | abs(AUC−0.5) | mean FR | mean US+IN |
|---|---|---|---|
| c_nq | 0.228 | 7.663 | 3.337 |
| grp_n_close | 0.218 | 16.396 | 9.789 |
| cos_addr_c3 | 0.159 | 0.798 | 0.645 |
| addr_c3_rank | 0.122 | 57.925 | 37.771 |
| c_cnt_addr_all | 0.119 | 1.132 | 0.893 |
| grp_n | 0.117 | 30.595 | 28.795 |
| c_cnt_nsp_s1 | 0.106 | 1.188 | 0.727 |
| c_cnt_core_all | 0.104 | 2.299 | 1.722 |
| c_cnt_core_s1 | 0.104 | 1.151 | 0.696 |
| c_cnt_ph_s1 | 0.102 | 1.359 | 0.866 |
| gap_score | 0.101 | 0.034 | 0.088 |
| gap_addr_c3 | 0.100 | 0.187 | 0.287 |
| cos_full_w | 0.097 | 0.751 | 0.680 |
| addr_len_q | 0.093 | 48.032 | 47.580 |
| n_views | 0.093 | 2.014 | 2.367 |

Multivariate (LightGBM 3-fold), dropping the top-gain feature each round:

| round | AUC | top-5 gain features (share) |
|---|---|---|
| 0 | 0.9976 | addr_len_q 0.30, c_nq 0.08, num_only_q 0.07, cos_addr_c3 0.05, addr_c3_rank 0.05 |
| 1 | 0.9957 | c_nq 0.13, cos_addr_c3 0.11, jw_addr 0.07, addr_c3_rank 0.07, addr_len_c 0.06 |
| 2 | 0.9953 | cos_addr_c3 0.11, grp_n_close 0.08, addr_c3_rank 0.07, num_only_q 0.06, r_addr_c3 0.06 |
| 3 | 0.9952 | grp_n_close 0.11, cos_full_w 0.07, num_only_q 0.06, addr_len_c 0.06, r_addr_c3 0.06 |
| 4 | 0.9947 | cos_full_w 0.09, num_only_q 0.08, r_addr_c3 0.08, addr_len_c 0.07, addr_c3_rank 0.07 |
| 5 | 0.9938 | gap_addr_c3 0.08, addr_c3_rank 0.08, num_only_q 0.08, r_addr_c3 0.07, jw_addr 0.07 |

## Likely matches (v2 p >= 0.5) (France 150,000 vs US+India 150,000)

Univariate |AUC − 0.5| (top 15; means France / rest):

| feature | abs(AUC−0.5) | mean FR | mean US+IN |
|---|---|---|---|
| addr_c3_rank | 0.243 | 53.455 | 15.004 |
| grp_n_close | 0.229 | 16.120 | 9.251 |
| c_nq | 0.207 | 5.499 | 2.376 |
| grp_n | 0.123 | 29.903 | 28.042 |
| c_cnt_core_s1 | 0.121 | 1.103 | 0.715 |
| c_cnt_ph_s1 | 0.117 | 1.344 | 0.918 |
| c_cnt_nsp_s1 | 0.116 | 1.131 | 0.761 |
| jw_core | 0.115 | 0.958 | 0.925 |
| jw_name | 0.113 | 0.941 | 0.916 |
| core_eq | 0.112 | 0.709 | 0.484 |
| addr_len_q | 0.110 | 48.200 | 47.004 |
| jw_nospace | 0.108 | 0.959 | 0.927 |
| nospace_eq | 0.107 | 0.728 | 0.514 |
| c_cnt_core_all | 0.101 | 2.234 | 1.808 |
| extra_c | 0.100 | 0.332 | 0.650 |

Multivariate (LightGBM 3-fold), dropping the top-gain feature each round:

| round | AUC | top-5 gain features (share) |
|---|---|---|
| 0 | 0.9968 | addr_len_q 0.26, addr_c3_rank 0.11, addr_len_c 0.09, c_nq 0.07, num_jacc 0.06 |
| 1 | 0.9942 | addr_c3_rank 0.16, cos_addr_c3 0.09, addr_len_c 0.08, c_nq 0.08, grp_n_close 0.06 |
| 2 | 0.9942 | grp_n_close 0.10, c_nq 0.10, addr_len_c 0.09, r_addr_c3 0.07, jw_addr 0.06 |
| 3 | 0.9936 | addr_len_c 0.10, c_nq 0.10, r_addr_c3 0.08, num_jacc 0.08, jw_addr 0.08 |
| 4 | 0.9884 | c_nq 0.12, r_addr_c3 0.09, cos_addr_c3 0.07, num_jacc 0.07, jw_addr 0.06 |
| 5 | 0.9849 | r_addr_c3 0.10, num_jacc 0.09, cos_addr_c3 0.09, jw_addr 0.07, cos_full_w 0.05 |
