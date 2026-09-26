# Submission log — 5/day HARD CAP. Fill BEFORE clicking submit.

| # | Day | Time (IST) | Who | File | Local CV | Public LB | What changed vs prev | Commit |
|---|-----|-----------|-----|------|----------|-----------|----------------------|--------|
| 1 | 2 (26 Sep) | 09:45 | Gaurav (box v2) → Soha uploads | v2_matching_results.tsv | 0.9623 (OOF, full data) | **0.947** (local→LB gap −1.5) | first full-data run: 4 views (name_c3,name_w,addr_c3,full_w), GPU top-k, cascade 9.2 cands/S1, LightGBM 150K S1; validator PASS | 48bc4e7 |
| 2 | 2 (26 Sep) | TBD | Gaurav (ensemble) → Soha uploads | ens1_consensus_matching_results.tsv | n/a (no labels; subset of v2) | _pending_ | consensus of v2 + Soha's 0.948 file: keep pairs both models predict; if no overlap or one side empty keep v2. Changes 16.1% of S1, drops 0.204 disputed pairs/S1. DIAGNOSTIC: LB up => disputed pairs mostly wrong (precision problem); LB down => recall problem. md5 c1a614f9 | 9199176 |
