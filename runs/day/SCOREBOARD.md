# Day-loop scoreboard

Q = 0.5·mix + 0.3·xc_us + 0.2·xc_in (hidden-holdout macro F0.5). KEEP if ΔQ ≥ +0.0015 (or cands −10% / runtime −20% with ΔQ ≥ −0.0005).

| cycle | time | change | Q | mix | xc_us | xc_in | cands/S1 | runtime | peak RSS | KEEP/REVERT | full-data file |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 26 Sep 07:45 | baseline v2 code (B4 table) | 0.9666 | 0.9805 | 0.9339 | 0.9810 | 9.2 | 4.9 h full | 7.3 GB | **champion** | submissions (v2, LB 0.947) |
| 1 | 26 Sep 14:07 | base = HEAD code on v2 stores (deterministic LightGBM, global o2o), new reference for n3/n3k/n3ph/n3r | 0.9669 | 0.9808 | 0.9344 | (0.9810 Q0) | mix 6.17 / xc_us 5.11 (slices) | mix 25.1 / xc_us 15.5 min (n-jobs 6) | 3.3 GB | reference | — |
| 2 | 26 Sep 14:55 (xc_in 15:55) | n3 = HQ France fixes (normalize.py + features.py generic set of 7defba7), rebuilt _n3 stores | 0.9652 | 0.9812 | 0.9273 | 0.9819 | mix 6.33 / xc_us 5.26 (+3%) / xc_in 8.18 | mix 29.5 / xc_us 17.7 min (incl. store build) | 3.4 GB | HOLD (ΔQ −0.0019 vs base; OOF 0.97644 vs 0.97653) -> ablation n3f running | — |
| 3 | 26 Sep 15:36 | n3f = n3 stores + v2 generic set (AML_GENERIC=v2): ablation | 0.9629 | 0.9810 | 0.9205 | (Q0 value) | mix 6.09 | mix ~24 min | 3.4 GB | REVERT (xc_us −0.0068 vs n3; mix OOF 0.97637) — generic set is not the cause of the n3 drop; keep HEAD set | — |
