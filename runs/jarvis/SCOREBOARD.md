# Jarvis SCOREBOARD
| time | change | OOF US / IN / overall | recall after cascade US / IN | cands/S1 | runtime | KEEP/REVERT | file |
|---|---|---|---|---|---|---|---|
| (v2 ref) | 4 views k=10, 150K S1, cascade top10 | — / — / 0.9623 | blocking 0.984 / 0.933; 0.953 overall after cascade | 9.17 | box ~5h test | — | LB 0.947 |
| 26 Sep 08:17 | QUEUE 0 smoke: pipeline_jv, 6 views k=15, 60K S1, key+reverse aug | stopped 08:50 (timings only) | India union+aug 0.960 (smoke) | — | — | — | runs/jarvis/LOG.md |
| 26 Sep 08:28 | jv1 (pipeline_jv) | superseded 08:50 by main-flags jv1 | — | — | — | — | — |
| 26 Sep 08:52 | jv1: MAIN pipeline, 6 views k=15, 600K S1, --key-rules 0.96 --reverse-k 3 --reverse-bypass 2, global o2o | running (train pass) | — | — | — | — | — |
| 26 Sep 09:45 | QUEUE 2 pilot: CE MiniLM on India block-0 pseudo-pool (30K S1, crude proxy ranking) | — | CE recall@10 0.919 vs proxy 0.863 (pool ceiling 0.919); AUC 0.9994 | — | 11.5 min | pilot | runs/jarvis/cepilot.log |
