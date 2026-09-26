# AML_PH=2 offline check (India train S1 of v2's full-density pass)

| GT subset | pairs | ph equal (v2/HEAD key) | ph equal (AML_PH=2) |
|---|---|---|---|
| all | 207,851 | 49.14% | 49.93% |
| blocked by v2 | 13,909 | 31.86% | 34.04% |
| cand Indic | 37,279 | 39.39% | 43.30% |
| blocked & cand Indic | 7,637 | 37.24% | 41.04% |
| blocked & ASCII | 6,272 | 25.32% | 25.51% |

Decoy proxy on a random 416,534-record sample of the India store: distinct ph keys 290,308 (old) vs 282,143 (AML_PH=2); mean records per key 1.435 vs 1.476.

Reading: AML_PH=2 raises exact ph-key agreement on India GT pairs with an Indic-script candidate from 39.4% to 43.3%
(blocked ones 37.2% -> 41.0%). ASCII pairs are unchanged. Keys merge 2.9% more records (decoy cost, small). A modest
blocking/feature gain for about 18% of India's GT pairs. The n4ph screen (chain5) measures the real effect.
("old" = v2's store in cache/, which has the same ph as HEAD's for India.)
