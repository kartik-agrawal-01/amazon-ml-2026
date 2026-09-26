# v2 OOF per country (QUEUE 2c)

Source: `output_v2_train/oof_pairs.tsv.gz` (1,382,533 OOF pairs, 149,998 S1 with >= 1 candidate). Rule: thr 0.70 + one-to-one over the whole OOF sample (v2's chosen rule). S1 without any candidate are not in the dump (so these numbers are slightly optimistic). Note v2's logged OOF 0.9623 was computed on a 100K-S1 subsample.

| country | S1 | OOF F0.5 | true matches/S1 (GT) | pred matches/S1 | pred-empty | GT-empty | TP/GT (recall) | precision | GT pairs in candidates |
|---|---|---|---|---|---|---|---|---|---|
| India | 60,029 | 0.94762 | 3.462 | 3.096 | 6.69% | 5.66% | 0.8859 | 0.9906 | 0.9191 |
| US | 89,969 | 0.97216 | 3.453 | 3.268 | 5.68% | 5.51% | 0.9396 | 0.9928 | 0.9762 |
| ALL | 149,998 | 0.96234 | 3.457 | 3.199 | 6.08% | 5.57% | 0.9181 | 0.9920 | 0.9533 |

Per GT bucket (mean F0.5 / pred-empty):

| country | 0 true | 1 true | 2 true | 3+ true |
|---|---|---|---|---|
| India | 0.9391 / 93.9% (n=3,399) | 0.8597 / 12.1% (n=3,222) | 0.9362 / 2.6% (n=10,066) | 0.9575 / 0.4% (n=43,342) |
| US | 0.9403 / 94.0% (n=4,959) | 0.9309 / 5.7% (n=5,034) | 0.9676 / 0.7% (n=15,415) | 0.9789 / 0.1% (n=64,561) |
| ALL | 0.9398 / 94.0% (n=8,358) | 0.9031 / 8.2% (n=8,256) | 0.9552 / 1.4% (n=25,481) | 0.9703 / 0.2% (n=107,903) |

Reading (loop, 26 Sep 16:00): the overall 0.96234 reproduces v2's logged 0.9623. India OOF 0.9476 vs US 0.9722 (−2.5 pt).
India loses 11.4% of its GT pairs: 8.1% never reach the model (blocking + cascade; US 2.4%) and 3.3% are model FNs
(US 3.7%). Precision is ~0.99 in both. So India's gap is mostly candidate recall (QUEUE 5), and the 1-true bucket is
the weak spot (India F 0.860, pred-empty 12.1% vs US 0.931 / 5.7%).
