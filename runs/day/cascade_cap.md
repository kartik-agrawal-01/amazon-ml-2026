# Candidate cap per S1 (QUEUE 8) — optimistic proxy on v2's full-density train OOF

Source `output_v2_train/oof_pairs.tsv.gz` (1,382,533 pairs, 149,998 S1), rule thr 0.70 + global one-to-one. Top-c per S1 by the FINAL model's OOF p (the real cascade ranks worse -> real losses are larger).

| cap | country | cands/S1 | GT pairs kept (of GT) | TP pairs lost vs cap 10 | OOF F0.5 | ΔF vs cap 10 |
|---|---|---|---|---|---|---|
| 10 | India | 9.21 | 0.9191 | 0 | 0.94762 | +0.00000 |
| 10 | US | 9.22 | 0.9762 | 0 | 0.97216 | +0.00000 |
| 9 | India | 8.48 | 0.9190 | 5 | 0.94762 | +0.00000 |
| 9 | US | 8.48 | 0.9761 | 11 | 0.97216 | -0.00000 |
| 8 | India | 7.68 | 0.9186 | 59 | 0.94761 | -0.00002 |
| 8 | US | 7.68 | 0.9755 | 132 | 0.97213 | -0.00003 |
| 7 | India | 6.82 | 0.9164 | 412 | 0.94748 | -0.00015 |
| 7 | US | 6.82 | 0.9726 | 779 | 0.97195 | -0.00021 |
| 6 | India | 5.92 | 0.9080 | 1,853 | 0.94679 | -0.00083 |
| 6 | US | 5.91 | 0.9623 | 3,372 | 0.97111 | -0.00105 |
| 5 | India | 4.97 | 0.8827 | 6,343 | 0.94406 | -0.00356 |
| 5 | US | 4.97 | 0.9315 | 11,561 | 0.96777 | -0.00440 |

True pairs by final-p rank within the S1 (share of the country's GT pairs in the candidates):

| rank | India | US |
|---|---|---|
| 1 | 0.2693 | 0.2730 |
| 2 | 0.2451 | 0.2539 |
| 3 | 0.1889 | 0.2018 |
| 4 | 0.1188 | 0.1326 |
| 5 | 0.0606 | 0.0703 |
| 6 | 0.0252 | 0.0308 |
| 7 | 0.0085 | 0.0104 |
| 8 | 0.0022 | 0.0028 |
| 9 | 0.0004 | 0.0007 |
| 10 | 0.0000 | 0.0001 |

Reading (loop, 26 Sep 19:40): even with a perfect ranker, cap 8 costs ≤ 0.00003 F0.5 per country (−17% cands/S1:
9.2 → 7.7) and cap 7 ≤ 0.0002 (−26%); cap 6 costs 0.001. The real cascade ranks on stage-A features only, so the true
cost is larger by however many GT pairs the cascade puts at rank 9–10. That number needs the cascade scores at full
density (pre-cascade unions are in cands_v2 for train/india + train/us; measure it on the fast lane after v3 with
`--cascade-top 8`, which reuses the cached unions). India and US behave the same, so a per-country cap is not needed for
them. France (test only, every S1 at the cap) can't be checked with labels.
Candidate_pairs.tsv must stay = the set fed to the model, so a cap can only be applied before the model (cascade), not
after the final p.
