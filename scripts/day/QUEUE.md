# Experiment queue — top = next. Humans/HQ may edit (push from the laptop). The loop marks items DONE/FAILED.

1. [P0] FAST LANE + correctness gate (see DAY_TASK.md). Nothing else until it passes.
2. [France] Label-free diagnosis using fast-lane test outputs: (a) adversarial validation — classifier to tell
   France test pairs from US/India test pairs on the pair features; list the most shifted features;
   (b) France vs US/India distributions of max prob per S1, stage-A scores, JW name/addr, zip-match rate;
   (c) inspect ~100 French S1 with predictions (plausible? missed?); (d) normalisation coverage for French:
   accents, rue/av/bd/chemin/allée/place, cedex, 5-digit postcodes, arrondissements, SARL/SAS/SA/EURL/SCI/SNC,
   "et"/"&". Turn findings into concrete fixes and evaluate them (xc_us is the proxy).
3. [France] Country-neutral model: drop or re-normalise the most country-shifted features found in 2(a)
   (e.g. rank/percentile within country instead of raw counts); evaluate Q.
4. [Accuracy, cheap] Model capacity on the fast lane: more trees/leaves, train S1 150K → 300K, 3-seed average.
5. [India recall] Slice: India-only extra view (name_ph / transliterated / dense MiniLM via src/blocking_b.py —
   see runs/night/track_b_block_eval.txt) or India k 10→15, keeping final cands/S1 ≤ 10 after the cascade;
   measure India pair recall + Q. If it wins → candidate-augmentation full-data run (India only).
6. [Efficiency] cascade_top 10 → 8 or 6 if Q holds (smaller candidate sets rank higher).
7. [Speed] Profile the full-data run by stage; vectorise the slowest stage-B features.
