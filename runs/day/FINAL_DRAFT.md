# FINAL (DRAFT, written 26 Sep 19:50 by session 14; the freeze session turns it into FINAL.md)

## Champion (slice Q)
SCOREBOARD row 6: n3ka = HQ France fixes (normalize/features 7defba7) + `--key-rules 0.96` + `--thr-adapt`
(R2 lower-only, full data `--thr-adapt-floor 0.30`) + global one-to-one; row 7 (rapidfuzz jaro, bit-exact) on top.
Q 0.9726 (mix 0.9821 / xc_us 0.9507 / xc_in 0.9818) vs Q0 0.9666.

## Full-data files (fill at freeze)
| rank | file | recipe | why / robustness | LB |
|---|---|---|---|---|
| ? | submissions/v4_matching_results.tsv | v3 + model variant from model_capacity.md (if any won) | OOF gain in both countries | — |
| ? | submissions/v3_matching_results.tsv | champion recipe on v2's candidates (output_v3/) | xc_us +0.017 from thr-adapt; France fixes | — |
| ref | submissions/v2_matching_results.tsv | v2 (48bc4e7) | LB 0.947 | 0.947 |
Ranking rule: final = PRIVATE leaderboard -> weight xc_us / xc_in (cross-country robustness) over slice mix.

## Packaging (humans run it, for the chosen file's output dir)
```bash
cd ~/amazon-ml-2026
python scripts/make_package.py --team <team_name> --output output_v3   # or output_v4; runs the official validator first
```
The output dir must hold BOTH matching_results.tsv and candidate_pairs.tsv of the same run
(candidate_pairs = exactly the set fed to the model).

## Open at freeze
- chain9 progress (runs/day/chain9.log): v3rs, cap, v4, then slice screens n4ph / n3r / n3ka_s7.
