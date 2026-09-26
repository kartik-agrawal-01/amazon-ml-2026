# n3 (HQ France fixes, 7defba7) on the slice screens (day loop, 26 Sep 14:55)

| run | base (HEAD code, v2 stores) | n3 (France fixes, _n3 stores) | Δ |
|---|---|---|---|
| slice_mix hidden F0.5 (US+IN -> US+IN) | 0.9808 | 0.9812 | +0.0004 |
| xc_us hidden F0.5 (US -> India) | 0.9344 | 0.9273 | **−0.0071** |
| slice_mix OOF F0.5 | 0.97653 | 0.97644 | −0.0001 |
| cands/S1 mix / xc_us | 6.17 / 5.11 | 6.33 / 5.26 | +3% |
| Q (xc_in = Q0 value) | 0.9669 | 0.9650 | −0.0019 |

Diagnosis of the xc_us drop (scripts/day/diff_runs.py, norm_diff.py; runs/day/n3_vs_base_xc_us.txt):
- n3 loses 2965 TP / gains 1443 on India test. 98% of the lost ones were still n3 candidates: the MODEL rejects them.
- 67% of lost TPs have an Indic-script candidate name (TP on Indic-name GT pairs 19478 -> 17770, −9%); ASCII-name
  pairs are net +148.
- For 759/800 lost-pair records, v2's and HEAD's normalize.py give identical name/core/legal/address. So the Indic
  pairs' inputs did not change; the US-trained model's decision surface did. The only features.py change is the
  larger `generic` set (dba/fka/aka/formerly/known/as/doing/business/ta/www/dr/mr/shri/sri/smt/incorporated/holding +
  French words), which changes extra_*_content on the US training pairs.
- Why this matters for France: xc_us is our proxy for an unseen country. A change that looks neutral on slice_mix
  can cost 0.7 pt on an unseen country, in either direction.

Running now: n3f = n3 stores + v2's generic set (env AML_GENERIC=v2). If it recovers xc_us, the generic set is the
cause and a narrower set (v2 + French words only) is the candidate. Otherwise the normalisation is (through the US
training pairs). Result goes in runs/day/SCOREBOARD.md and runs/day/LOG.md.
