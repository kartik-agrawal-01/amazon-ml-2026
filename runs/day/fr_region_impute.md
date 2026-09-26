# QUEUE 6 part 2: France region imputation from the city using S1 (evidence only; loop 26 Sep 19:40)

Script: `scripts/day/fr_region_impute.py` (streams the test TSVs, France rows; HQ's `_FR_CANON` region/département -> code).
Raw: `runs/day/fr_region_impute.txt`. The stores' `city`/`state` columns are empty for all countries (the data has one
`business_address` field), so "region" = a comma component that `_FR_CANON` maps to a region code.

- France test S1: 86.6% end in a region component (rest: shuffled component order). The French test covers only a
  handful of regions (Nouvelle-Aquitaine, Pays de la Loire, Hauts-de-France) and ~22 real cities (Bordeaux, Nantes,
  Lille, Tourcoing, Dunkerque, Roubaix, Calais, Saint-Nazaire, La Teste-de-Buch, Saint-Herblain ...), each 100% in one region.
  Table: city -> region with >= 3 S1 and purity >= 0.9 = 22 cities (the other ~8.6K "second-to-last components" are street
  fragments from shuffled addresses, < 3 S1 each).
- S2 / S3: 55.7% / 57.1% carry a region; of the region-less ones 74.6% / 74.1% end with a known city -> imputable
  (232K + 232K docs). Table accuracy on docs that DO carry a region: 1.000 (n 370K + 395K).

Implication: imputation is safe (label-free accuracy 100%) and would lift France S2/S3 region presence from ~56% to ~89%
(US-like 95%). Its matching value is only distribution alignment: with 3 regions a region token barely separates
decoys from matches (all candidates of an S1 are mostly in the same region), but region-less docs are shorter,
which moves address-overlap / length features (addr_len_q is a top adversarial feature, runs/day/adv_val.md).
It cannot be scored on Q (no France in any slice/screen). NOT implemented: v3 is queued on the n3 stores and a
store-level change would mean rebuilding France's store and would go into the fallback file unvalidated. If HQ wants it: a
post-normalisation pass per country (S1 city->region table, append the code to region-less S2/S3 n_addr/n_full), France only,
behind a flag, then compare v3 vs v3+impute France stats (empty rate, matches/S1, sure-key coverage).
