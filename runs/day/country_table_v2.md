# output_v2/matching_results.tsv vs itself (store: cache = v2 normaliser) — scripts/day/country_table.py baseline, 26 Sep 19:15

| country | S1 | empty | matches/S1 | recs under 2+ S1 |
|---|---|---|---|---|
| france | 259452 | 0.0518 | 3.304 | 6672 |
| india | 809986 | 0.0666 | 3.117 | 4899 |
| us | 663106 | 0.0548 | 3.349 | 2235 |

## france: exact-key pair coverage by rule (633958 key pairs with a rule)

| rule | pairs | coverage v2 |
|---|---|---|
| core_eq|num_eq | 360339 | 0.951 |
| core_eq|a_eq | 131493 | 1.000 |
| subset|a_eq | 29234 | 0.988 |
| disjoint|a_eq|invented | 22919 | 0.468 |
| disjoint|a_eq|realword | 18468 | 0.189 |
| core_eq|c_empty | 18023 | 0.900 |
| nsp_eq|num_eq | 12361 | 0.814 |
| swap1|a_eq|invented | 9301 | 0.929 |
| core_eq|street_eq_nonum | 7513 | 0.997 |
| swap1|a_eq|realword | 5859 | 0.861 |
| partial|a_eq|invented | 5291 | 0.878 |
| partial|a_eq|realword | 4480 | 0.757 |
| nsp_eq|a_eq | 4400 | 0.988 |
| reorder|a_eq | 4277 | 1.000 |

Note: key pairs here use the v2 normaliser; for v3 use --store-dir cache_n3 (HQ's France fixes change the keys), and
compare v2 on the same n3 keys (the script scores the --ref file on the --store-dir keys).
