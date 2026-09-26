# Sure key pairs missed by v2: model-rejected vs cut before the model

130678 pairs; S1 found in v2 candidate_pairs: 107715/107715

## by country

```
status   blocking_or_cascade_cut  model_rejected  total  cut_share
country                                                           
france                     40339           16261  56600      0.713
india                      23825           25458  49283      0.483
us                          4011           20784  24795      0.162
```

## by country / rule

```
status                           blocking_or_cascade_cut  model_rejected  total  cut_share
country rule                                                                              
india   core_eq|num_eq                             13085           16714  29799      0.439
france  disjoint|a_eq|invented                     18910            5911  24821      0.762
        core_eq|num_eq                             10646            2018  12664      0.841
india   disjoint|a_eq|invented                      3294            4576   7870      0.419
us      swap1|a_eq|invented                          105            5770   5875      0.018
        swap1|a_eq|realword                           23            5230   5253      0.004
india   nsp_eq|num_eq                               4478              28   4506      0.994
us      disjoint|a_eq|invented                       800            2704   3504      0.228
france  nsp_eq|num_eq                               3495               3   3498      0.999
        partial|a_eq|realword                       1537            1590   3127      0.492
us      core_eq|num_eq                               751            2227   2978      0.252
india   core_eq|c_empty                             2624             172   2796      0.938
us      partial|a_eq|invented                         26            2595   2621      0.010
france  swap1|a_eq|realword                          230            2270   2500      0.092
        partial|a_eq|invented                       1202            1050   2252      0.534
        swap1|a_eq|invented                          597            1583   2180      0.274
        core_eq|c_empty                             1857              57   1914      0.970
india   swap1|a_eq|invented                          114            1573   1687      0.068
france  core_eq|a_eq                                 342            1153   1495      0.229
us      core_eq|c_empty                             1089             146   1235      0.882
        nsp_eq|num_eq                               1137               4   1141      0.996
france  nsp_eq|a_eq                                 1129              10   1139      0.991
us      partial|a_eq|realword                          6             983    989      0.006
france  subset|a_eq                                  388             527    915      0.424
india   subset|a_eq                                   83             731    814      0.102
        partial|a_eq|invented                         47             751    798      0.059
us      core_eq|a_eq                                  14             610    624      0.022
        subset|a_eq                                   15             499    514      0.029
india   core_eq|a_eq                                  11             465    476      0.023
        swap1|a_eq|realword                           17             339    356      0.048
        partial|a_eq|realword                          5              87     92      0.054
france  core_eq|street_eq_nonum                        4              85     89      0.045
india   nsp_eq|a_eq                                   67               1     68      0.985
us      nsp_eq|a_eq                                   35               0     35      1.000
        core_eq|street_eq_nonum                       10              12     22      0.455
india   reorder|a_eq                                   0              21     21      0.000
france  reorder|a_eq                                   2               4      6      0.333
us      reorder|a_eq                                   0               4      4      0.000
```

## by country / rule / v2_gives_record_to_other_s1

```
status                                                       blocking_or_cascade_cut  model_rejected  total  cut_share
country rule                    v2_gives_record_to_other_s1                                                           
france  disjoint|a_eq|invented  0                                              18600            5794  24394      0.762
india   core_eq|num_eq          0                                               6314           15988  22302      0.283
                                1                                               6771             726   7497      0.903
france  core_eq|num_eq          0                                               5064            1457   6521      0.777
india   disjoint|a_eq|invented  0                                               2988            3480   6468      0.462
france  core_eq|num_eq          1                                               5582             561   6143      0.909
us      swap1|a_eq|invented     0                                                105            5770   5875      0.018
        swap1|a_eq|realword     0                                                 23            5230   5253      0.004
india   nsp_eq|num_eq           0                                               3938              28   3966      0.993
us      disjoint|a_eq|invented  0                                                775            2618   3393      0.228
france  nsp_eq|num_eq           0                                               3260               3   3263      0.999
        partial|a_eq|realword   0                                               1513            1581   3094      0.489
india   core_eq|c_empty         0                                               2624             172   2796      0.938
us      partial|a_eq|invented   0                                                 26            2594   2620      0.010
        core_eq|num_eq          0                                                511            2074   2585      0.198
france  swap1|a_eq|realword     0                                                228            2265   2493      0.091
        partial|a_eq|invented   0                                               1192            1041   2233      0.534
        swap1|a_eq|invented     0                                                592            1579   2171      0.273
        core_eq|c_empty         0                                               1857              56   1913      0.971
india   swap1|a_eq|invented     0                                                113            1569   1682      0.067
france  core_eq|a_eq            0                                                340            1109   1449      0.235
india   disjoint|a_eq|invented  1                                                306            1096   1402      0.218
us      core_eq|c_empty         0                                               1089             146   1235      0.882
france  nsp_eq|a_eq             0                                               1124               9   1133      0.992
us      nsp_eq|num_eq           0                                               1103               4   1107      0.996
        partial|a_eq|realword   0                                                  6             981    987      0.006
france  subset|a_eq             0                                                380             489    869      0.437
india   subset|a_eq             0                                                 81             725    806      0.100
        partial|a_eq|invented   0                                                 43             738    781      0.055
us      core_eq|a_eq            0                                                 14             610    624      0.022
india   nsp_eq|num_eq           1                                                540               0    540      1.000
us      subset|a_eq             0                                                 15             498    513      0.029
india   core_eq|a_eq            0                                                 11             465    476      0.023
france  disjoint|a_eq|invented  1                                                310             117    427      0.726
us      core_eq|num_eq          1                                                240             153    393      0.611
india   swap1|a_eq|realword     0                                                 17             339    356      0.048
france  nsp_eq|num_eq           1                                                235               0    235      1.000
us      disjoint|a_eq|invented  1                                                 25              86    111      0.225
france  core_eq|street_eq_nonum 0                                                  4              82     86      0.047
india   partial|a_eq|realword   0                                                  4              80     84      0.048
```

