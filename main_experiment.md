# FusionPortableV2 Main Experiment Summary

This table summarizes the current FusionPortableV2 handheld and UGV loop-verifier
experiments. Values are reported as `AP/MR@100P`.

## Data Sources

- Handheld batch:
  `workspace/robust_loop_verifier_runs/FusionPortableV2/fusionportablev2_batch_20260518_195158/self_calibrated_score_sweep/batch_score_sweep_per_sequence.csv`
- Handheld escalator00 support-ensemble run:
  `workspace/robust_loop_verifier_runs/FusionPortableV2/handheld/handheld_escalator00/handheld_escalator00_full_support_ensemble_20260518_172817_support_ensemble/self_calibrated_score_sweep/score_sweep.json`
- UGV batch:
  `workspace/robust_loop_verifier_runs/FusionPortableV2/fusionportablev2_batch_ugv_20260519_012615/score_sweep/batch_score_sweep_per_sequence.csv`
- ORB DBoW2 retrieval baseline:
  `workspace/baseline_runs/orb_dbow2_retrieval/fusionportablev2/20260519_205049/metrics.json`
- NetVLAD retrieval baseline:
  `workspace/baseline_runs/netvlad_retrieval/fusionportablev2/20260519_200600/metrics.json`

Excluded sequence:

| Platform | Sequence | Reason |
| --- | --- | --- |
| ugv | ugv_campus00 | All evaluated methods are `0/0`; current labels have no positive loop records. |

Notes:

- Support ensemble was not run for UGV, so UGV support-ensemble cells are `N/A`.
- `z_fusion` is now treated as a main-pipeline candidate. The compact paper
  table below uses the current all-sequence MR-best fixed configuration
  `z_fusion:salad=0,def=4,res=2`.
- The four named `z_fusion` columns below are selected from current sweeps:
  `z HH-best = z_fusion:salad=1,def=4,res=0.25`,
  `z UGV-best = z_fusion:salad=0,def=4,res=0.5`,
  `z All-AP = z_fusion:salad=0,def=4,res=1`,
  `z All-MR = z_fusion:salad=0,def=4,res=2`.

## Method Averages

| Method | Coverage | Handheld AP | Handheld MR | UGV AP | UGV MR | All AP | All MR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SALAD score only | 10 | 0.6630 | 0.2191 | 0.3125 | 0.0555 | 0.4877 | 0.1373 |
| ROVER deformation only | 10 | 0.8531 | 0.3243 | 0.5643 | 0.0750 | 0.7087 | 0.1997 |
| PGO residual only | 10 | 0.7365 | 0.2772 | 0.5876 | 0.1928 | 0.6620 | 0.2350 |
| DA3-ROVER++ support ensemble graph evidence | 5 | 0.8082 | 0.3383 | N/A | N/A | 0.8082 | 0.3383 |
| percentile_product:res,def | 10 | 0.8338 | 0.4204 | 0.6645 | 0.2503 | 0.7491 | 0.3354 |
| percentile_product:salad,def | 10 | 0.8270 | 0.3953 | 0.5271 | 0.1891 | 0.6770 | 0.2922 |
| percentile_product:salad,res | 10 | 0.7103 | 0.2558 | 0.4923 | 0.1716 | 0.6013 | 0.2137 |
| percentile_product:salad,res,def | 10 | 0.7884 | 0.3570 | 0.5701 | 0.2379 | 0.6793 | 0.2974 |
| percentile_min:salad,res,def | 10 | 0.7551 | 0.3056 | 0.5174 | 0.2105 | 0.6362 | 0.2581 |
| rank_product:res,def | 10 | 0.3966 | 0.0000 | 0.0696 | 0.0000 | 0.2331 | 0.0000 |
| rank_product:salad,def | 10 | 0.3920 | 0.0000 | 0.0722 | 0.0000 | 0.2321 | 0.0000 |
| rank_product:salad,res | 10 | 0.3747 | 0.0000 | 0.0708 | 0.0000 | 0.2227 | 0.0000 |
| rank_product:salad,res,def | 10 | 0.4198 | 0.0189 | 0.0901 | 0.0000 | 0.2550 | 0.0095 |
| z_fusion:salad=1,def=4,res=0.25 | 10 | 0.9049 | 0.5417 | 0.6452 | 0.2426 | 0.7750 | 0.3922 |
| z_fusion:salad=0.5,def=4,res=0.5 | 10 | 0.9016 | 0.5530 | 0.6699 | 0.2455 | 0.7858 | 0.3992 |
| z_fusion:salad=0,def=4,res=0.5 | 10 | 0.8826 | 0.4168 | 0.6973 | 0.2140 | 0.7899 | 0.3154 |
| z_fusion:salad=0.25,def=4,res=4 | 10 | 0.8584 | 0.4726 | 0.6795 | 0.2903 | 0.7689 | 0.3815 |
| z_fusion:salad=0,def=4,res=1 | 10 | 0.8903 | 0.5011 | 0.6971 | 0.2558 | 0.7937 | 0.3785 |
| z_fusion:salad=0,def=4,res=2 | 10 | 0.8864 | 0.5172 | 0.6936 | 0.2822 | 0.7900 | 0.3997 |

## Per-Sequence Key Results

| Platform | Sequence | SALAD | ROVER def | PGO res | Support ens | Pct res+def | Pct salad+def | z HH-best | z UGV-best | z All-AP | z All-MR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| handheld | handheld_escalator00 | 0.8339/0.0867 | 0.6951/0.0202 | 0.8198/0.1169 | 0.9063/0.1593 | 0.8850/0.2097 | 0.9111/0.2762 | 0.8971/0.3226 | 0.7984/0.0948 | 0.8569/0.1935 | 0.8964/0.2903 |
| handheld | handheld_escalator01 | 0.8407/0.1398 | 0.8603/0.1240 | 0.8462/0.2323 | 0.8913/0.1929 | 0.9229/0.3740 | 0.9380/0.3701 | 0.9418/0.5256 | 0.9077/0.2736 | 0.9260/0.4587 | 0.9348/0.4626 |
| handheld | handheld_grass00 | 0.0820/0.0000 | 0.7652/0.3000 | 0.2689/0.1000 | 0.3952/0.3000 | 0.4453/0.3000 | 0.5648/0.2000 | 0.7616/0.5000 | 0.7463/0.4000 | 0.7038/0.4000 | 0.6374/0.3000 |
| handheld | handheld_room00 | 0.6031/0.3962 | 0.9556/0.4340 | 0.7787/0.3962 | 0.8825/0.4717 | 0.9237/0.5472 | 0.7338/0.4906 | 0.9279/0.6038 | 0.9659/0.5094 | 0.9681/0.6604 | 0.9672/0.7358 |
| handheld | handheld_room01 | 0.9552/0.4730 | 0.9893/0.7432 | 0.9689/0.5405 | 0.9657/0.5676 | 0.9921/0.6712 | 0.9871/0.6396 | 0.9960/0.7568 | 0.9946/0.8063 | 0.9964/0.7928 | 0.9964/0.7973 |
| ugv | ugv_campus01 | 0.5729/0.1320 | 0.6246/0.0000 | 0.6932/0.0948 | N/A | 0.7729/0.1691 | 0.7126/0.2186 | 0.7568/0.2124 | 0.7904/0.1526 | 0.7945/0.1629 | 0.7937/0.1835 |
| ugv | ugv_parking00 | 0.0429/0.0000 | 0.6190/0.2286 | 0.6104/0.2000 | N/A | 0.6214/0.2286 | 0.3516/0.0571 | 0.5934/0.2857 | 0.6303/0.2286 | 0.6282/0.2286 | 0.6257/0.2286 |
| ugv | ugv_parking01 | 0.4173/0.0160 | 0.5086/0.0000 | 0.7530/0.4225 | N/A | 0.8037/0.4973 | 0.7131/0.3529 | 0.7909/0.3048 | 0.8370/0.2834 | 0.8383/0.4866 | 0.8326/0.5561 |
| ugv | ugv_parking02 | 0.1801/0.0044 | 0.6158/0.1467 | 0.5013/0.0800 | N/A | 0.5984/0.1067 | 0.3896/0.0667 | 0.5462/0.1600 | 0.6245/0.1556 | 0.6252/0.1511 | 0.6255/0.1511 |
| ugv | ugv_parking03 | 0.3493/0.1250 | 0.4532/0.0000 | 0.3799/0.1667 | N/A | 0.5260/0.2500 | 0.4687/0.2500 | 0.5386/0.2500 | 0.6044/0.2500 | 0.5994/0.2500 | 0.5904/0.2917 |

## Current Takeaways

- `percentile_product:res,def` is the strongest weight-free self-calibrated
  method among the current candidates: all-sequence average `0.7491/0.3354`.
- Rank-product methods are not competitive in the current setting.
- `z_fusion` is the strongest current main-pipeline candidate by all-sequence
  MR, using the fixed setting `salad=0,def=4,res=2` from the current sweep.
- UGV support-ensemble results are unavailable in this table because the UGV
  run was executed with support ensemble disabled.

## Paper Main Experiment Table (Bold Best, Underlined Second Best)

This paper-facing table keeps the per-sequence results for the main experiment
methods. Values are reported in percent as `AP / MR@100P`. `ugv_campus00` is
excluded because it has no positive loop records under the current labels. Best
and second-best methods in each row are selected by `MR@100P`, with AP used as
the tie-breaker.

| Platform | Sequence | ORB DBoW2 score only | NetVLAD score only | SALAD score only | ROVER deformation only | percentile_product:res,def | z_fusion:salad=0,def=4,res=2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| handheld | handheld_escalator00 | 67.18 / 4.08 | 84.22 / 9.69 | 83.39 / 8.67 | 69.51 / 2.02 | <u>88.50 / 20.97</u> | **89.64 / 29.03** |
| handheld | handheld_escalator01 | 56.57 / 1.59 | 83.93 / 12.26 | 84.07 / 13.98 | 86.03 / 12.40 | <u>92.29 / 37.40</u> | **93.48 / 46.26** |
| handheld | handheld_grass00 | 1.94 / 0.00 | 17.28 / 5.26 | 8.20 / 0.00 | **76.52 / 30.00** | 44.53 / 30.00 | <u>63.74 / 30.00</u> |
| handheld | handheld_room00 | 59.46 / 34.09 | 52.00 / 27.45 | 60.31 / 39.62 | 95.56 / 43.40 | <u>92.37 / 54.72</u> | **96.72 / 73.58** |
| handheld | handheld_room01 | 92.07 / 16.38 | 95.86 / 48.48 | 95.52 / 47.30 | <u>98.93 / 74.32</u> | 99.21 / 67.12 | **99.64 / 79.73** |
| ugv | ugv_campus01 | 59.44 / 8.35 | 46.02 / 1.20 | 57.29 / 13.20 | 62.46 / 0.00 | <u>77.29 / 16.91</u> | **79.37 / 18.35** |
| ugv | ugv_parking00 | 1.26 / 0.00 | 3.67 / 0.00 | 4.29 / 0.00 | 61.90 / 22.86 | <u>62.14 / 22.86</u> | **62.57 / 22.86** |
| ugv | ugv_parking01 | 0.36 / 0.00 | 3.31 / 0.00 | 41.73 / 1.60 | 50.86 / 0.00 | <u>80.37 / 49.73</u> | **83.26 / 55.61** |
| ugv | ugv_parking02 | 19.01 / 1.23 | 11.81 / 0.39 | 18.01 / 0.44 | <u>61.58 / 14.67</u> | 59.84 / 10.67 | **62.55 / 15.11** |
| ugv | ugv_parking03 | 11.58 / 3.12 | 46.35 / 7.41 | 34.93 / 12.50 | 45.32 / 0.00 | <u>52.60 / 25.00</u> | **59.04 / 29.17** |
| average | 10 sequences | 36.88 / 6.89 | 44.44 / 11.21 | 48.77 / 13.73 | 70.87 / 19.97 | <u>74.91 / 33.54</u> | **79.00 / 39.97** |

For completeness, the current AP-best `z_fusion` variant is
`z_fusion:salad=0,def=4,res=1`, with average `79.37 / 37.85`. The table above
uses the MR-best variant because `MR@100P` is the stricter loop-verification
metric for high-precision deployment.
