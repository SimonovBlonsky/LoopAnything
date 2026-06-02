# LoopAnything Main Experiment Summary

> **Historical diagnostic results only.** The tables in this document were
> computed from automatic distance-and-rotation positive labels. Failure-case
> review found that this label definition rejects visually and geometrically
> valid loop factors. These numbers must not be used as paper-facing results.
> The replacement protocol is specified in
> `docs/superpowers/specs/2026-06-02-handheld-escalator00-safe-loop-oracle-prototype-design.md`.

This table summarizes the current FusionPortableV2 handheld and UGV loop-verifier
experiments, with GEODE and NTU-VIRAL reference experiments appended separately.
Values are reported as `AP/MR@100P`.

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
- GEODE Offroad05_beta reference run:
  `workspace/robust_loop_verifier_runs/GEODE/geode_batch_20260519_161713/batch_metrics.csv`
- GEODE Offroad01_beta/Offroad02_beta reference run:
  `workspace/robust_loop_verifier_runs/GEODE/geode_offroad_beta_full_20260520_152209/`
- NTU-VIRAL reference run with AsterSLAM keyframe trajectory labels:
  `workspace/robust_loop_verifier_runs/NTU-VIRAL/ntu_viral_batch_20260521_aster_slam_labels/`
- Causal rank/percentile rows:
  recomputed from the listed `candidate_records.jsonl` files with the current
  `robust_loop_verifier.score_sweep` implementation.

Excluded sequence:

| Platform | Sequence | Reason |
| --- | --- | --- |
| ugv | ugv_campus00 | All evaluated methods are `0/0`; current labels have no positive loop records. |

Notes:

- Support ensemble was not run for UGV, so UGV support-ensemble cells are `N/A`.
- Causal verification is now a hard boundary for paper-facing main methods:
  a method may only use the current query, its retrieved top-K candidates, and
  current/history-only SLAM state or statistics.
- Full-sequence `percentile_product:*` and `z_fusion:*` use the whole evaluated
  candidate distribution for calibration. They are kept only as
  sequence-calibrated offline references, not as main methods.

## Causal Method Averages

| Method | Coverage | Handheld AP | Handheld MR | UGV AP | UGV MR | All AP | All MR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SALAD score only | 10 | 0.6630 | 0.2191 | 0.3125 | 0.0555 | 0.4877 | 0.1373 |
| ROVER deformation only | 10 | 0.8531 | 0.3243 | 0.5643 | 0.0750 | 0.7087 | 0.1997 |
| PGO residual only | 10 | 0.7365 | 0.2772 | 0.5876 | 0.1928 | 0.6620 | 0.2350 |
| absolute_graph:def=0.5,res=0.25 | 10 | 0.8741 | 0.4923 | 0.6960 | 0.2058 | 0.7850 | 0.3491 |
| query_gate_graph:def=0.5,res=0.25,margin=0 | 10 | 0.8693 | 0.4812 | 0.7498 | 0.2606 | 0.8096 | 0.3709 |
| DA3-ROVER++ support ensemble graph evidence | 5 | 0.8082 | 0.3383 | N/A | N/A | 0.8082 | 0.3383 |
| rank_product:res,def | 10 | 0.3966 | 0.0000 | 0.0696 | 0.0000 | 0.2331 | 0.0000 |
| query_percentile_product:res,def | 10 | 0.4295 | 0.0000 | 0.0794 | 0.0000 | 0.2543 | 0.0000 |

## Sequence-Calibrated References

These rows are non-causal because they use full-sequence candidate statistics.
They should be read as offline references or upper-bound diagnostics only.

| Method | Coverage | Handheld AP | Handheld MR | UGV AP | UGV MR | All AP | All MR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| percentile_product:res,def | 10 | 0.8338 | 0.4204 | 0.6645 | 0.2503 | 0.7491 | 0.3354 |
| percentile_product:salad,def | 10 | 0.8270 | 0.3953 | 0.5271 | 0.1891 | 0.6770 | 0.2922 |
| percentile_product:salad,res | 10 | 0.7103 | 0.2558 | 0.4923 | 0.1716 | 0.6013 | 0.2137 |
| percentile_product:salad,res,def | 10 | 0.7884 | 0.3570 | 0.5701 | 0.2379 | 0.6793 | 0.2974 |
| percentile_min:salad,res,def | 10 | 0.7551 | 0.3056 | 0.5174 | 0.2105 | 0.6362 | 0.2581 |
| z_fusion:salad=1,def=4,res=0.25 | 10 | 0.9049 | 0.5417 | 0.6452 | 0.2426 | 0.7750 | 0.3922 |
| z_fusion:salad=0.5,def=4,res=0.5 | 10 | 0.9016 | 0.5530 | 0.6699 | 0.2455 | 0.7858 | 0.3992 |
| z_fusion:salad=0,def=4,res=0.5 | 10 | 0.8826 | 0.4168 | 0.6973 | 0.2140 | 0.7899 | 0.3154 |
| z_fusion:salad=0.25,def=4,res=4 | 10 | 0.8584 | 0.4726 | 0.6795 | 0.2903 | 0.7689 | 0.3815 |
| z_fusion:salad=0,def=4,res=1 | 10 | 0.8903 | 0.5011 | 0.6971 | 0.2558 | 0.7937 | 0.3785 |
| z_fusion:salad=0,def=4,res=2 | 10 | 0.8864 | 0.5172 | 0.6936 | 0.2822 | 0.7900 | 0.3997 |

## Per-Sequence Causal Key Results

| Platform | Sequence | SALAD | ROVER def | PGO res | Abs graph | Query-gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| handheld | handheld_escalator00 | 0.8339/0.0867 | 0.6951/0.0202 | 0.8198/0.1169 | 0.8997/0.2782 | 0.8672/0.2117 |
| handheld | handheld_escalator01 | 0.8407/0.1398 | 0.8603/0.1240 | 0.8462/0.2323 | 0.9339/0.4429 | 0.8938/0.2953 |
| handheld | handheld_grass00 | 0.0820/0.0000 | 0.7652/0.3000 | 0.2689/0.1000 | 0.5837/0.3000 | 0.6453/0.4000 |
| handheld | handheld_room00 | 0.6031/0.3962 | 0.9556/0.4340 | 0.7787/0.3962 | 0.9582/0.6792 | 0.9561/0.6792 |
| handheld | handheld_room01 | 0.9552/0.4730 | 0.9893/0.7432 | 0.9689/0.5405 | 0.9950/0.7613 | 0.9841/0.8198 |
| ugv | ugv_campus01 | 0.5729/0.1320 | 0.6246/0.0000 | 0.6932/0.0948 | 0.7936/0.1588 | 0.8207/0.1794 |
| ugv | ugv_parking00 | 0.0429/0.0000 | 0.6190/0.2286 | 0.6104/0.2000 | 0.6276/0.2286 | 0.6868/0.2571 |
| ugv | ugv_parking01 | 0.4173/0.0160 | 0.5086/0.0000 | 0.7530/0.4225 | 0.8344/0.2406 | 0.8141/0.3690 |
| ugv | ugv_parking02 | 0.1801/0.0044 | 0.6158/0.1467 | 0.5013/0.0800 | 0.6256/0.1511 | 0.7327/0.1644 |
| ugv | ugv_parking03 | 0.3493/0.1250 | 0.4532/0.0000 | 0.3799/0.1667 | 0.5987/0.2500 | 0.6949/0.3333 |

## Current Takeaways

- Among strict causal methods, the current best all-sequence result is
  `query_gate_graph:def=0.5,res=0.25,margin=0` with average `0.8096/0.3709`.
  This uses the candidate's absolute graph evidence plus the current query's
  best top-K graph evidence as a causal query-level loop/no-loop confidence.
- `absolute_graph:def=0.5,res=0.25` is the best simpler absolute graph score,
  with average `0.7850/0.3491`.
- History-only calibration and pure per-query rank/percentile fusion were
  tested but are not competitive under the current AP/MR protocol. The best
  history-only score found in this sweep is
  `history_percentile_product:salad,def`, with average `0.6127/0.1492`.
- Sequence-calibrated `z_fusion` and global `percentile_product` remain useful
  references showing that residual/deformation fusion has high offline
  potential, but they are not paper-facing causal main methods.
- UGV support-ensemble results are unavailable in this table because the UGV
  run was executed with support ensemble disabled.

## Paper Main Experiment Table (Bold Best, Underlined Second Best)

This paper-facing table keeps the per-sequence results for the main experiment
methods. Values are reported in percent as `AP / MR@100P`. `ugv_campus00` is
excluded because it has no positive loop records under the current labels. Best
and second-best methods in each row are selected by `MR@100P`, with AP used as
the tie-breaker.

| Platform | Sequence | ORB DBoW2 score only | NetVLAD score only | SALAD score only | ROVER deformation only | PGO residual only | absolute_graph:def=0.5,res=0.25 | query_gate_graph:def=0.5,res=0.25,margin=0 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| handheld | handheld_escalator00 | 67.18 / 4.08 | 84.22 / 9.69 | 83.39 / 8.67 | 69.51 / 2.02 | 81.98 / 11.69 | **89.97 / 27.82** | <u>86.72 / 21.17</u> |
| handheld | handheld_escalator01 | 56.57 / 1.59 | 83.93 / 12.26 | 84.07 / 13.98 | 86.03 / 12.40 | 84.62 / 23.23 | **93.39 / 44.29** | <u>89.38 / 29.53</u> |
| handheld | handheld_grass00 | 1.94 / 0.00 | 17.28 / 5.26 | 8.20 / 0.00 | <u>76.52 / 30.00</u> | 26.89 / 10.00 | 58.37 / 30.00 | **64.53 / 40.00** |
| handheld | handheld_room00 | 59.46 / 34.09 | 52.00 / 27.45 | 60.31 / 39.62 | 95.56 / 43.40 | 77.87 / 39.62 | **95.82 / 67.92** | <u>95.61 / 67.92</u> |
| handheld | handheld_room01 | 92.07 / 16.38 | 95.86 / 48.48 | 95.52 / 47.30 | 98.93 / 74.32 | 96.89 / 54.05 | <u>99.50 / 76.13</u> | **98.41 / 81.98** |
| ugv | ugv_campus01 | 59.44 / 8.35 | 46.02 / 1.20 | 57.29 / 13.20 | 62.46 / 0.00 | 69.32 / 9.48 | <u>79.36 / 15.88</u> | **82.07 / 17.94** |
| ugv | ugv_parking00 | 1.26 / 0.00 | 3.67 / 0.00 | 4.29 / 0.00 | 61.90 / 22.86 | 61.04 / 20.00 | <u>62.76 / 22.86</u> | **68.68 / 25.71** |
| ugv | ugv_parking01 | 0.36 / 0.00 | 3.31 / 0.00 | 41.73 / 1.60 | 50.86 / 0.00 | **75.30 / 42.25** | 83.44 / 24.06 | <u>81.41 / 36.90</u> |
| ugv | ugv_parking02 | 19.01 / 1.23 | 11.81 / 0.39 | 18.01 / 0.44 | 61.58 / 14.67 | 50.13 / 8.00 | <u>62.56 / 15.11</u> | **73.27 / 16.44** |
| ugv | ugv_parking03 | 11.58 / 3.12 | 46.35 / 7.41 | 34.93 / 12.50 | 45.32 / 0.00 | 37.99 / 16.67 | <u>59.87 / 25.00</u> | **69.49 / 33.33** |
| average | 10 sequences | 36.88 / 6.89 | 44.44 / 11.21 | 48.77 / 13.73 | 70.87 / 19.97 | 66.20 / 23.50 | <u>78.50 / 34.91</u> | **80.96 / 37.09** |
| GEODE Offroad | Offroad02_beta | N/A | N/A | 11.97 / 0.58 | 27.77 / 0.00 | 45.77 / 4.07 | <u>68.91 / 7.56</u> | **73.12 / 8.14** |

The sequence-calibrated rows above are intentionally excluded from this
paper-facing table because they are not causal. The best strict causal method
by average `MR@100P` is currently
`query_gate_graph:def=0.5,res=0.25,margin=0`.

## GEODE Reference Experiments

GEODE offroad runs are included as reference experiments because they expose a
different operating regime from FusionPortable: visual retrieval is highly
imbalanced and offroad false loops are frequent.

`Offroad05_beta` is extremely imbalanced: `11835` retrieval candidates contain
only `62` positives, spanning `32` query frames. The newer beta run adds
`Offroad01_beta` and `Offroad02_beta`: `Offroad01_beta` has `6905` evaluated
candidates with `76` retrieved positives, while `Offroad02_beta` has `9275`
evaluated candidates with `172` retrieved positives.

Values are reported in percent as `AP / MR@100P`. Full-sequence
`percentile_product` and `z_fusion` are non-causal references here as well.
Best and second-best methods are selected by `MR@100P`, with AP used as the
tie-breaker.

| Dataset | Sequence | SALAD score only | ROVER deformation only | absolute_graph:def=0.5,res=0.25 | query_gate_graph:def=0.5,res=0.25,margin=0 | percentile_product:res,def | z_fusion:salad=0,def=4,res=2 | Support ensemble graph evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GEODE Offroad | Offroad05_beta | 7.71 / 1.61 | 2.40 / 0.00 | N/A | N/A | <u>21.08 / 3.23</u> | 20.44 / 3.23 | **77.42 / 29.03** |
| GEODE Offroad | Offroad01_beta | 0.74 / 0.00 | 8.82 / 0.00 | 22.97 / 0.00 | 19.82 / 0.00 | 10.68 / 0.00 | **24.48 / 5.26** | <u>27.48 / 0.00</u> |
| GEODE Offroad | Offroad02_beta | 11.97 / 0.58 | 27.77 / 0.00 | 68.91 / 7.56 | 73.12 / 8.14 | <u>71.70 / 24.42</u> | **76.70 / 28.49** | 81.78 / 22.67 |

These results are not folded into the FusionPortable average above. They are
kept as a separate reference because the current GEODE experiments use external GT
timestamp association and, when enabled, the expensive support-ensemble
configuration.

## NTU-VIRAL Reference Experiments

NTU-VIRAL is included as a reference generalization experiment. The current
labels use the AsterSLAM-exported keyframe trajectory
`raw/trajectory_keyframes.txt` as the reference trajectory
(`gt_label_source=aster_slam_trajectory_keyframes`), rather than the previous
external GT heading approximation. This fixed the severe label noise observed in
the first NTU-VIRAL run, but the resulting `MR@100P` is still low, indicating
that false positives appear early in the ranked loop candidates.

The evaluated candidate/positive counts are: `eee_01` `3915/286`, `eee_02`
`2745/672`, `nya_01` `2375/211`, `nya_02` `3855/1411`, and `nya_03`
`5185/1606`. Values are reported in percent as `AP / MR@100P`. Best and
second-best causal methods are selected by `MR@100P`, with AP used as the
tie-breaker.

| Dataset | Sequence | SALAD score only | ROVER deformation only | PGO residual only | absolute_graph:def=0.5,res=0.25 | query_gate_graph:def=0.5,res=0.25,margin=0 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| NTU-VIRAL | eee_01 | **66.13 / 9.09** | 71.58 / 1.05 | 62.17 / 2.80 | 75.56 / 1.75 | <u>74.02 / 3.50</u> |
| NTU-VIRAL | eee_02 | **63.04 / 7.14** | 62.46 / 0.45 | 54.92 / 0.15 | <u>64.86 / 0.60</u> | 64.40 / 0.60 |
| NTU-VIRAL | nya_01 | **49.99 / 6.64** | 58.23 / 0.95 | 40.50 / 0.95 | <u>61.04 / 0.95</u> | 58.19 / 0.95 |
| NTU-VIRAL | nya_02 | 77.79 / 4.96 | <u>84.67 / 6.52</u> | 84.17 / 4.11 | 87.75 / 6.02 | **88.28 / 8.86** |
| NTU-VIRAL | nya_03 | **75.33 / 4.86** | 83.87 / 0.68 | 80.41 / 1.43 | 86.70 / 2.62 | <u>86.43 / 3.42</u> |
| average | 5 sequences | **66.46 / 6.54** | 72.16 / 1.93 | 64.43 / 1.89 | 75.18 / 2.39 | <u>74.27 / 3.46</u> |

The sequence-calibrated NTU-VIRAL references show that adding retrieval score
back into the graph evidence improves ranking, but these rows are not causal
and should be interpreted only as offline diagnostics: `percentile_product:res,def`
averages `75.57 / 5.67`, `percentile_product:salad,def` averages
`80.19 / 17.90`, `percentile_product:salad,res,def` averages `80.42 / 16.15`,
`z_fusion:salad=0.25,def=0.5,res=0.25` averages `80.89 / 16.50`, and
`z_fusion:salad=0.25,def=0.25,res=0.5` averages `80.07 / 19.52`.

These results are not folded into the FusionPortable average above. NTU-VIRAL
currently uses AsterSLAM reference/pseudo-GT labels, and the low `MR@100P`
suggests it is better treated as a generalization and failure-analysis dataset
than as primary evidence for safe online loop acceptance.
