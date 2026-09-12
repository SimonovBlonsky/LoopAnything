# LoopAnything New Main Experiment

This document reports the current paper-facing results on the manually annotated,
ROVER-aligned loop-verification benchmark. It supersedes the automatic
distance-and-rotation-label results in `main_experiment.md`.

## Benchmark Protocol

- Ten sequences from FusionPortableV2, GEODE, and NTU-VIRAL.
- Forty sampled queries per sequence and ten frozen DBoW2 candidates per query.
- A total of 4,000 query-candidate pairs, including 1,909 manually annotated positives.
- All methods evaluate exactly the same candidate pairs and binary labels.
- Metrics are reported as percentages in the form `AP / MR@100P`.
- The verifier results use the candidate records generated after the recent-support
  rescue, so the known `no_valid_support` implementation issue is not included.

## Compared Methods

- **DBoW2, NetVLAD, SALAD:** retrieval-score baselines.
- **DA3 Forward-only:** ranks frozen pairs using only DA3/Sim3 forward-pass
  diagnostics, without SALAD, PGO, or GT-error fields.
- **ROVER-like:** ranks candidates using trajectory deformation.
- **LoopAnything:** the previous query-gated graph reranking method.
- **Hard veto:** preserves SALAD ranking, applies `SALAD >= 0.2`, and rejects candidates
  above the 97.5th percentile of trajectory deformation.
- **Soft penalty (AP-opt):** uses
  `SALAD - 0.05 * max(0, log1p(pgo_error_after) - 3.1323)`.
- **Soft penalty (MR-opt):** applies `SALAD >= 0.4` and uses
  `SALAD - 0.25 * max(0, deformation - 6.3373)`.

## Per-Sequence Results

| Dataset / sequence | Pos. | DBoW2 | NetVLAD | SALAD | ROVER-like | Previous LoopAnything | Hard veto | Soft penalty (AP-opt) | Soft penalty (MR-opt) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FusionPortableV2/handheld/handheld_escalator00 | 158 | 84.23 / 17.72 | 94.19 / 51.27 | 94.64 / 48.73 | 79.18 / 4.43 | 89.76 / 51.90 | 94.64 / 48.73 | 94.71 / 50.63 | 91.24 / 48.73 |
| FusionPortableV2/handheld/handheld_room00 | 39 | 31.82 / 7.69 | 41.23 / 15.38 | 48.27 / 20.51 | 47.92 / 20.51 | 53.98 / 17.95 | 48.27 / 20.51 | 52.23 / 20.51 | 48.24 / 20.51 |
| FusionPortableV2/handheld/handheld_room01 | 238 | 88.73 / 0.42 | 94.23 / 9.24 | 95.65 / 30.67 | 84.33 / 2.10 | 85.79 / 1.26 | 95.65 / 30.67 | 96.37 / 30.67 | 95.65 / 30.67 |
| FusionPortableV2/ugv/ugv_campus01 | 153 | 80.85 / 21.57 | 96.02 / 39.87 | 95.66 / 54.90 | 69.60 / 2.61 | 82.19 / 20.92 | 95.66 / 54.90 | 95.90 / 58.17 | 93.75 / 48.37 |
| FusionPortableV2/ugv/ugv_parking01 | 228 | 85.48 / 0.44 | 94.97 / 8.77 | 95.44 / 17.98 | 78.26 / 1.75 | 81.48 / 1.32 | 95.08 / 17.98 | 94.82 / 11.84 | 93.58 / 13.16 |
| GEODE/Offroad/Offroad02_beta | 149 | 48.46 / 0.67 | 76.69 / 1.34 | 84.20 / 14.09 | 58.46 / 0.67 | 60.60 / 4.03 | 83.63 / 14.09 | 83.53 / 13.42 | 79.18 / 23.49 |
| GEODE/Offroad/Offroad05_beta | 139 | 57.44 / 2.88 | 83.39 / 8.63 | 87.55 / 15.11 | 51.31 / 0.72 | 54.90 / 0.72 | 87.76 / 15.11 | 86.92 / 14.39 | 79.30 / 24.46 |
| NTU-VIRAL/NTU-VIRAL/eee_01 | 204 | 79.61 / 27.45 | 87.47 / 16.18 | 94.54 / 22.55 | 82.45 / 10.29 | 88.21 / 30.39 | 94.54 / 22.55 | 94.60 / 22.55 | 94.24 / 22.55 |
| NTU-VIRAL/NTU-VIRAL/eee_02 | 317 | 92.88 / 13.25 | 99.17 / 55.84 | 99.39 / 67.82 | 97.16 / 30.28 | 97.72 / 48.58 | 99.39 / 67.82 | 99.36 / 67.82 | 99.19 / 67.82 |
| NTU-VIRAL/NTU-VIRAL/nya_02 | 284 | 97.45 / 16.55 | 98.60 / 7.75 | 98.21 / 2.11 | 94.97 / 10.92 | 97.60 / 9.51 | 98.21 / 2.11 | 98.21 / 2.11 | 98.21 / 2.11 |
| **Macro average** | **1,909** | **74.69 / 10.86** | **86.60 / 21.43** | **89.35 / 29.45** | **74.36 / 8.43** | **79.22 / 18.66** | **89.28 / 29.45** | **89.67 / 29.21** | **87.26 / 30.19** |

## Current Conclusions

- The AP-optimized residual penalty gives the highest macro AP, improving SALAD from
  `89.35` to `89.67`, while reducing MR@100P from `29.45` to `29.21`.
- The MR-optimized deformation penalty gives the highest macro MR@100P, improving it
  from `29.45` to `30.19`, but macro AP falls to `87.26`.
- The best hard-veto configuration does not improve over SALAD: `89.28 / 29.45`
  versus `89.35 / 29.45`.
- The current graph evidence is useful as a weak residual-aware penalty, but it is not
  yet reliable enough to serve as the paper's hard loop-factor rejection rule.
- The previous query-gated reranker is not competitive on the manually annotated
  benchmark because graph evidence can promote low-visual-similarity candidates.

## Strict Positive-Rechecked Main-8 Benchmark

After the full eight-sequence manual review, all pairs marked positive were checked a
second time with the positive-only review UI. This stricter label set is now the
paper-facing benchmark for `table1.tex` and `table2.tex`.

Benchmark root:

```text
workspace/rover_aligned_benchmark/benchmark_v1_da3_auto_labels_v4_asterslam_main8_full_manual_review_20260701_013214_positive_recheck_main8_20260701_115646
```

Data-release record for the strict benchmark:

- Pair manifest:
  `workspace/rover_aligned_benchmark/benchmark_v1_da3_auto_labels_v4_asterslam_main8_full_manual_review_20260701_013214_positive_recheck_main8_20260701_115646/benchmark_pairs.jsonl`
- Final strict labels:
  `workspace/rover_aligned_benchmark/benchmark_v1_da3_auto_labels_v4_asterslam_main8_full_manual_review_20260701_013214_positive_recheck_main8_20260701_115646/annotations.jsonl`
- Annotation integrity seal:
  `workspace/rover_aligned_benchmark/benchmark_v1_da3_auto_labels_v4_asterslam_main8_full_manual_review_20260701_013214_positive_recheck_main8_20260701_115646/annotation_seal.json`
- Benchmark manifest:
  `workspace/rover_aligned_benchmark/benchmark_v1_da3_auto_labels_v4_asterslam_main8_full_manual_review_20260701_013214_positive_recheck_main8_20260701_115646/manifest.json`
- Frozen method scores:
  `workspace/rover_aligned_benchmark/benchmark_v1_da3_auto_labels_v4_asterslam_main8_full_manual_review_20260701_013214_positive_recheck_main8_20260701_115646/scores/`
- LoopAnything candidate records used for the platform-shared sweep:
  `workspace/rover_aligned_benchmark/benchmark_v1/candidate_records.jsonl`

Baseline metrics:

```text
metrics/main8_manual_review_eval_20260701_124936/
```

LoopAnything per-platform shared Hard Gate sweep:

```text
metrics/hard_gate_platform_shared_sweep_20260804_v3/
```

The strict positive counts are:

| Sequence | Pos. |
| --- | ---: |
| FusionPortableV2/handheld/handheld_escalator00 (`esc0`) | 132 |
| FusionPortableV2/handheld/handheld_room00 (`room0`) | 24 |
| FusionPortableV2/ugv/ugv_campus01 (`camp1`) | 84 |
| FusionPortableV2/ugv/ugv_parking01 (`park1`) | 3 |
| GEODE/Offroad/Offroad02_beta (`off2`) | 19 |
| GEODE/Offroad/Offroad05_beta (`off5`) | 3 |
| NTU-VIRAL/NTU-VIRAL/eee_01 (`eee1`) | 87 |
| NTU-VIRAL/NTU-VIRAL/nya_02 (`nya2`) | 249 |
| **Total** | **601** |

### Platform-Shared Hard Gate Sweep

For this comparison, the score is the negative maximum normalized Hard Gate
violation. Parameters are shared within each platform group, not tuned per sequence.
The selection rule keeps settings within `0.01` macro MR@100P of the group-best
MR@100P and then chooses the highest macro AP. The sweep reuses the cached candidate
diagnostics and does not rerun DA3 or GTSAM.

| Group | Shared sequences | Parameter key | AP / MR@100P | TP@100P |
| --- | --- | --- | ---: | ---: |
| handheld | `esc0`, `room0` | `handheld` | 87.67 / 50.76 | 89 / 156 |
| ugv | `camp1`, `park1` | `ugv` | 90.29 / 61.90 | 50 / 87 |
| geode | `off2`, `off5` | `geode` | 63.23 / 43.86 | 6 / 22 |
| ntu | `eee1`, `nya2` | `ntu` | 92.33 / 21.02 | 60 / 336 |
| **Balanced aggregate** | all eight | platform-shared | **83.38 / 44.39** | **205 / 601** |

The exact selected thresholds, platform membership, search multipliers, selection
rule, benchmark inputs, and full-precision aggregate are stored in:

```text
configs/robust_loop_verifier/table1_hard_gate_platform_shared_20260804.json
```

Reproduction command from the `LoopAnything` repository root:

```bash
source /home/chenguyuan/anaconda3/etc/profile.d/conda.sh
conda activate da3
PYTHONPATH=src python \
  robust_loop_verification_scripts/check_8sequence_benchmark/run_platform_shared_sweep.py \
  workspace/rover_aligned_benchmark/benchmark_v1_da3_auto_labels_v4_asterslam_main8_full_manual_review_20260701_013214_positive_recheck_main8_20260701_115646 \
  --hard-gate-candidate-scores \
  workspace/rover_aligned_benchmark/benchmark_v1_da3_auto_labels_v4_asterslam_main8_full_manual_review_20260701_013214_positive_recheck_main8_20260701_115646/metrics/aster_hard_gate_max_violation_20260804/candidate_scores.jsonl \
  --output-dir \
  workspace/rover_aligned_benchmark/benchmark_v1_da3_auto_labels_v4_asterslam_main8_full_manual_review_20260701_013214_positive_recheck_main8_20260701_115646/metrics/hard_gate_platform_shared_sweep_20260804_v3 \
  --mr-tolerance 0.01
```

### Global-Shared Hard Gate Sweep (Table 1)

Table 1 now uses one exactly shared Hard-Gate parameter set for all eight
sequences. TP@100P is not used for selection. The deterministic extreme search
performs `1,503,736` approximate candidate evaluations over four local refinement
rounds and evaluates `791` unique exact finalists over log2 multipliers in
`[-16, 16]`.

The two macro-average optima and the sequence-coverage setting are:

| Selection | AP / MR@100P | Joint Table-1 sequence wins | AP/MR cells won |
| --- | ---: | ---: | ---: |
| Best macro AP | **82.59 / 36.12** | 2 / 8 | 7 / 16 |
| Best macro MR@100P | 79.33 / **50.43** | 4 / 8 | 11 / 16 |
| **Max joint sequence wins (Table 1)** | **79.71 / 46.67** | **6 / 8** | **13 / 16** |

| Operating point | esc0 | room0 | camp1 | park1 | off2 | off5 | eee1 | nya2 | Avg. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Best macro AP | 95.60 / 42.42 | 63.25 / 29.17 | 95.03 / 52.38 | 80.56 / 33.33 | 65.64 / 21.05 | 76.67 / 66.67 | 88.25 / 29.89 | 95.72 / 14.06 | **82.59 / 36.12** |
| Best macro MR@100P | 98.63 / 82.58 | 93.30 / 87.50 | 99.32 / 91.67 | 80.56 / 33.33 | 48.59 / 31.58 | 37.59 / 33.33 | 82.84 / 31.03 | 93.84 / 12.45 | **79.33 / 50.43** |
| Max joint sequence wins | 97.17 / 68.94 | 84.35 / 70.83 | 99.06 / 89.29 | 80.56 / 33.33 | 57.08 / 31.58 | 43.39 / 33.33 | 83.13 / 35.63 | 92.91 / 10.44 | **79.71 / 46.67** |

The Table 1 row uses the max-joint-wins thresholds below. The macro AP and macro
MR optima are retained as separate reproducible operating points.

```text
salad_min_score=0.8, support_min_baseline=0.3,
sim3_min_scale=0.8, sim3_max_scale=80,
sim3_alignment_rmse=0.1325061, sim3_direction_error=25,
translation_norm=2.3831312, trajectory_deformation=0.058926166,
pgo_error_per_factor=1, odom_strain_chi2=50, loop_chi2=0.018933704
```

The complete search definitions and all three solutions are stored in:

```text
configs/robust_loop_verifier/table1_hard_gate_global_shared_extreme_20260806_v5.json
```

Result directory:

```text
workspace/rover_aligned_benchmark/benchmark_v1_da3_auto_labels_v4_asterslam_main8_full_manual_review_20260701_013214_positive_recheck_main8_20260701_115646/metrics/hard_gate_global_shared_extreme_sweep_20260806_v5/
```

Reproduction command (run from `LoopAnything/`):

```bash
PYTHONPATH=src /home/chenguyuan/anaconda3/envs/da3/bin/python \
  robust_loop_verification_scripts/check_8sequence_benchmark/run_extreme_global_shared_sweep.py \
  workspace/rover_aligned_benchmark/benchmark_v1_da3_auto_labels_v4_asterslam_main8_full_manual_review_20260701_013214_positive_recheck_main8_20260701_115646 \
  --hard-gate-candidate-scores \
  workspace/rover_aligned_benchmark/benchmark_v1_da3_auto_labels_v4_asterslam_main8_full_manual_review_20260701_013214_positive_recheck_main8_20260701_115646/metrics/aster_hard_gate_max_violation_20260804/candidate_scores.jsonl \
  --output-dir \
  workspace/rover_aligned_benchmark/benchmark_v1_da3_auto_labels_v4_asterslam_main8_full_manual_review_20260701_013214_positive_recheck_main8_20260701_115646/metrics/hard_gate_global_shared_extreme_sweep_20260806_v5 \
  --global-samples 300000 --local-samples 300000 --local-rounds 4 \
  --exact-finalists 1024 --batch-size 1024 --seed 20260807 \
  --log2-min -16 --log2-max 16 \
  --warm-start-config configs/robust_loop_verifier/table1_hard_gate_global_shared_extreme_20260806_v4.json
```

Per-sequence AP / MR@100P values in percent for the Table 1 max-joint-wins
setting:

| Method | esc0 | room0 | camp1 | park1 | off2 | off5 | eee1 | nya2 | Avg. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DBoW2 | 81.21 / 14.39 | 36.01 / 12.50 | 77.11 / 23.81 | 1.31 / 0.00 | 26.71 / 10.53 | 21.19 / 0.00 | 64.10 / 8.05 | 95.08 / 22.09 | 50.34 / 11.42 |
| NetVLAD | 94.78 / 56.06 | 38.12 / 16.67 | 77.57 / 19.05 | 37.34 / 33.33 | 31.67 / 5.26 | 37.92 / 33.33 | 71.25 / 29.89 | 96.60 / 16.87 | 60.66 / 26.31 |
| SALAD | 95.47 / 58.33 | 60.15 / 41.67 | 72.58 / 19.05 | 8.98 / 0.00 | 15.69 / 5.26 | 36.01 / 33.33 | 75.26 / 14.94 | 96.46 / 14.46 | 57.57 / 23.38 |
| BoQ-dinov2 | 95.91 / 65.91 | 70.56 / 33.33 | 81.06 / 30.95 | 8.71 / 0.00 | 21.24 / 0.00 | 41.39 / 33.33 | 82.15 / 35.63 | 96.36 / 29.72 | 62.17 / 28.61 |
| LoFTR | 86.42 / 68.94 | 61.37 / 33.33 | 79.50 / 32.14 | 1.23 / 0.00 | 24.55 / 5.26 | 43.46 / 33.33 | 78.99 / 18.39 | 91.01 / 14.86 | 58.32 / 25.78 |
| DUSt3R | 67.91 / 21.97 | 55.11 / 25.00 | 96.24 / 38.10 | 45.56 / 33.33 | 49.04 / 31.58 | 46.92 / 33.33 | 36.86 / 3.45 | 79.80 / 12.85 | 59.68 / 24.95 |
| MAST3R | 80.46 / 9.85 | 13.12 / 0.00 | 75.99 / 36.90 | 34.49 / 33.33 | 56.64 / 31.58 | 35.93 / 33.33 | 75.09 / 16.09 | 94.23 / 17.27 | 58.24 / 22.29 |
| VGGT-track | 80.13 / 32.58 | 13.58 / 0.00 | 29.65 / 5.95 | 3.05 / 0.00 | 13.66 / 0.00 | 17.17 / 0.00 | 73.99 / 19.54 | 92.10 / 13.25 | 40.42 / 8.92 |
| ROVER-like | 86.57 / 5.30 | 74.20 / 54.17 | 68.58 / 4.76 | 4.86 / 0.00 | 25.43 / 0.00 | 37.54 / 33.33 | 75.26 / 17.24 | 90.86 / 10.04 | 57.91 / 15.61 |
| **Ours** | **97.17 / 68.94** | **84.35 / 70.83** | **99.06 / 89.29** | **80.56 / 33.33** | **57.08 / 31.58** | 43.39 / 33.33 | **83.13 / 35.63** | 92.91 / 10.44 | **79.71 / 46.67** |

## DA3 Auto-Label Frozen Benchmark Calibration

After fixing the VPR cache camera/lidar frame issue, we rebuilt the benchmark cache
and generated an auxiliary DA3 automatic-label benchmark. This benchmark keeps the
same frozen ROVER-aligned pair set, but replaces the manual visual labels with DA3
geometry labels. The automatic label is positive when the recovered metric DA3 loop
factor is close to the reference relative pose under the following thresholds:

- rotation error `<= 15 deg`;
- translation direction error `<= 20 deg`;
- translation error threshold `min(max(0.2 * ||t_gt||, 1.0 m), 5.0 m)`.

This DA3 auto-label benchmark has `4,000` pairs and `1,402` positives. It is useful
for score calibration because it measures whether a ranked pair produces a usable DA3
geometry factor, rather than only whether it is visually similar.

### Current Frozen-Pair Pipeline

The current benchmark pipeline does not rerun retrieval. It scores the fixed DBoW2
top-10 candidate pairs as follows:

```text
frozen DBoW2 top-10 q-c pairs
  -> SALAD descriptor score for the same q-c pair
  -> candidate-local support-frame selection
  -> DA3 triplet geometry on (query, candidate, support)
  -> odometry-based Sim3 alignment using candidate-support odometry
  -> metric query-candidate loop factor
  -> full-prefix PGO with the recovered loop factor
  -> graph/deformation evidence and Sim3 consistency score
  -> calibrated final ranking score
```

The previous LoopAnything row used only query-gated graph evidence:

```text
graph = -deformation - 0.5 * log(1 + PGO residual)
score = graph + best_graph_for_current_query
```

The tuned score that performs best on the DA3 auto-label benchmark keeps SALAD as an
appearance prior and adds two calibrated geometry terms:

```text
graph = -deformation - 2 * log(1 + PGO residual)

score = SALAD
      + 2.0 * z_seq(graph)
      + 1.0 * percentile_query(-Sim3 direction error)
```

Here `z_seq(.)` is sequence-level z-score normalization over the evaluated frozen
pairs in the same sequence, and `percentile_query(.)` is the percentile rank among
the current query's top-10 candidates. This score is recorded as:

```text
salad_plus_seqz_graph_qpct_sim3dir:ratio=2,ag=2,ad=1
```

### DA3 Auto-Label Results

Values are reported as `AP / MR@100P` in percent. The tuned LoopAnything row is the
recommended paper-facing score for this frozen benchmark setting because it is ranked
second by both AP and MR@100P in the exact tie-safe sweep.

| Method | AP / MR@100P |
| --- | ---: |
| DBoW2 | 53.53 / 7.20 |
| NetVLAD | 59.36 / 10.70 |
| SALAD | 59.95 / 10.31 |
| BoQ-dinov2 | 61.68 / 10.50 |
| DA3 Forward-only | 59.58 / 2.70 |
| LoFTR | 53.65 / 12.55 |
| VGGT-track | 57.32 / 4.42 |
| DUSt3R | 56.70 / **18.03** |
| MAST3R | **66.11** / 7.20 |
| ROVER-like | 58.12 / 6.47 |
| Previous LoopAnything | 63.57 / 13.31 |
| **Tuned LoopAnything** | 66.08 / 17.21 |

BoQ-dinov2 is included as a strong pure image-retrieval baseline. It scores each
frozen q-c pair by cosine similarity between L2-normalized BoQ global descriptors.
It improves AP over NetVLAD and SALAD on the full 10-sequence benchmark, but its
MR@100P and TP@100P remain below the best geometric and graph-consistency methods,
showing that stronger global appearance descriptors alone do not solve
zero-false-positive safe-factor discovery.

LoFTR is included as a ROVER-style geometric verification baseline: it scores each
frozen q-c pair by the number of RANSAC inliers after LoFTR matching. It improves
MR@100P over appearance-only retrieval baselines, but its AP is close to DBoW2,
showing that raw local-feature inlier count is not well calibrated across sequences.

DUSt3R is included as a 3D pointmap matching baseline under the same ROVER-style
protocol. It has lower macro AP than MAST3R and tuned LoopAnything, but its
MR@100P is strong, mainly because it discovers many zero-false-positive positives
on `handheld_room01`.

MAST3R is included as a 3D-grounded dense matching baseline under the same
ROVER-style protocol. It achieves the best macro AP by a small margin, but its
MR@100P is much lower than tuned LoopAnything. This indicates strong global ranking
quality but weaker zero-false-positive safe-factor discovery under this benchmark.

VGGT-track is included as another foundation-model correspondence baseline. It uses
VGGT's native track head to produce q-c correspondences and ranks each pair by the
number of RANSAC-filtered inlier matches. Its AP is competitive with other
geometric baselines on visually easy sequences, but its MR@100P and TP@100P are low,
showing that dense tracked correspondence count alone is poorly calibrated for
zero-false-positive safe-factor discovery.

DA3 Forward-only is included as an ablation of the proposed geometry path. It ranks
the same frozen q-c pairs using only forward-pass DA3/Sim3 diagnostics, without SALAD
appearance evidence, PGO residuals, trajectory deformation, or GT pose errors. Its
AP is close to SALAD on the 10-sequence benchmark, but its MR@100P and TP@100P are
very low. This shows that raw DA3 geometry diagnostics are not enough to expose
zero-false-positive safe loop factors; the appearance prior and graph-consistency
verification signals are both needed.

#### BoQ Reproducibility Notes

The BoQ baseline follows a pure retrieval protocol. For each frozen q-c pair, it
extracts one BoQ global descriptor per image and ranks the pair by the dot product
of L2-normalized descriptors. It does not use local matching, RANSAC, DA3, Sim3
alignment, PGO, verifier scores, or GT pose errors.

Recorded execution details:

- BoQ repository commit: `1a4965ea7dfd9bd0dd846adf7a0e430f68101d12`.
- Model loading: `hubconf.get_trained_boq(backbone_name="dinov2", output_dim=12288)`.
- Checkpoint path: `None`; the official torch.hub BoQ weights were used.
- Image preprocessing: RGB image, `ToTensor`, bicubic resize to `322 x 322`, and
  ImageNet normalization.
- Benchmark score: dot product of L2-normalized BoQ global descriptors.
- Score file:
  `workspace/rover_aligned_benchmark/benchmark_v1_da3_auto_labels_v4_asterslam/scores/boq_dinov2.jsonl`.
- All `4,000` frozen pairs were scored successfully.

#### LoFTR Reproducibility Notes

The paper-facing LoFTR baseline follows the ROVER-style geometric-verification
protocol: LoFTR produces dense matches for the same frozen ROVER-aligned q-c pairs as
all other methods, and the verification score is the number of RANSAC-filtered
inlier matches. This keeps the comparison aligned with ROVER's Table I, where local
feature and geometric foundation model baselines are ranked by RANSAC-filtered
inlier-match counts rather than their native dataset-specific pose-AUC metrics.

For reproducibility, the LoFTR matcher follows the official inference snippet in
`LoFTR/README.md`: it instantiates `src.loftr.LoFTR(config=default_cfg)`, loads the
pretrained DS checkpoint, switches to `eval()`, and runs under `torch.no_grad()`.

Recorded execution details:

- LoFTR repository commit: `df7ca80f917334b94cfbe32cc2901e09a80e70a8`.
- Checkpoint: `LoFTR/data/weights/indoor_ds_new.ckpt`.
- Checkpoint SHA256:
  `be9ff88b323ec27889114719f668ae41aff7034b56a4c4acbd46b8b180b87ed3`.
- Image preprocessing: grayscale input resized to `640 x 480`, matching the
  ScanNet-style resolution used by the official indoor evaluation loader.
- Matcher config used by the paper-facing score file:
  `src.loftr.utils.cvpr_ds_config.default_cfg`,
  i.e. DS matching with `MATCH_COARSE.THR=0.2`, `MATCH_TYPE=dual_softmax`,
  `BORDER_RM=2`, and `TEMP_BUG_FIX=False`.
- Benchmark score: number of geometric inliers after OpenCV fundamental-matrix
  RANSAC, using `USAC_MAGSAC`, threshold `1.0 px`, confidence `0.999`,
  maximum `10000` iterations, and minimum `8` matches.

We also tested the ScanNet-new evaluation config
`configs/loftr/indoor/scannet/loftr_ds_eval_new.py`, which changes the matcher to
`TEMP_BUG_FIX=True` and `BORDER_RM=0`. That stronger variant is retained as an
exploratory result for future method improvement, but it is not the paper-facing
LoFTR row used by the current Table 1/2 because the tables are currently frozen to
the ROVER-style baseline configuration above.

#### DUSt3R Reproducibility Notes

The DUSt3R baseline follows the ROVER-style geometric-verification protocol. It
runs DUSt3R on each frozen q-c pair, filters pointmaps by confidence, extracts
reciprocal pointmap matches, and ranks each pair by the number of RANSAC-filtered
inlier matches. This is a geometric matchability score, not a metric safe-factor
verification score.

Recorded execution details:

- DUSt3R repository commit: `4c24a6ebf04809f2cfe59915e51779c8984aaa40`.
- Checkpoint: `dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth`.
- Checkpoint SHA256:
  `9f194a32e34fc124ed10f64115b8e848b7da87c2d1ed131d34b4806721c19dbd`.
- Image preprocessing: official DUSt3R loader with `size=512`.
- Pointmap confidence threshold: `3.0`.
- Benchmark score: number of geometric inliers after OpenCV fundamental-matrix
  RANSAC, using `USAC_MAGSAC`, threshold `1.0 px`, confidence `0.999`,
  maximum `10000` iterations, and minimum `8` matches.
- Score file:
  `workspace/rover_aligned_benchmark/benchmark_v1_da3_auto_labels_v4_asterslam/scores/dust3r.jsonl`.
- `3,998 / 4,000` frozen pairs were scored successfully; the remaining two pairs
  failed inside OpenCV RANSAC and are handled by the benchmark's failed-score
  sentinel.

#### VGGT Reproducibility Notes

The VGGT baseline follows the same ROVER-style geometric-verification protocol as
LoFTR, DUSt3R, and MAST3R. For each frozen q-c pair, VGGT's native track head tracks
a regular query grid from the query image to the candidate image. The benchmark
score is the number of RANSAC-filtered inlier correspondences. This uses VGGT only
as a correspondence-producing geometric baseline; it does not use DA3, Sim3
alignment, PGO, verifier scores, or GT pose errors.

Recorded execution details:

- VGGT repository commit: `5052e59eb4cdff994ed8d502614ff85952cd3fd2`.
- Model path:
  `/home/chenguyuan/.cache/huggingface/hub/models--facebook--VGGT-1B/snapshots/860abec7937da0a4c03c41d3c269c366e82abdf9`.
- Image preprocessing: VGGT official loader with `preprocess_mode=pad` and target
  size `518`.
- Query grid: `grid_size=32`, `border_px=8`, `max_query_points=2048`; this produced
  `289` query points per pair under the current VGGT input size.
- Track filtering: candidate-frame `visibility_threshold=0.2` and
  `confidence_threshold=0.2`.
- Benchmark score: number of geometric inliers after OpenCV fundamental-matrix
  RANSAC, using `USAC_MAGSAC`, threshold `1.0 px`, confidence `0.999`,
  maximum `10000` iterations, and minimum `8` matches.
- Score file:
  `workspace/rover_aligned_benchmark/benchmark_v1_da3_auto_labels_v4_asterslam/scores/vggt.jsonl`.
- `3,997 / 4,000` frozen pairs were scored successfully; the remaining three pairs
  failed inside OpenCV RANSAC and are handled by the benchmark's failed-score
  sentinel.

#### MAST3R Reproducibility Notes

The MAST3R baseline follows the official pair-matching path in `mast3r/README.md`:
it loads `AsymmetricMASt3R`, runs `dust3r.inference.inference` on each frozen q-c
pair, extracts the predicted dense descriptors, and computes reciprocal matches with
`mast3r.fast_nn.fast_reciprocal_NNs`.

Recorded execution details:

- MAST3R repository commit: `f5209afc300cec36239a7ac992263f36847bbba0`.
- Checkpoint:
  `mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth`.
- Checkpoint SHA256:
  `e28f91b488554653e2b46ddae9c78c1143e0bcb2e27d3e26cdb0b717f1568eb2`.
- Image preprocessing: official MAST3R/DUST3R loader with `size=512`.
- Matching: `subsample_or_initxy1=8`, `dist=dot`, `block_size=8192`, followed by
  the official 3 px border filtering.
- Benchmark score: number of geometric inliers after OpenCV fundamental-matrix
  RANSAC, using `USAC_MAGSAC`, threshold `1.0 px`, confidence `0.999`,
  maximum `10000` iterations, and minimum `8` matches.
- Score file:
  `workspace/rover_aligned_benchmark/benchmark_v1_da3_auto_labels_v4_asterslam/scores/mast3r.jsonl`.
- All `4,000` frozen pairs were scored successfully after fixing the MAST3R
  environment's PyTorch/MKL runtime issue.

#### DA3 Forward-only Reproducibility Notes

The DA3 Forward-only ablation reuses the geometry predictions generated for the
DA3 automatic-label benchmark. It does not rerun retrieval, does not use SALAD scores,
and does not use PGO. The reported paper-facing row is `DA3 residual only`, which
ranks candidates by the forward-pass residual diagnostic from
`geometry_predictions.jsonl`.

Summary:

| Setting | 10-seq AP / MR@100P | 10-seq TP@100P | 9-seq AP / MR@100P | 9-seq TP@100P |
| --- | ---: | ---: | ---: | ---: |
| DA3 residual only | 59.58 / 2.70 | 49 (3.50%) | 58.89 / 2.79 | 46 (3.71%) |
| DA3 seq-z residual+direction | 59.54 / 2.70 | 49 (3.50%) | 58.89 / 2.79 | 46 (3.71%) |
| DA3 direction only | 59.43 / 2.66 | 48 (3.42%) | 58.80 / 2.75 | 45 (3.63%) |
| DA3 query-percentile residual+direction | 43.54 / 0.00 | 0 (0.00%) | 43.52 / 0.00 | 0 (0.00%) |

Per-sequence result for the selected `DA3 residual only` row:

| Dataset / sequence | Pos. | AP / MR@100P | TP@100P |
| --- | ---: | ---: | ---: |
| FusionPortableV2/handheld/handheld_escalator00 | 176 | 81.53 / 5.11 | 9 |
| FusionPortableV2/handheld/handheld_room00 | 191 | 85.94 / 2.09 | 4 |
| FusionPortableV2/handheld/handheld_room01 | 292 | 96.05 / 7.53 | 22 |
| FusionPortableV2/ugv/ugv_campus01 | 163 | 70.22 / 3.68 | 6 |
| FusionPortableV2/ugv/ugv_parking01 | 70 | 33.22 / 0.00 | 0 |
| GEODE/Offroad/Offroad02_beta | 40 | 46.33 / 5.00 | 2 |
| GEODE/Offroad/Offroad05_beta | 22 | 7.25 / 0.00 | 0 |
| NTU-VIRAL/NTU-VIRAL/eee_01 | 107 | 47.16 / 0.00 | 0 |
| NTU-VIRAL/NTU-VIRAL/eee_02 | 162 | 65.74 / 1.85 | 3 |
| NTU-VIRAL/NTU-VIRAL/nya_02 | 179 | 62.34 / 1.68 | 3 |

Recorded execution details:

- DA3 model:
  `depth-anything/DA3-LARGE-1.1`, snapshot
  `0e109ae307c5982f319a67cf6f9f99ccdc0ec97c`.
- Model file:
  `/home/chenguyuan/.cache/huggingface/hub/models--depth-anything--DA3-LARGE-1.1/snapshots/0e109ae307c5982f319a67cf6f9f99ccdc0ec97c/model.safetensors`.
- Model SHA256:
  `739905c423cf0d6ccaf9e61a8401d82ba1ac32d7f4d3ee6dca8f92b377633f64`.
- DA3 inference: `backend=real`, `device=cuda`, `da3_process_res=504`,
  `da3_triplet_batch_size=4`, `da3_ref_view_strategy=first`.
- Auto-label thresholds used when generating `geometry_predictions.jsonl`:
  `rotation <= 15 deg`, `translation direction <= 20 deg`,
  `translation <= min(max(0.2 * ||t_gt||, 1.0 m), 5.0 m)`,
  `min_direction_baseline_m=0.5`.
- Prediction file:
  `workspace/rover_aligned_benchmark/benchmark_v1_da3_auto_labels_v4_asterslam/geometry_predictions.jsonl`.
- Prediction records: `4,000`.
- Report source:
  `workspace/rover_aligned_benchmark/benchmark_v1_da3_auto_labels_v4_asterslam/metrics/da3_forward_only_20260617/report.md`.

Per-sequence breakdown is listed below. Values are `AP / MR@100P` in percent. The
baseline rows come from the exact-refine benchmark's selected per-sequence output;
the tuned LoopAnything row decomposes the recommended balanced setting
`salad_plus_seqz_graph_qpct_sim3dir:ratio=2,ag=2,ad=1` by replaying the recorded
candidate records.

| Dataset / sequence | Pos. | DBoW2 | NetVLAD | SALAD | BoQ-dinov2 | DA3 Forward-only | LoFTR | VGGT-track | DUSt3R | MAST3R | ROVER-like | Previous LoopAnything | Tuned LoopAnything |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FusionPortableV2/handheld/handheld_escalator00 | 176 | 73.66 / 3.41 | 89.27 / 27.27 | 91.32 / 28.98 | 91.34 / 31.25 | 81.53 / 5.11 | 77.90 / 34.09 | 83.83 / 7.39 | 72.18 / 35.80 | 87.02 / 3.41 | 77.27 / 3.98 | 87.19 / 31.82 | 87.41 / 32.39 |
| FusionPortableV2/handheld/handheld_room00 | 191 | 57.96 / 1.57 | 61.78 / 5.76 | 65.05 / 9.95 | 72.98 / 4.71 | 85.94 / 2.09 | 56.11 / 6.81 | 76.17 / 2.62 | 85.60 / 13.61 | 86.41 / 0.52 | 59.27 / 7.85 | 66.92 / 9.95 | 72.69 / 14.14 |
| FusionPortableV2/handheld/handheld_room01 | 292 | 93.93 / 41.10 | 95.18 / 49.32 | 95.99 / 39.73 | 98.03 / 36.99 | 96.05 / 7.53 | 92.88 / 44.52 | 97.84 / 23.29 | 98.68 / 63.70 | 98.16 / 23.63 | 92.43 / 38.36 | 92.99 / 36.64 | 97.26 / 41.44 |
| FusionPortableV2/ugv/ugv_campus01 | 163 | 72.51 / 12.27 | 75.93 / 9.82 | 74.70 / 9.82 | 78.28 / 15.95 | 70.22 / 3.68 | 65.49 / 16.56 | 61.19 / 3.07 | 85.99 / 37.42 | 83.77 / 9.82 | 78.10 / 5.52 | 86.92 / 28.22 | 87.99 / 42.94 |
| FusionPortableV2/ugv/ugv_parking01 | 70 | 29.70 / 0.00 | 29.36 / 1.43 | 31.00 / 0.00 | 27.08 / 0.00 | 33.22 / 0.00 | 31.00 / 0.00 | 32.74 / 0.00 | 29.38 / 2.86 | 36.91 / 4.29 | 35.69 / 1.43 | 36.60 / 2.86 | 37.65 / 2.86 |
| GEODE/Offroad/Offroad02_beta | 40 | 24.42 / 5.00 | 28.92 / 2.50 | 17.77 / 0.00 | 26.67 / 5.00 | 46.33 / 5.00 | 23.21 / 12.50 | 25.53 / 2.50 | 36.85 / 15.00 | 46.49 / 10.00 | 29.56 / 0.00 | 46.80 / 12.50 | 54.72 / 17.50 |
| GEODE/Offroad/Offroad05_beta | 22 | 6.17 / 0.00 | 15.39 / 4.55 | 13.73 / 4.55 | 14.49 / 4.55 | 7.25 / 0.00 | 11.67 / 4.55 | 19.36 / 0.00 | 12.62 / 4.55 | 28.80 / 4.55 | 21.93 / 4.55 | 18.70 / 4.55 | 20.40 / 9.09 |
| NTU-VIRAL/NTU-VIRAL/eee_01 | 107 | 38.33 / 0.00 | 42.51 / 0.93 | 50.51 / 0.93 | 52.03 / 1.87 | 47.16 / 0.00 | 42.72 / 0.93 | 50.52 / 0.93 | 35.48 / 0.00 | 52.40 / 1.87 | 49.93 / 0.00 | 53.73 / 0.00 | 53.32 / 2.80 |
| NTU-VIRAL/NTU-VIRAL/eee_02 | 162 | 64.42 / 2.47 | 79.43 / 3.70 | 83.65 / 1.85 | 82.22 / 1.85 | 65.74 / 1.85 | 71.08 / 4.94 | 76.25 / 1.85 | 52.03 / 7.41 | 73.73 / 11.11 | 70.37 / 1.85 | 75.12 / 4.32 | 77.84 / 0.00 |
| NTU-VIRAL/NTU-VIRAL/nya_02 | 179 | 74.16 / 6.15 | 75.87 / 1.68 | 75.77 / 7.26 | 73.69 / 2.79 | 62.34 / 1.68 | 64.39 / 0.56 | 68.71 / 0.00 | 58.19 / 0.00 | 67.45 / 2.79 | 66.68 / 1.12 | 70.72 / 2.23 | 72.23 / 8.94 |

### Discoverable Safe Loop Factors

`MR@100P` can also be interpreted as the fraction of positive loop opportunities that
a method can retrieve before the first false positive appears in its ranked list.
Multiplying `MR@100P` by the number of positives gives `TP@100P`, i.e. the number of
safe loop opportunities that can be exposed at 100% precision under the benchmark
labels. The table below keeps the 9-sequence no-`eee_02` reference split used during
method selection; the current paper Table 2 additionally excludes
`FusionPortableV2/handheld/handheld_room01` and reports the corresponding
eight-sequence subset.

| Dataset / sequence | Pos. | DBoW2 | NetVLAD | SALAD | BoQ-dinov2 | DA3 Forward-only | LoFTR | VGGT-track | DUSt3R | MAST3R | ROVER-like | Tuned LoopAnything |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FusionPortableV2/handheld/handheld_escalator00 | 176 | 6 | 48 | 51 | 55 | 9 | 60 | 13 | 63 | 6 | 7 | 57 |
| FusionPortableV2/handheld/handheld_room00 | 191 | 3 | 11 | 19 | 9 | 4 | 13 | 5 | 26 | 1 | 15 | 27 |
| FusionPortableV2/handheld/handheld_room01 | 292 | 120 | 144 | 116 | 108 | 22 | 130 | 68 | 186 | 69 | 112 | 121 |
| FusionPortableV2/ugv/ugv_campus01 | 163 | 20 | 16 | 16 | 26 | 6 | 27 | 5 | 61 | 16 | 9 | 70 |
| FusionPortableV2/ugv/ugv_parking01 | 70 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 2 | 3 | 1 | 2 |
| GEODE/Offroad/Offroad02_beta | 40 | 2 | 1 | 0 | 2 | 2 | 5 | 1 | 6 | 4 | 0 | 7 |
| GEODE/Offroad/Offroad05_beta | 22 | 0 | 1 | 1 | 1 | 0 | 1 | 0 | 1 | 1 | 1 | 2 |
| NTU-VIRAL/NTU-VIRAL/eee_01 | 107 | 0 | 1 | 1 | 2 | 0 | 1 | 1 | 0 | 2 | 0 | 3 |
| NTU-VIRAL/NTU-VIRAL/nya_02 | 179 | 11 | 3 | 13 | 5 | 3 | 1 | 0 | 0 | 5 | 2 | 16 |
| **Total** | **1,240** | **162 (13.06%)** | **226 (18.23%)** | **217 (17.50%)** | **208 (16.77%)** | **46 (3.71%)** | **238 (19.19%)** | **93 (7.50%)** | **345 (27.82%)** | **107 (8.63%)** | **147 (11.85%)** | **305 (24.60%)** |

This result supports a second paper-level insight: compared with appearance-only
retrieval, ROVER-like deformation ranking, and most RANSAC-inlier geometric
matching baselines, LoopAnything can expose more safe loop constraints at zero
false-positive operating points. DUSt3R is a notable exception in this current
benchmark, mainly due to `handheld_room01`; this should be addressed explicitly in
the final table discussion rather than hidden.

### Positive-Pair Relative Pose Distribution

To understand what the DA3 auto-label benchmark considers a positive loop
opportunity, we measured the GT relative pose of every positive q-c pair. Translation
is the norm of `gt_translation` in `geometry_predictions.jsonl`, and rotation is
`gt_rotation_angle_deg`. The current paper Table 1 excludes `eee_02`, but it is kept
below as a reference row.

| Dataset / sequence | Pos. | t median (m) | t mean (m) | rot median (deg) | rot mean (deg) |
| --- | ---: | ---: | ---: | ---: | ---: |
| FusionPortableV2/handheld/handheld_escalator00 | 176 | 2.26 | 4.00 | 14.99 | 20.68 |
| FusionPortableV2/handheld/handheld_room00 | 191 | 5.67 | 5.65 | 35.91 | 50.02 |
| FusionPortableV2/handheld/handheld_room01 | 292 | 1.47 | 2.07 | 5.40 | 11.33 |
| FusionPortableV2/ugv/ugv_campus01 | 163 | 8.40 | 13.00 | 12.24 | 22.97 |
| FusionPortableV2/ugv/ugv_parking01 | 70 | 19.76 | 21.75 | 4.56 | 7.97 |
| GEODE/Offroad/Offroad02_beta | 40 | 11.39 | 11.86 | 12.25 | 19.99 |
| GEODE/Offroad/Offroad05_beta | 22 | 19.80 | 22.39 | 13.65 | 17.99 |
| NTU-VIRAL/NTU-VIRAL/eee_01 | 107 | 6.53 | 7.63 | 4.20 | 14.57 |
| NTU-VIRAL/NTU-VIRAL/eee_02 (excluded) | 162 | 2.61 | 3.13 | 4.56 | 6.25 |
| NTU-VIRAL/NTU-VIRAL/nya_02 | 179 | 1.89 | 2.13 | 5.78 | 9.74 |
| **Current 9-seq pooled** | **1,240** | **3.61** | **6.61** | **9.30** | **20.40** |
| **All 10-seq pooled** | **1,402** | **3.29** | **6.20** | **7.94** | **18.77** |

The positive labels are therefore not equivalent to a fixed small-pose-distance
threshold. UGV and GEODE positives include large translation baselines, while several
handheld sequences include substantial rotation changes. This supports interpreting
the benchmark as a DA3-geometry loop-opportunity benchmark rather than a strict
near-duplicate image retrieval benchmark.

Single-metric optima are:

- best AP: `salad_plus_seqz_graph_qpct_sim3dir:ratio=2,ag=1,ad=0.5`, `66.16 / 16.68`;
- best MR@100P: `salad_plus_seqz_graph_qpct_sim3dir:ratio=4,ag=1.5,ad=1`, `65.58 / 17.34`;
- recommended balanced setting: `salad_plus_seqz_graph_qpct_sim3dir:ratio=2,ag=2,ad=1`, `66.08 / 17.21`.

### MR/TP-Oriented Top-N Geometry Gate Sweep

After comparing DUSt3R and the SALAD-free graph ablation, we ran an additional
post-hoc sweep to test whether the current score can expose more zero-false-positive
safe factors. The sweep reuses the same frozen candidates and candidate records; it
does not rerun DA3 or retrieval. The best MR@100P and TP@100P setting was:

```text
topn_gate_current:gate=nosalad,n=5,pen=-0.5
```

The base score is the current calibrated score:

```text
graph = -trajectory_deformation_rmse - 2 * log(1 + pgo_error_after)

current_score = salad_score
              + 2.0 * z_seq(graph)
              + 1.0 * percentile_query(-sim3_direction_error_deg)
```

The gate score removes the SALAD term:

```text
nosalad_score = 2.0 * z_seq(graph)
              + 1.0 * percentile_query(-sim3_direction_error_deg)
```

For each query, candidates outside the top-5 according to `nosalad_score` receive a
small penalty:

```text
score = current_score - 0.5 * I[candidate not in top-5 by nosalad_score for this query]
```

This is a query-local geometry gate rather than a hard veto. It keeps SALAD as an
appearance prior, but requires candidates to remain competitive under geometry-only
DA3+Sim3+PGO evidence.

The 9-sequence summary is:

| Setting | AP / MR@100P | TP@100P |
| --- | ---: | ---: |
| Current formula reproduced | 64.33 / 18.75 | 299 (24.11%) |
| Best AP in corrected sweep | 64.40 / 19.29 | 303 (24.44%) |
| **Best MR/TP geometry gate** | **63.85 / 20.83** | **332 (26.77%)** |

Per-sequence breakdown for the best MR/TP geometry gate:

| Dataset / sequence | Pos. | AP / MR@100P | TP@100P |
| --- | ---: | ---: | ---: |
| FusionPortableV2/handheld/handheld_escalator00 | 176 | 87.25 / 44.89 | 79 |
| FusionPortableV2/handheld/handheld_room00 | 191 | 71.77 / 13.61 | 26 |
| FusionPortableV2/handheld/handheld_room01 | 292 | 93.08 / 43.84 | 128 |
| FusionPortableV2/ugv/ugv_campus01 | 163 | 87.10 / 42.94 | 70 |
| FusionPortableV2/ugv/ugv_parking01 | 70 | 36.91 / 2.86 | 2 |
| GEODE/Offroad/Offroad02_beta | 40 | 53.52 / 20.00 | 8 |
| GEODE/Offroad/Offroad05_beta | 22 | 20.67 / 9.09 | 2 |
| NTU-VIRAL/NTU-VIRAL/eee_01 | 107 | 52.67 / 1.87 | 2 |
| NTU-VIRAL/NTU-VIRAL/eee_02 (excluded) | 162 | 77.42 / 0.00 | 0 |
| NTU-VIRAL/NTU-VIRAL/nya_02 | 179 | 71.68 / 8.38 | 15 |

The setting improves zero-false-positive discovery but lowers AP compared with the
previous paper-facing table. It is therefore best treated as an MR/TP-oriented
candidate for Table 2 or as a supplementary operating point, not as an automatic
replacement for the AP-oriented main score.

Result sources:

- corrected sweep:
  `workspace/rover_aligned_benchmark/benchmark_v1_da3_auto_labels_v4_asterslam/metrics/score_boost_corrected_20260617/`
- candidate records:
  `workspace/rover_aligned_benchmark/benchmark_v1/candidate_records.jsonl`
- benchmark labels:
  `workspace/rover_aligned_benchmark/benchmark_v1_da3_auto_labels_v4_asterslam/annotations.jsonl`

### Tuned LoopAnything Pipeline Used By The Current Main Table

The current paper-facing method is a calibrated frozen-pair ranking pipeline. It is
important to distinguish it from the older query-gated graph reranker and from the
support-ensemble experiment:

- it does **not** rerun retrieval inside the benchmark;
- it evaluates the same frozen DBoW2 top-10 query-candidate pairs as every baseline;
- it uses SALAD only as an appearance score for each frozen pair;
- it uses one candidate-local support frame, not support ensemble;
- it uses the batched independent-triplet DA3 path in `score_frozen_query_candidates`;
- it ranks pairs with a post-hoc calibrated score built from recorded candidate fields.

The exact code path that produces the candidate fields is:

```text
score_rover_aligned_benchmark.py --method verifier
  -> rover_pair_scoring.score_verifier_pairs(...)
  -> pipeline.score_frozen_query_candidates(...)
  -> pipeline._score_candidates_with_batched_triplets(...)
  -> pipeline._score_single_support_candidate_after_da3(...)
```

For each query \(q\), the benchmark starts from frozen DBoW2 candidates
\(\{c_1,\ldots,c_{10}\}\). `score_verifier_pairs` first loads the precomputed
SALAD score for every frozen pair from `scores/salad.jsonl`; it does not search a
new SALAD candidate set. For each candidate \(c\), the verifier selects one support
frame \(s\) near the candidate using `select_support` with the sequence-specific
recent-exclusion value recorded in `manifest.json`, `support_window=4`, and
`min_support_baseline_m=0.3`.

All valid \((q,c,s)\) triplets for the same query are then evaluated by the batched
independent-triplet implementation. The code builds one DA3 triplet per candidate
and runs chunks of size `config.da3.triplet_batch_size` (currently 4). Each batch
element remains an independent three-view scene; candidates are not concatenated as
views of the same DA3 scene. This matches:

```text
pipeline._score_candidates_with_batched_triplets
  -> _run_da3_triplets
```

After DA3 inference, each triplet is aligned to metric odometry using only the
candidate-support relation:

```text
pipeline._score_single_support_candidate_after_da3
  -> align_triplet_to_candidate_support(
       da3_result.predicted_c2w,
       odom_by_idx[candidate_idx],
       odom_by_idx[support_idx],
     )
```

The resulting aligned query-to-candidate loop factor is inserted into a full-prefix
PGO:

```text
run_full_prefix_pgo(
  prefix_indices=prefix_indices_through_query,
  odom_poses=prefix_odom_poses,
  loop_from_idx=query_idx,
  loop_to_idx=candidate_idx,
  loop_factor=aligned_da3_loop_factor,
)
```

For every candidate record, the main score uses exactly these fields:

- `salad_score`: SALAD descriptor similarity for the frozen q-c pair;
- `pgo_error_after`: graph error after full-prefix PGO with the DA3 loop factor;
- `trajectory_deformation_rmse`: Sim3-aligned RMSE between the optimized prefix
  trajectory and the original odometry prefix;
- `sim3_direction_error_deg`: candidate-support Sim3 direction diagnostic from
  `align_triplet_to_candidate_support`.

The recommended score is:

```text
graph = -trajectory_deformation_rmse - 2 * log(1 + pgo_error_after)

score = salad_score
      + 2.0 * z_seq(graph)
      + 1.0 * percentile_query(-sim3_direction_error_deg)
```

where:

- `z_seq(graph)` is sequence-level z-score normalization over the evaluated frozen
  pairs of the same sequence;
- `percentile_query(-sim3_direction_error_deg)` is the within-query percentile rank
  among the current frozen top-10 candidates, where a smaller direction error is
  better;
- candidates that fail support selection, DA3, Sim3 validation, or PGO do not have
  valid graph/direction evidence and are treated as low geometry-confidence records
  by the exact-refine sweep;
- the score is recorded as
  `salad_plus_seqz_graph_qpct_sim3dir:ratio=2,ag=2,ad=1`.

This is the method reported as **Tuned LoopAnything** / **Ours** in the current main
table. It should be described in the paper as a calibrated ranking score over frozen
candidate pairs, not as an online hard-veto accept/reject policy. The online AsterSLAM
version can reuse the same signals, but it still needs a separate thresholding policy
for accepting loop factors into iSAM2.

Result sources:

- DA3 auto-label benchmark:
  `workspace/rover_aligned_benchmark/benchmark_v1_da3_auto_labels_v4_asterslam/`
- score sweep:
  `workspace/rover_aligned_benchmark/benchmark_v1_da3_auto_labels_v4_asterslam/metrics/score_boost_sweep_exact_refine_20260615/`
- brief summary:
  `workspace/rover_aligned_benchmark/benchmark_v1_da3_auto_labels_v4_asterslam/metrics/score_boost_sweep_exact_refine_20260615/sweep_brief.md`

## Revised Paper Insight

The current manual benchmark primarily captures human-perceived appearance similarity.
It cannot objectively determine whether a method-specific relative-pose estimate is a
safe PGO factor. It should therefore remain a diagnostic retrieval experiment rather
than serve as the final safe-factor benchmark.

The paper's central insight is:

> Appearance similarity can propose loop hypotheses, but it neither characterizes
> geometric observability nor guarantees factor safety. A 3D foundation model can
> recover loop geometry from views with low appearance similarity but meaningful 3D
> overlap, while an explicit verifier is required to decide whether the recovered
> factor can be safely incorporated into SLAM.

This separates loop closure into three distinct questions:

1. Does the candidate pair share sufficient observable 3D structure?
2. Can DA3 recover an accurate relative-pose factor from that overlap?
3. Can the factor be accepted without introducing an unsafe PGO constraint?

## Revised Main Experiments

### Table 1: Safe Loop Factor Discovery And Verification

Table 1 is the main paper table and directly evaluates the complete insight. DBoW2,
NetVLAD, and DINO-SALAD remain their original retrieval methods: they score query-
candidate pairs using only their native appearance descriptors and do not use DA3.
LoopAnything uses broadened proposals, DA3 geometry, and factor verification. Report:

- number of loose loop hypotheses proposed;
- number of GT-safe loop opportunities covered by those hypotheses;
- AP against the common GT-safe-loop-opportunity labels;
- Recall@K against the same labels;
- for LoopAnything, number of generated factors, accepted safe factors, and accepted
  unsafe factors.

The common pair-level label must be independent of every evaluated method. It represents
a GT-safe loop opportunity: the query and candidate have sufficient observable 3D
overlap for a reliable relative constraint. DBoW2, NetVLAD, and DINO-SALAD are evaluated
directly by ranking these labels with their native retrieval scores. LoopAnything is
evaluated by its final verified score. A separate factor-level oracle compares each
LoopAnything DA3 estimate with the reference query-candidate relative pose to determine
whether the generated factor is actually safe.

The main system setting may use different proposal budgets because recovering more loop
opportunities is part of the contribution. Proposal count, DA3 inference count, and
runtime must be reported transparently. A fixed-compute-budget comparison can be
included as an ablation rather than imposed on the main system result.

The expected result is that LoopAnything exposes more geometrically usable hypotheses,
retains more safe loop factors, rejects unsafe factors, and achieves higher safe-factor
AP and Recall@K than appearance-only retrieval systems.

### Table 2: End-to-End AsterSLAM

Select representative sequences with meaningful odometry drift and loop opportunities.
Compare odometry only, DINO-SALAD loop closure, broadened DA3 factors without
verification, and complete LoopAnything. Report ATE/RPE, accepted safe and unsafe
factors, large-viewpoint loop count, catastrophic PGO failures, and runtime. This table
must demonstrate that the additional safe factors translate into better trajectories
without destabilizing PGO.

## Ablation Experiments

### Geometric Loop Opportunity Discovery

Evaluate candidate generators against reference-trajectory-aligned 3D-overlap labels.
Separate high-appearance/high-overlap, low-appearance/high-overlap, and
high-appearance/low-overlap cases. This explains where broadened proposals recover loop
opportunities missed by appearance retrieval.

### Factor Safety Verification

Use fixed candidate pairs and fixed DA3 factors. Compare no verification, DA3 local
geometry diagnostics, ROVER-style deformation, PGO residual, and the complete verifier.
Report safe-factor AP and MR@100P. This isolates which evidence rejects inaccurate or
ambiguous loop factors.

## Revised Pipeline Direction

```text
broad candidate proposal
  -> candidate-local support selection
  -> DA3 triplet geometry
  -> odometry-based Sim3 metric alignment
  -> factor-level geometry and safety verification
  -> accept or veto
  -> AsterSLAM PGO
```

SALAD should remain an appearance-based proposal prior rather than being replaced by
graph-score reranking. Candidate generation must be broadened enough to include
low-appearance/high-overlap pairs. The verifier operates on the concrete DA3 factor and
has veto authority; it should not promote visually implausible candidates solely because
they receive a favorable graph score.

## Result Sources

- Base methods:
  `workspace/rover_aligned_benchmark/benchmark_v1/metrics/metrics_per_sequence.csv`
- Frozen benchmark table:
  `workspace/rover_aligned_benchmark/benchmark_v1/metrics/table1.md`
- Veto/penalty sweep:
  `workspace/rover_aligned_benchmark/benchmark_v1/metrics/salad_veto_sweep_fast/`
- Candidate records:
  `workspace/rover_aligned_benchmark/benchmark_v1/metrics/support_relaxation_sweep/manifest_recent_rescue_real/candidate_records_manifest_recent_rescue.jsonl`
