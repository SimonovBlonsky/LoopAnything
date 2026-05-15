# Robust Loop Verifier Offline Design

Status: active design, approved 2026-05-15.

This spec defines the first offline stage of the robust loop verifier project.
The target is an interpretable, training-free, ROVER-like loop verification
system:

```text
online-causal VPR cache -> SALAD retrieval -> DA3 metric loop factor
-> full-prefix temporary GTSAM PGO -> ROVER-style trajectory deformation score
-> AP / MR@100 precision
```

The first implementation focuses on FusionPortableV2. KITTI and GEODE are
explicitly deferred until FusionPortableV2 is stable and diagnostically useful.
AsterSLAM online integration is out of scope for this spec.

## Non-Goals And Forbidden Priors

The legacy learned-policy documents have been moved to:

- `LoopAnything/docs/superpowers/specs/legacy/`
- `LoopAnything/docs/superpowers/plans/legacy/`
- `docs/deferred/`

The robust verifier must not use learned-policy labels, feature schemas, hard
gates, score definitions, threshold rules, cache artifacts, or acceptance logic.
In particular, do not use names or concepts such as `safe_loop_factor_v1`,
`x_geom`, learned-policy feature records, learned-policy labels, or learned
policy hard gates to define ground truth, verifier validity, scores, or
evaluation.

The only allowed use of legacy learned-policy notes is to preserve known DA3
implementation bugfixes:

- DA3 `prediction.extrinsics` are native world-to-camera (`w2c`). Offline code
  must invert them to camera-to-world (`c2w`) before Sim3 alignment, matching
  AsterSLAM's runtime DA3 pose node behavior.
- DA3 reference-view strategy must be `first` for stable
  `[query, candidate, support]` slot ordering. Do not rely on a default strategy
  that can internally reorder views.
- `process_res=112` is smoke-test only. Real geometry, Sim3 alignment, and PGO
  scoring use normal DA3 runtime resolution, default `process_res=504`.
- Support selection should use the support nearest to the candidate keyframe
  index among frames that pass validity and baseline filters, instead of sorting
  by maximum candidate-support baseline.

No other legacy learned-policy content may be consulted as a design source.

## Stage 1 Scope

Stage 1 implements an offline DA3-ROVER baseline. It is not a learned policy and
does not train a model. It does not learn or manually choose a score acceptance
threshold. Each candidate emits a continuous score, and evaluation computes AP
and maximum recall at 100 percent precision from the full precision-recall curve.

Stage 1 includes:

- FusionPortableV2 preprocess into an online-causal VPR cache.
- SALAD historical retrieval with `top_k=10` as the main setting.
- `top_k=5` and `top_k=20` retrieval ablations.
- DA3 triplet pose estimation for `(query, candidate, support)`.
- Candidate-support Sim3 metric alignment to convert DA3's unscaled pose into a
  metric query-candidate loop factor.
- Full-prefix temporary GTSAM pose graph optimization for each candidate.
- ROVER-style trajectory deformation score from Sim3-aligned translation RMSE
  between original and optimized prefix trajectories.
- Candidate-level AP and MR@100 precision.
- Per-candidate records, PR curve data, triplet visualizations, and trajectory
  before/after visualizations.

Stage 1 excludes:

- AsterSLAM online runtime integration.
- Shadow-mode or hard-gate runtime verifier logic.
- KITTI and GEODE adapters in the first implementation pass.
- Traditional GV / ORB / SIFT / LoFTR / RANSAC baselines.
- Learned models, learned thresholds, or learned uncertainty calibration.

## Dataset Preprocess

Each dataset/platform gets an explicit YAML config. For the first pass, only
FusionPortableV2 is required.

Required config fields:

- `dataset_name`
- `platform`
- `input_root`
- `output_root`
- `positive_radius_m`
- `recent_exclusion_keyframes`
- `retrieval_top_k_main`
- `retrieval_top_k_ablations`
- `support_window`
- `support_count`
- `min_support_baseline_m`
- `pgo_noise`
- DA3 runtime settings, including `process_res` and `ref_view_strategy`

`positive_radius_m` and `recent_exclusion_keyframes` must be explicit. There are
no hidden defaults for ground-truth label generation.

The preprocess output is an online-causal sequence cache, not a fixed
SALAD-style `ref/query` split. It preserves keyframe order and writes:

```text
<cache_root>/<dataset>/<platform>/<sequence>/
  manifest.json
  keyframes.jsonl
  positives.jsonl
  images/
```

Images are stored under `images/` using symlinks by default. If symlink creation
fails, the preprocessor copies the image. Manifests store relative cache paths so
the cache can be moved and evaluated independently.

For each query keyframe `q`, positives are generated only from data-source GT or
high-precision reference trajectories:

```text
c.idx < q.idx - recent_exclusion_keyframes
norm(p_gt(q) - p_gt(c)) <= positive_radius_m
```

Ground truth uses position only. It must not use viewpoint, yaw, rotation,
DA3, Sim3, odometry consistency, retrieval score, verifier score, or any
learned-policy output. The rationale is that a true loop is defined by revisiting
the same physical place; viewpoint is a method difficulty factor, not the GT
definition.

Sequences without usable GT are skipped. Private dog data is out of scope for
this spec because it will be manually annotated separately.

## Retrieval

Retrieval uses SALAD descriptors in an online-causal historical database. For
each query, the candidate database contains only frames that are older than the
query and outside the configured recent exclusion.

Main setting:

```text
retrieval_top_k = 10
```

Ablations:

```text
retrieval_top_k in {5, 20}
```

The runtime-facing main table uses `top10` because AsterSLAM runtime retrieval
cannot practically use very large top-k. `top20` is only an ablation and
retrieval-ceiling diagnostic.

`SALAD score only` is a required baseline: all candidate pairs are ranked by the
retrieval score alone and evaluated with the same GT labels.

## Support Selection

Stage 1 uses one support frame by default:

```text
support_count = 1
```

`support_window`, `support_count`, and `min_support_baseline_m` are system
configuration parameters. They are defined per dataset/platform config and may
be swept as engineering parameters. They are not learned thresholds and do not
define GT.

Support candidates must satisfy:

- They are historical relative to the query.
- They are not the candidate itself.
- They lie within the configured candidate-centered keyframe-index window.
- They have a usable image and pose.
- Their candidate-support metric baseline is at least
  `min_support_baseline_m`.
- They satisfy the query-relative recent exclusion:

```text
abs(query.idx - support.idx) > recent_exclusion_keyframes
```

After filtering, supports are sorted by nearest candidate keyframe index:

```text
abs(support.idx - candidate.idx) ascending
```

Ties are broken by smaller support index. This rule is based on prior debugging
showing that the nearest candidate-neighborhood support tends to preserve visual
overlap better than maximizing baseline.

## DA3 Metric Loop Factor

For each `(query, candidate, support)` triplet:

1. Run DA3 with stable view order `[query, candidate, support]`.
2. Use `ref_view_strategy="first"`.
3. Use `process_res=504` by default.
4. Treat DA3 native extrinsics as `w2c` and invert to `c2w`.
5. Align DA3 triplet poses to metric scale using the candidate-support odometry
   prior.
6. Extract the aligned query-candidate relative pose as the metric loop factor.

DA3 itself has no metric translation scale. Sim3 scale magnitude is therefore
not a hard rejection criterion. Invalid alignment is limited to undefined
computation, such as non-finite poses, non-finite scale, non-positive scale, or
zero/near-zero DA3 candidate-support baseline.

The weak DA3/Sim3 baseline ranks candidates using only factor self-consistency
diagnostics, primarily candidate-support alignment residual and direction error.
It does not use query-candidate odometry residual, because false loops should not
be judged by forcing query-candidate odometry to agree with the retrieved pair.

## Full-Prefix Temporary PGO

For each query-candidate pair, Stage 1 builds a temporary full-prefix pose graph
up to the query keyframe:

- Add a prior on the first keyframe in the prefix.
- Add odometry `BetweenFactor<Pose3>` edges between consecutive keyframes in the
  prefix.
- Add the DA3-derived query-candidate loop `BetweenFactor<Pose3>`.
- Optimize the temporary graph with Python GTSAM 4.1.1.
- Never mutate the source cache, AsterSLAM logs, or any persistent runtime
  backend state.

This is intentionally full-prefix rather than local-window. It matches the
ROVER offline formulation and avoids prematurely designing around online runtime
constraints. AsterSLAM runtime integration will be designed only after Stage 1
experimental results and failure modes are understood.

PGO noise parameters are dataset/platform system configuration, similar to SLAM
benchmark config files. They are not network training, and they are not verifier
score thresholds. The offline pipeline may sweep support and PGO parameters to
select dataset/platform configs. Selected configs must be saved with the run
artifacts and later can be ported into AsterSLAM if needed.

This config selection is a system-configuration process, not a learned model and
not a learned acceptance threshold. The selected dataset/platform configs must
be explicit files in the experiment artifacts and in any released code. The
paper can report the selected system configuration results directly, following
normal SLAM benchmark practice where each dataset/platform has an explicit
configuration file.

The causality direction is:

```text
offline robust-verifier experiments -> selected system config -> future AsterSLAM port
```

It is not:

```text
current AsterSLAM backend defaults -> offline verifier definition
```

## ROVER-Style Score

Let `X` be the original prefix trajectory and `X*` be the trajectory after
temporary PGO with the candidate loop factor. Let `P` and `P*` be their
translation point sets at matching keyframe timestamps.

Stage 1 computes the ROVER-style score:

1. Find the best Sim3 alignment from `P*` to `P`.
2. Compute translation RMSE after alignment.
3. Convert RMSE to the canonical evaluation score with
   `score_rover = -trajectory_deformation_rmse`.

Rotation residuals are not part of the Stage 1 main score. They may be logged
only as diagnostics if useful, but they must not affect the main ROVER baseline.

## Methods In Stage 1 Table

The first implementation reports:

- `SALAD score only`
- `SALAD + DA3/Sim3 self-consistency score`
- `SALAD + DA3-ROVER full-prefix trajectory score`

Traditional geometric-verification baselines are handled outside this spec.

## Metrics

Evaluation is candidate-level. For every retrieved candidate, use the precomputed
GT positive list to assign:

```text
label(q, c) = c in positives[q]
```

Metrics:

- `AP`: average precision over canonical candidate scores.
- `MR@100P`: maximum recall at 100 percent precision.

Canonical evaluation scores are always larger-is-better:

- `SALAD score only` uses the SALAD similarity score directly.
- `SALAD + DA3/Sim3 self-consistency score` uses the negative self-consistency
  residual, so lower residual ranks higher.
- `SALAD + DA3-ROVER full-prefix trajectory score` uses
  `-trajectory_deformation_rmse`.

No fixed acceptance threshold is part of Stage 1. Thresholds are swept only by
the metric computation to produce the precision-recall curve.

Report metrics for:

- Main `top10` setting.
- `top5` retrieval ablation.
- `top20` retrieval ablation.
- Dataset/platform selected configs.

## Artifacts

Each run writes:

```text
<run_root>/
  run_config.yaml
  run_manifest.json
  sequence_summaries.jsonl
  candidate_records.jsonl
  metrics.json
  metrics.md
  pr_curves/
  visual_records/
  trajectory_plots/
```

`candidate_records.jsonl` includes at least:

- dataset/platform/sequence
- query/candidate/support indices and image paths
- SALAD rank and score
- DA3 runtime settings
- DA3/Sim3 validity diagnostics
- metric loop factor
- PGO convergence status
- trajectory deformation RMSE
- GT label
- method scores

`visual_records/` stores `(query, candidate, support)` triplets for manual false
positive and false negative inspection.

`trajectory_plots/` stores original-vs-optimized prefix trajectory plots for
sampled high-score true positives, high-score false positives, and false
negatives near the decision frontier.

## Error Handling

Candidates with missing images, missing poses, failed DA3 inference, invalid
non-finite transforms, undefined Sim3 alignment, or failed GTSAM optimization are
recorded with a structured failure reason. They remain visible in summaries.

Validity failures are not learned-policy gates. They are computation failures
that prevent a candidate from producing the normal method score.

Every retrieved candidate must still receive an evaluation score for every
reported method. If a method cannot compute its normal score for a candidate,
that method assigns the candidate its worst finite score for that run, after all
normally scored candidates. Ties among failed candidates are stable-sorted by
query index, candidate index, and failure reason. Failed candidates must not be
excluded from AP or MR@100P, because exclusion would act as an implicit hard
gate and inflate metrics.

## Implementation Order

1. Implement FusionPortableV2 online-causal VPR cache preprocess.
2. Implement SALAD historical retrieval over the cache.
3. Implement support selection with nearest-candidate ordering.
4. Implement DA3 triplet runner with corrected pose convention and stable view
   ordering.
5. Implement candidate-support Sim3 metric alignment and loop factor extraction.
6. Implement full-prefix Python GTSAM temporary PGO.
7. Implement ROVER-style trajectory deformation score.
8. Implement AP/MR evaluation and artifacts.
9. Run FusionPortableV2 diagnostics, visualizations, and failure analysis.

KITTI and GEODE adapters are designed after FusionPortableV2 is working and
diagnostically understood.

## Acceptance Criteria

The Stage 1 implementation is acceptable when it can:

- Preprocess FusionPortableV2 into an online-causal VPR cache with explicit
  positive lists.
- Generate GT positive lists only from dataset GT or high-precision reference
  trajectory positions, `positive_radius_m`, and
  `recent_exclusion_keyframes`; do not use viewpoint, yaw, rotation, DA3, Sim3,
  odometry consistency, retrieval score, verifier score, or learned-policy
  artifacts for GT labels.
- Run SALAD historical retrieval with top10 main setting and top5/top20
  ablations.
- Run DA3 triplets using `w2c -> c2w`, `ref_view_strategy="first"`, and
  non-smoke DA3 resolution.
- Select supports by nearest candidate keyframe index after validity and
  baseline filtering.
- Convert DA3 triplets into metric query-candidate loop factors using
  candidate-support Sim3 alignment.
- Run full-prefix temporary GTSAM PGO per candidate.
- Score candidates with ROVER-style Sim3-aligned translation RMSE.
- Report AP and MR@100 precision.
- Assign a larger-is-better canonical score to every retrieved candidate for
  every reported method, with computation failures ranked after all normal
  scores rather than excluded.
- Save candidate records, PR curve data, triplet visualizations, and trajectory
  plots.
- Avoid all legacy learned-policy labels, features, hard gates, score rules, and
  cache artifacts.
- Avoid AsterSLAM online runtime implementation.
