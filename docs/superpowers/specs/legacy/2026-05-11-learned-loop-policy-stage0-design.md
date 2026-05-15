# LEGACY: Learned Loop Policy Stage0 Design

Status: legacy as of 2026-05-15.

This document is retained only as a historical record. The learned-policy
Stage0 design is not considered converged and must not be used as the basis for
the active robust loop verifier.

Do not use this document's labels, hard gates, feature schema, thresholds,
support rules, score definitions, or data contracts as priors for robust loop
verifier design or implementation. In particular, do not use
`safe_loop_factor_v1`, `x_geom`, or any learned-policy cache outputs to define
ground truth or verifier acceptance. The learned-policy direction is deferred
until an interpretable training-free robust loop verifier is completed and
achieves strong experimental results.

---

# Causal Loop Policy Dataset Builder Design

Date: 2026-05-11

## Purpose

The causal loop-policy dataset builder creates the offline data contract for
the learned loop verification policy. It consumes AsterSLAM's loop-disabled
dataset export and produces runtime-consistent candidate, support, geometry,
feature, and label caches for downstream training.

The dataset builder is not the learning contribution. Its job is to make later
learning experiments reproducible, auditable, and aligned with the online SLAM
system.

## Package Boundary

All new LoopAnything code for this work lives in:

```text
LoopAnything/src/loop_policy/
```

The implementation must not modify `LoopAnything/src/depth_anything_3`. The
`loop_policy` package may import DA3 and existing LoopAnything utilities through
their public Python interfaces, but it should keep learned-policy dataset logic
outside the DA3 source tree.

Packaging may update `LoopAnything/pyproject.toml` so editable installs include
`src/loop_policy`.

## Naming Rule

Code names must describe the function they implement, not the roadmap phase or
plan step that introduced them. Do not add Python symbols, module names, CLI
commands, output artifact names, schema versions, log messages, or comments
named after vague labels such as `stage0`, `stage_0`, `stage 0`, `stage1`,
`task1`, or `task 2`.

Use functional names instead, for example `LoopPolicyDatasetConfig`,
`dataset_builder.py`, `sequence_summary.json`, and `dataset_manifest.json`.
Research notes may mention phases in prose when discussing the paper roadmap,
but implementation-facing names must remain self-explanatory to collaborators.

## Inputs

Each sequence input is an AsterSLAM-derived raw cache:

```text
<dataset-root>/<platform>/<sequence>/raw/
  sequence_meta.json
  keyframes.jsonl
  keyframes_with_images.jsonl
  trajectory.txt
  trajectory_indices.txt
  trajectory_keyframes.txt
  trajectory_keyframe_indices.txt
  keyframe_images/
```

Optional GT inputs:

```text
/data/datasets/FusionPortable/<platform>/<sequence>/<sequence>.txt
```

`sequence_meta.json` provides calibration, sequence name, image topic metadata,
loop-closure status, and `T_camera_lidar`. `keyframes_with_images.jsonl` is the
source of query/candidate/support image paths.

## Main Protocol: Causal Online Cache

The main causal loop-policy dataset cache must simulate the online SLAM
information boundary.
For a query keyframe `q`, the retrieval database, support set, and verifier
inputs may only use keyframes that already exist before `q`.

Descriptor extraction may be precomputed for the full sequence for efficiency.
That is allowed because descriptor extraction is an offline cache of image
features. The retrieval search itself must still be causal: the index searched
for query `q` contains only historical keyframes.

Retrieval database:

```text
retrieval_db(q) =
    keyframes where timestamp < q.timestamp
    AND abs(keyframe_idx - q.idx) > exclude_recent_keyframes
    AND image_path is not null
```

Support database for candidate `c`:

```text
support_db(q, c) =
    keyframes where timestamp < q.timestamp
    AND support.idx != c.idx
    AND support.idx in [c.idx - support_window, c.idx + support_window]
    AND abs(q.idx - support.idx) > exclude_recent_keyframes
    AND image_path is not null
    AND camera_center_baseline(c, support) >= min_support_baseline
```

Support selection sorts valid supports by candidate-support camera-center
baseline descending, tie-breaking by keyframe index ascending, then keeps
`support_count`. This mirrors AsterSLAM's `Da3LoopVerifier` behavior.

Every output record must include enough audit fields to prove the cache is
causal:

```text
causal: true
query_idx
query_timestamp
database_max_idx
database_max_timestamp
retrieval_db_size
support_snapshot_max_idx
support_snapshot_max_timestamp
candidate_source
candidate_idx
candidate_timestamp
selected_support_indices
selected_support_timestamps
```

If `candidate_source` is `retrieval_topk`, the candidate must come from the
causal retrieval database. If later augmentation adds `gt_backfill`, the record
must remain causal and must not be used for runtime top-K policy evaluation.

## Outputs

The dataset builder writes one cache directory per sequence:

```text
<loop-policy-cache-root>/<platform>/<sequence>/
  sequence_index.json
  descriptors.npz
  retrieval_topk.jsonl
  support_selection.jsonl
  candidate_features.jsonl
  sequence_summary.json
```

Root-level summaries:

```text
<loop-policy-cache-root>/
  dataset_manifest.json
  sequence_summaries.jsonl
```

`sequence_index.json` records sequence metadata, thresholds, calibration,
keyframe counts, image counts, GT availability, and protocol settings.

`descriptors.npz` stores one descriptor per keyframe with an image. It also
stores `keyframe_idx`, `timestamp`, and descriptor normalization metadata.

`retrieval_topk.jsonl` is query-level and causal. Each line contains the query
keyframe, causal database size, and ranked historical candidates with keyframe
indices, timestamps, and SALAD scores.

`support_selection.jsonl` records candidate-level support decisions, including
selected support indices, support timestamps, baselines, and rejection reasons
such as `no_valid_support`.

`candidate_features.jsonl` is the main training source. Each line is one
query-candidate record with retrieval scores, candidate timestamp, support
metadata, DA3/Sim3 metrics, `x_geom`, labels, and audit labels.

## Feature Contract

The first policy feature vector is `x_geom[32]`, aligned with the learned policy
draft:

```text
01 rank_norm
02 salad_score_qc
03 salad_score_qs
04 salad_score_cs
05 salad_score_qc - top1_score
06 salad_score_qc - topK_score
07 da3_rot_qc_deg
08 da3_trans_qc_norm
09 da3_rot_cs_deg
10 da3_trans_cs_norm
11 da3_rot_qs_deg
12 da3_trans_qs_norm
13 odom_rot_qc_deg
14 odom_trans_qc_norm
15 odom_rot_cs_deg
16 odom_trans_cs_norm
17 support_baseline
18 sim3_scale
19 abs_log_sim3_scale
20 support_align_rmse
21 direction_error_deg
22 aligned_vs_odom_rot_residual_deg
23 aligned_vs_odom_trans_residual_norm
24 aligned_loop_rot_deg
25 aligned_loop_trans_norm
26 q_depth_conf_median
27 c_depth_conf_median
28 s_depth_conf_median
29 q_valid_depth_ratio
30 c_valid_depth_ratio
31 s_valid_depth_ratio
32 min_depth_conf_median
```

The main `x_geom[32]` contract is defined for `support_count=1`, using the
primary selected support `selected_supports[0]`. If an offline ablation uses
`support_count > 1`, per-support values must be stored in separate array fields
such as `support_baselines`, `salad_scores_qs`, `salad_scores_cs`,
`da3_rot_cs_deg_by_support`, and `support_depth_conf_medians`. The first 32
scalar fields must continue to use the primary support unless the ablation
declares a different schema version. Multi-support mean/min/max aggregates must
not silently replace the main feature meanings.

If a deterministic precondition fails before DA3/Sim3, the candidate may be
written for analysis with `precondition_valid=false`, a `negative_reason`, and
missing DA3-dependent fields. Such records are not valid network inputs until a
training loader explicitly filters or pads them.

## DA3 And Sim3 Semantics

Candidate-local DA3 groups are formed as:

```text
[query, candidate, support_0, support_1, ...]
```

The DA3 batch shape is:

```text
[G, S, 3, H, W]
```

where `G` is the number of query-candidate groups in the batch and `S` is the
number of views in each group. Candidates must not be placed in the same DA3
view group, because cross-view attention would contaminate candidate-local
geometry.

The Sim3 prior postprocess must match AsterSLAM's
`alignDa3PosesWithCandidateSupportPrior`:

- use predicted DA3 camera poses for query, candidate, and supports;
- use odometry-derived camera poses for candidate and supports as the prior;
- estimate scale from candidate-support baseline;
- align candidate rotation to the odometry candidate rotation;
- reject invalid scale, excessive support alignment RMSE, and excessive prior
  direction error;
- output the aligned query-candidate loop factor from the same DA3+Sim3 pass.

The accepted loop factor must reuse the already-computed DA3+Sim3 result. Stage
0 must not model a second DA3 inference after reranking.

## Labels

The main training label is:

```text
safe_loop_factor_v1 =
    precondition_valid
    AND sim3_quality_good
    AND odom_consistent_loose
```

where:

```text
sim3_quality_good =
    abs_log_sim3_scale <= abs_log_sim3_scale_thr
    AND support_align_rmse <= support_align_rmse_thr
    AND direction_error_deg <= direction_error_thr

odom_consistent_loose =
    aligned_vs_odom_rot_residual_deg <= loose_rot_thr
    AND aligned_vs_odom_trans_residual_norm <= loose_trans_thr
```

GT audit labels are written when GT is available:

```text
place_true_gt
factor_good_gt
valid_loop_gt = place_true_gt AND factor_good_gt
```

GT audit labels must not be used to construct causal retrieval candidates in
the main `retrieval_topk` source. They are used for evaluation, error analysis,
and optional marked backfill.

## Causal-Preserving Augmentation

The dataset builder should first generate the strict causal main cache, then
report whether the data volume is sufficient. If the main cache is sparse, the
following augmentations are allowed because they preserve the online
information boundary:

- cache a larger causal retrieval pool, for example `retrieval_pool_size=50` or
  `100`, while preserving the runtime top-K rank;
- sample more query frames around loop-rich temporal windows;
- mine hard negatives from high-score historical candidates that fail Sim3,
  odometry consistency, or GT audit checks;
- write `sample_weight` and `class_balance_group` for later training samplers
  instead of duplicating records in the cache;
- optionally add historical GT-positive candidates outside retrieval top-K with
  `candidate_source="gt_backfill"` for offline factor-quality training only.

The following are not allowed in the main cache:

- future retrieval candidates;
- future support frames;
- full-sequence all-to-all FAISS retrieval for main top-K records;
- reverse-time samples presented as online data;
- non-causal support that improves Sim3 alignment.

## Module Layout

```text
LoopAnything/src/loop_policy/
  __init__.py
  schema.py          # dataclasses and JSON/NPZ field names
  io.py              # AsterSLAM raw cache and GT trajectory readers
  geometry.py        # SE3, T_camera_lidar, pose residual helpers
  support.py         # AsterSLAM-compatible causal support selection
  retrieval.py       # descriptor extraction and causal top-K retrieval
  da3_runner.py      # candidate-local batched DA3 inference
  sim3_prior.py      # Python port of AsterSLAM Sim3 prior alignment
  labels.py          # safe_loop_factor_v1 and GT audit labels
  dataset_builder.py # CLI orchestration and resume logic
```

The CLI entrypoint should be:

```bash
python -m loop_policy.dataset_builder \
  --dataset-root /data/datasets/FusionPortable/fusionportable_loop_dataset \
  --gt-root /data/datasets/FusionPortable \
  --output-root /data/datasets/FusionPortable/loop_policy_dataset_cache \
  --sequences handheld_room00 handheld_room01 \
  --retrieval-pool-size 50 \
  --runtime-top-k 4 \
  --causal
```

`--causal` is the default and should be recorded in every manifest. A future
non-causal diagnostic mode must require an explicit flag and write to a
separate output root.

## Error Handling

The dataset builder should continue at candidate granularity where possible:

- missing query image: skip query and count it in `sequence_summary.json`;
- insufficient historical database: skip query or write an empty retrieval
  record, depending on `--write-empty-queries`;
- missing candidate image: reject candidate before DA3;
- no valid support: write candidate record with `negative_reason=no_valid_support`;
- DA3 failure or non-finite output: write `negative_reason=da3_failed`;
- Sim3 rejection: write the exact Sim3 rejection reason.

Sequence-level failures, such as missing `sequence_meta.json` or unreadable
trajectory files, should fail the sequence and write an error record to
`sequence_summaries.jsonl`.

## Validation

Unit tests should cover:

- TUM trajectory and AsterSLAM raw cache parsing;
- causal retrieval database truncation for several query indices;
- support selection parity with AsterSLAM's documented selection rules;
- Sim3 prior alignment on synthetic poses, including scale and rejection cases;
- label generation from synthetic metrics;
- output schema round-trip for JSONL and NPZ records.

A smoke test should run on a small exported sequence, for example
`handheld_room01`, with a small query limit and `retrieval_pool_size=10`. The
smoke summary must report:

```text
sequence
keyframe_count
image_keyframe_count
query_count
retrieval_candidate_count
valid_support_count
da3_success_count
safe_loop_factor_positive_count
negative_reason histogram
causal leakage audit passed
```

The leakage audit passes only if every retrieval candidate and every selected
support has `timestamp < query_timestamp`.

## Non-Goals

The dataset builder does not train the policy network, implement downstream
models, integrate with AsterSLAM runtime, or define final paper metrics. It
produces the offline cache consumed by those later experiments.
