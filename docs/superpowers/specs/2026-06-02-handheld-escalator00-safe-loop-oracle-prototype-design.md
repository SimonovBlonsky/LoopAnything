# Handheld Escalator00 Safe-Loop Oracle Prototype Design

**Date:** 2026-06-02
**Status:** Approved design for a small-scale offline prototype

## 1. Objective

Build an offline, auditable labeling prototype for
`FusionPortableV2/handheld/handheld_escalator00`.

The prototype evaluates whether each fixed retrieval candidate can yield a
LiDAR-derived loop factor that is safe to add to a full-sequence pose graph. It
must generate a visual review record for every query-candidate pair and support
manual assignment of:

- `positive`
- `negative`
- `ambiguous`

The prototype is for ground-truth construction and audit. It is not an online
verifier and must not use the proposed DA3-based method to define labels.

## 2. Scope

### 2.1 Included

- One sequence:
  `FusionPortableV2/handheld/handheld_escalator00`
- Fixed candidate set: causal SALAD top-10 retrieval candidates.
- One-time extraction of keyframe-aligned LiDAR scans from:
  `/data/datasets/FusionPortable/handheld/handheld_escalator00/handheld_escalator00.bag`
- LiDAR topic:
  `/os_cloud_node/points`
- Local-submap NanoGICP registration.
- Single-factor full-sequence PGO counterfactual analysis.
- Static HTML index, per-pair HTML pages, plots, point-cloud visualizations,
  machine-readable records, and editable manual annotation files.

### 2.2 Excluded

- DA3 inference, DA3 depth, DA3 pose, DA3 confidence, and Sim3 alignment.
- Proposed causal-verifier scores, residual-aware fusion scores, deformation
  scores, and graph-gate outputs.
- Learned loop-policy artifacts.
- Automatic final labels derived from a weighted score.
- Multi-sequence execution.
- Online AsterSLAM integration.
- Stateful annotation web application.

SALAD may only define the fixed candidate set. The labeling pipeline must not
read SALAD similarity scores when producing evidence or labels.

## 3. Data Sources and Provenance

### 3.1 Reference Trajectory

Use the loop-disabled AsterSLAM keyframe trajectory:

```text
/data/datasets/FusionPortable/fusionportable_loop_dataset/
  handheld/handheld_escalator00/raw/trajectory_keyframes.txt
```

This trajectory is used as a **reference trajectory**, not as sensor ground
truth. It has complete SE(3) poses and exact keyframe correspondence. The
existing export reports translation ATE RMSE `0.091073m` after SE(3) alignment.

The original FusionPortable handheld trajectory cannot provide orientation
ground truth because its quaternion columns are identity rotations.

Every output manifest must state:

```text
reference_pose_source=aster_slam_loop_disabled_trajectory_keyframes
reference_pose_is_sensor_gt=false
```

### 3.2 LiDAR Scans

Read `sensor_msgs/PointCloud2` messages from:

```text
/os_cloud_node/points
```

The source rosbag contains 2,480 LiDAR scans over approximately 247 seconds.
Match each exported keyframe timestamp to its nearest LiDAR scan timestamp.
Record the synchronization delta and reject extraction if it exceeds a
configurable tolerance.

The scan extractor runs under the system ROS Python environment. Downstream
analysis reads cached files and runs independently under the `da3` conda
environment.

## 4. Coordinate Frames and Local Submaps

Store each extracted LiDAR scan in its native LiDAR frame.

For a center keyframe `i`, construct a local submap in the center-frame
coordinate system:

```math
M_i = \bigcup_{j \in \mathcal{N}(i)} T_{ij}^{ref} L_j
```

where:

- `L_j` is the native scan for keyframe `j`.
- `T_ij^ref` is the short-range relative transform from the reference
  keyframe trajectory.
- `N(i)` is a configurable local neighborhood around `i`.

Using center-relative local coordinates prevents the two submaps from being
pre-aligned in a shared global frame before registration. Voxel downsampling is
applied after submap construction.

## 5. LiDAR Registration

For each fixed causal SALAD top-10 candidate pair `(q, c)`:

1. Construct `M_q` and `M_c`.
2. Compute the reference relative transform:

   ```math
   T_{cq}^{ref} = (T_{wc}^{ref})^{-1} T_{wq}^{ref}
   ```

3. Run NanoGICP from `T_cq^ref` and configurable perturbations around it.
4. Run registration in both directions.
5. Record:
   - convergence
   - inlier RMSE / NanoGICP fitness
   - symmetric overlap ratio
   - correction relative to `T_cq^ref`
   - multi-start endpoint dispersion
   - bidirectional inverse-consistency error
   - source and target point counts

The reference transform is an initialization and audit reference. NanoGICP
must return an optimized transform.

## 6. Single-Factor Full-Sequence PGO

For each candidate pair independently:

1. Build an odometry-only full-sequence graph from the loop-disabled AsterSLAM
   keyframe trajectory.
2. Add exactly one LiDAR NanoGICP loop factor for `(q, c)`.
3. Run full-sequence PGO with fixed factor covariance and no robust kernel in
   the primary audit.
4. Compare the optimized trajectory with the odometry-only trajectory and the
   available external translation reference.
5. Record:
   - optimization convergence
   - loop-factor residual before and after optimization
   - odometry-edge strain distribution
   - trajectory deformation profile
   - maximum local deformation
   - translation ATE before and after optimization
   - ATE change

Only one candidate factor may be added per PGO run. This isolates the
counterfactual effect of that factor.

Future trajectory information is allowed because this is an offline oracle
audit. The online verifier remains causal and must not access these outputs at
runtime.

## 7. Manual Annotation Protocol

The prototype does not automatically emit final binary labels. It emits
evidence records for complete manual review.

### 7.1 Label Semantics

Assign `positive` when:

- the query and candidate local submaps exhibit broad static geometric overlap;
- multi-start and bidirectional NanoGICP converge to a stable metric relative
  pose;
- adding the single LiDAR factor results in a plausible full-sequence
  correction without obvious folding, discontinuity, or concentrated
  odometry-edge strain.

Assign `negative` when:

- the local submaps lack meaningful overlap;
- registration aligns repetitive but distinct structures;
- NanoGICP is unstable or inconsistent;
- the factor causes implausible full-sequence deformation.

Assign `ambiguous` when:

- multiple geometrically plausible registrations remain;
- the scene is observability-degenerate;
- registration and PGO evidence conflict;
- visual and geometric evidence are insufficient for a defensible decision.

`ambiguous` records are excluded from AP and MR@100P and reported separately.

### 7.2 Bias Control

The manual review page must not display:

- DA3 outputs
- proposed verifier outputs
- SALAD similarity scores
- previous distance-threshold labels

The review order must support deterministic shuffling by seed. Labels must be
stored separately from generated evidence so annotations can be reset or
reviewed independently.

## 8. Visual Audit Artifacts

Generate one static review directory per query-candidate pair:

```text
q000101_c000029/
  record.json
  query.png
  candidate.png
  lidar_overlay_top.png
  lidar_overlay_side.png
  trajectory_before_after.png
  odom_strain_profile.png
  pgo_summary.json
  review.html
```

Generate a sequence-level `index.html` with:

- previous / next navigation
- query and candidate indices
- retrieval rank
- annotation status
- links to pair review pages

Each pair page displays:

- query and candidate images
- local-submap overlay before and after registration
- trajectory before and after single-factor PGO
- odometry-edge strain profile
- reference and registration diagnostics
- `positive`, `negative`, `ambiguous`, and `skip` annotation controls

The initial implementation uses browser `localStorage` for annotation state and
provides an explicit JSONL export action. It does not need a server-side
database. The exported JSONL file is the reviewer-approved annotation artifact;
generated evidence remains read-only.

## 9. Output Layout

Use a dedicated ignored workspace directory:

```text
LoopAnything/workspace/safe_loop_oracle/
  handheld_escalator00/
    manifest.json
    extracted_scans/
    scan_manifest.jsonl
    candidates.jsonl
    pair_records.jsonl
    annotations.jsonl
    index.html
    pairs/
```

`annotations.jsonl` is the only reviewer-approved artifact. It is exported from
the static review UI after annotation. Generated evidence is reproducible and
must not be manually modified.

## 10. Prototype Validation

The prototype is accepted for broader rollout only after:

1. Keyframe scan extraction succeeds for all available handheld_escalator00
   keyframes within the configured timestamp tolerance.
2. Candidate count matches causal SALAD top-10 retrieval output.
3. Every pair has a complete `record.json`, visual page, and required plots.
4. A targeted audit covers:
   - visually obvious true loops
   - visually obvious false loops
   - repetitive escalator boundary cases
   - cases that violated the previous distance-only labels
5. Human review confirms that the evidence is sufficient to assign
   `positive`, `negative`, or `ambiguous` without viewing any proposed-method
   output.

## 11. Contamination Boundary

The oracle prototype is an independent dataset-construction tool. It must not
import from or consume artifacts produced by:

```text
LoopAnything/src/loop_policy
LoopAnything/tests/loop_policy
DA3 inference outputs
robust_loop_verifier candidate score sweeps
causal verifier predictions
```

After the annotations are frozen, a separate evaluation pipeline may compare
DA3 and verifier outputs against the frozen labels.
