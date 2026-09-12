# VINS-Fusion SLAM Accuracy Experiment Design

**Date:** 2026-06-19
**Status:** Draft design for implementation planning

## 1. Workspace Boundary

This design document is stored under `LoopAnything/docs/` because it belongs to
the LoopAnything paper experiment design. It must not be interpreted as permission
to modify LoopAnything code, configurations, benchmark data, or result tables.

The implementation workspace for this experiment is strictly:

```text
/home/chenguyuan/code/NeurIPS26/VINS_Fusion_ws/src/VINS-Fusion
```

The active VINS-Fusion branch is:

```text
loopClosure
```

All experiment scripts, launch changes, C++ integration code, logging, and VINS
runtime outputs for this phase should be added only inside the VINS-Fusion
workspace above, unless explicitly approved otherwise.

## 2. Objective

Design the VINS-Fusion part of the LoopAnything SLAM Accuracy experiment.

The experiment asks whether replacing VINS-Fusion's original loop candidate
retrieval with stronger retrieval or retrieval-plus-verification improves final
SLAM trajectory accuracy while keeping the rest of the SLAM system fixed.

The controlled variable is the loop-closure front end:

- original VINS-Fusion DBoW2 retrieval;
- NetVLAD retrieval;
- SALAD retrieval;
- LoopAnything retrieval plus factor-safety verification.

The controlled back end is VINS-Fusion's original loop-closure backend:

- same VIO frontend;
- same keyframe stream;
- same PnP geometric verification for retrieval-only baselines;
- same VINS pose graph optimization;
- same trajectory export and evo evaluation pipeline.

## 3. Dataset Scope

Use only the nine NTU-VIRAL sequences that did not show the RTP degradation:

```text
eee_01
eee_02
eee_03
nya_01
nya_02
nya_03
sbs_01
sbs_02
sbs_03
```

Do not include:

```text
rtp_01
rtp_02
rtp_03
```

The RTP sequences are excluded because the current VINS-Fusion baseline exhibits
severe odometry degradation there, making loop-closure comparison dominated by
front-end failure rather than loop-closure quality.

The data roots are:

```text
bags: /data/datasets/NTU-VIRAL/data
raw ground truth: /data/datasets/NTU-VIRAL/groundtruth
TUM ground truth: /data/datasets/NTU-VIRAL/processed_gt_tum
```

For this VINS-Fusion SLAM Accuracy experiment, the TUM ground-truth converter
must preserve the raw CSV quaternion when it is valid and use identity only when
the raw quaternion is invalid. It must not synthesize heading from position.

## 4. Method Matrix

All methods run the same VINS-Fusion estimator and the same VINS-Fusion pose
graph optimizer. They differ only in how loop candidates are proposed and, for
LoopAnything, whether the proposed loop is accepted as a safe factor.

| Method | Candidate source | Verification before PGO | PGO backend |
| --- | --- | --- | --- |
| VINS-Fusion + DBoW2 | Native BRIEF-DBoW2 in `loop_fusion` | Native BRIEF matching + PnP RANSAC | Native VINS pose graph |
| VINS-Fusion + NetVLAD | NetVLAD top-K historical keyframes | Native BRIEF matching + PnP RANSAC | Native VINS pose graph |
| VINS-Fusion + SALAD | SALAD top-K historical keyframes | Native BRIEF matching + PnP RANSAC | Native VINS pose graph |
| VINS-Fusion + LoopAnything | Broad retrieval candidates plus LoopAnything verifier | LoopAnything safety gate, then VINS-compatible loop insertion | Native VINS pose graph |

## 5. Existing VINS-Fusion Loop-Closure Data Flow

The current VINS-Fusion loop-closure path is:

```text
loop_fusion_node
  -> receives VIO odometry, keyframe pose, keyframe image, and keyframe points
  -> constructs KeyFrame
  -> PoseGraph::addKeyFrame(...)
  -> PoseGraph::detectLoop(...)
  -> KeyFrame::findConnection(old_kf)
  -> BRIEF descriptor matching
  -> PnP RANSAC using current keyframe 3D points and old keyframe 2D observations
  -> writes KeyFrame::loop_info
  -> PoseGraph::optimize4DoF or PoseGraph::optimize6DoF
  -> writes vio_loop.csv
```

Important implementation boundary:

`PoseGraph::detectLoop(...)` is the candidate-retrieval boundary. Replacing this
component is sufficient for DBoW2, NetVLAD, and SALAD comparison because
`KeyFrame::findConnection(...)` can remain unchanged.

`KeyFrame::loop_info` is the accepted-loop-factor boundary. LoopAnything should
ultimately accept or reject loop factors before they enter this boundary.

## 6. Candidate-Retrieval Design

### 6.1 Native DBoW2 Baseline

Keep the existing VINS-Fusion implementation unchanged:

```text
PoseGraph::detectLoop(...)
  -> DBoW2 query on BRIEF descriptors
  -> candidate keyframe id
  -> KeyFrame::findConnection(...)
```

This is the baseline currently produced by `run_loop_fusion:=1`.

### 6.2 NetVLAD and SALAD Baselines

NetVLAD and SALAD should be integrated as alternative candidate providers:

```text
current keyframe image
  -> descriptor extraction
  -> historical descriptor database
  -> top-K legal historical candidates
  -> candidate keyframe id(s)
  -> KeyFrame::findConnection(...)
```

For fairness, NetVLAD and SALAD must not use LoopAnything geometry, DA3, PGO
residuals, or benchmark labels. They only change the image-retrieval candidate
source. The geometric verification remains VINS-Fusion's native BRIEF matching
and PnP RANSAC.

Candidate legality should follow the VINS runtime convention:

- candidate index must be sufficiently older than the current keyframe;
- candidate must have an available keyframe image and stored keyframe object;
- candidate selection must be causal and must not look into future frames.

The exact top-K and score threshold should be fixed before running the final
experiment and recorded in the run manifest. They should be shared between
NetVLAD and SALAD where possible, except for method-specific score scales.

## 7. LoopAnything Design

LoopAnything is not just another retrieval score. It should be evaluated as a
retrieval-plus-verification loop-closure front end.

The intended runtime logic is:

```text
current keyframe
  -> broad loop candidate proposal
  -> candidate-local support selection
  -> DA3 triplet geometry
  -> odometry-based Sim3 metric alignment
  -> factor-safety verification
  -> accept or reject candidate
  -> if accepted, insert a VINS-compatible loop factor
  -> native VINS pose graph optimization
```

For the first reliable paper experiment, the recommended implementation is an
offline-computed replay mode:

```text
offline LoopAnything candidate/verification records
  -> runtime VINS loop_fusion reads accepted pairs by keyframe index or timestamp
  -> accepted pair is passed to the same VINS loop insertion path
  -> VINS writes vio_loop.csv
```

This avoids putting GPU-heavy DA3 inference directly into ROS runtime during the
paper deadline. It also makes the experiment deterministic and reproducible.

The replay mode must still be causal: for a current keyframe `q`, only candidates
with index or timestamp earlier than `q` are allowed. Offline computation may be
used for speed, but the accepted edge set must be equivalent to a causal policy.

## 8. Loop Factor Insertion Policy

For retrieval-only methods, accepted loop factors are generated by VINS-Fusion:

```text
candidate id from retrieval
  -> KeyFrame::findConnection(old_kf)
  -> PnP RANSAC
  -> loop_info
  -> pose graph
```

For LoopAnything, there are two possible insertion policies.

### 8.1 Recommended Initial Policy: Candidate Gate + Native PnP Factor

LoopAnything determines whether a candidate is safe enough to try. If accepted,
VINS still calls `KeyFrame::findConnection(old_kf)` and inserts the native PnP
factor only if VINS PnP succeeds.

This policy is conservative and easiest to defend:

- all methods use the same VINS loop factor type;
- LoopAnything contributes by selecting safer candidates;
- PGO backend and loop residual model remain unchanged.

Its limitation is that it may understate LoopAnything's geometric factor ability,
because VINS's BRIEF/PnP may fail on candidates where DA3 could estimate a valid
factor.

### 8.2 Follow-up Policy: Direct DA3 Factor Insertion

LoopAnything directly writes a metric relative pose into `loop_info` and inserts
that factor into the native VINS pose graph.

This better reflects the full LoopAnything claim, but it requires careful frame
conversion, covariance/weight choice, and additional validation. It should be a
second-stage implementation after the conservative replay path is working.

## 9. Outputs and Metrics

Each method-sequence run must write a self-contained output directory containing:

```text
run_manifest.json
vio.csv
vio_loop.csv
vio.tum
vio_loop.tum
evo/evo_summary.json
evo/ape_vio.zip or ape_vio.json
evo/ape_loop.zip or ape_loop.json
evo/rpe_vio.zip or rpe_vio.json
evo/rpe_loop.zip or rpe_loop.json
evo/trajectory_gt_vio_loop.png
loop_events.jsonl
```

`loop_events.jsonl` should record at least:

```json
{
  "timestamp": 123.456,
  "query_index": 120,
  "candidate_index": 42,
  "method": "salad",
  "retrieval_score": 0.71,
  "retrieval_rank": 1,
  "verifier_status": "not_applicable",
  "pnp_status": "accepted",
  "pnp_inliers": 43,
  "relative_translation_norm": 3.2,
  "relative_yaw_deg": 8.5
}
```

For LoopAnything, `verifier_status` should distinguish:

- candidate not proposed;
- verifier rejected;
- verifier accepted but native PnP failed;
- verifier accepted and loop factor inserted.

The main reported metrics are:

- ATE RMSE of odometry-only `vio`;
- ATE RMSE of loop-optimized `vio_loop`;
- ATE improvement percentage;
- RPE RMSE before and after loop closure;
- number of inserted loop factors;
- number of rejected or failed candidate loops.

The paper table should report per-sequence ATE RMSE and an average over the nine
selected sequences. A supplementary table can report loop counts and failure
statistics.

## 10. Batch Execution Design

The final runner should support:

```text
METHOD=dbow2|netvlad|salad|loopanything
SEQUENCES=eee_01,eee_02,eee_03,nya_01,nya_02,nya_03,sbs_01,sbs_02,sbs_03
PLAY_RATE=1.0
RUN_PREFIX=<stable experiment id>
```

The existing NTU-VIRAL batch runner should be changed from "all directories under
data root" to an explicit sequence list, defaulting to the nine selected
sequences. This avoids accidentally rerunning RTP sequences.

Each run should use the same VINS-Fusion config unless the method requires only
loop-retrieval-specific parameters.

## 11. Fairness Rules

The experiment is fair only if these constraints hold:

1. All methods run the same VINS-Fusion VIO frontend.
2. All methods use the same NTU-VIRAL bags and same ground-truth conversion.
3. All retrieval-only methods use the same native VINS PnP verification.
4. All methods use the same VINS pose graph optimizer and trajectory export.
5. No method uses future frames to propose causal loop edges.
6. No method uses GT trajectory or benchmark labels at runtime.
7. Method-specific thresholds are fixed before final batch execution and recorded
   in each manifest.

## 12. Risks and Mitigations

### 12.1 Runtime GPU Integration Risk

Running DA3, SALAD, or NetVLAD inside ROS may cause scheduling stalls and
nondeterministic timing. The initial implementation should use offline descriptor
or verifier replay where possible.

### 12.2 Index Alignment Risk

VINS keyframe indices are runtime-generated and may differ across methods if
method runtime affects estimator timing. The replay mechanism should prefer
timestamp matching or explicitly verify that keyframe index streams are identical
across methods.

### 12.3 Native PnP Bottleneck Risk

If LoopAnything uses candidate gate plus native PnP insertion, improvements may
be limited by VINS's BRIEF/PnP factor generation. This is acceptable for the first
controlled experiment, but a direct DA3 factor insertion experiment should be
kept as a follow-up if time allows.

### 12.4 Over-Optimization Risk

Do not tune thresholds per sequence. If thresholds are swept, report the sweep
protocol separately and freeze one global setting for the final table.

## 13. Implementation Plan Entry Point

The implementation should start with the smallest change that produces a valid
controlled experiment:

1. Restrict the existing NTU-VIRAL batch runner to the nine selected sequences.
2. Add method selection to `loop_fusion`.
3. Refactor native DBoW2 candidate retrieval behind a provider interface.
4. Add offline replay candidate provider for NetVLAD and SALAD.
5. Add LoopAnything replay verifier records and event logging.
6. Run DBoW2, NetVLAD, SALAD, and LoopAnything over all nine sequences.
7. Summarize ATE/RPE/loop-count results into the paper experiment table.
