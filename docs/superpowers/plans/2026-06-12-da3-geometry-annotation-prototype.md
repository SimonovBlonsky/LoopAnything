# DA3 Geometry Annotation Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-label the existing frozen 4,000 query-candidate pairs using the current
LoopAnything support + DA3 + odometry-Sim3 pose generator, then manually inspect every
pair in one day.

**Architecture:** Keep the existing benchmark pair manifest and image caches unchanged.
Run only support selection, batched DA3 triplet inference, and Sim3 alignment; explicitly
skip PGO and all verifier scores. Compare each generated metric loop factor with the
cache GT relative pose, checkpoint one query at a time, and expose the automatic label
and pose errors in the existing append-only annotation UI.

**Tech Stack:** Python, NumPy, existing `robust_loop_verifier` package, DA3-LARGE-1.1,
HTML/JavaScript annotation UI, pytest.

---

## Scope And Time Budget

Output root:

```text
workspace/rover_aligned_benchmark/benchmark_v1/da3_geometry_annotation_v1/
  benchmark_pairs.jsonl
  manifest.json
  geometry_predictions.jsonl
  geometry_prediction_manifest.json
  annotation_events.jsonl
  annotations.jsonl
  annotation_seal.json
  label_comparison.csv
  label_comparison.md
  label_comparison.json
```

The original `benchmark_v1` files and sealed annotations remain read-only.

Initial automatic-positive rule:

```text
effective_translation_threshold_m =
  clamp(
    0.2 * gt_translation_norm_m,
    min=1.0 m,
    max=5.0 m
  )

factor generation succeeded
AND translation_error_m <= effective_translation_threshold_m
AND rotation_error_deg <= 15.0
AND (
  gt_translation_norm_m < 0.5
  OR translation_direction_error_deg <= 20.0
)
```

All thresholds are CLI arguments and recorded in
`geometry_prediction_manifest.json`. Each prediction also records its pair-specific
`effective_translation_error_threshold_m`. Translation-direction error is displayed
as `N/A` when either translation norm is numerically degenerate. The scale-adaptive
translation rule is shared across all sequences; it does not use per-sequence
hand-tuned thresholds.

Expected one-day schedule:

```text
implementation + unit tests       2-3 h
20-pair convention smoke test     0.5 h
4,000-pair DA3 generation         1-2 h, checkpointed
manual review at 3-6 s/pair       3.5-7 h
comparison report                 <0.5 h
```

## File Structure

- Create `src/robust_loop_verifier/geometry_annotation.py`
  - Compute GT relative pose, factor errors, automatic labels, and comparison summary.
- Modify `src/robust_loop_verifier/pipeline.py`
  - Preserve the generated Sim3 loop factor and support a pose-only path that returns
    before PGO.
- Modify `src/robust_loop_verifier/rover_pair_scoring.py`
  - Iterate frozen pairs by sequence/query and produce pose-only records.
- Create `robust_loop_verification_scripts/build_da3_geometry_annotation.py`
  - Prepare the new annotation bundle, run resumable DA3 inference, and write manifests.
- Modify `robust_loop_verification_scripts/annotate_rover_aligned_benchmark.py`
  - Optionally load and display geometry predictions; add accept-auto keyboard control.
- Create `robust_loop_verification_scripts/summarize_da3_geometry_annotations.py`
  - Compare previous manual labels, automatic geometry labels, and reviewed labels.
- Create `robust_loop_verification_scripts/run_da3_geometry_annotation_prototype.sh`
  - One command for build/resume, validation, and launching the UI.
- Create `tests/robust_loop_verifier/test_geometry_annotation.py`
- Modify `tests/robust_loop_verifier/test_pipeline.py`
- Modify `tests/robust_loop_verifier/test_rover_pair_scoring.py`
- Modify `tests/robust_loop_verifier/test_rover_annotation.py`

### Task 1: Define Pose-Error And Automatic-Label Semantics

**Files:**
- Create: `src/robust_loop_verifier/geometry_annotation.py`
- Create: `tests/robust_loop_verifier/test_geometry_annotation.py`

- [ ] **Step 1: Write tests for exact-pose, rotated-pose, translated-pose, and degenerate-direction cases**

Test the public interface:

```python
@dataclass(frozen=True)
class GeometryLabelThresholds:
    min_translation_error_m: float = 1.0
    max_translation_error_m: float = 5.0
    translation_error_scale_ratio: float = 0.2
    max_rotation_error_deg: float = 15.0
    max_translation_direction_error_deg: float = 20.0
    min_direction_baseline_m: float = 0.5


def evaluate_metric_loop_factor(
    gt_query_c2w: np.ndarray,
    gt_candidate_c2w: np.ndarray,
    estimated_query_to_candidate: np.ndarray | None,
    *,
    factor_status: str,
    thresholds: GeometryLabelThresholds,
) -> dict[str, object]:
    ...
```

Assert:

```text
GT factor = inverse(gt_query_c2w) * gt_candidate_c2w
exact estimate -> zero errors and auto_label=1
20-degree rotation error -> auto_label=0
2 m translation error at a 2 m GT baseline -> auto_label=0
1.5 m translation error at a 10 m GT baseline -> auto_label=1
translation threshold is capped at 5 m
GT baseline below 0.5 m -> direction error does not reject
missing/invalid estimate -> auto_label=0 with explicit reason
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
PYTHONPATH=src pytest tests/robust_loop_verifier/test_geometry_annotation.py -q
```

Expected: import or symbol failure because the module does not exist.

- [ ] **Step 3: Implement pose-error computation**

Return JSON-safe fields:

```text
gt_relative_pose
estimated_relative_pose
gt_rotation_axis
gt_rotation_angle_deg
estimated_rotation_axis
estimated_rotation_angle_deg
gt_translation
gt_translation_norm_m
estimated_translation
estimated_translation_norm_m
translation_error_m
rotation_error_deg
translation_direction_error_deg
factor_status
automatic_label
automatic_label_reason
```

Use the existing `pose_between()` convention for both GT and estimate. Compute rotation
error from the angle of:

```text
inverse(R_gt) * R_est
```

Compute metric translation error from the translation part of:

```text
inverse(T_gt) * T_est
```

Compute direction error from the angle between the GT and estimated relative
translation vectors. Validate finite SE(3) inputs and threshold values.

- [ ] **Step 4: Run the focused tests**

Run:

```bash
PYTHONPATH=src pytest tests/robust_loop_verifier/test_geometry_annotation.py -q
```

Expected: all tests pass.

### Task 2: Export LoopAnything Factors Without Running The Verifier

**Files:**
- Modify: `src/robust_loop_verifier/pipeline.py`
- Modify: `src/robust_loop_verifier/rover_pair_scoring.py`
- Modify: `tests/robust_loop_verifier/test_pipeline.py`
- Modify: `tests/robust_loop_verifier/test_rover_pair_scoring.py`

- [ ] **Step 1: Add failing tests for a pose-only frozen-candidate path**

Extend:

```python
def score_frozen_query_candidates(
    ...,
    run_pgo: bool = True,
) -> list[dict[str, Any]]:
```

The test must verify that `run_pgo=False`:

```text
still performs support selection
still performs batched DA3 inference
still performs candidate-support Sim3 alignment
writes loop_factor as a flattened 4x4 matrix
does not call run_full_prefix_pgo
leaves all PGO/verifier fields unset
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
PYTHONPATH=src pytest \
  tests/robust_loop_verifier/test_pipeline.py \
  tests/robust_loop_verifier/test_rover_pair_scoring.py -q
```

Expected: failures for the missing `run_pgo` behavior and `loop_factor` field.

- [ ] **Step 3: Implement the minimal pose-only path**

In `_new_candidate_record()`, initialize:

```python
"loop_factor": None,
```

Immediately after successful Sim3 alignment:

```python
record["loop_factor"] = np.asarray(sim3.loop_factor, dtype=np.float64).reshape(-1)
```

When `run_pgo=False`, finish the candidate after setting `score_da3_sim3`; do not build
the prefix graph. Keep the default `run_pgo=True` so existing experiments are unchanged.

Add a query-batch iterator in `rover_pair_scoring.py`:

```python
def iter_geometry_factor_batches(
    benchmark_root: Path,
    config,
    da3_runner,
    *,
    completed_pair_ids: set[str] | None = None,
) -> Iterator[list[dict[str, Any]]]:
    ...
```

It must:

```text
preserve benchmark pair order
group by sequence and query
load image, odom_pose, and gt_pose from keyframes.jsonl
call score_frozen_query_candidates(..., run_pgo=False)
attach the corresponding GT query/candidate poses
yield one completed query batch at a time
skip pair IDs already checkpointed
not read old annotations or positives.jsonl
```

- [ ] **Step 4: Run focused and regression tests**

Run:

```bash
PYTHONPATH=src pytest \
  tests/robust_loop_verifier/test_pipeline.py \
  tests/robust_loop_verifier/test_rover_pair_scoring.py -q
```

Expected: all tests pass and existing verifier behavior remains unchanged.

### Task 3: Build A Resumable 4,000-Pair Geometry Prediction Bundle

**Files:**
- Create: `robust_loop_verification_scripts/build_da3_geometry_annotation.py`
- Create: `tests/robust_loop_verifier/test_da3_geometry_annotation_script.py`

- [ ] **Step 1: Write tests for preparation, checkpoint resume, and source-hash validation**

The script interface is:

```bash
python robust_loop_verification_scripts/build_da3_geometry_annotation.py \
  workspace/rover_aligned_benchmark/benchmark_v1 \
  --output-root workspace/rover_aligned_benchmark/benchmark_v1/da3_geometry_annotation_v1 \
  --backend real \
  --device cuda \
  --min-translation-error-m 1.0 \
  --max-translation-error-m 5.0 \
  --translation-error-scale-ratio 0.2 \
  --max-rotation-error-deg 15.0 \
  --max-translation-direction-error-deg 20.0 \
  --min-direction-baseline-m 0.5
```

Tests must verify:

```text
benchmark_pairs.jsonl and manifest.json are copied exactly once
existing non-matching output bundle is rejected
each completed query appends and fsyncs its rows
restart skips completed pair IDs without duplicates
partial final line is rejected rather than silently ignored
all 4,000 records are required before marking generation complete
source pair-manifest hash, DA3 config, thresholds, command, and source commit are recorded
```

- [ ] **Step 2: Run the script tests and verify they fail**

Run:

```bash
PYTHONPATH=src pytest \
  tests/robust_loop_verifier/test_da3_geometry_annotation_script.py -q
```

Expected: import or script behavior failures.

- [ ] **Step 3: Implement bundle generation**

For each yielded query batch:

1. Convert `loop_factor` to a 4x4 estimate or `None`.
2. Call `evaluate_metric_loop_factor()`.
3. Merge identity, support, Sim3, GT, estimate, errors, and automatic label.
4. Append ten rows to `geometry_predictions.jsonl`.
5. Flush and `fsync` before processing the next query.

Do not write:

```text
pgo_error_after
trajectory_deformation_rmse
query_gate_graph score
old manual label
```

At full coverage, atomically write `geometry_prediction_manifest.json` with the
prediction-file hash and `complete=true`.

- [ ] **Step 4: Run script tests and a 20-pair mock smoke test**

Run:

```bash
PYTHONPATH=src pytest \
  tests/robust_loop_verifier/test_da3_geometry_annotation_script.py -q

PYTHONPATH=src python \
  robust_loop_verification_scripts/build_da3_geometry_annotation.py \
  workspace/rover_aligned_benchmark/benchmark_v1 \
  --output-root /tmp/da3_geometry_annotation_smoke \
  --backend mock \
  --pair-limit 20
```

Expected: tests pass and 20 ordered prediction rows are produced.

### Task 4: Add Fast Geometry Review To The Existing Annotation UI

**Files:**
- Modify: `robust_loop_verification_scripts/annotate_rover_aligned_benchmark.py`
- Modify: `tests/robust_loop_verifier/test_rover_annotation.py`

- [ ] **Step 1: Add failing state and HTML tests**

When `geometry_predictions.jsonl` exists, `/api/state` must include:

```json
{
  "geometry": {
    "automatic_label": 1,
    "support_idx": 28,
    "factor_status": "ok",
    "gt_rotation_angle_deg": 34.1,
    "gt_rotation_axis": [0.0, 0.0, 1.0],
    "gt_translation": [1.2, -0.1, 0.0],
    "gt_translation_norm_m": 1.204,
    "estimated_rotation_angle_deg": 31.2,
    "estimated_rotation_axis": [0.0, 0.0, 1.0],
    "estimated_translation": [1.1, -0.2, 0.0],
    "translation_error_m": 0.14,
    "rotation_error_deg": 3.2,
    "translation_direction_error_deg": 4.8,
    "automatic_label_reason": "within_thresholds"
  }
}
```

The UI tests must verify:

```text
q and c remain visually dominant
geometry card displays plain numeric text, not LaTeX
DA3 automatic positive/negative label is displayed prominently
Space and Enter submit automatic_label
P explicitly submits positive
N explicitly submits negative
Backspace still undoes
old benchmark roots without geometry_predictions.jsonl still work
```

- [ ] **Step 2: Run annotation tests and verify they fail**

Run:

```bash
PYTHONPATH=src pytest tests/robust_loop_verifier/test_rover_annotation.py -q
```

Expected: missing geometry state and keyboard behavior failures.

- [ ] **Step 3: Implement optional geometry display**

Load `geometry_predictions.jsonl` only if present. Require exactly one prediction per
pair and exact pair-order coverage before starting the server. Never expose the old
manual label in the UI.

Add a compact card:

```text
DA3 AUTO LABEL: POSITIVE / NEGATIVE
GT rotation: angle + axis
GT translation: [x, y, z], norm
Estimated rotation: angle + axis
Estimated translation: [x, y, z], norm
Rotation error
Metric translation error
Effective translation error threshold
Translation-direction error
Support index / failure reason
```

Render `DA3 AUTO LABEL` as a prominent green `POSITIVE` or red `NEGATIVE` badge. This
is the thresholded result of the support + DA3 + Sim3 pose generator, not the final
human annotation. The reviewer may accept it with Space/Enter or override it with P/N.

Controls:

```text
Space or Enter: accept automatic label
P: positive override
N: negative override
Backspace: undo
```

All actions continue to use the existing append-only event log and immediate `fsync`.

- [ ] **Step 4: Run annotation tests**

Run:

```bash
PYTHONPATH=src pytest tests/robust_loop_verifier/test_rover_annotation.py -q
```

Expected: all tests pass.

### Task 5: Produce The Label-Difference Report And One-Command Workflow

**Files:**
- Create: `robust_loop_verification_scripts/summarize_da3_geometry_annotations.py`
- Create: `robust_loop_verification_scripts/run_da3_geometry_annotation_prototype.sh`
- Create: `tests/robust_loop_verifier/test_da3_geometry_annotation_summary.py`

- [ ] **Step 1: Write summary tests**

Compare:

```text
old reviewed appearance label
DA3 automatic geometry label
new reviewed geometry label
```

Report per sequence and macro:

```text
pair count
old positive count/rate
automatic positive count/rate
reviewed positive count/rate
old negative -> reviewed positive
old positive -> reviewed negative
automatic-label precision/recall against reviewed labels
support/DA3/Sim3 failure counts
median and p90 pose errors for reviewed positives and negatives
```

- [ ] **Step 2: Implement the summary script**

Require valid seals for the old and new reviewed annotations. Write CSV, Markdown, and
JSON atomically. Include threshold and prediction-manifest hashes in the report.

- [ ] **Step 3: Implement the wrapper**

The wrapper accepts:

```text
--benchmark-root
--output-root
--device
--backend
--pair-limit
--port
--skip-build
--summary-only
```

Default workflow:

```text
build or resume geometry_predictions.jsonl
validate complete prediction coverage
launch annotation server with --open
```

After finalization:

```bash
robust_loop_verification_scripts/run_da3_geometry_annotation_prototype.sh \
  --summary-only
```

- [ ] **Step 4: Run all targeted tests**

Run:

```bash
PYTHONPATH=src pytest \
  tests/robust_loop_verifier/test_geometry_annotation.py \
  tests/robust_loop_verifier/test_da3_geometry_annotation_script.py \
  tests/robust_loop_verifier/test_da3_geometry_annotation_summary.py \
  tests/robust_loop_verifier/test_rover_annotation.py \
  tests/robust_loop_verifier/test_rover_pair_scoring.py \
  tests/robust_loop_verifier/test_pipeline.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Run a real 20-pair GPU convention check before the full dataset**

Run:

```bash
PYTHONPATH=src /home/chenguyuan/anaconda3/envs/da3/bin/python \
  robust_loop_verification_scripts/build_da3_geometry_annotation.py \
  workspace/rover_aligned_benchmark/benchmark_v1 \
  --output-root workspace/rover_aligned_benchmark/benchmark_v1/da3_geometry_smoke20 \
  --backend real \
  --device cuda \
  --pair-limit 20
```

Manually verify at least:

```text
one near-zero-error synthetic or obvious loop case
one obvious non-loop case
query-to-candidate convention is not inverted
translation is expressed in the expected query frame
all displayed values are finite and understandable
```

- [ ] **Step 6: Launch the full resumable generation and annotation**

Run:

```bash
robust_loop_verification_scripts/run_da3_geometry_annotation_prototype.sh \
  --benchmark-root workspace/rover_aligned_benchmark/benchmark_v1 \
  --output-root workspace/rover_aligned_benchmark/benchmark_v1/da3_geometry_annotation_v1 \
  --device cuda
```

If interrupted, rerun the same command. It must resume from the next incomplete query.

## Acceptance Criteria

1. All 4,000 frozen pairs have exactly one geometry prediction.
2. PGO, residual, deformation, and verifier scores are absent from label generation.
3. Every successful factor is compared against `gt_pose` using the same relative-pose
   convention.
4. Every pair is manually reviewed with immediate append-only persistence.
5. The old sealed annotations remain unchanged.
6. The final report quantifies how geometry labels differ from the previous
   appearance-driven labels.
7. The prototype is explicitly documented as method-bound and excluded from the final
   paper main table; RoMa v2 will replace DA3 in the final benchmark labeler.
