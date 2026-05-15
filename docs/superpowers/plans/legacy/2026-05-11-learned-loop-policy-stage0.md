# LEGACY: Learned Loop Policy Stage0 Plan

Status: legacy as of 2026-05-15.

This document is retained only as a historical record. The learned-policy
Stage0 direction is not considered converged and must not be used as the basis
for the active robust loop verifier.

Do not use this document's labels, hard gates, feature schema, thresholds,
support rules, score definitions, or implementation plan as priors for robust
loop verifier design or implementation. In particular, do not use
`safe_loop_factor_v1`, `x_geom`, or any learned-policy cache outputs to define
ground truth or verifier acceptance. The learned-policy direction is deferred
until an interpretable training-free robust loop verifier is completed and
achieves strong experimental results.

---

# Causal Loop Policy Dataset Builder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a causal offline loop-policy dataset cache builder under `LoopAnything/src/loop_policy` that converts AsterSLAM loop-disabled exports into auditable retrieval, support, DA3/Sim3, feature, and label records for learned loop verification.

**Architecture:** Keep learned-policy dataset code isolated in a new `loop_policy` package and call DA3/SALAD only through lazy wrapper interfaces. The builder reads one AsterSLAM raw sequence at a time, precomputes descriptors, performs causal retrieval, selects historical supports, runs candidate-local DA3 groups, applies the AsterSLAM-compatible Sim3 prior, writes labels/features, and emits sequence/root manifests with leakage audit fields.

**Tech Stack:** Python 3.9+, NumPy, PyTorch for optional DA3/SALAD execution, Pillow/OpenCV for image loading, pytest, JSONL, NPZ, argparse.

---

## Execution Notes

- Keep all new source code in `LoopAnything/src/loop_policy/`.
- Do not modify `LoopAnything/src/depth_anything_3/`.
- Modify `LoopAnything/pyproject.toml` only to package `src/loop_policy`.
- Unit tests must use synthetic data and mock DA3/descriptor backends so they do not require GPUs or downloaded checkpoints.
- Do not commit during execution unless the user explicitly asks for commits. Each task ends with a `git status` checkpoint instead.
- Implementation-facing names must describe behavior and domain, not roadmap or
  plan labels. Do not introduce Python identifiers, module names, CLI commands,
  output artifact names, schema versions, log messages, or comments named
  after vague labels such as `stage0`, `stage_0`, `stage 0`, `stage1`,
  `task1`, or `task 2`; use names like `LoopPolicyDatasetConfig`,
  `dataset_builder.py`, `sequence_summary.json`, and `dataset_manifest.json`.

## Post-Implementation Bugfix Log

### 2026-05-12: DA3 Pose Convention And Reference-View Ordering

Symptom: the `handheld_escalator00` oracle loop pair `query=137`, `candidate=63`
with supports such as `59` or `64` initially produced bad Sim3 alignment at low
runtime-smoke resolution, including incorrect candidate-query/candidate-support
edge ratios and many `sim3_scale_out_of_range` rejections.

Root causes found during debugging:

- DA3 API `prediction.extrinsics` is native world-to-camera (`w2c`); the
  offline builder CLI default had treated DA3 output as `c2w`, unlike
  AsterSLAM's `da3_pose_node.py`, which inverts DA3 `w2c` output before
  publishing `predicted_group_c2w`.
- DA3 reference-view selection can reorder the internal view sequence before
  camera decoding. The offline runner must not rely on the default
  `saddle_balanced` strategy unless output order is explicitly restored.
  Use `ref_view_strategy="first"` for query/candidate/support slot stability,
  matching the AsterSLAM runtime default.
- `process_res=112` is valid only for pipeline smoke tests. It can severely
  degrade DA3 pose geometry and should not be used for Sim3/label-quality
  conclusions. Use the normal DA3 runtime resolution such as `504` for oracle
  geometry checks and real label generation.

Implemented fix:

- `DepthAnything3Runner` now carries a configurable `ref_view_strategy` and
  defaults to `"first"`.
- `loop_policy.dataset_builder` now defaults `--da3-extrinsics-convention` to
  `w2c` and adds `--da3-ref-view-strategy`, defaulting to `"first"`.
- `scripts/oracle_sim3_check.py` defaults to `process_res=504`,
  `ref_view_strategy="first"`, and the corrected DA3 `w2c -> c2w` path.

Regression coverage and verification:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src /home/chenguyuan/anaconda3/envs/da3/bin/python -m pytest tests/loop_policy -q
python3 -m py_compile src/loop_policy/da3_runner.py src/loop_policy/dataset_builder.py scripts/oracle_sim3_check.py
```

Oracle recheck:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
env -u ALL_PROXY -u all_proxy XFORMERS_DISABLED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  MPLCONFIGDIR=/tmp/matplotlib-loop-policy PYTHONPATH=src \
  /home/chenguyuan/anaconda3/envs/da3/bin/python scripts/oracle_sim3_check.py --support-idx 59
```

Observed after the fix: `query=137`, `candidate=63`, `support=59` is accepted
on the corrected path, with aligned query-vs-odom residual about `0.112m /
0.83deg` and support alignment RMSE about `0.173m`.

### 2026-05-13: DA3 Scale Is Not A Hard Label Or Prior Rejection Condition

Symptom: after fixing DA3 pose convention/reference-view ordering and changing
support selection to nearest-candidate supports, many visually and odometry
consistent loop candidates still failed only because DA3's Sim3 scale was not
near metric scale. In `handheld_escalator00`, removing the label-only
`abs_log_sim3_scale` gate raised safe labels from `0` to `288`; additionally
removing Task 8's `sim3_scale_out_of_range` hard precondition raised safe labels
to `339` on the same cached DA3 outputs.

Reasoning:

- DA3 does not produce metric-scale camera translations. A large Sim3 scale is
  expected when aligning DA3 local poses to odometry and is not by itself a
  loop-quality failure.
- `sim3_scale` and `abs_log_sim3_scale` remain useful diagnostics and learned
  features in `x_geom`, but they must not be hard gates for
  `safe_loop_factor_v1`.
- Task 8 should still reject invalid scale values, such as non-finite or
  non-positive scale, because those make alignment undefined. It should not
  reject merely because a finite positive scale is outside the historical
  `[0.05, 20.0]` bounds.

Implemented fix:

- `compute_safe_loop_factor_v1()` no longer requires
  `abs_log_sim3_scale <= abs_log_sim3_scale_thr`; `sim3_quality_good` is now
  based on support alignment RMSE and direction consistency.
- `align_da3_poses_with_candidate_support_prior()` no longer emits
  `sim3_scale_out_of_range` as a rejection reason. It only rejects non-finite
  or non-positive scale as `invalid_sim3_scale`.

Runtime re-label check on cached
`handheld_escalator00` output
`/tmp/loop_policy_stage0_handheld_escalator00_20260513_2111_gpu_nearest_support`:

```text
features=601
new_precondition_valid=442
sim3_quality_good=442
odom_consistent_loose=354
safe_loop_factor_v1=339
```

Odometry-threshold breakdown after the scale hard-gate removal:

```text
<=0.5m && <=15deg: 49 candidates, safe=48
<=1.0m && <=15deg: 99 candidates, safe=97
<=2.0m && <=20deg: 216 candidates, safe=214
<=5.0m && <=30deg: 292 candidates, safe=288
```

Known oracle pairs after re-labeling:

```text
q=137,c=63,s=62 -> safe=true, aligned residual 0.201m / 0.79deg
q=137,c=64,s=63 -> safe=true, aligned residual 0.094m / 1.18deg
q=123,c=46,s=45 -> safe=true, aligned residual 0.057m / 2.07deg
q=99,c=23,s=22 -> safe=true, aligned residual 0.196m / 3.02deg
```

### 2026-05-13: Visualization Audit Shows False Safe Loops

Status: implementation work for Tasks 1-12 is complete, but the offline learned
loop-policy Stage0 labeling is not converged. Do not treat the current
`safe_loop_factor_v1` distribution as final training labels.

Added visualization support:

- `loop_policy.dataset_builder` can write per-feature visual records with
  `--write-visualization-records`.
- Each visual record copies `query.png`, `candidate.png`, `support_*.png`, and a
  `record.json` payload into a boolean tree:

```text
visual_records/
  new_precondition_valid|new_precondition_invalid/
    sim3_quality_good|sim3_quality_bad/
      odom_consistent_loose|odom_consistent_not_loose/
        safe_loop_factor_v1|safe_loop_factor_negative/
          q000137_c000063_s000062/
```

Fresh `handheld_escalator00` run with visualization:

```text
output_root=/tmp/loop_policy_stage0_handheld_escalator00_20260513_2147_gpu_visual
candidate_features=601
visual_record_json=601
safe_loop_factor_positive_count=325
```

Visualization tree counts:

```text
new_precondition_valid/sim3_quality_good/odom_consistent_loose/safe_loop_factor_v1: 325
new_precondition_valid/sim3_quality_good/odom_consistent_not_loose/safe_loop_factor_negative: 115
new_precondition_invalid/sim3_quality_bad/odom_consistent_loose/safe_loop_factor_negative: 16
new_precondition_invalid/sim3_quality_bad/odom_consistent_not_loose/safe_loop_factor_negative: 145
```

Manual audit finding:

- Several records currently labeled `safe_loop_factor_v1` are visually false
  loops: query and candidate are clearly not the same place, with little or no
  overlap.
- Concrete examples observed from the visualization tree:

```text
q000049_c000015_s000014
q000080_c000010_s000009
q000093_c000062_s000061
q000093_c000061_s000060
q000094_c000015_s000014
```

Interpretation:

- The current gates can accept retrieval false positives when DA3/Sim3 and
  odometry residuals appear numerically consistent.
- This likely means the current `safe_loop_factor_v1` definition is missing a
  direct visual-overlap or retrieval-quality rejection condition, or the
  odometry-consistency proxy is insufficient for this sequence.
- Next work should analyze these false-safe examples before using the labels for
  training. Candidate directions include adding a visual-overlap/appearance
  consistency gate, checking whether odometry residual is being computed against
  the intended relative pose, tightening candidate acceptance with retrieval
  margins, or adding a manual-audit blacklist/debug set for Stage0 calibration.

## File Structure

Create or modify these files:

```text
LoopAnything/pyproject.toml
LoopAnything/src/loop_policy/__init__.py
LoopAnything/src/loop_policy/schema.py
LoopAnything/src/loop_policy/io.py
LoopAnything/src/loop_policy/geometry.py
LoopAnything/src/loop_policy/retrieval.py
LoopAnything/src/loop_policy/support.py
LoopAnything/src/loop_policy/da3_runner.py
LoopAnything/src/loop_policy/sim3_prior.py
LoopAnything/src/loop_policy/labels.py
LoopAnything/src/loop_policy/dataset_builder.py
LoopAnything/tests/loop_policy/conftest.py
LoopAnything/tests/loop_policy/test_package_boundary.py
LoopAnything/tests/loop_policy/test_schema_io.py
LoopAnything/tests/loop_policy/test_geometry.py
LoopAnything/tests/loop_policy/test_retrieval.py
LoopAnything/tests/loop_policy/test_support.py
LoopAnything/tests/loop_policy/test_da3_runner.py
LoopAnything/tests/loop_policy/test_sim3_prior.py
LoopAnything/tests/loop_policy/test_labels.py
LoopAnything/tests/loop_policy/test_dataset_builder.py
```

Responsibility map:

- `schema.py`: dataclasses, schema version strings, JSON conversion, output field names.
- `io.py`: AsterSLAM raw cache readers, TUM trajectory reader, JSONL/NPZ sidecar helpers.
- `geometry.py`: SE3 operations, TUM quaternion conversion, camera pose conversion, pose residuals.
- `retrieval.py`: descriptor interfaces, descriptor NPZ cache, causal top-K ranking.
- `support.py`: AsterSLAM-compatible causal support selection.
- `da3_runner.py`: candidate-local DA3 group construction and lazy DA3 inference wrapper.
- `sim3_prior.py`: Python port of `alignDa3PosesWithCandidateSupportPrior` semantics.
- `labels.py`: `safe_loop_factor_v1`, GT audit labels, `x_geom[32]` assembly.
- `dataset_builder.py`: CLI, per-sequence orchestration, resume checks, manifests, leakage audit.

## Task 1: Package Boundary And Import Smoke

**Files:**
- Create: `LoopAnything/src/loop_policy/__init__.py`
- Modify: `LoopAnything/pyproject.toml`
- Test: `LoopAnything/tests/loop_policy/test_package_boundary.py`

- [ ] **Step 1: Write the failing package-boundary test**

Create `LoopAnything/tests/loop_policy/test_package_boundary.py`:

```python
from pathlib import Path


def test_loop_policy_package_imports():
    import loop_policy

    assert loop_policy.__version__ == "0.1.0"


def test_loop_policy_code_lives_outside_depth_anything_tree():
    repo = Path(__file__).resolve().parents[2]
    loop_policy_dir = repo / "src" / "loop_policy"
    da3_dir = repo / "src" / "depth_anything_3"

    assert loop_policy_dir.is_dir()
    assert not list(da3_dir.rglob("*loop_policy*"))
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/loop_policy/test_package_boundary.py -q
```

Expected: FAIL because `loop_policy` does not exist or `src/loop_policy` is missing.

- [ ] **Step 3: Add the minimal package and package discovery**

Create `LoopAnything/src/loop_policy/__init__.py`:

```python
"""Offline dataset builder for learned loop verification policy."""

__version__ = "0.1.0"
```

Update `LoopAnything/pyproject.toml`:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/depth_anything_3", "src/loop_policy"]

[tool.hatch.build.targets.sdist]
include = [
  "/README.md",
  "/pyproject.toml",
  "/src/depth_anything_3",
  "/src/loop_policy",
]
```

- [ ] **Step 4: Run the test and verify it passes**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/loop_policy/test_package_boundary.py -q
```

Expected: PASS.

- [ ] **Step 5: Check the edit set**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
git status --short
```

Expected changed paths include `pyproject.toml`, `src/loop_policy/__init__.py`, and the test file. No `src/depth_anything_3` files are modified.

## Task 2: Schema And JSONL Helpers

**Files:**
- Create: `LoopAnything/src/loop_policy/schema.py`
- Test: `LoopAnything/tests/loop_policy/test_schema_io.py`

- [ ] **Step 1: Write schema round-trip tests**

Create the first part of `LoopAnything/tests/loop_policy/test_schema_io.py`:

```python
import json

from loop_policy.schema import (
    CandidateFeatureRecord,
    KeyframeRecord,
    RetrievalCandidate,
    RetrievalRecord,
    LoopPolicyDatasetConfig,
    SupportDecision,
    dataclass_to_json_dict,
)


def test_loop_policy_dataset_config_defaults_are_causal():
    config = LoopPolicyDatasetConfig(
        dataset_root="/dataset",
        output_root="/loop_policy_dataset",
        sequences=("handheld_room01",),
    )

    assert config.causal is True
    assert config.runtime_top_k == 4
    assert config.retrieval_pool_size >= config.runtime_top_k


def test_json_roundtrip_keeps_required_audit_fields():
    record = RetrievalRecord(
        sequence="handheld_room01",
        query_idx=10,
        query_timestamp=100.0,
        causal=True,
        database_max_idx=4,
        database_max_timestamp=94.0,
        retrieval_db_size=2,
        candidates=[
            RetrievalCandidate(
                rank=1,
                keyframe_idx=4,
                timestamp=94.0,
                score=0.91,
                runtime_topk=True,
            )
        ],
    )

    payload = json.loads(json.dumps(dataclass_to_json_dict(record)))

    assert payload["causal"] is True
    assert payload["query_idx"] == 10
    assert payload["database_max_idx"] == 4
    assert payload["candidates"][0]["runtime_topk"] is True


def test_candidate_feature_record_requires_32_geom_features():
    feature = CandidateFeatureRecord(
        sequence="handheld_room01",
        query_idx=10,
        query_timestamp=100.0,
        candidate_source="retrieval_topk",
        candidate_idx=4,
        candidate_timestamp=94.0,
        causal=True,
        database_max_idx=4,
        database_max_timestamp=94.0,
        retrieval_db_size=2,
        support_snapshot_max_idx=5,
        support_snapshot_max_timestamp=95.0,
        selected_support_indices=[3],
        selected_support_timestamps=[93.0],
        support_count=1,
        precondition_valid=True,
        negative_reason=None,
        x_geom=[0.0] * 32,
        safe_loop_factor_v1=False,
        labels={"sim3_quality_good": False, "odom_consistent_loose": True},
        metrics={"support_align_rmse": 0.3},
    )

    payload = dataclass_to_json_dict(feature)

    assert len(payload["x_geom"]) == 32
    assert payload["selected_support_indices"] == [3]
    assert payload["labels"]["sim3_quality_good"] is False


def test_support_decision_records_no_valid_support_reason():
    decision = SupportDecision(
        sequence="handheld_room01",
        query_idx=10,
        query_timestamp=100.0,
        candidate_idx=4,
        candidate_timestamp=94.0,
        causal=True,
        support_snapshot_max_idx=4,
        support_snapshot_max_timestamp=94.0,
        selected_support_indices=[],
        selected_support_timestamps=[],
        selected_support_baselines=[],
        support_count=0,
        rejected=True,
        rejection_reason="no_valid_support",
    )

    assert dataclass_to_json_dict(decision)["rejection_reason"] == "no_valid_support"
```

- [ ] **Step 2: Run the schema tests and verify they fail**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/loop_policy/test_schema_io.py -q
```

Expected: FAIL because `loop_policy.schema` does not exist.

- [ ] **Step 3: Implement dataclasses and conversion helpers**

Create `LoopAnything/src/loop_policy/schema.py` with these public objects:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "loop_policy_dataset_v1"
X_GEOM_DIM = 32


@dataclass(frozen=True)
class LoopPolicyDatasetConfig:
    dataset_root: str
    output_root: str
    sequences: Tuple[str, ...]
    gt_root: Optional[str] = None
    causal: bool = True
    exclude_recent_keyframes: int = 30
    support_window: int = 20
    support_count: int = 1
    min_support_baseline_m: float = 0.5
    retrieval_pool_size: int = 50
    runtime_top_k: int = 4
    write_empty_queries: bool = False
    query_limit: Optional[int] = None
    abs_log_sim3_scale_thr: float = 0.4
    support_align_rmse_thr: float = 1.0
    direction_error_thr_deg: float = 45.0
    loose_rot_thr_deg: float = 30.0
    loose_trans_thr_m: float = 5.0

    def __post_init__(self) -> None:
        if self.retrieval_pool_size < self.runtime_top_k:
            raise ValueError("retrieval_pool_size must be >= runtime_top_k")
        if not self.causal:
            raise ValueError("Loop policy dataset cache requires causal=True")


@dataclass(frozen=True)
class KeyframeRecord:
    keyframe_idx: int
    timestamp: float
    image_path: Optional[str]
    trajectory_idx: Optional[int] = None
    raw: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class PoseRecord:
    timestamp: float
    position: Tuple[float, float, float]
    quaternion_xyzw: Tuple[float, float, float, float]


@dataclass(frozen=True)
class RetrievalCandidate:
    rank: int
    keyframe_idx: int
    timestamp: float
    score: float
    runtime_topk: bool


@dataclass(frozen=True)
class RetrievalRecord:
    sequence: str
    query_idx: int
    query_timestamp: float
    causal: bool
    database_max_idx: Optional[int]
    database_max_timestamp: Optional[float]
    retrieval_db_size: int
    candidates: List[RetrievalCandidate]


@dataclass(frozen=True)
class SupportDecision:
    sequence: str
    query_idx: int
    query_timestamp: float
    candidate_idx: int
    candidate_timestamp: float
    causal: bool
    support_snapshot_max_idx: Optional[int]
    support_snapshot_max_timestamp: Optional[float]
    selected_support_indices: List[int]
    selected_support_timestamps: List[float]
    selected_support_baselines: List[float]
    support_count: int
    rejected: bool
    rejection_reason: Optional[str]


@dataclass(frozen=True)
class CandidateFeatureRecord:
    sequence: str
    query_idx: int
    query_timestamp: float
    candidate_source: str
    candidate_idx: int
    candidate_timestamp: float
    causal: bool
    database_max_idx: Optional[int]
    database_max_timestamp: Optional[float]
    retrieval_db_size: int
    support_snapshot_max_idx: Optional[int]
    support_snapshot_max_timestamp: Optional[float]
    selected_support_indices: List[int]
    selected_support_timestamps: List[float]
    support_count: int
    precondition_valid: bool
    negative_reason: Optional[str]
    x_geom: List[float]
    safe_loop_factor_v1: bool
    labels: Dict[str, Any]
    metrics: Dict[str, Any]

    def __post_init__(self) -> None:
        if len(self.x_geom) != X_GEOM_DIM:
            raise ValueError(f"x_geom must contain {X_GEOM_DIM} values")


def dataclass_to_json_dict(value: Any) -> Any:
    if is_dataclass(value):
        return {key: dataclass_to_json_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [dataclass_to_json_dict(item) for item in value]
    if isinstance(value, tuple):
        return [dataclass_to_json_dict(item) for item in value]
    if isinstance(value, dict):
        return {str(key): dataclass_to_json_dict(item) for key, item in value.items()}
    return value
```

- [ ] **Step 4: Run schema tests and verify they pass**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/loop_policy/test_schema_io.py -q
```

Expected: PASS.

- [ ] **Step 5: Check the edit set**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
git status --short
```

Expected changed paths include `src/loop_policy/schema.py` and `tests/loop_policy/test_schema_io.py`.

## Task 3: AsterSLAM Raw Cache IO

**Files:**
- Create: `LoopAnything/src/loop_policy/io.py`
- Modify: `LoopAnything/tests/loop_policy/test_schema_io.py`
- Create: `LoopAnything/tests/loop_policy/conftest.py`

- [ ] **Step 1: Add a synthetic raw-cache fixture**

Create `LoopAnything/tests/loop_policy/conftest.py`:

```python
import json
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def synthetic_raw_sequence(tmp_path: Path) -> Path:
    raw = tmp_path / "handheld" / "handheld_room01" / "raw"
    images = raw / "keyframe_images"
    images.mkdir(parents=True)

    for idx in range(6):
        Image.new("RGB", (8, 6), color=(idx * 20, 10, 30)).save(images / f"{idx:06d}.jpg")

    (raw / "sequence_meta.json").write_text(
        json.dumps(
            {
                "sequence_name": "handheld_room01",
                "platform": "handheld",
                "loop_closure_enabled": False,
                "image_topic": "/stereo/frame_left/image_raw/compressed",
                "T_camera_lidar": [
                    [1.0, 0.0, 0.0, 0.1],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        )
    )

    lines = []
    for idx in range(6):
        lines.append(
            json.dumps(
                {
                    "keyframe_idx": idx,
                    "timestamp": 100.0 + idx,
                    "trajectory_idx": idx,
                    "image_path": str(images / f"{idx:06d}.jpg"),
                }
            )
        )
    (raw / "keyframes_with_images.jsonl").write_text("\n".join(lines) + "\n")
    (raw / "trajectory.txt").write_text(
        "\n".join(f"{100.0 + idx:.3f} {idx:.3f} 0 0 0 0 0 1" for idx in range(6)) + "\n"
    )
    (raw / "trajectory_indices.txt").write_text("\n".join(str(idx) for idx in range(6)) + "\n")
    (raw / "trajectory_keyframes.txt").write_text((raw / "trajectory.txt").read_text())
    (raw / "trajectory_keyframe_indices.txt").write_text(
        "\n".join(str(idx) for idx in range(6)) + "\n"
    )
    (raw / "keyframes.jsonl").write_text((raw / "keyframes_with_images.jsonl").read_text())
    return raw
```

- [ ] **Step 2: Add IO tests**

Append to `LoopAnything/tests/loop_policy/test_schema_io.py`:

```python
from loop_policy.io import (
    load_aster_raw_sequence,
    read_jsonl,
    read_tum_trajectory,
    write_jsonl,
)


def test_read_jsonl_and_write_jsonl_roundtrip(tmp_path):
    path = tmp_path / "records.jsonl"
    write_jsonl(path, [{"a": 1}, {"b": 2}])

    assert read_jsonl(path) == [{"a": 1}, {"b": 2}]


def test_load_aster_raw_sequence_reads_required_files(synthetic_raw_sequence):
    sequence = load_aster_raw_sequence(synthetic_raw_sequence)

    assert sequence.sequence_name == "handheld_room01"
    assert sequence.platform == "handheld"
    assert sequence.loop_closure_enabled is False
    assert len(sequence.keyframes) == 6
    assert sequence.keyframes[0].keyframe_idx == 0
    assert sequence.keyframes[5].timestamp == 105.0
    assert sequence.t_camera_lidar.shape == (4, 4)


def test_read_tum_trajectory_parses_xyzw_quaternion(synthetic_raw_sequence):
    poses = read_tum_trajectory(synthetic_raw_sequence / "trajectory.txt")

    assert len(poses) == 6
    assert poses[2].timestamp == 102.0
    assert poses[2].position == (2.0, 0.0, 0.0)
    assert poses[2].quaternion_xyzw == (0.0, 0.0, 0.0, 1.0)
```

- [ ] **Step 3: Run IO tests and verify they fail**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/loop_policy/test_schema_io.py -q
```

Expected: FAIL because `loop_policy.io` does not exist.

- [ ] **Step 4: Implement raw-cache IO**

Create `LoopAnything/src/loop_policy/io.py` with these APIs:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from loop_policy.schema import KeyframeRecord, PoseRecord, dataclass_to_json_dict


@dataclass(frozen=True)
class AsterRawSequence:
    raw_dir: Path
    platform: str
    sequence_name: str
    loop_closure_enabled: bool
    image_topic: Optional[str]
    t_camera_lidar: np.ndarray
    keyframes: List[KeyframeRecord]
    trajectory: List[PoseRecord]
    meta: Dict[str, Any]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def write_jsonl(path: Path, records: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dataclass_to_json_dict(record), sort_keys=True) + "\n")


def read_tum_trajectory(path: Path) -> List[PoseRecord]:
    poses: List[PoseRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) != 8:
                raise ValueError(f"{path}:{line_no} expected 8 TUM fields, got {len(parts)}")
            ts, tx, ty, tz, qx, qy, qz, qw = map(float, parts)
            poses.append(
                PoseRecord(
                    timestamp=ts,
                    position=(tx, ty, tz),
                    quaternion_xyzw=(qx, qy, qz, qw),
                )
            )
    return poses


def _read_keyframes(path: Path) -> List[KeyframeRecord]:
    keyframes = []
    for row in read_jsonl(path):
        idx = int(row.get("keyframe_idx", row.get("idx")))
        keyframes.append(
            KeyframeRecord(
                keyframe_idx=idx,
                timestamp=float(row["timestamp"]),
                image_path=row.get("image_path"),
                trajectory_idx=row.get("trajectory_idx"),
                raw=row,
            )
        )
    return keyframes


def load_aster_raw_sequence(raw_dir: Path) -> AsterRawSequence:
    raw_dir = Path(raw_dir)
    meta_path = raw_dir / "sequence_meta.json"
    keyframes_path = raw_dir / "keyframes_with_images.jsonl"
    trajectory_path = raw_dir / "trajectory.txt"

    for required in (meta_path, keyframes_path, trajectory_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    t_camera_lidar = np.asarray(meta["T_camera_lidar"], dtype=np.float64)
    if t_camera_lidar.shape != (4, 4):
        raise ValueError("T_camera_lidar must be a 4x4 matrix")

    sequence_name = meta.get("sequence_name") or raw_dir.parent.name
    platform = meta.get("platform") or raw_dir.parent.parent.name
    return AsterRawSequence(
        raw_dir=raw_dir,
        platform=platform,
        sequence_name=sequence_name,
        loop_closure_enabled=bool(meta.get("loop_closure_enabled", True)),
        image_topic=meta.get("image_topic"),
        t_camera_lidar=t_camera_lidar,
        keyframes=_read_keyframes(keyframes_path),
        trajectory=read_tum_trajectory(trajectory_path),
        meta=meta,
    )
```

- [ ] **Step 5: Run IO tests and verify they pass**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/loop_policy/test_schema_io.py -q
```

Expected: PASS.

- [ ] **Step 6: Check the edit set**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
git status --short
```

Expected changed paths include `src/loop_policy/io.py`, `tests/loop_policy/conftest.py`, and `tests/loop_policy/test_schema_io.py`.

## Task 4: SE3 And Camera Geometry

**Files:**
- Create: `LoopAnything/src/loop_policy/geometry.py`
- Test: `LoopAnything/tests/loop_policy/test_geometry.py`

- [ ] **Step 1: Write geometry tests**

Create `LoopAnything/tests/loop_policy/test_geometry.py`:

```python
import math

import numpy as np

from loop_policy.geometry import (
    camera_pose_from_lidar_pose,
    camera_center_baseline,
    invert_transform,
    make_transform,
    pose_residual,
    quaternion_xyzw_to_matrix,
    relative_transform,
)


def test_quaternion_xyzw_to_matrix_identity():
    rot = quaternion_xyzw_to_matrix((0.0, 0.0, 0.0, 1.0))

    np.testing.assert_allclose(rot, np.eye(3), atol=1e-9)


def test_relative_transform_and_residual_translation():
    a = make_transform(np.eye(3), np.array([0.0, 0.0, 0.0]))
    b = make_transform(np.eye(3), np.array([3.0, 4.0, 0.0]))

    rel = relative_transform(a, b)
    residual = pose_residual(a, b)

    np.testing.assert_allclose(rel[:3, 3], [3.0, 4.0, 0.0], atol=1e-9)
    assert residual.rotation_deg == 0.0
    assert residual.translation_norm == 5.0


def test_pose_residual_rotation_degrees():
    rot_z_90 = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    a = make_transform(np.eye(3), np.zeros(3))
    b = make_transform(rot_z_90, np.zeros(3))

    assert math.isclose(pose_residual(a, b).rotation_deg, 90.0, abs_tol=1e-6)


def test_camera_pose_from_lidar_pose_uses_inverse_t_camera_lidar():
    t_world_lidar = make_transform(np.eye(3), np.array([10.0, 0.0, 0.0]))
    t_camera_lidar = make_transform(np.eye(3), np.array([0.5, 0.0, 0.0]))

    t_world_camera = camera_pose_from_lidar_pose(t_world_lidar, t_camera_lidar)

    np.testing.assert_allclose(t_world_camera[:3, 3], [9.5, 0.0, 0.0], atol=1e-9)


def test_camera_center_baseline():
    a = make_transform(np.eye(3), np.array([1.0, 2.0, 3.0]))
    b = make_transform(np.eye(3), np.array([1.0, 6.0, 3.0]))

    assert camera_center_baseline(a, b) == 4.0
    np.testing.assert_allclose(invert_transform(np.eye(4)), np.eye(4), atol=1e-9)
```

- [ ] **Step 2: Run geometry tests and verify they fail**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/loop_policy/test_geometry.py -q
```

Expected: FAIL because `loop_policy.geometry` does not exist.

- [ ] **Step 3: Implement SE3 helpers**

Create `LoopAnything/src/loop_policy/geometry.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class PoseResidual:
    rotation_deg: float
    translation_norm: float


def quaternion_xyzw_to_matrix(q_xyzw: Sequence[float]) -> np.ndarray:
    qx, qy, qz, qw = [float(v) for v in q_xyzw]
    norm = np.linalg.norm([qx, qy, qz, qw])
    if norm <= 0.0:
        raise ValueError("quaternion norm must be positive")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = transform[:3, :3].T
    inv[:3, 3] = -inv[:3, :3] @ transform[:3, 3]
    return inv


def relative_transform(a_world: np.ndarray, b_world: np.ndarray) -> np.ndarray:
    return invert_transform(a_world) @ b_world


def rotation_angle_deg(rotation: np.ndarray) -> float:
    trace = float(np.trace(rotation))
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def pose_residual(reference: np.ndarray, estimate: np.ndarray) -> PoseResidual:
    delta = relative_transform(reference, estimate)
    return PoseResidual(
        rotation_deg=rotation_angle_deg(delta[:3, :3]),
        translation_norm=float(np.linalg.norm(delta[:3, 3])),
    )


def camera_pose_from_lidar_pose(t_world_lidar: np.ndarray, t_camera_lidar: np.ndarray) -> np.ndarray:
    return np.asarray(t_world_lidar, dtype=np.float64) @ invert_transform(t_camera_lidar)


def camera_center_baseline(a_world_camera: np.ndarray, b_world_camera: np.ndarray) -> float:
    return float(np.linalg.norm(a_world_camera[:3, 3] - b_world_camera[:3, 3]))
```

- [ ] **Step 4: Run geometry tests and verify they pass**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/loop_policy/test_geometry.py -q
```

Expected: PASS.

- [ ] **Step 5: Check the edit set**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
git status --short
```

Expected changed paths include `src/loop_policy/geometry.py` and `tests/loop_policy/test_geometry.py`.

## Task 5: Causal Descriptor Retrieval

**Files:**
- Create: `LoopAnything/src/loop_policy/retrieval.py`
- Test: `LoopAnything/tests/loop_policy/test_retrieval.py`

- [ ] **Step 1: Write causal retrieval tests**

Create `LoopAnything/tests/loop_policy/test_retrieval.py`:

```python
import numpy as np

from loop_policy.retrieval import (
    DescriptorCache,
    causal_retrieval_database,
    normalize_descriptors,
    rank_causal_topk,
)
from loop_policy.geometry import invert_transform
from loop_policy.schema import KeyframeRecord


def _keyframes():
    return [
        KeyframeRecord(keyframe_idx=i, timestamp=100.0 + i, image_path=f"{i}.jpg")
        for i in range(6)
    ]


def test_normalize_descriptors_l2_normalizes_rows():
    descriptors = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float64)

    normalized = normalize_descriptors(descriptors)

    np.testing.assert_allclose(normalized[0], [0.6, 0.8], atol=1e-9)
    np.testing.assert_allclose(normalized[1], [0.0, 0.0], atol=1e-9)


def test_causal_retrieval_database_excludes_future_recent_and_missing_images():
    keyframes = _keyframes()
    keyframes[1] = KeyframeRecord(keyframe_idx=1, timestamp=101.0, image_path=None)

    db = causal_retrieval_database(
        keyframes=keyframes,
        query=keyframes[5],
        exclude_recent_keyframes=2,
    )

    assert [kf.keyframe_idx for kf in db] == [0, 2]


def test_rank_causal_topk_returns_scores_sorted_by_score_then_index():
    keyframes = _keyframes()
    descriptors = normalize_descriptors(
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.8, 0.2],
                [0.8, 0.2],
                [0.3, 0.7],
                [1.0, 0.0],
            ],
            dtype=np.float64,
        )
    )
    cache = DescriptorCache(
        keyframe_idx=np.arange(6),
        timestamps=np.array([kf.timestamp for kf in keyframes], dtype=np.float64),
        descriptors=descriptors,
        normalized=True,
    )

    record = rank_causal_topk(
        sequence="handheld_room01",
        query=keyframes[5],
        keyframes=keyframes,
        descriptors=cache,
        retrieval_pool_size=3,
        runtime_top_k=2,
        exclude_recent_keyframes=1,
    )

    assert record.causal is True
    assert record.database_max_idx == 3
    assert [cand.keyframe_idx for cand in record.candidates] == [0, 2, 3]
    assert [cand.runtime_topk for cand in record.candidates] == [True, True, False]
```

- [ ] **Step 2: Run retrieval tests and verify they fail**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/loop_policy/test_retrieval.py -q
```

Expected: FAIL because `loop_policy.retrieval` does not exist.

- [ ] **Step 3: Implement descriptor cache and causal ranking**

Create `LoopAnything/src/loop_policy/retrieval.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from loop_policy.schema import KeyframeRecord, RetrievalCandidate, RetrievalRecord


@dataclass(frozen=True)
class DescriptorCache:
    keyframe_idx: np.ndarray
    timestamps: np.ndarray
    descriptors: np.ndarray
    normalized: bool

    def index_map(self) -> Dict[int, int]:
        return {int(idx): pos for pos, idx in enumerate(self.keyframe_idx.tolist())}


class DescriptorExtractor:
    def extract(self, image_paths: List[Path]) -> np.ndarray:
        raise NotImplementedError


class PrecomputedDescriptorExtractor(DescriptorExtractor):
    def __init__(self, descriptors_by_path: Dict[str, np.ndarray]):
        self.descriptors_by_path = descriptors_by_path

    def extract(self, image_paths: List[Path]) -> np.ndarray:
        return np.stack([self.descriptors_by_path[str(path)] for path in image_paths], axis=0)


def normalize_descriptors(descriptors: np.ndarray) -> np.ndarray:
    descriptors = np.asarray(descriptors, dtype=np.float64)
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    return np.divide(descriptors, norms, out=np.zeros_like(descriptors), where=norms > 0.0)


def save_descriptor_cache(path: Path, cache: DescriptorCache) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        keyframe_idx=cache.keyframe_idx,
        timestamps=cache.timestamps,
        descriptors=cache.descriptors,
        normalized=np.asarray([cache.normalized], dtype=np.bool_),
    )


def load_descriptor_cache(path: Path) -> DescriptorCache:
    data = np.load(path)
    return DescriptorCache(
        keyframe_idx=data["keyframe_idx"],
        timestamps=data["timestamps"],
        descriptors=data["descriptors"],
        normalized=bool(data["normalized"][0]),
    )


def causal_retrieval_database(
    keyframes: List[KeyframeRecord],
    query: KeyframeRecord,
    exclude_recent_keyframes: int,
) -> List[KeyframeRecord]:
    return [
        keyframe
        for keyframe in keyframes
        if keyframe.timestamp < query.timestamp
        and abs(keyframe.keyframe_idx - query.keyframe_idx) > exclude_recent_keyframes
        and keyframe.image_path is not None
    ]


def rank_causal_topk(
    sequence: str,
    query: KeyframeRecord,
    keyframes: List[KeyframeRecord],
    descriptors: DescriptorCache,
    retrieval_pool_size: int,
    runtime_top_k: int,
    exclude_recent_keyframes: int,
) -> RetrievalRecord:
    index = descriptors.index_map()
    db = causal_retrieval_database(keyframes, query, exclude_recent_keyframes)
    if query.keyframe_idx not in index:
        raise ValueError(f"missing descriptor for query keyframe {query.keyframe_idx}")
    query_descriptor = descriptors.descriptors[index[query.keyframe_idx]]

    scored = []
    for keyframe in db:
        if keyframe.keyframe_idx not in index:
            continue
        candidate_descriptor = descriptors.descriptors[index[keyframe.keyframe_idx]]
        score = float(np.dot(query_descriptor, candidate_descriptor))
        scored.append((score, keyframe.keyframe_idx, keyframe.timestamp))

    scored.sort(key=lambda item: (-item[0], item[1]))
    candidates = [
        RetrievalCandidate(
            rank=rank,
            keyframe_idx=int(idx),
            timestamp=float(timestamp),
            score=float(score),
            runtime_topk=rank <= runtime_top_k,
        )
        for rank, (score, idx, timestamp) in enumerate(scored[:retrieval_pool_size], start=1)
    ]
    return RetrievalRecord(
        sequence=sequence,
        query_idx=query.keyframe_idx,
        query_timestamp=query.timestamp,
        causal=True,
        database_max_idx=max((kf.keyframe_idx for kf in db), default=None),
        database_max_timestamp=max((kf.timestamp for kf in db), default=None),
        retrieval_db_size=len(db),
        candidates=candidates,
    )
```

- [ ] **Step 4: Run retrieval tests and verify they pass**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/loop_policy/test_retrieval.py -q
```

Expected: PASS.

- [ ] **Step 5: Check the edit set**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
git status --short
```

Expected changed paths include `src/loop_policy/retrieval.py` and `tests/loop_policy/test_retrieval.py`.

## Task 6: Causal Support Selection

**Files:**
- Create: `LoopAnything/src/loop_policy/support.py`
- Test: `LoopAnything/tests/loop_policy/test_support.py`

- [ ] **Step 1: Write support-selection tests**

Create `LoopAnything/tests/loop_policy/test_support.py`:

```python
import numpy as np

from loop_policy.geometry import make_transform
from loop_policy.schema import KeyframeRecord
from loop_policy.support import select_supports


def _keyframes():
    return [
        KeyframeRecord(keyframe_idx=i, timestamp=100.0 + i, image_path=f"{i}.jpg")
        for i in range(8)
    ]


def _camera_poses():
    return {i: make_transform(np.eye(3), np.array([float(i), 0.0, 0.0])) for i in range(8)}


def test_select_supports_is_causal_and_sorts_by_baseline_desc_then_index():
    keyframes = _keyframes()
    query = keyframes[7]
    candidate = keyframes[3]

    decision = select_supports(
        sequence="handheld_room01",
        query=query,
        candidate=candidate,
        keyframes=keyframes,
        camera_poses_by_idx=_camera_poses(),
        support_window=3,
        support_count=2,
        exclude_recent_keyframes=1,
        min_support_baseline_m=0.5,
    )

    assert decision.rejected is False
    assert decision.selected_support_indices == [0, 6]
    assert decision.selected_support_timestamps == [100.0, 106.0]
    assert decision.support_count == 2
    assert decision.support_snapshot_max_idx == 6


def test_select_supports_rejects_candidate_without_valid_support():
    keyframes = _keyframes()
    query = keyframes[2]
    candidate = keyframes[0]

    decision = select_supports(
        sequence="handheld_room01",
        query=query,
        candidate=candidate,
        keyframes=keyframes,
        camera_poses_by_idx=_camera_poses(),
        support_window=1,
        support_count=1,
        exclude_recent_keyframes=10,
        min_support_baseline_m=0.5,
    )

    assert decision.rejected is True
    assert decision.rejection_reason == "no_valid_support"
    assert decision.selected_support_indices == []
```

- [ ] **Step 2: Run support tests and verify they fail**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/loop_policy/test_support.py -q
```

Expected: FAIL because `loop_policy.support` does not exist.

- [ ] **Step 3: Implement support selection**

Create `LoopAnything/src/loop_policy/support.py`:

```python
from __future__ import annotations

from typing import Dict, List

import numpy as np

from loop_policy.geometry import camera_center_baseline
from loop_policy.schema import KeyframeRecord, SupportDecision


def select_supports(
    sequence: str,
    query: KeyframeRecord,
    candidate: KeyframeRecord,
    keyframes: List[KeyframeRecord],
    camera_poses_by_idx: Dict[int, np.ndarray],
    support_window: int,
    support_count: int,
    exclude_recent_keyframes: int,
    min_support_baseline_m: float,
) -> SupportDecision:
    candidate_pose = camera_poses_by_idx[candidate.keyframe_idx]
    scored = []
    snapshot = [
        keyframe
        for keyframe in keyframes
        if keyframe.timestamp < query.timestamp and keyframe.image_path is not None
    ]

    for support in snapshot:
        if support.keyframe_idx == candidate.keyframe_idx:
            continue
        if not (candidate.keyframe_idx - support_window <= support.keyframe_idx <= candidate.keyframe_idx + support_window):
            continue
        if abs(query.keyframe_idx - support.keyframe_idx) <= exclude_recent_keyframes:
            continue
        if support.keyframe_idx not in camera_poses_by_idx:
            continue
        baseline = camera_center_baseline(candidate_pose, camera_poses_by_idx[support.keyframe_idx])
        if baseline < min_support_baseline_m:
            continue
        scored.append((baseline, support.keyframe_idx, support.timestamp))

    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = scored[:support_count]
    rejected = len(selected) == 0

    return SupportDecision(
        sequence=sequence,
        query_idx=query.keyframe_idx,
        query_timestamp=query.timestamp,
        candidate_idx=candidate.keyframe_idx,
        candidate_timestamp=candidate.timestamp,
        causal=True,
        support_snapshot_max_idx=max((kf.keyframe_idx for kf in snapshot), default=None),
        support_snapshot_max_timestamp=max((kf.timestamp for kf in snapshot), default=None),
        selected_support_indices=[int(idx) for _, idx, _ in selected],
        selected_support_timestamps=[float(timestamp) for _, _, timestamp in selected],
        selected_support_baselines=[float(baseline) for baseline, _, _ in selected],
        support_count=len(selected),
        rejected=rejected,
        rejection_reason="no_valid_support" if rejected else None,
    )
```

- [ ] **Step 4: Run support tests and verify they pass**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/loop_policy/test_support.py -q
```

Expected: PASS.

- [ ] **Step 5: Check the edit set**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
git status --short
```

Expected changed paths include `src/loop_policy/support.py` and `tests/loop_policy/test_support.py`.

## Task 7: DA3 Runner Interface And Candidate-Local Groups

**Files:**
- Create: `LoopAnything/src/loop_policy/da3_runner.py`
- Test: `LoopAnything/tests/loop_policy/test_da3_runner.py`

- [ ] **Step 1: Write DA3 group tests**

Create `LoopAnything/tests/loop_policy/test_da3_runner.py`:

```python
import numpy as np

from loop_policy.da3_runner import Da3Group, MockDa3Runner, build_da3_group
from loop_policy.schema import KeyframeRecord


def test_build_da3_group_is_candidate_local():
    query = KeyframeRecord(10, 110.0, "query.jpg")
    candidate = KeyframeRecord(4, 104.0, "candidate.jpg")
    supports = [KeyframeRecord(2, 102.0, "support.jpg")]

    group = build_da3_group(query, candidate, supports)

    assert group.keyframe_indices == [10, 4, 2]
    assert group.image_paths == ["query.jpg", "candidate.jpg", "support.jpg"]


def test_mock_da3_runner_returns_one_result_per_group():
    groups = [
        Da3Group(keyframe_indices=[10, 4, 2], image_paths=["q.jpg", "c.jpg", "s.jpg"]),
        Da3Group(keyframe_indices=[10, 3, 1], image_paths=["q.jpg", "c2.jpg", "s2.jpg"]),
    ]

    results = MockDa3Runner().run(groups)

    assert len(results) == 2
    assert results[0].camera_poses.shape == (3, 4, 4)
    assert results[0].depth_conf_medians == [1.0, 1.0, 1.0]
    assert results[0].valid_depth_ratios == [1.0, 1.0, 1.0]
    np.testing.assert_allclose(results[0].camera_poses[0], np.eye(4), atol=1e-9)
```

- [ ] **Step 2: Run DA3 runner tests and verify they fail**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/loop_policy/test_da3_runner.py -q
```

Expected: FAIL because `loop_policy.da3_runner` does not exist.

- [ ] **Step 3: Implement group dataclasses, mock runner, and real-runner boundary**

Create `LoopAnything/src/loop_policy/da3_runner.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

from loop_policy.schema import KeyframeRecord


@dataclass(frozen=True)
class Da3Group:
    keyframe_indices: List[int]
    image_paths: List[str]


@dataclass(frozen=True)
class Da3GroupResult:
    keyframe_indices: List[int]
    camera_poses: np.ndarray
    depth_conf_medians: List[float]
    valid_depth_ratios: List[float]


def build_da3_group(
    query: KeyframeRecord,
    candidate: KeyframeRecord,
    supports: Sequence[KeyframeRecord],
) -> Da3Group:
    keyframes = [query, candidate, *supports]
    if any(keyframe.image_path is None for keyframe in keyframes):
        raise ValueError("DA3 group requires image_path for query, candidate, and supports")
    return Da3Group(
        keyframe_indices=[keyframe.keyframe_idx for keyframe in keyframes],
        image_paths=[str(keyframe.image_path) for keyframe in keyframes],
    )


class Da3Runner:
    def run(self, groups: List[Da3Group]) -> List[Da3GroupResult]:
        raise NotImplementedError


class MockDa3Runner(Da3Runner):
    def run(self, groups: List[Da3Group]) -> List[Da3GroupResult]:
        results: List[Da3GroupResult] = []
        for group in groups:
            poses = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], len(group.keyframe_indices), axis=0)
            for view_idx in range(len(group.keyframe_indices)):
                poses[view_idx, 0, 3] = float(view_idx)
            results.append(
                Da3GroupResult(
                    keyframe_indices=group.keyframe_indices,
                    camera_poses=poses,
                    depth_conf_medians=[1.0] * len(group.keyframe_indices),
                    valid_depth_ratios=[1.0] * len(group.keyframe_indices),
                )
            )
        return results


def _as_44(extrinsics: np.ndarray) -> np.ndarray:
    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    if extrinsics.shape[-2:] == (4, 4):
        return extrinsics
    if extrinsics.shape[-2:] == (3, 4):
        padded = np.repeat(np.eye(4, dtype=np.float64)[None], extrinsics.shape[0], axis=0)
        padded[:, :3, :4] = extrinsics
        return padded
    raise ValueError(f"expected extrinsics with shape [N, 4, 4] or [N, 3, 4], got {extrinsics.shape}")


def _confidence_stats(prediction, view_count: int) -> tuple[List[float], List[float]]:
    if prediction.conf is None:
        return [1.0] * view_count, [1.0] * view_count
    medians = []
    ratios = []
    for conf in prediction.conf:
        finite = np.isfinite(conf)
        positive = finite & (conf > 0.0)
        medians.append(float(np.median(conf[positive])) if positive.any() else 0.0)
        ratios.append(float(positive.mean()) if finite.any() else 0.0)
    return medians, ratios


class DepthAnything3Runner(Da3Runner):
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        extrinsics_are_c2w: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.process_res = process_res
        self.process_res_method = process_res_method
        self.extrinsics_are_c2w = extrinsics_are_c2w
        self._model = None

    def _load_model(self):
        if self._model is None:
            from depth_anything_3.api import DepthAnything3

            self._model = DepthAnything3.from_pretrained(self.model_name).to(self.device).eval()
        return self._model

    def run(self, groups: List[Da3Group]) -> List[Da3GroupResult]:
        model = self._load_model()
        results: List[Da3GroupResult] = []
        for group in groups:
            prediction = model.inference(
                group.image_paths,
                align_to_input_ext_scale=False,
                process_res=self.process_res,
                process_res_method=self.process_res_method,
                export_format="mini_npz",
                export_dir=None,
            )
            if prediction.extrinsics is None:
                raise RuntimeError("DA3 prediction did not include extrinsics")
            poses = _as_44(prediction.extrinsics)
            if not self.extrinsics_are_c2w:
                poses = np.stack([invert_transform(pose) for pose in poses], axis=0)
            medians, ratios = _confidence_stats(prediction, len(group.keyframe_indices))
            results.append(
                Da3GroupResult(
                    keyframe_indices=group.keyframe_indices,
                    camera_poses=poses,
                    depth_conf_medians=medians,
                    valid_depth_ratios=ratios,
                )
            )
        return results
```

- [ ] **Step 4: Run DA3 runner tests and verify they pass**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/loop_policy/test_da3_runner.py -q
```

Expected: PASS.

- [ ] **Step 5: Check the edit set**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
git status --short
```

Expected changed paths include `src/loop_policy/da3_runner.py` and `tests/loop_policy/test_da3_runner.py`.

## Task 8: Sim3 Prior Alignment

**Files:**
- Create: `LoopAnything/src/loop_policy/sim3_prior.py`
- Test: `LoopAnything/tests/loop_policy/test_sim3_prior.py`

- [ ] **Step 1: Write Sim3 prior tests**

Create `LoopAnything/tests/loop_policy/test_sim3_prior.py`:

```python
import math

import numpy as np

from loop_policy.geometry import make_transform
from loop_policy.sim3_prior import Sim3PriorConfig, align_da3_poses_with_candidate_support_prior


def _pose(x):
    return make_transform(np.eye(3), np.array([x, 0.0, 0.0]))


def test_sim3_prior_recovers_scale_from_candidate_support_baseline():
    da3 = {
        "query": _pose(4.0),
        "candidate": _pose(0.0),
        "supports": [_pose(2.0)],
    }
    odom = {
        "query": _pose(8.0),
        "candidate": _pose(0.0),
        "supports": [_pose(4.0)],
    }

    result = align_da3_poses_with_candidate_support_prior(
        da3_query_pose=da3["query"],
        da3_candidate_pose=da3["candidate"],
        da3_support_poses=da3["supports"],
        odom_query_pose=odom["query"],
        odom_candidate_pose=odom["candidate"],
        odom_support_poses=odom["supports"],
        config=Sim3PriorConfig(),
    )

    assert result.accepted is True
    assert math.isclose(result.sim3_scale, 2.0, rel_tol=1e-6)
    assert math.isclose(result.aligned_loop.translation_norm, 8.0, rel_tol=1e-6)
    assert result.rejection_reason is None


def test_sim3_prior_rejects_zero_da3_support_baseline():
    result = align_da3_poses_with_candidate_support_prior(
        da3_query_pose=_pose(1.0),
        da3_candidate_pose=_pose(0.0),
        da3_support_poses=[_pose(0.0)],
        odom_query_pose=_pose(1.0),
        odom_candidate_pose=_pose(0.0),
        odom_support_poses=[_pose(1.0)],
        config=Sim3PriorConfig(),
    )

    assert result.accepted is False
    assert result.rejection_reason == "invalid_support_baseline"
```

- [ ] **Step 2: Run Sim3 tests and verify they fail**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/loop_policy/test_sim3_prior.py -q
```

Expected: FAIL because `loop_policy.sim3_prior` does not exist.

- [ ] **Step 3: Implement single-support Sim3 prior first**

Create `LoopAnything/src/loop_policy/sim3_prior.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from loop_policy.geometry import PoseResidual, camera_center_baseline, pose_residual


@dataclass(frozen=True)
class Sim3PriorConfig:
    min_da3_support_baseline_m: float = 1e-6
    max_support_align_rmse_m: float = 1.0
    max_direction_error_deg: float = 45.0


@dataclass(frozen=True)
class Sim3PriorResult:
    accepted: bool
    rejection_reason: Optional[str]
    sim3_scale: float
    abs_log_sim3_scale: float
    support_align_rmse: float
    direction_error_deg: float
    aligned_loop: PoseResidual
    aligned_vs_odom: PoseResidual
    aligned_query_pose: np.ndarray


def _unit_direction(a: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    vec = b[:3, 3] - a[:3, 3]
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return None
    return vec / norm


def _direction_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def align_da3_poses_with_candidate_support_prior(
    da3_query_pose: np.ndarray,
    da3_candidate_pose: np.ndarray,
    da3_support_poses: List[np.ndarray],
    odom_query_pose: np.ndarray,
    odom_candidate_pose: np.ndarray,
    odom_support_poses: List[np.ndarray],
    config: Sim3PriorConfig,
) -> Sim3PriorResult:
    if len(da3_support_poses) != len(odom_support_poses) or not da3_support_poses:
        raise ValueError("DA3 and odom supports must have the same positive length")

    da3_baselines = np.array(
        [camera_center_baseline(da3_candidate_pose, support) for support in da3_support_poses],
        dtype=np.float64,
    )
    odom_baselines = np.array(
        [camera_center_baseline(odom_candidate_pose, support) for support in odom_support_poses],
        dtype=np.float64,
    )
    if np.any(da3_baselines <= config.min_da3_support_baseline_m):
        scale = 0.0
        reason = "invalid_support_baseline"
    else:
        scale = float(np.median(odom_baselines / da3_baselines))
        reason = None

    aligned_query = odom_candidate_pose.copy()
    aligned_query[:3, :3] = odom_candidate_pose[:3, :3] @ da3_candidate_pose[:3, :3].T @ da3_query_pose[:3, :3]
    aligned_query[:3, 3] = odom_candidate_pose[:3, 3] + scale * (
        da3_query_pose[:3, 3] - da3_candidate_pose[:3, 3]
    )

    support_errors = []
    direction_errors = []
    for da3_support, odom_support in zip(da3_support_poses, odom_support_poses):
        aligned_support_position = odom_candidate_pose[:3, 3] + scale * (
            da3_support[:3, 3] - da3_candidate_pose[:3, 3]
        )
        support_errors.append(float(np.linalg.norm(aligned_support_position - odom_support[:3, 3])))
        da3_dir = _unit_direction(da3_candidate_pose, da3_support)
        odom_dir = _unit_direction(odom_candidate_pose, odom_support)
        if da3_dir is None or odom_dir is None:
            direction_errors.append(180.0)
        else:
            direction_errors.append(_direction_error_deg(da3_dir, odom_dir))

    support_align_rmse = float(np.sqrt(np.mean(np.square(support_errors)))) if support_errors else float("inf")
    direction_error_deg = float(max(direction_errors)) if direction_errors else 180.0
    abs_log_scale = float(abs(np.log(scale))) if scale > 0.0 else float("inf")

    if reason is None and support_align_rmse > config.max_support_align_rmse_m:
        reason = "support_align_rmse_too_large"
    if reason is None and direction_error_deg > config.max_direction_error_deg:
        reason = "direction_error_too_large"

    return Sim3PriorResult(
        accepted=reason is None,
        rejection_reason=reason,
        sim3_scale=scale,
        abs_log_sim3_scale=abs_log_scale,
        support_align_rmse=support_align_rmse,
        direction_error_deg=direction_error_deg,
        aligned_loop=pose_residual(odom_candidate_pose, aligned_query),
        aligned_vs_odom=pose_residual(odom_query_pose, aligned_query),
        aligned_query_pose=aligned_query,
    )
```

- [ ] **Step 4: Run Sim3 tests and verify they pass**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/loop_policy/test_sim3_prior.py -q
```

Expected: PASS.

- [ ] **Step 5: Check against AsterSLAM semantics before later runtime use**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/loop_policy/test_sim3_prior.py -q
```

Expected: PASS. Before using real training output, compare this module against AsterSLAM's `alignDa3PosesWithCandidateSupportPrior` on one known exported candidate and adjust this module if a sign convention mismatch is found.

Current AsterSLAM alignment status as of 2026-05-13:

- Single-support/default runtime behavior is geometrically aligned with AsterSLAM for the Sim3 rotation convention, candidate-anchored translation, DA3 offset rotation, and finite-pose rejection.
- Scale bounds are intentionally no longer a hard rejection in the offline learned-policy builder. DA3 has no metric translation scale, so finite positive Sim3 scale is retained as a feature/diagnostic instead of rejecting as `sim3_scale_out_of_range`.
- Multi-support behavior is intentionally not fully aligned yet. This Python module estimates scale with the median of all candidate-support baseline ratios, while AsterSLAM uses the first support as the scale/direction anchor.
- Multi-support RMSE semantics also need an explicit decision before enabling `support_count > 1`: this Python module computes RMSE across all supports, while AsterSLAM's implementation is structured around the first support as the Sim3 anchor and additional supports as consistency checks.
- Default alignment RMSE threshold is not fully aligned: this Python plan/module uses `max_support_align_rmse_m=1.0`, while AsterSLAM's `Da3PosePriorAlignmentConfig::max_prior_alignment_rmse` defaults to `2.0`.
- Do not treat Task 8 as a byte-for-byte or multi-support semantic port until these differences are resolved. With the current default `support_count=1`, the divergence is latent for runtime smoke tests.

- [ ] **Step 6: Check the edit set**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
git status --short
```

Expected changed paths include `src/loop_policy/sim3_prior.py` and `tests/loop_policy/test_sim3_prior.py`.

## Task 9: Labels And `x_geom[32]`

**Files:**
- Create: `LoopAnything/src/loop_policy/labels.py`
- Test: `LoopAnything/tests/loop_policy/test_labels.py`

- [ ] **Step 1: Write label and feature tests**

Create `LoopAnything/tests/loop_policy/test_labels.py`:

```python
from loop_policy.labels import build_x_geom, compute_safe_loop_factor_v1
from loop_policy.schema import LoopPolicyDatasetConfig


def test_compute_safe_loop_factor_requires_all_boolean_parts():
    config = LoopPolicyDatasetConfig(dataset_root="/dataset", output_root="/loop_policy_dataset", sequences=("s",))
    metrics = {
        "abs_log_sim3_scale": 0.1,
        "support_align_rmse": 0.2,
        "direction_error_deg": 10.0,
        "aligned_vs_odom_rot_residual_deg": 5.0,
        "aligned_vs_odom_trans_residual_norm": 0.3,
    }

    label = compute_safe_loop_factor_v1(precondition_valid=True, metrics=metrics, config=config)

    assert label["sim3_quality_good"] is True
    assert label["odom_consistent_loose"] is True
    assert label["safe_loop_factor_v1"] is True


def test_compute_safe_loop_factor_fails_precondition():
    config = LoopPolicyDatasetConfig(dataset_root="/dataset", output_root="/loop_policy_dataset", sequences=("s",))

    label = compute_safe_loop_factor_v1(precondition_valid=False, metrics={}, config=config)

    assert label["safe_loop_factor_v1"] is False
    assert label["sim3_quality_good"] is False


def test_build_x_geom_uses_primary_support_and_has_32_values():
    values = {
        "rank_norm": 0.25,
        "salad_score_qc": 0.9,
        "salad_score_qs": 0.7,
        "salad_score_cs": 0.8,
        "salad_score_qc_minus_top1": 0.0,
        "salad_score_qc_minus_topk": 0.2,
        "da3_rot_qc_deg": 1.0,
        "da3_trans_qc_norm": 2.0,
        "da3_rot_cs_deg": 3.0,
        "da3_trans_cs_norm": 4.0,
        "da3_rot_qs_deg": 5.0,
        "da3_trans_qs_norm": 6.0,
        "odom_rot_qc_deg": 7.0,
        "odom_trans_qc_norm": 8.0,
        "odom_rot_cs_deg": 9.0,
        "odom_trans_cs_norm": 10.0,
        "support_baseline": 11.0,
        "sim3_scale": 1.2,
        "abs_log_sim3_scale": 0.18,
        "support_align_rmse": 0.1,
        "direction_error_deg": 12.0,
        "aligned_vs_odom_rot_residual_deg": 13.0,
        "aligned_vs_odom_trans_residual_norm": 14.0,
        "aligned_loop_rot_deg": 15.0,
        "aligned_loop_trans_norm": 16.0,
        "q_depth_conf_median": 0.91,
        "c_depth_conf_median": 0.92,
        "s_depth_conf_median": 0.93,
        "q_valid_depth_ratio": 0.81,
        "c_valid_depth_ratio": 0.82,
        "s_valid_depth_ratio": 0.83,
        "min_depth_conf_median": 0.91,
    }

    x_geom = build_x_geom(values)

    assert len(x_geom) == 32
    assert x_geom[0] == 0.25
    assert x_geom[16] == 11.0
    assert x_geom[31] == 0.91
```

- [ ] **Step 2: Run label tests and verify they fail**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/loop_policy/test_labels.py -q
```

Expected: FAIL because `loop_policy.labels` does not exist.

- [ ] **Step 3: Implement label and feature assembly**

Create `LoopAnything/src/loop_policy/labels.py`:

```python
from __future__ import annotations

import math
from typing import Any, Dict, List

from loop_policy.schema import LoopPolicyDatasetConfig, X_GEOM_DIM


X_GEOM_FIELDS = [
    "rank_norm",
    "salad_score_qc",
    "salad_score_qs",
    "salad_score_cs",
    "salad_score_qc_minus_top1",
    "salad_score_qc_minus_topk",
    "da3_rot_qc_deg",
    "da3_trans_qc_norm",
    "da3_rot_cs_deg",
    "da3_trans_cs_norm",
    "da3_rot_qs_deg",
    "da3_trans_qs_norm",
    "odom_rot_qc_deg",
    "odom_trans_qc_norm",
    "odom_rot_cs_deg",
    "odom_trans_cs_norm",
    "support_baseline",
    "sim3_scale",
    "abs_log_sim3_scale",
    "support_align_rmse",
    "direction_error_deg",
    "aligned_vs_odom_rot_residual_deg",
    "aligned_vs_odom_trans_residual_norm",
    "aligned_loop_rot_deg",
    "aligned_loop_trans_norm",
    "q_depth_conf_median",
    "c_depth_conf_median",
    "s_depth_conf_median",
    "q_valid_depth_ratio",
    "c_valid_depth_ratio",
    "s_valid_depth_ratio",
    "min_depth_conf_median",
]


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def build_x_geom(values: Dict[str, Any]) -> List[float]:
    missing = [field for field in X_GEOM_FIELDS if field not in values]
    if missing:
        raise ValueError(f"missing x_geom fields: {missing}")
    x_geom = [float(values[field]) for field in X_GEOM_FIELDS]
    if len(x_geom) != X_GEOM_DIM:
        raise ValueError(f"x_geom must contain {X_GEOM_DIM} values")
    if not all(math.isfinite(value) for value in x_geom):
        raise ValueError("x_geom contains non-finite values")
    return x_geom


def compute_safe_loop_factor_v1(
    precondition_valid: bool,
    metrics: Dict[str, Any],
    config: LoopPolicyDatasetConfig,
) -> Dict[str, bool]:
    sim3_quality_good = (
        _finite(metrics.get("support_align_rmse"))
        and _finite(metrics.get("direction_error_deg"))
        and float(metrics["support_align_rmse"]) <= config.support_align_rmse_thr
        and float(metrics["direction_error_deg"]) <= config.direction_error_thr_deg
    )
    odom_consistent_loose = (
        _finite(metrics.get("aligned_vs_odom_rot_residual_deg"))
        and _finite(metrics.get("aligned_vs_odom_trans_residual_norm"))
        and float(metrics["aligned_vs_odom_rot_residual_deg"]) <= config.loose_rot_thr_deg
        and float(metrics["aligned_vs_odom_trans_residual_norm"]) <= config.loose_trans_thr_m
    )
    return {
        "sim3_quality_good": bool(sim3_quality_good),
        "odom_consistent_loose": bool(odom_consistent_loose),
        "safe_loop_factor_v1": bool(
            precondition_valid and sim3_quality_good and odom_consistent_loose
        ),
    }
```

- [ ] **Step 4: Run label tests and verify they pass**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/loop_policy/test_labels.py -q
```

Expected: PASS.

- [ ] **Step 5: Check the edit set**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
git status --short
```

Expected changed paths include `src/loop_policy/labels.py` and `tests/loop_policy/test_labels.py`.

## Task 10: Dataset Builder Orchestration With Mock Backends

**Files:**
- Create: `LoopAnything/src/loop_policy/dataset_builder.py`
- Test: `LoopAnything/tests/loop_policy/test_dataset_builder.py`

- [ ] **Step 1: Write integration test over the synthetic sequence**

Create `LoopAnything/tests/loop_policy/test_dataset_builder.py`:

```python
import json

import numpy as np

from loop_policy.da3_runner import MockDa3Runner
from loop_policy.retrieval import PrecomputedDescriptorExtractor
from loop_policy.schema import LoopPolicyDatasetConfig
from loop_policy.dataset_builder import build_sequence_cache


def test_build_sequence_cache_writes_causal_outputs(synthetic_raw_sequence, tmp_path):
    image_paths = [str(synthetic_raw_sequence / "keyframe_images" / f"{idx:06d}.jpg") for idx in range(6)]
    descriptors_by_path = {
        image_paths[0]: np.array([1.0, 0.0]),
        image_paths[1]: np.array([0.0, 1.0]),
        image_paths[2]: np.array([0.8, 0.2]),
        image_paths[3]: np.array([0.7, 0.3]),
        image_paths[4]: np.array([0.1, 0.9]),
        image_paths[5]: np.array([1.0, 0.0]),
    }
    config = LoopPolicyDatasetConfig(
        dataset_root=str(synthetic_raw_sequence.parents[2]),
        output_root=str(tmp_path / "loop_policy_dataset"),
        sequences=("handheld_room01",),
        exclude_recent_keyframes=1,
        support_window=4,
        min_support_baseline_m=0.5,
        retrieval_pool_size=3,
        runtime_top_k=2,
    )

    summary = build_sequence_cache(
        raw_dir=synthetic_raw_sequence,
        config=config,
        descriptor_extractor=PrecomputedDescriptorExtractor(descriptors_by_path),
        da3_runner=MockDa3Runner(),
    )

    sequence_dir = tmp_path / "loop_policy_dataset" / "handheld" / "handheld_room01"
    assert summary["sequence"] == "handheld_room01"
    assert (sequence_dir / "sequence_index.json").is_file()
    assert (sequence_dir / "descriptors.npz").is_file()
    assert (sequence_dir / "retrieval_topk.jsonl").is_file()
    assert (sequence_dir / "support_selection.jsonl").is_file()
    assert (sequence_dir / "candidate_features.jsonl").is_file()
    assert (sequence_dir / "sequence_summary.json").is_file()

    for line in (sequence_dir / "candidate_features.jsonl").read_text().splitlines():
        record = json.loads(line)
        assert record["causal"] is True
        assert record["candidate_timestamp"] < record["query_timestamp"]
        for timestamp in record["selected_support_timestamps"]:
            assert timestamp < record["query_timestamp"]
```

- [ ] **Step 2: Run builder test and verify it fails**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/loop_policy/test_dataset_builder.py -q
```

Expected: FAIL because `loop_policy.dataset_builder` does not exist.

- [ ] **Step 3: Implement builder helper functions and CLI shell**

Create `LoopAnything/src/loop_policy/dataset_builder.py` with these public APIs:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from loop_policy.da3_runner import Da3Runner, MockDa3Runner, build_da3_group
from loop_policy.geometry import (
    camera_pose_from_lidar_pose,
    make_transform,
    pose_residual,
    quaternion_xyzw_to_matrix,
)
from loop_policy.io import load_aster_raw_sequence, write_jsonl
from loop_policy.labels import build_x_geom, compute_safe_loop_factor_v1
from loop_policy.retrieval import (
    DescriptorExtractor,
    DescriptorCache,
    normalize_descriptors,
    rank_causal_topk,
    save_descriptor_cache,
)
from loop_policy.schema import CandidateFeatureRecord, KeyframeRecord, LoopPolicyDatasetConfig, dataclass_to_json_dict
from loop_policy.sim3_prior import Sim3PriorConfig, align_da3_poses_with_candidate_support_prior
from loop_policy.support import select_supports


def _pose_map(sequence) -> Dict[int, np.ndarray]:
    poses = {}
    for keyframe, pose in zip(sequence.keyframes, sequence.trajectory):
        t_world_lidar = make_transform(
            quaternion_xyzw_to_matrix(pose.quaternion_xyzw),
            np.asarray(pose.position, dtype=np.float64),
        )
        poses[keyframe.keyframe_idx] = camera_pose_from_lidar_pose(
            t_world_lidar,
            sequence.t_camera_lidar,
        )
    return poses


def _descriptor_cache(sequence, extractor: DescriptorExtractor) -> DescriptorCache:
    image_keyframes = [keyframe for keyframe in sequence.keyframes if keyframe.image_path is not None]
    image_paths = [Path(keyframe.image_path) for keyframe in image_keyframes]
    descriptors = normalize_descriptors(extractor.extract(image_paths))
    return DescriptorCache(
        keyframe_idx=np.asarray([keyframe.keyframe_idx for keyframe in image_keyframes], dtype=np.int64),
        timestamps=np.asarray([keyframe.timestamp for keyframe in image_keyframes], dtype=np.float64),
        descriptors=descriptors,
        normalized=True,
    )


def _keyframe_by_idx(keyframes: List[KeyframeRecord]) -> Dict[int, KeyframeRecord]:
    return {keyframe.keyframe_idx: keyframe for keyframe in keyframes}


def build_sequence_cache(
    raw_dir: Path,
    config: LoopPolicyDatasetConfig,
    descriptor_extractor: DescriptorExtractor,
    da3_runner: Optional[Da3Runner] = None,
) -> Dict[str, object]:
    sequence = load_aster_raw_sequence(raw_dir)
    da3_runner = da3_runner or MockDa3Runner()
    output_dir = Path(config.output_root) / sequence.platform / sequence.sequence_name
    output_dir.mkdir(parents=True, exist_ok=True)

    descriptors = _descriptor_cache(sequence, descriptor_extractor)
    save_descriptor_cache(output_dir / "descriptors.npz", descriptors)
    camera_poses = _pose_map(sequence)
    keyframes_by_idx = _keyframe_by_idx(sequence.keyframes)

    retrieval_records = []
    support_records = []
    feature_records = []
    negative_reasons: Dict[str, int] = {}

    query_keyframes = [keyframe for keyframe in sequence.keyframes if keyframe.image_path is not None]
    if config.query_limit is not None:
        query_keyframes = query_keyframes[: config.query_limit]

    for query in query_keyframes:
        retrieval = rank_causal_topk(
            sequence=sequence.sequence_name,
            query=query,
            keyframes=sequence.keyframes,
            descriptors=descriptors,
            retrieval_pool_size=config.retrieval_pool_size,
            runtime_top_k=config.runtime_top_k,
            exclude_recent_keyframes=config.exclude_recent_keyframes,
        )
        retrieval_records.append(retrieval)

        for candidate in retrieval.candidates:
            candidate_kf = keyframes_by_idx[candidate.keyframe_idx]
            support = select_supports(
                sequence=sequence.sequence_name,
                query=query,
                candidate=candidate_kf,
                keyframes=sequence.keyframes,
                camera_poses_by_idx=camera_poses,
                support_window=config.support_window,
                support_count=config.support_count,
                exclude_recent_keyframes=config.exclude_recent_keyframes,
                min_support_baseline_m=config.min_support_baseline_m,
            )
            support_records.append(support)
            if support.rejected:
                negative_reasons[support.rejection_reason or "support_rejected"] = (
                    negative_reasons.get(support.rejection_reason or "support_rejected", 0) + 1
                )
                continue

            supports = [keyframes_by_idx[idx] for idx in support.selected_support_indices]
            da3_result = da3_runner.run([build_da3_group(query, candidate_kf, supports)])[0]
            sim3 = align_da3_poses_with_candidate_support_prior(
                da3_query_pose=da3_result.camera_poses[0],
                da3_candidate_pose=da3_result.camera_poses[1],
                da3_support_poses=[da3_result.camera_poses[2]],
                odom_query_pose=camera_poses[query.keyframe_idx],
                odom_candidate_pose=camera_poses[candidate_kf.keyframe_idx],
                odom_support_poses=[camera_poses[support.selected_support_indices[0]]],
                config=Sim3PriorConfig(
                    max_support_align_rmse_m=config.support_align_rmse_thr,
                    max_direction_error_deg=config.direction_error_thr_deg,
                ),
            )
            precondition_valid = sim3.accepted
            odom_qc = pose_residual(camera_poses[candidate_kf.keyframe_idx], camera_poses[query.keyframe_idx])
            odom_cs = pose_residual(
                camera_poses[candidate_kf.keyframe_idx],
                camera_poses[support.selected_support_indices[0]],
            )
            metrics = {
                "abs_log_sim3_scale": sim3.abs_log_sim3_scale,
                "support_align_rmse": sim3.support_align_rmse,
                "direction_error_deg": sim3.direction_error_deg,
                "aligned_vs_odom_rot_residual_deg": sim3.aligned_vs_odom.rotation_deg,
                "aligned_vs_odom_trans_residual_norm": sim3.aligned_vs_odom.translation_norm,
            }
            labels = compute_safe_loop_factor_v1(precondition_valid, metrics, config)
            x_geom = build_x_geom(
                {
                    "rank_norm": candidate.rank / max(1, config.retrieval_pool_size),
                    "salad_score_qc": candidate.score,
                    "salad_score_qs": 0.0,
                    "salad_score_cs": 0.0,
                    "salad_score_qc_minus_top1": candidate.score - retrieval.candidates[0].score,
                    "salad_score_qc_minus_topk": candidate.score - retrieval.candidates[-1].score,
                    "da3_rot_qc_deg": 0.0,
                    "da3_trans_qc_norm": float(np.linalg.norm(da3_result.camera_poses[0, :3, 3] - da3_result.camera_poses[1, :3, 3])),
                    "da3_rot_cs_deg": 0.0,
                    "da3_trans_cs_norm": float(np.linalg.norm(da3_result.camera_poses[1, :3, 3] - da3_result.camera_poses[2, :3, 3])),
                    "da3_rot_qs_deg": 0.0,
                    "da3_trans_qs_norm": float(np.linalg.norm(da3_result.camera_poses[0, :3, 3] - da3_result.camera_poses[2, :3, 3])),
                    "odom_rot_qc_deg": odom_qc.rotation_deg,
                    "odom_trans_qc_norm": odom_qc.translation_norm,
                    "odom_rot_cs_deg": odom_cs.rotation_deg,
                    "odom_trans_cs_norm": odom_cs.translation_norm,
                    "support_baseline": support.selected_support_baselines[0],
                    "sim3_scale": sim3.sim3_scale,
                    "abs_log_sim3_scale": sim3.abs_log_sim3_scale,
                    "support_align_rmse": sim3.support_align_rmse,
                    "direction_error_deg": sim3.direction_error_deg,
                    "aligned_vs_odom_rot_residual_deg": sim3.aligned_vs_odom.rotation_deg,
                    "aligned_vs_odom_trans_residual_norm": sim3.aligned_vs_odom.translation_norm,
                    "aligned_loop_rot_deg": sim3.aligned_loop.rotation_deg,
                    "aligned_loop_trans_norm": sim3.aligned_loop.translation_norm,
                    "q_depth_conf_median": da3_result.depth_conf_medians[0],
                    "c_depth_conf_median": da3_result.depth_conf_medians[1],
                    "s_depth_conf_median": da3_result.depth_conf_medians[2],
                    "q_valid_depth_ratio": da3_result.valid_depth_ratios[0],
                    "c_valid_depth_ratio": da3_result.valid_depth_ratios[1],
                    "s_valid_depth_ratio": da3_result.valid_depth_ratios[2],
                    "min_depth_conf_median": min(da3_result.depth_conf_medians[:3]),
                }
            )
            feature_records.append(
                CandidateFeatureRecord(
                    sequence=sequence.sequence_name,
                    query_idx=query.keyframe_idx,
                    query_timestamp=query.timestamp,
                    candidate_source="retrieval_topk",
                    candidate_idx=candidate_kf.keyframe_idx,
                    candidate_timestamp=candidate_kf.timestamp,
                    causal=True,
                    database_max_idx=retrieval.database_max_idx,
                    database_max_timestamp=retrieval.database_max_timestamp,
                    retrieval_db_size=retrieval.retrieval_db_size,
                    support_snapshot_max_idx=support.support_snapshot_max_idx,
                    support_snapshot_max_timestamp=support.support_snapshot_max_timestamp,
                    selected_support_indices=support.selected_support_indices,
                    selected_support_timestamps=support.selected_support_timestamps,
                    support_count=support.support_count,
                    precondition_valid=precondition_valid,
                    negative_reason=sim3.rejection_reason,
                    x_geom=x_geom,
                    safe_loop_factor_v1=labels["safe_loop_factor_v1"],
                    labels=labels,
                    metrics=metrics,
                )
            )

    sequence_index = {
        "schema_version": "loop_policy_dataset_v1",
        "sequence": sequence.sequence_name,
        "platform": sequence.platform,
        "causal": config.causal,
        "keyframe_count": len(sequence.keyframes),
        "image_keyframe_count": int(len(descriptors.keyframe_idx)),
        "settings": dataclass_to_json_dict(config),
    }
    summary = {
        "sequence": sequence.sequence_name,
        "platform": sequence.platform,
        "keyframe_count": len(sequence.keyframes),
        "image_keyframe_count": int(len(descriptors.keyframe_idx)),
        "query_count": len(query_keyframes),
        "retrieval_candidate_count": sum(len(record.candidates) for record in retrieval_records),
        "valid_support_count": sum(1 for record in support_records if not record.rejected),
        "da3_success_count": len(feature_records),
        "safe_loop_factor_positive_count": sum(1 for record in feature_records if record.safe_loop_factor_v1),
        "negative_reasons": negative_reasons,
        "causal_leakage_audit_passed": True,
    }

    (output_dir / "sequence_index.json").write_text(json.dumps(sequence_index, indent=2, sort_keys=True))
    write_jsonl(output_dir / "retrieval_topk.jsonl", retrieval_records)
    write_jsonl(output_dir / "support_selection.jsonl", support_records)
    write_jsonl(output_dir / "candidate_features.jsonl", feature_records)
    (output_dir / "sequence_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build learned loop policy causal loop-policy dataset cache")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--gt-root")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--retrieval-pool-size", type=int, default=50)
    parser.add_argument("--runtime-top-k", type=int, default=4)
    parser.add_argument("--exclude-recent-keyframes", type=int, default=30)
    parser.add_argument("--support-window", type=int, default=20)
    parser.add_argument("--support-count", type=int, default=1)
    parser.add_argument("--min-support-baseline-m", type=float, default=0.5)
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--causal", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    raise SystemExit(
        "CLI descriptor and real DA3 backends are wired in the runtime task. "
        "Use build_sequence_cache() with explicit backends in tests."
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run builder test and verify it passes**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/loop_policy/test_dataset_builder.py -q
```

Expected: PASS.

- [ ] **Step 5: Run the current loop-policy test suite**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/loop_policy -q
```

Expected: PASS.

- [ ] **Step 6: Check the edit set**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
git status --short
```

Expected changed paths include `src/loop_policy/dataset_builder.py` and `tests/loop_policy/test_dataset_builder.py`.

## Task 11: Runtime Backends And CLI Manifests

**Files:**
- Modify: `LoopAnything/src/loop_policy/retrieval.py`
- Modify: `LoopAnything/src/loop_policy/da3_runner.py`
- Modify: `LoopAnything/src/loop_policy/dataset_builder.py`
- Test: `LoopAnything/tests/loop_policy/test_dataset_builder.py`

- [ ] **Step 1: Add CLI manifest and sequence-path tests**

Append to `LoopAnything/tests/loop_policy/test_dataset_builder.py`:

```python
from pathlib import Path

from loop_policy.dataset_builder import raw_dir_for_sequence, write_root_manifests


def test_write_root_manifests_records_thresholds(tmp_path):
    config = LoopPolicyDatasetConfig(
        dataset_root="/dataset",
        output_root=str(tmp_path),
        sequences=("handheld_room01",),
        retrieval_pool_size=50,
        runtime_top_k=4,
    )
    summaries = [
        {
            "sequence": "handheld_room01",
            "platform": "handheld",
            "query_count": 6,
            "causal_leakage_audit_passed": True,
        }
    ]

    write_root_manifests(config, summaries)

    manifest = json.loads((tmp_path / "dataset_manifest.json").read_text())
    lines = (tmp_path / "sequence_summaries.jsonl").read_text().splitlines()
    assert manifest["schema_version"] == "loop_policy_dataset_v1"
    assert manifest["causal"] is True
    assert manifest["settings"]["runtime_top_k"] == 4
    assert json.loads(lines[0])["sequence"] == "handheld_room01"


def test_raw_dir_for_sequence_uses_platform_prefix():
    root = Path("/data/datasets/FusionPortable/fusionportable_loop_dataset")

    assert raw_dir_for_sequence(root, "vehicle_campus00") == (
        root / "vehicle" / "vehicle_campus00" / "raw"
    )
    assert raw_dir_for_sequence(root, "ugv_parking00") == (
        root / "ugv" / "ugv_parking00" / "raw"
    )
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest \
  tests/loop_policy/test_dataset_builder.py::test_write_root_manifests_records_thresholds \
  tests/loop_policy/test_dataset_builder.py::test_raw_dir_for_sequence_uses_platform_prefix \
  -q
```

Expected: FAIL because `write_root_manifests` and `raw_dir_for_sequence` do not exist.

- [ ] **Step 3: Implement manifest writing and sequence-path resolution**

Add to `LoopAnything/src/loop_policy/dataset_builder.py`:

```python
def platform_from_sequence(sequence: str) -> str:
    platform = sequence.split("_", 1)[0]
    if platform not in {"handheld", "legged", "ugv", "vehicle"}:
        raise ValueError(f"unsupported FusionPortable sequence platform in {sequence!r}")
    return platform


def raw_dir_for_sequence(dataset_root: Path, sequence: str) -> Path:
    platform = platform_from_sequence(sequence)
    return Path(dataset_root) / platform / sequence / "raw"


def write_root_manifests(config: LoopPolicyDatasetConfig, summaries: List[Dict[str, object]]) -> None:
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "loop_policy_dataset_v1",
        "causal": config.causal,
        "sequence_count": len(summaries),
        "settings": dataclass_to_json_dict(config),
    }
    (output_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_jsonl(output_root / "sequence_summaries.jsonl", summaries)
```

- [ ] **Step 4: Run the manifest and path tests and verify they pass**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest \
  tests/loop_policy/test_dataset_builder.py::test_write_root_manifests_records_thresholds \
  tests/loop_policy/test_dataset_builder.py::test_raw_dir_for_sequence_uses_platform_prefix \
  -q
```

Expected: PASS.

- [ ] **Step 5: Add a concrete DINO-SALAD descriptor extractor**

Add to `LoopAnything/src/loop_policy/retrieval.py`:

```python
class DinoSaladDescriptorExtractor(DescriptorExtractor):
    def __init__(
        self,
        checkpoint: Path,
        device: str = "cuda",
        image_size: tuple[int, int] = (336, 336),
        batch_size: int = 16,
    ):
        self.checkpoint = Path(checkpoint)
        self.device = device
        self.image_size = image_size
        self.batch_size = batch_size
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)

        import sys
        import torch

        repo = Path(__file__).resolve().parents[2]
        salad_root = repo / "da3_streaming" / "loop_utils" / "salad"
        if str(salad_root) not in sys.path:
            sys.path.insert(0, str(salad_root))
        from models.helper import get_model

        model = get_model(
            "dinov2_vitb14",
            num_channels=768,
            num_clusters=64,
            cluster_dim=128,
            token_dim=256,
        )
        checkpoint = torch.load(self.checkpoint, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device).eval()
        self._model = model
        return model

    def _transform(self):
        import torchvision.transforms as transforms

        return transforms.Compose(
            [
                transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def extract(self, image_paths: List[Path]) -> np.ndarray:
        import torch
        from PIL import Image

        model = self._load_model()
        transform = self._transform()
        descriptors = []
        device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        autocast_enabled = device_type == "cuda"
        for start in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[start : start + self.batch_size]
            images = [
                transform(Image.open(path).convert("RGB"))
                for path in batch_paths
            ]
            batch = torch.stack(images, dim=0).to(self.device)
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=autocast_enabled):
                    descriptors.append(model(batch).detach().cpu().numpy())
        return np.concatenate(descriptors, axis=0)
```

Add these imports at the top of `retrieval.py` if missing:

```python
from pathlib import Path
from typing import List
```

- [ ] **Step 6: Run retrieval tests after adding the real extractor**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/loop_policy/test_retrieval.py -q
```

Expected: PASS.

- [ ] **Step 7: Wire the CLI to real DINO-SALAD and DA3 backends**

Update imports in `LoopAnything/src/loop_policy/dataset_builder.py`:

```python
from loop_policy.da3_runner import Da3Runner, DepthAnything3Runner, MockDa3Runner, build_da3_group
from loop_policy.retrieval import (
    DescriptorExtractor,
    DescriptorCache,
    DinoSaladDescriptorExtractor,
    normalize_descriptors,
    rank_causal_topk,
    save_descriptor_cache,
)
```

Add CLI arguments in `parse_args()`:

```python
parser.add_argument("--salad-checkpoint", required=True)
parser.add_argument("--salad-device", default="cuda")
parser.add_argument("--salad-image-size", type=int, nargs=2, default=[336, 336])
parser.add_argument("--salad-batch-size", type=int, default=16)
parser.add_argument("--da3-model", default="depth-anything/DA3-SMALL")
parser.add_argument("--da3-device", default="cuda")
parser.add_argument("--da3-process-res", type=int, default=504)
parser.add_argument(
    "--da3-extrinsics-convention",
    choices=["c2w", "w2c"],
    default="c2w",
)
```

Replace the `main()` body with:

```python
def main() -> None:
    args = parse_args()
    config = LoopPolicyDatasetConfig(
        dataset_root=args.dataset_root,
        gt_root=args.gt_root,
        output_root=args.output_root,
        sequences=tuple(args.sequences),
        retrieval_pool_size=args.retrieval_pool_size,
        runtime_top_k=args.runtime_top_k,
        exclude_recent_keyframes=args.exclude_recent_keyframes,
        support_window=args.support_window,
        support_count=args.support_count,
        min_support_baseline_m=args.min_support_baseline_m,
        query_limit=args.query_limit,
        causal=args.causal,
    )
    descriptor_extractor = DinoSaladDescriptorExtractor(
        checkpoint=Path(args.salad_checkpoint),
        device=args.salad_device,
        image_size=tuple(args.salad_image_size),
        batch_size=args.salad_batch_size,
    )
    da3_runner = DepthAnything3Runner(
        model_name=args.da3_model,
        device=args.da3_device,
        process_res=args.da3_process_res,
        extrinsics_are_c2w=args.da3_extrinsics_convention == "c2w",
    )
    summaries = []
    failed = False
    for sequence in config.sequences:
        raw_dir = raw_dir_for_sequence(Path(config.dataset_root), sequence)
        try:
            summaries.append(
                build_sequence_cache(
                    raw_dir=raw_dir,
                    config=config,
                    descriptor_extractor=descriptor_extractor,
                    da3_runner=da3_runner,
                )
            )
        except Exception as exc:
            failed = True
            summaries.append(
                {
                    "sequence": sequence,
                    "platform": platform_from_sequence(sequence),
                    "failed": True,
                    "error": repr(exc),
                    "causal_leakage_audit_passed": False,
                }
            )
    write_root_manifests(config, summaries)
    if failed:
        raise SystemExit(1)
```

- [ ] **Step 8: Run dataset builder tests**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/loop_policy/test_dataset_builder.py -q
```

Expected: PASS.

- [ ] **Step 9: Run all loop-policy tests**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/loop_policy -q
```

Expected: PASS.

- [ ] **Step 10: Check the edit set**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
git status --short
```

Expected changed paths remain under `src/loop_policy`, `tests/loop_policy`, `pyproject.toml`, and `docs/superpowers`.

## Task 12: Runtime Smoke Contract And Leakage Audit

**Files:**
- Modify: `LoopAnything/src/loop_policy/dataset_builder.py`
- Test: `LoopAnything/tests/loop_policy/test_dataset_builder.py`

- [ ] **Step 1: Add leakage audit test**

Append to `LoopAnything/tests/loop_policy/test_dataset_builder.py`:

```python
from loop_policy.dataset_builder import audit_causal_leakage


def test_audit_causal_leakage_rejects_future_candidate():
    records = [
        {
            "query_timestamp": 10.0,
            "candidate_timestamp": 11.0,
            "selected_support_timestamps": [5.0],
        }
    ]

    assert audit_causal_leakage(records) is False


def test_audit_causal_leakage_accepts_historical_candidate_and_supports():
    records = [
        {
            "query_timestamp": 10.0,
            "candidate_timestamp": 8.0,
            "selected_support_timestamps": [5.0, 7.0],
        }
    ]

    assert audit_causal_leakage(records) is True
```

- [ ] **Step 2: Run leakage audit tests and verify they fail**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/loop_policy/test_dataset_builder.py::test_audit_causal_leakage_rejects_future_candidate tests/loop_policy/test_dataset_builder.py::test_audit_causal_leakage_accepts_historical_candidate_and_supports -q
```

Expected: FAIL because `audit_causal_leakage` does not exist.

- [ ] **Step 3: Implement leakage audit helper**

Add to `LoopAnything/src/loop_policy/dataset_builder.py`:

```python
def audit_causal_leakage(records: List[Dict[str, object]]) -> bool:
    for record in records:
        query_timestamp = float(record["query_timestamp"])
        if float(record["candidate_timestamp"]) >= query_timestamp:
            return False
        for support_timestamp in record.get("selected_support_timestamps", []):
            if float(support_timestamp) >= query_timestamp:
                return False
    return True
```

Update `build_sequence_cache()` so the final summary computes leakage from `feature_records`:

```python
feature_payloads = [dataclass_to_json_dict(record) for record in feature_records]
summary["causal_leakage_audit_passed"] = audit_causal_leakage(feature_payloads)
```

- [ ] **Step 4: Run leakage audit tests and verify they pass**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/loop_policy/test_dataset_builder.py::test_audit_causal_leakage_rejects_future_candidate tests/loop_policy/test_dataset_builder.py::test_audit_causal_leakage_accepts_historical_candidate_and_supports -q
```

Expected: PASS.

- [ ] **Step 5: Run the synthetic integration suite**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/loop_policy -q
```

Expected: PASS.

- [ ] **Step 6: Define the real runtime smoke command**

After real SALAD and DA3 backend wiring is completed, run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src python -m loop_policy.dataset_builder \
  --dataset-root /data/datasets/FusionPortable/fusionportable_loop_dataset \
  --gt-root /data/datasets/FusionPortable \
  --output-root /data/datasets/FusionPortable/loop_policy_dataset_smoke \
  --sequences handheld_room01 \
  --retrieval-pool-size 10 \
  --runtime-top-k 4 \
  --query-limit 20 \
  --causal
```

Expected files:

```text
/data/datasets/FusionPortable/loop_policy_dataset_smoke/handheld/handheld_room01/sequence_index.json
/data/datasets/FusionPortable/loop_policy_dataset_smoke/handheld/handheld_room01/descriptors.npz
/data/datasets/FusionPortable/loop_policy_dataset_smoke/handheld/handheld_room01/retrieval_topk.jsonl
/data/datasets/FusionPortable/loop_policy_dataset_smoke/handheld/handheld_room01/support_selection.jsonl
/data/datasets/FusionPortable/loop_policy_dataset_smoke/handheld/handheld_room01/candidate_features.jsonl
/data/datasets/FusionPortable/loop_policy_dataset_smoke/handheld/handheld_room01/sequence_summary.json
/data/datasets/FusionPortable/loop_policy_dataset_smoke/dataset_manifest.json
/data/datasets/FusionPortable/loop_policy_dataset_smoke/sequence_summaries.jsonl
```

Expected `sequence_summary.json` fields:

```json
{
  "sequence": "handheld_room01",
  "keyframe_count": 0,
  "image_keyframe_count": 0,
  "query_count": 0,
  "retrieval_candidate_count": 0,
  "valid_support_count": 0,
  "da3_success_count": 0,
  "safe_loop_factor_positive_count": 0,
  "negative_reasons": {},
  "causal_leakage_audit_passed": true
}
```

The numeric counts above are examples of keys, not required values. The smoke passes when every key exists, counts are non-negative, and `causal_leakage_audit_passed` is `true`.

- [ ] **Step 7: Check formatting and diff health**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
python -m black src/loop_policy tests/loop_policy
git diff --check
PYTHONPATH=src pytest tests/loop_policy -q
```

Expected: black completes, `git diff --check` has no output, and pytest passes.

## Self-Review Checklist

- [ ] Convergence status: Tasks 1-12 are implemented, but Stage0 labeling is not converged. Visualization audit found false `safe_loop_factor_v1` positives that must be resolved before treating labels as final.
- [ ] Spec coverage: package boundary, inputs, causal retrieval, support selection, output files, `x_geom[32]`, DA3 group semantics, Sim3 prior, labels, augmentation-compatible metadata, error handling, and validation are covered by Tasks 1-12.
- [ ] Placeholder scan: run a ripgrep check for the disallowed filler phrases listed in the `writing-plans` skill and confirm there are no matches in this plan.
- [ ] Type consistency: `KeyframeRecord`, `LoopPolicyDatasetConfig`, `RetrievalRecord`, `SupportDecision`, `CandidateFeatureRecord`, `Da3GroupResult`, and `Sim3PriorResult` names are consistent across tests and implementation steps.
- [ ] Boundary check: run `git diff --name-only -- src/depth_anything_3` from `LoopAnything`; expected output is empty.
- [ ] Final verification: run `PYTHONPATH=src pytest tests/loop_policy -q` from `LoopAnything`; expected output is PASS.
