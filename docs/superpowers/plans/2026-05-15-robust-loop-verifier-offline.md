# Robust Loop Verifier Offline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Status:** Stage 1 offline baseline converged on 2026-05-17. The
implementation is accepted as a working foundation and now enters an
improvement phase.

**Goal:** Build the FusionPortableV2 Stage 1 offline DA3-ROVER verifier from `2026-05-15-robust-loop-verifier-design.md`.

**Architecture:** Add a new `robust_loop_verifier` package under `LoopAnything/src/` with no dependency on legacy `loop_policy`. The package builds an online-causal VPR cache, runs SALAD retrieval, selects candidate-neighborhood supports, converts DA3 triplets into metric loop factors, scores each candidate with full-prefix GTSAM PGO trajectory diagnostics, and writes metrics plus visual artifacts.

**Source Isolation Constraint:** Do not reference, import, copy, or use `LoopAnything/src/loop_policy` or `LoopAnything/tests/loop_policy` while implementing this plan. Those modules and tests belong to the deferred learned-loop-policy draft, which is explicitly not converged as recorded in commit `626131774e1533c55e81d248a09e102b4c8609cf`. The causal direction must remain: the robust loop verifier is an experiment-backed system that will later guide loop-policy design, not a system contaminated by unvalidated loop-policy priors.

**Tech Stack:** Python 3.9+, NumPy, PyYAML/OmegaConf-compatible YAML parsing, Pillow, OpenCV, Matplotlib, PyTorch/DA3 for real inference, Python GTSAM 4.1.1, pytest.

**Git Policy:** Do not commit during execution unless the user explicitly requests it. Each task ends with `git status --short` as a checkpoint.

---

## Convergence Review 2026-05-17

Lightweight review against the spec found no blocker to converging this plan as
the Stage 1 offline baseline:

- The implementation is isolated under `src/robust_loop_verifier` and does not
  import legacy `loop_policy`.
- The pipeline runs the intended offline path: FusionPortableV2 cache
  preprocess, SALAD historical retrieval, support selection, real DA3 triplets,
  candidate-support Sim3 alignment, full-prefix Python GTSAM PGO, candidate
  records, AP, and MR@100 precision.
- FusionPortableV2 `handheld` and `legged` default to AsterSLAM
  `raw/trajectory_keyframes.txt` as the GT label pose source. This avoids the
  external GT timestamp gaps and missing orientation observed in those
  platforms.
- Positive generation now uses translation plus configurable rotation overlap
  for pure-visual labels: `positive_radius_m`, `positive_max_rotation_deg`
  defaulting to `45.0`, and `recent_exclusion_keyframes`.
- Real `handheld_escalator00` cache regeneration produced 184 keyframes, 77
  positive queries, and 553 positive pairs with
  `gt_label_source=aster_slam_trajectory_keyframes`.

The real run at
`workspace/robust_loop_verifier_runs/FusionPortableV2/handheld/handheld_escalator00/20260517_211913/metrics.json`
reported:

```text
SALAD score only: AP=0.8339, MR@100P=0.0867
SALAD + DA3/Sim3 self-consistency score: AP=0.2923, MR@100P=0.0000
SALAD + DA3-ROVER full-prefix trajectory score: AP=0.7642, MR@100P=0.0665
```

The main remaining finding is methodological rather than a blocker: deformation
alone is not a sufficient ROVER-like score with the current GTSAM setup. False
loops can remain as high-residual constraints without strongly deforming the
optimized trajectory. Post-analysis on the same candidate records showed that
`deformation + log1p(pgo_error_after)` improves the run to approximately
`AP=0.907` and `MR@100P=0.252`. The improvement phase should therefore focus
on residual-aware trajectory-prior scoring, PGO noise sweeps, top-k ablations,
and richer visualization/curve artifacts.

---

## File Structure

Create a new package. Do not import from `src/loop_policy` or reuse its labels,
feature records, hard gates, cache artifacts, tests, or implementation patterns.

- Create `LoopAnything/src/robust_loop_verifier/__init__.py`
  - Package marker and version string.
- Create `LoopAnything/src/robust_loop_verifier/schema.py`
  - Dataclasses for configs, keyframes, positives, retrieval records, DA3 records, PGO records, candidate records, and metrics.
- Create `LoopAnything/src/robust_loop_verifier/io.py`
  - JSONL/YAML helpers, TUM trajectory parser, relative symlink/copy helpers.
- Create `LoopAnything/src/robust_loop_verifier/geometry.py`
  - SE(3), quaternion, camera pose, Sim3 Umeyama alignment, pose residual helpers.
- Create `LoopAnything/src/robust_loop_verifier/fusionportable.py`
  - FusionPortableV2 raw export reader and online-causal VPR cache preprocessor.
- Create `LoopAnything/src/robust_loop_verifier/retrieval.py`
  - Descriptor interface, mock descriptor backend, SALAD backend adapter, online-causal top-k retrieval.
- Create `LoopAnything/src/robust_loop_verifier/support.py`
  - Candidate-neighborhood support selection, nearest candidate index ordering after filtering.
- Create `LoopAnything/src/robust_loop_verifier/da3_runner.py`
  - DA3 group construction, mock runner, real DA3 runner with `w2c -> c2w`, `ref_view_strategy="first"`, `process_res=504`.
- Create `LoopAnything/src/robust_loop_verifier/sim3_factor.py`
  - Candidate-support Sim3 metric alignment and query-candidate loop factor extraction.
- Create `LoopAnything/src/robust_loop_verifier/pgo.py`
  - Full-prefix GTSAM temporary PGO and ROVER trajectory deformation scoring.
- Create `LoopAnything/src/robust_loop_verifier/metrics.py`
  - Larger-is-better canonical scores, AP, MR@100 precision, failure worst-score ordering.
- Create `LoopAnything/src/robust_loop_verifier/artifacts.py`
  - Candidate JSONL, metrics JSON/MD, PR curve data, visual triplets, trajectory plots.
- Create `LoopAnything/src/robust_loop_verifier/pipeline.py`
  - End-to-end orchestration for preprocess, retrieval, DA3, Sim3, PGO, metrics, artifacts.
- Create `LoopAnything/src/robust_loop_verifier/cli.py`
  - `preprocess-fusionportable`, `run-mock`, and `run-cache` command entrypoints.
- Modify `LoopAnything/pyproject.toml`
  - Include `src/robust_loop_verifier` in wheel/sdist packages.
- Create `LoopAnything/configs/robust_loop_verifier/fusionportablev2_handheld.yaml`
  - Explicit FusionPortableV2 handheld system config.
- Create `LoopAnything/tests/robust_loop_verifier/`
  - Focused unit tests using synthetic data and mock DA3/SALAD/GTSAM inputs.

---

## Task 1: Package Boundary, Config, And Schema

**Files:**
- Create: `LoopAnything/src/robust_loop_verifier/__init__.py`
- Create: `LoopAnything/src/robust_loop_verifier/schema.py`
- Create: `LoopAnything/src/robust_loop_verifier/io.py`
- Modify: `LoopAnything/pyproject.toml`
- Create: `LoopAnything/tests/robust_loop_verifier/test_package_boundary.py`
- Create: `LoopAnything/tests/robust_loop_verifier/test_schema_io.py`

- [ ] **Step 1: Write package-boundary tests**

Create `tests/robust_loop_verifier/test_package_boundary.py`:

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "robust_loop_verifier"


def test_robust_loop_verifier_does_not_import_legacy_loop_policy():
    forbidden = "loop_policy"
    offenders = []
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if forbidden in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_plan_does_not_allow_loop_policy_as_reference_source():
    plan = ROOT / "docs" / "superpowers" / "plans" / "2026-05-15-robust-loop-verifier-offline.md"
    text = plan.read_text(encoding="utf-8")
    assert "Do not reference, import, copy, or use `LoopAnything/src/loop_policy`" in text
    assert "`LoopAnything/tests/loop_policy`" in text


def test_pyproject_packages_robust_loop_verifier():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"src/robust_loop_verifier"' in text
```

- [ ] **Step 2: Write schema and IO tests**

Create `tests/robust_loop_verifier/test_schema_io.py`:

```python
from pathlib import Path

import pytest

from robust_loop_verifier.io import read_json, read_jsonl, read_yaml, write_json, write_jsonl, write_yaml
from robust_loop_verifier.schema import RobustLoopVerifierConfig


def test_config_requires_gt_fields(tmp_path: Path):
    path = tmp_path / "config.yaml"
    write_yaml(
        path,
        {
            "dataset_name": "FusionPortableV2",
            "platform": "handheld",
            "input_root": "/data/datasets/FusionPortable/fusionportable_loop_dataset",
            "output_root": str(tmp_path / "cache"),
            "retrieval_top_k_main": 10,
            "retrieval_top_k_ablations": [5, 20],
            "support_window": 4,
            "support_count": 1,
            "min_support_baseline_m": 0.3,
            "pgo_noise": {
                "prior_sigmas": [0.01, 0.01, 0.01, 0.1, 0.1, 0.1],
                "odom_sigmas": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                "loop_sigmas": [0.1, 0.1, 0.1, 1.0, 1.0, 1.0],
            },
            "da3": {"process_res": 504, "ref_view_strategy": "first"},
        },
    )
    with pytest.raises(ValueError, match="positive_radius_m"):
        RobustLoopVerifierConfig.from_yaml(path)


def test_jsonl_roundtrip(tmp_path: Path):
    path = tmp_path / "records.jsonl"
    rows = [{"idx": 1, "name": "a"}, {"idx": 2, "name": "b"}]
    write_jsonl(path, rows)
    assert list(read_jsonl(path)) == rows


def test_json_roundtrip(tmp_path: Path):
    path = tmp_path / "manifest.json"
    data = {"dataset_name": "FusionPortableV2", "keyframe_count": 2}
    write_json(path, data)
    assert read_json(path) == data
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/robust_loop_verifier/test_package_boundary.py tests/robust_loop_verifier/test_schema_io.py -q
```

Expected: FAIL because `robust_loop_verifier` package does not exist.

- [ ] **Step 4: Implement minimal package, schema, and IO**

Create `src/robust_loop_verifier/__init__.py`:

```python
"""Offline robust loop verifier for DA3-ROVER experiments."""

__all__ = ["__version__"]
__version__ = "0.1.0"
```

Create `src/robust_loop_verifier/io.py` with:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping, MutableMapping

import yaml


JsonDict = MutableMapping[str, object]


def read_jsonl(path: Path) -> Iterator[JsonDict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_json(path: Path) -> JsonDict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be a mapping: {path}")
    return data


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def write_json(path: Path, data: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(data), indent=2, sort_keys=True), encoding="utf-8")


def read_yaml(path: Path) -> JsonDict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def write_yaml(path: Path, data: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(data), handle, sort_keys=True)
```

Create `src/robust_loop_verifier/schema.py` with these dataclasses and config validation:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from robust_loop_verifier.io import read_yaml


@dataclass(frozen=True)
class PgoNoiseConfig:
    prior_sigmas: List[float]
    odom_sigmas: List[float]
    loop_sigmas: List[float]

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "PgoNoiseConfig":
        return cls(
            prior_sigmas=_six_floats(data, "prior_sigmas"),
            odom_sigmas=_six_floats(data, "odom_sigmas"),
            loop_sigmas=_six_floats(data, "loop_sigmas"),
        )


@dataclass(frozen=True)
class Da3RuntimeConfig:
    process_res: int
    ref_view_strategy: str

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "Da3RuntimeConfig":
        process_res = int(_required(data, "process_res"))
        ref_view_strategy = str(_required(data, "ref_view_strategy"))
        if process_res < 224:
            raise ValueError("da3.process_res must be a real geometry resolution")
        if ref_view_strategy != "first":
            raise ValueError('da3.ref_view_strategy must be "first"')
        return cls(process_res=process_res, ref_view_strategy=ref_view_strategy)


@dataclass(frozen=True)
class RobustLoopVerifierConfig:
    dataset_name: str
    platform: str
    input_root: Path
    output_root: Path
    gt_root: Path
    positive_radius_m: float
    recent_exclusion_keyframes: int
    retrieval_top_k_main: int
    retrieval_top_k_ablations: List[int]
    support_window: int
    support_count: int
    min_support_baseline_m: float
    pgo_noise: PgoNoiseConfig
    da3: Da3RuntimeConfig

    @classmethod
    def from_yaml(cls, path: Path) -> "RobustLoopVerifierConfig":
        data = read_yaml(path)
        return cls.from_mapping(data)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "RobustLoopVerifierConfig":
        positive_radius_m = float(_required(data, "positive_radius_m"))
        recent_exclusion = int(_required(data, "recent_exclusion_keyframes"))
        if positive_radius_m <= 0.0:
            raise ValueError("positive_radius_m must be positive")
        if recent_exclusion < 0:
            raise ValueError("recent_exclusion_keyframes must be non-negative")
        return cls(
            dataset_name=str(_required(data, "dataset_name")),
            platform=str(_required(data, "platform")),
            input_root=Path(str(_required(data, "input_root"))),
            output_root=Path(str(_required(data, "output_root"))),
            gt_root=Path(str(_required(data, "gt_root"))),
            positive_radius_m=positive_radius_m,
            recent_exclusion_keyframes=recent_exclusion,
            retrieval_top_k_main=int(_required(data, "retrieval_top_k_main")),
            retrieval_top_k_ablations=[int(v) for v in _required(data, "retrieval_top_k_ablations")],
            support_window=int(_required(data, "support_window")),
            support_count=int(_required(data, "support_count")),
            min_support_baseline_m=float(_required(data, "min_support_baseline_m")),
            pgo_noise=PgoNoiseConfig.from_mapping(_mapping(data, "pgo_noise")),
            da3=Da3RuntimeConfig.from_mapping(_mapping(data, "da3")),
        )


@dataclass(frozen=True)
class KeyframeRecord:
    idx: int
    timestamp: float
    image_path: Optional[str]
    odom_pose: np.ndarray
    gt_pose: np.ndarray


@dataclass(frozen=True)
class PositiveRecord:
    query_idx: int
    positive_indices: List[int]


def _required(data: Mapping[str, object], key: str) -> object:
    if key not in data:
        raise ValueError(f"missing required config field: {key}")
    return data[key]


def _mapping(data: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = _required(data, key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _six_floats(data: Mapping[str, object], key: str) -> List[float]:
    value = _required(data, key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 6:
        raise ValueError(f"{key} must contain six numeric values")
    return [float(v) for v in value]
```

Modify `pyproject.toml`:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/depth_anything_3", "src/loop_policy", "src/robust_loop_verifier"]

[tool.hatch.build.targets.sdist]
include = [
  "/README.md",
  "/pyproject.toml",
  "/src/depth_anything_3",
  "/src/loop_policy",
  "/src/robust_loop_verifier",
]
```

- [ ] **Step 5: Run tests and checkpoint**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/robust_loop_verifier/test_package_boundary.py tests/robust_loop_verifier/test_schema_io.py -q
git status --short
```

Expected: PASS. `git status` shows only intended new package/test/config changes plus pre-existing unrelated worktree changes.

---

## Task 2: Geometry, TUM Parsing, And GT Association

**Files:**
- Create: `LoopAnything/src/robust_loop_verifier/geometry.py`
- Modify: `LoopAnything/src/robust_loop_verifier/io.py`
- Create: `LoopAnything/tests/robust_loop_verifier/test_geometry.py`
- Create: `LoopAnything/tests/robust_loop_verifier/test_tum_io.py`

- [ ] **Step 1: Write geometry tests**

Create `tests/robust_loop_verifier/test_geometry.py`:

```python
import math

import numpy as np

from robust_loop_verifier.geometry import (
    invert_transform,
    make_transform,
    pose_between,
    rotation_matrix_from_quat_xyzw,
    sim3_align_points,
)


def test_quaternion_identity_to_rotation_matrix():
    rot = rotation_matrix_from_quat_xyzw([0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(rot, np.eye(3), atol=1e-9)


def test_pose_between_translation():
    a = make_transform(np.eye(3), [1.0, 0.0, 0.0])
    b = make_transform(np.eye(3), [3.5, 0.0, 0.0])
    rel = pose_between(a, b)
    np.testing.assert_allclose(rel[:3, 3], [2.5, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(invert_transform(invert_transform(a)), a, atol=1e-9)


def test_sim3_align_points_recovers_scale_rotation_translation():
    src = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    angle = math.pi / 2.0
    rot = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    dst = 3.0 * (rot @ src.T).T + np.array([5.0, -2.0, 1.0])
    result = sim3_align_points(src, dst)
    aligned = result.scale * (result.rotation @ src.T).T + result.translation
    np.testing.assert_allclose(aligned, dst, atol=1e-8)
    assert result.rmse < 1e-8
```

- [ ] **Step 2: Write TUM IO tests**

Create `tests/robust_loop_verifier/test_tum_io.py`:

```python
from pathlib import Path

import numpy as np

from robust_loop_verifier.io import associate_tum_by_timestamp, read_tum_trajectory


def test_read_tum_trajectory(tmp_path: Path):
    path = tmp_path / "traj.txt"
    path.write_text(
        "1.0 0 0 0 0 0 0 1\n"
        "1.1 1 2 3 0 0 0 1\n",
        encoding="utf-8",
    )
    records = read_tum_trajectory(path)
    assert [record.timestamp for record in records] == [1.0, 1.1]
    np.testing.assert_allclose(records[1].pose[:3, 3], [1.0, 2.0, 3.0])


def test_associate_tum_by_timestamp_nearest_with_tolerance(tmp_path: Path):
    path = tmp_path / "traj.txt"
    path.write_text(
        "10.00 0 0 0 0 0 0 1\n"
        "10.05 5 0 0 0 0 0 1\n",
        encoding="utf-8",
    )
    records = read_tum_trajectory(path)
    matched = associate_tum_by_timestamp(records, 10.049, max_delta_sec=0.01)
    np.testing.assert_allclose(matched.pose[:3, 3], [5.0, 0.0, 0.0])
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/robust_loop_verifier/test_geometry.py tests/robust_loop_verifier/test_tum_io.py -q
```

Expected: FAIL because geometry and TUM helpers do not exist.

- [ ] **Step 4: Implement geometry and TUM helpers**

Create `src/robust_loop_verifier/geometry.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Sim3Alignment:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    rmse: float


def rotation_matrix_from_quat_xyzw(quat_xyzw: Sequence[float]) -> np.ndarray:
    x, y, z, w = [float(v) for v in quat_xyzw]
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("quaternion norm must be finite and positive")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def make_transform(rotation: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    inv = np.eye(4, dtype=np.float64)
    rot = transform[:3, :3]
    inv[:3, :3] = rot.T
    inv[:3, 3] = -rot.T @ transform[:3, 3]
    return inv


def pose_between(a_c2w: np.ndarray, b_c2w: np.ndarray) -> np.ndarray:
    return invert_transform(a_c2w) @ b_c2w


def sim3_align_points(src_points: np.ndarray, dst_points: np.ndarray) -> Sim3Alignment:
    src = np.asarray(src_points, dtype=np.float64)
    dst = np.asarray(dst_points, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or src.shape[0] < 3:
        raise ValueError("Sim3 alignment requires matching Nx3 point arrays with N >= 3")
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    covariance = (dst_centered.T @ src_centered) / src.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0.0:
        correction[2, 2] = -1.0
    rotation = u @ correction @ vt
    variance = float(np.mean(np.sum(src_centered * src_centered, axis=1)))
    if variance <= 0.0:
        raise ValueError("source points are degenerate")
    scale = float(np.trace(np.diag(singular_values) @ correction) / variance)
    translation = dst_mean - scale * rotation @ src_mean
    aligned = scale * (rotation @ src.T).T + translation
    rmse = float(np.sqrt(np.mean(np.sum((aligned - dst) ** 2, axis=1))))
    return Sim3Alignment(scale=scale, rotation=rotation, translation=translation, rmse=rmse)
```

Extend `src/robust_loop_verifier/io.py` with TUM helpers:

```python
from dataclasses import dataclass
import bisect
import numpy as np

from robust_loop_verifier.geometry import make_transform, rotation_matrix_from_quat_xyzw


@dataclass(frozen=True)
class TumPoseRecord:
    timestamp: float
    pose: np.ndarray


def read_tum_trajectory(path: Path) -> list[TumPoseRecord]:
    records: list[TumPoseRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) != 8:
                raise ValueError(f"expected TUM row with 8 fields in {path}: {stripped}")
            timestamp = float(parts[0])
            xyz = [float(v) for v in parts[1:4]]
            quat = [float(v) for v in parts[4:8]]
            records.append(TumPoseRecord(timestamp, make_transform(rotation_matrix_from_quat_xyzw(quat), xyz)))
    records.sort(key=lambda item: item.timestamp)
    return records


def associate_tum_by_timestamp(
    records: list[TumPoseRecord], timestamp: float, max_delta_sec: float
) -> TumPoseRecord:
    if not records:
        raise ValueError("cannot associate against an empty TUM trajectory")
    stamps = [record.timestamp for record in records]
    pos = bisect.bisect_left(stamps, timestamp)
    candidates = []
    if pos < len(records):
        candidates.append(records[pos])
    if pos > 0:
        candidates.append(records[pos - 1])
    best = min(candidates, key=lambda record: abs(record.timestamp - timestamp))
    if abs(best.timestamp - timestamp) > max_delta_sec:
        raise ValueError(
            f"nearest GT timestamp delta {abs(best.timestamp - timestamp):.6f}s exceeds {max_delta_sec:.6f}s"
        )
    return best
```

- [ ] **Step 5: Run tests and checkpoint**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/robust_loop_verifier/test_geometry.py tests/robust_loop_verifier/test_tum_io.py -q
git status --short
```

Expected: PASS.

---

## Task 3: FusionPortableV2 Online-Causal VPR Cache Preprocess

**Files:**
- Create: `LoopAnything/src/robust_loop_verifier/fusionportable.py`
- Create: `LoopAnything/tests/robust_loop_verifier/test_fusionportable_preprocess.py`
- Create: `LoopAnything/configs/robust_loop_verifier/fusionportablev2_handheld.yaml`

- [ ] **Step 1: Write synthetic preprocess test**

Create `tests/robust_loop_verifier/test_fusionportable_preprocess.py`:

```python
import json
from pathlib import Path

from PIL import Image

from robust_loop_verifier.fusionportable import preprocess_fusionportable_sequence
from robust_loop_verifier.io import read_jsonl
from robust_loop_verifier.schema import RobustLoopVerifierConfig


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(path)


def test_preprocess_writes_online_causal_cache(tmp_path: Path):
    raw = tmp_path / "raw"
    (raw / "keyframe_images").mkdir(parents=True)
    for idx in range(6):
        _write_png(raw / "keyframe_images" / f"{idx:06d}.png")
    (raw / "sequence_meta.json").write_text(
        json.dumps({"sequence_name": "handheld_test", "trajectory_keyframes_file": "trajectory_keyframes.txt"}),
        encoding="utf-8",
    )
    with (raw / "keyframes_with_images.jsonl").open("w", encoding="utf-8") as handle:
        for idx in range(6):
            handle.write(
                json.dumps(
                    {
                        "keyframe_idx": idx,
                        "timestamp": float(idx),
                        "has_image": True,
                        "image_path": f"keyframe_images/{idx:06d}.png",
                    }
                )
                + "\n"
            )
    (raw / "trajectory_keyframes.txt").write_text(
        "\n".join(f"{idx}.0 {float(idx)} 0 0 0 0 0 1" for idx in range(6)) + "\n",
        encoding="utf-8",
    )
    gt = tmp_path / "gt" / "handheld_test.txt"
    gt.parent.mkdir()
    gt.write_text(
        "\n".join(
            [
                "0.0 0 0 0 0 0 0 1",
                "1.0 1 0 0 0 0 0 1",
                "2.0 2 0 0 0 0 0 1",
                "3.0 10 0 0 0 0 0 1",
                "4.0 0.2 0 0 0 0 0 1",
                "5.0 0.1 0 0 0 0 0 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = RobustLoopVerifierConfig.from_mapping(
        {
            "dataset_name": "FusionPortableV2",
            "platform": "handheld",
            "input_root": str(tmp_path / "dataset"),
            "output_root": str(tmp_path / "cache"),
            "gt_root": str(tmp_path / "gt"),
            "positive_radius_m": 0.5,
            "recent_exclusion_keyframes": 2,
            "retrieval_top_k_main": 10,
            "retrieval_top_k_ablations": [5, 20],
            "support_window": 4,
            "support_count": 1,
            "min_support_baseline_m": 0.3,
            "pgo_noise": {
                "prior_sigmas": [0.01, 0.01, 0.01, 0.1, 0.1, 0.1],
                "odom_sigmas": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                "loop_sigmas": [0.1, 0.1, 0.1, 1.0, 1.0, 1.0],
            },
            "da3": {"process_res": 504, "ref_view_strategy": "first"},
        }
    )
    out = preprocess_fusionportable_sequence(
        raw_dir=raw,
        gt_trajectory_file=gt,
        sequence_name="handheld_test",
        config=config,
        max_gt_delta_sec=0.01,
    )
    positives = list(read_jsonl(out / "positives.jsonl"))
    assert positives[-1]["query_idx"] == 5
    assert positives[-1]["positive_indices"] == [0]
    assert (out / "images" / "000005.png").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/robust_loop_verifier/test_fusionportable_preprocess.py -q
```

Expected: FAIL because `fusionportable.py` does not exist.

- [ ] **Step 3: Implement FusionPortable preprocess**

Create `src/robust_loop_verifier/fusionportable.py` with functions:

```python
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np

from robust_loop_verifier.io import (
    associate_tum_by_timestamp,
    read_jsonl,
    read_tum_trajectory,
    write_json,
    write_jsonl,
)
from robust_loop_verifier.schema import RobustLoopVerifierConfig


def preprocess_fusionportable_sequence(
    raw_dir: Path,
    gt_trajectory_file: Path,
    sequence_name: str,
    config: RobustLoopVerifierConfig,
    max_gt_delta_sec: float = 0.05,
) -> Path:
    keyframe_rows = list(read_jsonl(raw_dir / "keyframes_with_images.jsonl"))
    odom_records = read_tum_trajectory(raw_dir / "trajectory_keyframes.txt")
    gt_records = read_tum_trajectory(gt_trajectory_file)
    if len(odom_records) != len(keyframe_rows):
        raise ValueError("trajectory_keyframes.txt length must match keyframes_with_images.jsonl")

    out_dir = config.output_root / config.dataset_name / config.platform / sequence_name
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    keyframe_records = []
    gt_positions = {}
    for row, odom in zip(keyframe_rows, odom_records):
        idx = int(row["keyframe_idx"])
        timestamp = float(row["timestamp"])
        gt = associate_tum_by_timestamp(gt_records, timestamp, max_gt_delta_sec)
        image_rel = _link_image(raw_dir, image_dir, idx, row.get("image_path"))
        gt_positions[idx] = gt.pose[:3, 3]
        keyframe_records.append(
            {
                "idx": idx,
                "timestamp": timestamp,
                "image_path": image_rel,
                "odom_pose": odom.pose.reshape(-1).tolist(),
                "gt_pose": gt.pose.reshape(-1).tolist(),
                "source_raw_dir": str(raw_dir),
                "source_gt_trajectory_file": str(gt_trajectory_file),
            }
        )

    positives = _build_positive_rows(
        keyframe_records,
        gt_positions,
        config.positive_radius_m,
        config.recent_exclusion_keyframes,
    )
    write_jsonl(out_dir / "keyframes.jsonl", keyframe_records)
    write_jsonl(out_dir / "positives.jsonl", positives)
    write_json(
        out_dir / "manifest.json",
        {
            "dataset_name": config.dataset_name,
            "platform": config.platform,
            "sequence_name": sequence_name,
            "positive_radius_m": config.positive_radius_m,
            "recent_exclusion_keyframes": config.recent_exclusion_keyframes,
            "keyframe_count": len(keyframe_records),
        },
    )
    return out_dir


def _link_image(raw_dir: Path, image_dir: Path, idx: int, image_path: object) -> str | None:
    if image_path is None:
        return None
    src = raw_dir / str(image_path)
    dst = image_dir / f"{idx:06d}{src.suffix.lower() or '.png'}"
    if dst.exists():
        return str(dst.relative_to(image_dir.parent))
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return str(dst.relative_to(image_dir.parent))


def _build_positive_rows(
    keyframes: Iterable[dict],
    gt_positions: dict[int, np.ndarray],
    positive_radius_m: float,
    recent_exclusion_keyframes: int,
) -> list[dict]:
    rows = []
    ordered = list(keyframes)
    for query in ordered:
        q_idx = int(query["idx"])
        q_pos = gt_positions[q_idx]
        positives = []
        for candidate in ordered:
            c_idx = int(candidate["idx"])
            if c_idx >= q_idx - recent_exclusion_keyframes:
                continue
            if float(np.linalg.norm(q_pos - gt_positions[c_idx])) <= positive_radius_m:
                positives.append(c_idx)
        rows.append({"query_idx": q_idx, "positive_indices": positives})
    return rows
```

- [ ] **Step 4: Add FusionPortableV2 handheld config**

Create `configs/robust_loop_verifier/fusionportablev2_handheld.yaml`:

```yaml
dataset_name: FusionPortableV2
platform: handheld
input_root: /data/datasets/FusionPortable/fusionportable_loop_dataset
gt_root: /data/datasets/FusionPortable/handheld
output_root: /data/datasets/FusionPortable/robust_loop_verifier_cache
positive_radius_m: 2.0
recent_exclusion_keyframes: 30
retrieval_top_k_main: 10
retrieval_top_k_ablations: [5, 20]
support_window: 4
support_count: 1
min_support_baseline_m: 0.3
pgo_noise:
  prior_sigmas: [0.01, 0.01, 0.01, 0.1, 0.1, 0.1]
  odom_sigmas: [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]
  loop_sigmas: [0.1, 0.1, 0.1, 1.0, 1.0, 1.0]
da3:
  process_res: 504
  ref_view_strategy: first
```

- [ ] **Step 5: Run tests and checkpoint**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/robust_loop_verifier/test_fusionportable_preprocess.py -q
git status --short
```

Expected: PASS.

---

## Task 4: Online-Causal Retrieval With Mock Descriptor Backend

**Files:**
- Create: `LoopAnything/src/robust_loop_verifier/retrieval.py`
- Create: `LoopAnything/tests/robust_loop_verifier/test_retrieval.py`

- [ ] **Step 1: Write retrieval tests**

Create `tests/robust_loop_verifier/test_retrieval.py`:

```python
import numpy as np

from robust_loop_verifier.retrieval import DescriptorSet, retrieve_historical_topk


def test_retrieval_excludes_future_and_recent_candidates():
    descriptors = DescriptorSet(
        keyframe_indices=[0, 1, 2, 3, 4],
        descriptors=np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.8, 0.2],
                [1.0, 0.0],
            ],
            dtype=np.float64,
        ),
    )
    record = retrieve_historical_topk(
        query_idx=4,
        descriptors=descriptors,
        top_k=3,
        recent_exclusion_keyframes=1,
    )
    assert [candidate.candidate_idx for candidate in record.candidates] == [0, 1]
    assert record.query_idx == 4


def test_retrieval_uses_larger_similarity_as_better():
    descriptors = DescriptorSet(
        keyframe_indices=[0, 1, 3],
        descriptors=np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float64),
    )
    record = retrieve_historical_topk(
        query_idx=3,
        descriptors=descriptors,
        top_k=2,
        recent_exclusion_keyframes=0,
    )
    assert record.candidates[0].candidate_idx == 1
    assert record.candidates[0].score > record.candidates[1].score
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/robust_loop_verifier/test_retrieval.py -q
```

Expected: FAIL because `retrieval.py` does not exist.

- [ ] **Step 3: Implement retrieval**

Create `src/robust_loop_verifier/retrieval.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class DescriptorSet:
    keyframe_indices: List[int]
    descriptors: np.ndarray


@dataclass(frozen=True)
class RetrievalCandidate:
    query_idx: int
    candidate_idx: int
    rank: int
    score: float


@dataclass(frozen=True)
class RetrievalRecord:
    query_idx: int
    candidates: List[RetrievalCandidate]


def retrieve_historical_topk(
    query_idx: int,
    descriptors: DescriptorSet,
    top_k: int,
    recent_exclusion_keyframes: int,
) -> RetrievalRecord:
    keyframes = list(descriptors.keyframe_indices)
    if query_idx not in keyframes:
        raise ValueError(f"missing descriptor for query {query_idx}")
    query_pos = keyframes.index(query_idx)
    matrix = np.asarray(descriptors.descriptors, dtype=np.float64)
    query_descriptor = _l2_normalize(matrix[query_pos])
    candidates = []
    for pos, candidate_idx in enumerate(keyframes):
        if candidate_idx == query_idx:
            continue
        if candidate_idx >= query_idx - recent_exclusion_keyframes:
            continue
        score = float(np.dot(query_descriptor, _l2_normalize(matrix[pos])))
        candidates.append((score, candidate_idx))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return RetrievalRecord(
        query_idx=query_idx,
        candidates=[
            RetrievalCandidate(query_idx=query_idx, candidate_idx=idx, rank=rank + 1, score=score)
            for rank, (score, idx) in enumerate(candidates[:top_k])
        ],
    )


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("descriptor norm must be finite and positive")
    return vector / norm
```

- [ ] **Step 4: Run tests and checkpoint**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/robust_loop_verifier/test_retrieval.py -q
git status --short
```

Expected: PASS.

---

## Task 5: Support Selection

**Files:**
- Create: `LoopAnything/src/robust_loop_verifier/support.py`
- Create: `LoopAnything/tests/robust_loop_verifier/test_support.py`

- [ ] **Step 1: Write support tests**

Create `tests/robust_loop_verifier/test_support.py`:

```python
import numpy as np

from robust_loop_verifier.geometry import make_transform
from robust_loop_verifier.support import select_support


def _poses():
    return {idx: make_transform(np.eye(3), [float(idx), 0.0, 0.0]) for idx in range(10)}


def test_support_selects_nearest_candidate_index_after_filters():
    result = select_support(
        query_idx=9,
        candidate_idx=3,
        available_indices=[0, 1, 2, 3, 4, 5, 6],
        image_indices={0, 1, 2, 3, 4, 5, 6},
        camera_poses=_poses(),
        support_window=4,
        recent_exclusion_keyframes=2,
        min_support_baseline_m=0.3,
    )
    assert result.support_idx == 2
    assert result.rejection_reason is None


def test_support_rejects_when_recent_filter_removes_all():
    result = select_support(
        query_idx=5,
        candidate_idx=3,
        available_indices=[2, 4],
        image_indices={2, 4},
        camera_poses=_poses(),
        support_window=2,
        recent_exclusion_keyframes=10,
        min_support_baseline_m=0.3,
    )
    assert result.support_idx is None
    assert result.rejection_reason == "no_valid_support"
```

- [ ] **Step 2: Implement support selection**

Create `src/robust_loop_verifier/support.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Set

import numpy as np


@dataclass(frozen=True)
class SupportSelection:
    query_idx: int
    candidate_idx: int
    support_idx: Optional[int]
    support_baseline_m: Optional[float]
    rejection_reason: Optional[str]


def select_support(
    query_idx: int,
    candidate_idx: int,
    available_indices: Sequence[int],
    image_indices: Set[int],
    camera_poses: Mapping[int, np.ndarray],
    support_window: int,
    recent_exclusion_keyframes: int,
    min_support_baseline_m: float,
) -> SupportSelection:
    scored = []
    candidate_pose = camera_poses.get(candidate_idx)
    if candidate_pose is None:
        return SupportSelection(query_idx, candidate_idx, None, None, "missing_candidate_pose")
    for support_idx in available_indices:
        if support_idx == candidate_idx:
            continue
        if abs(support_idx - candidate_idx) > support_window:
            continue
        if abs(query_idx - support_idx) <= recent_exclusion_keyframes:
            continue
        if support_idx not in image_indices:
            continue
        support_pose = camera_poses.get(support_idx)
        if support_pose is None:
            continue
        baseline = float(np.linalg.norm(candidate_pose[:3, 3] - support_pose[:3, 3]))
        if baseline < min_support_baseline_m:
            continue
        scored.append((abs(support_idx - candidate_idx), support_idx, baseline))
    if not scored:
        return SupportSelection(query_idx, candidate_idx, None, None, "no_valid_support")
    scored.sort(key=lambda item: (item[0], item[1]))
    _, support_idx, baseline = scored[0]
    return SupportSelection(query_idx, candidate_idx, support_idx, baseline, None)
```

- [ ] **Step 3: Run tests and checkpoint**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/robust_loop_verifier/test_support.py -q
git status --short
```

Expected: PASS.

---

## Task 6: DA3 Triplet Runner Interface And Sim3 Metric Loop Factor

**Files:**
- Create: `LoopAnything/src/robust_loop_verifier/da3_runner.py`
- Create: `LoopAnything/src/robust_loop_verifier/sim3_factor.py`
- Create: `LoopAnything/tests/robust_loop_verifier/test_da3_runner.py`
- Create: `LoopAnything/tests/robust_loop_verifier/test_sim3_factor.py`

- [ ] **Step 1: Write DA3 runner and Sim3 tests**

Create `tests/robust_loop_verifier/test_da3_runner.py`:

```python
import numpy as np

from robust_loop_verifier.da3_runner import MockDa3Runner, build_da3_triplet


def test_build_triplet_preserves_query_candidate_support_order():
    triplet = build_da3_triplet("q.png", "c.png", "s.png", query_idx=10, candidate_idx=3, support_idx=2)
    assert triplet.view_roles == ["query", "candidate", "support"]
    assert triplet.keyframe_indices == [10, 3, 2]


def test_mock_da3_runner_returns_c2w_poses():
    triplet = build_da3_triplet("q.png", "c.png", "s.png", query_idx=10, candidate_idx=3, support_idx=2)
    result = MockDa3Runner().run_triplet(triplet)
    assert result.predicted_c2w.shape == (3, 4, 4)
    np.testing.assert_allclose(result.predicted_c2w[0], np.eye(4))
```

Create `tests/robust_loop_verifier/test_sim3_factor.py`:

```python
import math

import numpy as np

from robust_loop_verifier.geometry import make_transform, pose_between
from robust_loop_verifier.sim3_factor import align_triplet_to_candidate_support


def _pose(x):
    return make_transform(np.eye(3), [x, 0.0, 0.0])


def test_align_triplet_recovers_metric_query_candidate_factor():
    da3_query = _pose(0.5)
    da3_candidate = _pose(0.0)
    da3_support = _pose(0.25)
    odom_candidate = _pose(10.0)
    odom_support = _pose(11.0)
    result = align_triplet_to_candidate_support(
        da3_query_c2w=da3_query,
        da3_candidate_c2w=da3_candidate,
        da3_support_c2w=da3_support,
        odom_candidate_c2w=odom_candidate,
        odom_support_c2w=odom_support,
    )
    assert result.valid
    assert math.isclose(result.sim3_scale, 4.0)
    np.testing.assert_allclose(result.loop_factor[:3, 3], [-2.0, 0.0, 0.0], atol=1e-9)


def test_align_triplet_rejects_zero_da3_baseline():
    result = align_triplet_to_candidate_support(
        da3_query_c2w=_pose(1.0),
        da3_candidate_c2w=_pose(0.0),
        da3_support_c2w=_pose(0.0),
        odom_candidate_c2w=_pose(0.0),
        odom_support_c2w=_pose(1.0),
    )
    assert not result.valid
    assert result.rejection_reason == "invalid_da3_support_baseline"
```

- [ ] **Step 2: Implement DA3 runner interface and Sim3 factor**

Create `src/robust_loop_verifier/da3_runner.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class Da3Triplet:
    image_paths: List[str]
    keyframe_indices: List[int]
    view_roles: List[str]


@dataclass(frozen=True)
class Da3TripletResult:
    keyframe_indices: List[int]
    view_roles: List[str]
    predicted_c2w: np.ndarray
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0


def build_da3_triplet(
    query_image: str,
    candidate_image: str,
    support_image: str,
    query_idx: int,
    candidate_idx: int,
    support_idx: int,
) -> Da3Triplet:
    return Da3Triplet(
        image_paths=[query_image, candidate_image, support_image],
        keyframe_indices=[query_idx, candidate_idx, support_idx],
        view_roles=["query", "candidate", "support"],
    )


class MockDa3Runner:
    def run_triplet(self, triplet: Da3Triplet) -> Da3TripletResult:
        poses = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], 3, axis=0)
        poses[1, 0, 3] = 0.25
        poses[2, 0, 3] = 0.5
        return Da3TripletResult(
            keyframe_indices=triplet.keyframe_indices,
            view_roles=triplet.view_roles,
            predicted_c2w=poses,
        )
```

Create `src/robust_loop_verifier/sim3_factor.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from robust_loop_verifier.geometry import pose_between


@dataclass(frozen=True)
class Sim3LoopFactorResult:
    valid: bool
    loop_factor: Optional[np.ndarray]
    sim3_scale: float
    support_alignment_residual_m: float
    direction_error_deg: float
    rejection_reason: Optional[str]


def align_triplet_to_candidate_support(
    da3_query_c2w: np.ndarray,
    da3_candidate_c2w: np.ndarray,
    da3_support_c2w: np.ndarray,
    odom_candidate_c2w: np.ndarray,
    odom_support_c2w: np.ndarray,
) -> Sim3LoopFactorResult:
    da3_delta = da3_support_c2w[:3, 3] - da3_candidate_c2w[:3, 3]
    odom_delta = odom_support_c2w[:3, 3] - odom_candidate_c2w[:3, 3]
    da3_norm = float(np.linalg.norm(da3_delta))
    odom_norm = float(np.linalg.norm(odom_delta))
    if da3_norm <= 1e-9 or not np.isfinite(da3_norm) or odom_norm <= 1e-9:
        return Sim3LoopFactorResult(False, None, float("nan"), float("inf"), float("inf"), "invalid_da3_support_baseline")
    scale = odom_norm / da3_norm
    if not np.isfinite(scale) or scale <= 0.0:
        return Sim3LoopFactorResult(False, None, scale, float("inf"), float("inf"), "invalid_sim3_scale")
    rotation = odom_candidate_c2w[:3, :3] @ da3_candidate_c2w[:3, :3].T
    translation = odom_candidate_c2w[:3, 3] - scale * rotation @ da3_candidate_c2w[:3, 3]
    aligned_query = _apply_sim3(da3_query_c2w, rotation, scale, translation)
    aligned_candidate = _apply_sim3(da3_candidate_c2w, rotation, scale, translation)
    aligned_support = _apply_sim3(da3_support_c2w, rotation, scale, translation)
    residual = float(np.linalg.norm(aligned_support[:3, 3] - odom_support_c2w[:3, 3]))
    direction_error_deg = _direction_error_deg(aligned_support[:3, 3] - aligned_candidate[:3, 3], odom_delta)
    return Sim3LoopFactorResult(
        valid=True,
        loop_factor=pose_between(aligned_query, aligned_candidate),
        sim3_scale=scale,
        support_alignment_residual_m=residual,
        direction_error_deg=direction_error_deg,
        rejection_reason=None,
    )


def _apply_sim3(pose: np.ndarray, rotation: np.ndarray, scale: float, translation: np.ndarray) -> np.ndarray:
    aligned = np.eye(4, dtype=np.float64)
    aligned[:3, :3] = rotation @ pose[:3, :3]
    aligned[:3, 3] = scale * rotation @ pose[:3, 3] + translation
    return aligned


def _direction_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm <= 0.0 or b_norm <= 0.0:
        return float("inf")
    dot = float(np.clip(np.dot(a / a_norm, b / b_norm), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))
```

- [ ] **Step 3: Run tests and checkpoint**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/robust_loop_verifier/test_da3_runner.py tests/robust_loop_verifier/test_sim3_factor.py -q
git status --short
```

Expected: PASS.

---

## Task 7: Full-Prefix GTSAM PGO And ROVER Scoring

**Files:**
- Create: `LoopAnything/src/robust_loop_verifier/pgo.py`
- Create: `LoopAnything/tests/robust_loop_verifier/test_pgo.py`

- [ ] **Step 1: Write PGO tests**

Create `tests/robust_loop_verifier/test_pgo.py`:

```python
import numpy as np
import pytest

gtsam = pytest.importorskip("gtsam")

from robust_loop_verifier.geometry import make_transform
from robust_loop_verifier.pgo import PgoNoise, run_full_prefix_pgo, trajectory_deformation_rmse


def _poses(count):
    return [make_transform(np.eye(3), [float(i), 0.0, 0.0]) for i in range(count)]


def test_full_prefix_pgo_true_loop_has_small_deformation():
    poses = _poses(6)
    loop_factor = np.linalg.inv(poses[5]) @ poses[0]
    result = run_full_prefix_pgo(
        prefix_indices=list(range(6)),
        odom_poses=poses,
        loop_from_idx=5,
        loop_to_idx=0,
        loop_factor=loop_factor,
        noise=PgoNoise.default_for_tests(),
    )
    assert result.converged
    assert trajectory_deformation_rmse(poses, result.optimized_poses) < 1e-4


def test_full_prefix_pgo_false_loop_has_larger_deformation():
    poses = _poses(6)
    bad_loop = make_transform(np.eye(3), [-20.0, 0.0, 0.0])
    result = run_full_prefix_pgo(
        prefix_indices=list(range(6)),
        odom_poses=poses,
        loop_from_idx=5,
        loop_to_idx=0,
        loop_factor=bad_loop,
        noise=PgoNoise.default_for_tests(),
    )
    assert result.converged
    assert trajectory_deformation_rmse(poses, result.optimized_poses) > 0.1
```

- [ ] **Step 2: Implement PGO and ROVER scoring**

Create `src/robust_loop_verifier/pgo.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from robust_loop_verifier.geometry import pose_between, sim3_align_points


@dataclass(frozen=True)
class PgoNoise:
    prior_sigmas: List[float]
    odom_sigmas: List[float]
    loop_sigmas: List[float]

    @classmethod
    def default_for_tests(cls) -> "PgoNoise":
        return cls(
            prior_sigmas=[0.01, 0.01, 0.01, 0.1, 0.1, 0.1],
            odom_sigmas=[0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
            loop_sigmas=[0.1, 0.1, 0.1, 1.0, 1.0, 1.0],
        )


@dataclass(frozen=True)
class PgoResult:
    converged: bool
    optimized_poses: List[np.ndarray]
    error_before: float
    error_after: float
    failure_reason: str | None


def run_full_prefix_pgo(
    prefix_indices: List[int],
    odom_poses: List[np.ndarray],
    loop_from_idx: int,
    loop_to_idx: int,
    loop_factor: np.ndarray,
    noise: PgoNoise,
) -> PgoResult:
    import gtsam

    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.asarray(noise.prior_sigmas, dtype=np.float64))
    odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.asarray(noise.odom_sigmas, dtype=np.float64))
    loop_noise = gtsam.noiseModel.Diagonal.Sigmas(np.asarray(noise.loop_sigmas, dtype=np.float64))

    for idx, pose in zip(prefix_indices, odom_poses):
        values.insert(idx, _pose3(pose, gtsam))
    graph.add(gtsam.PriorFactorPose3(prefix_indices[0], _pose3(odom_poses[0], gtsam), prior_noise))
    for left_idx, right_idx, left_pose, right_pose in zip(prefix_indices[:-1], prefix_indices[1:], odom_poses[:-1], odom_poses[1:]):
        graph.add(gtsam.BetweenFactorPose3(left_idx, right_idx, _pose3(pose_between(left_pose, right_pose), gtsam), odom_noise))
    graph.add(gtsam.BetweenFactorPose3(loop_from_idx, loop_to_idx, _pose3(loop_factor, gtsam), loop_noise))

    try:
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values)
        initial_error = float(graph.error(values))
        optimized = optimizer.optimize()
        final_error = float(graph.error(optimized))
        optimized_poses = [_matrix_from_pose3(optimized.atPose3(idx)) for idx in prefix_indices]
        return PgoResult(True, optimized_poses, initial_error, final_error, None)
    except Exception as exc:
        return PgoResult(False, list(odom_poses), float("inf"), float("inf"), str(exc))


def trajectory_deformation_rmse(original_poses: List[np.ndarray], optimized_poses: List[np.ndarray]) -> float:
    original = np.asarray([pose[:3, 3] for pose in original_poses], dtype=np.float64)
    optimized = np.asarray([pose[:3, 3] for pose in optimized_poses], dtype=np.float64)
    if len(original) < 3:
        return float(np.sqrt(np.mean(np.sum((original - optimized) ** 2, axis=1))))
    return sim3_align_points(optimized, original).rmse


def _pose3(matrix: np.ndarray, gtsam_module):
    return gtsam_module.Pose3(
        gtsam_module.Rot3(np.asarray(matrix[:3, :3], dtype=np.float64)),
        gtsam_module.Point3(
            float(matrix[0, 3]),
            float(matrix[1, 3]),
            float(matrix[2, 3]),
        ),
    )


def _matrix_from_pose3(pose3) -> np.ndarray:
    return np.asarray(pose3.matrix(), dtype=np.float64)
```

- [ ] **Step 3: Run tests in DA3 environment**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src /home/chenguyuan/anaconda3/envs/da3/bin/python -m pytest tests/robust_loop_verifier/test_pgo.py -q
git status --short
```

Expected: PASS.

---

## Task 8: Metrics, Canonical Scores, And Failure Ordering

**Files:**
- Create: `LoopAnything/src/robust_loop_verifier/metrics.py`
- Create: `LoopAnything/tests/robust_loop_verifier/test_metrics.py`

- [ ] **Step 1: Write metrics tests**

Create `tests/robust_loop_verifier/test_metrics.py`:

```python
import math

from robust_loop_verifier.metrics import assign_failure_worst_scores, average_precision, max_recall_at_100_precision


def test_average_precision_larger_score_is_better():
    labels = [True, False, True]
    scores = [0.9, 0.8, 0.1]
    assert math.isclose(average_precision(labels, scores), (1.0 + 2.0 / 3.0) / 2.0)


def test_max_recall_at_100_precision_stops_before_false_positive():
    labels = [True, True, False, True]
    scores = [0.9, 0.8, 0.7, 0.1]
    assert math.isclose(max_recall_at_100_precision(labels, scores), 2.0 / 3.0)


def test_failure_scores_rank_after_normal_scores():
    scores = [1.0, None, 0.0, None]
    fixed = assign_failure_worst_scores(scores)
    assert fixed[1] < fixed[2]
    assert fixed[3] < fixed[2]
    assert fixed[1] != fixed[3]
```

- [ ] **Step 2: Implement metrics**

Create `src/robust_loop_verifier/metrics.py`:

```python
from __future__ import annotations

from typing import Iterable, List, Optional


def assign_failure_worst_scores(scores: Iterable[Optional[float]]) -> List[float]:
    raw = list(scores)
    finite = [float(score) for score in raw if score is not None]
    worst = min(finite) if finite else 0.0
    step = max(abs(worst), 1.0) * 1e-9
    fixed = []
    failure_rank = 1
    for score in raw:
        if score is None:
            fixed.append(worst - step * failure_rank)
            failure_rank += 1
        else:
            fixed.append(float(score))
    return fixed


def average_precision(labels: Iterable[bool], scores: Iterable[float]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda item: -item[0])
    positives = sum(1 for _, label in pairs if label)
    if positives == 0:
        return 0.0
    hit_count = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(pairs, start=1):
        if label:
            hit_count += 1
            precision_sum += hit_count / rank
    return precision_sum / positives


def max_recall_at_100_precision(labels: Iterable[bool], scores: Iterable[float]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda item: -item[0])
    positives = sum(1 for _, label in pairs if label)
    if positives == 0:
        return 0.0
    true_positives = 0
    false_positives = 0
    best_recall = 0.0
    for _, label in pairs:
        if label:
            true_positives += 1
        else:
            false_positives += 1
        if false_positives == 0:
            best_recall = max(best_recall, true_positives / positives)
    return best_recall
```

- [ ] **Step 3: Run tests and checkpoint**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/robust_loop_verifier/test_metrics.py -q
git status --short
```

Expected: PASS.

---

## Task 9: Artifacts And Visualizations

**Files:**
- Create: `LoopAnything/src/robust_loop_verifier/artifacts.py`
- Create: `LoopAnything/tests/robust_loop_verifier/test_artifacts.py`

- [ ] **Step 1: Write artifact tests**

Create `tests/robust_loop_verifier/test_artifacts.py`:

```python
from pathlib import Path

from PIL import Image

from robust_loop_verifier.artifacts import write_metrics_markdown, write_triplet_visual_record


def _image(path: Path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 8), color=color).save(path)


def test_write_metrics_markdown(tmp_path: Path):
    out = tmp_path / "metrics.md"
    write_metrics_markdown(out, {"salad": {"AP": 0.5, "MR@100P": 0.25}})
    text = out.read_text(encoding="utf-8")
    assert "| salad | 0.5000 | 0.2500 |" in text


def test_write_triplet_visual_record(tmp_path: Path):
    q, c, s = tmp_path / "q.png", tmp_path / "c.png", tmp_path / "s.png"
    _image(q, (255, 0, 0))
    _image(c, (0, 255, 0))
    _image(s, (0, 0, 255))
    out_dir = tmp_path / "visual"
    write_triplet_visual_record(out_dir, "q000010_c000003_s000002", q, c, s)
    assert (out_dir / "q000010_c000003_s000002" / "triplet.png").is_file()
```

- [ ] **Step 2: Implement artifacts**

Create `src/robust_loop_verifier/artifacts.py`:

```python
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Mapping

from PIL import Image, ImageDraw


def write_json(path: Path, data: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(data), indent=2, sort_keys=True), encoding="utf-8")


def write_metrics_markdown(path: Path, metrics: Mapping[str, Mapping[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| method | AP | MR@100P |", "|---|---:|---:|"]
    for method, values in metrics.items():
        lines.append(f"| {method} | {values['AP']:.4f} | {values['MR@100P']:.4f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_triplet_visual_record(
    root: Path,
    record_name: str,
    query_image: Path,
    candidate_image: Path,
    support_image: Path,
) -> None:
    out_dir = root / record_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for src, name in [(query_image, "query.png"), (candidate_image, "candidate.png"), (support_image, "support.png")]:
        shutil.copy2(src, out_dir / name)
    images = [Image.open(path).convert("RGB") for path in [query_image, candidate_image, support_image]]
    width = sum(image.width for image in images)
    height = max(image.height for image in images) + 20
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for label, image in zip(["query", "candidate", "support"], images):
        canvas.paste(image, (x, 20))
        draw.text((x + 2, 2), label, fill=(0, 0, 0))
        x += image.width
    canvas.save(out_dir / "triplet.png")
```

- [ ] **Step 3: Run tests and checkpoint**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/robust_loop_verifier/test_artifacts.py -q
git status --short
```

Expected: PASS.

---

## Task 10: Pipeline Orchestration With Mock Backends

**Files:**
- Create: `LoopAnything/src/robust_loop_verifier/pipeline.py`
- Create: `LoopAnything/tests/robust_loop_verifier/test_pipeline.py`

- [ ] **Step 1: Write mock pipeline test**

Create `tests/robust_loop_verifier/test_pipeline.py`:

```python
from pathlib import Path

from robust_loop_verifier.pipeline import run_mock_sequence_evaluation


def test_mock_pipeline_writes_candidate_records_and_metrics(tmp_path: Path):
    run_root = tmp_path / "run"
    result = run_mock_sequence_evaluation(run_root)
    assert result["candidate_count"] > 0
    assert (run_root / "candidate_records.jsonl").is_file()
    assert (run_root / "metrics.json").is_file()
    assert (run_root / "metrics.md").is_file()
```

- [ ] **Step 2: Implement mock pipeline**

Create `src/robust_loop_verifier/pipeline.py` with a synthetic end-to-end path:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np

from robust_loop_verifier.artifacts import write_json, write_metrics_markdown
from robust_loop_verifier.geometry import make_transform
from robust_loop_verifier.io import write_jsonl
from robust_loop_verifier.metrics import average_precision, assign_failure_worst_scores, max_recall_at_100_precision
from robust_loop_verifier.pgo import PgoNoise, run_full_prefix_pgo, trajectory_deformation_rmse
from robust_loop_verifier.retrieval import DescriptorSet, retrieve_historical_topk


def run_mock_sequence_evaluation(run_root: Path) -> dict:
    run_root.mkdir(parents=True, exist_ok=True)
    poses = [make_transform(np.eye(3), [float(i), 0.0, 0.0]) for i in range(8)]
    descriptors = DescriptorSet(
        keyframe_indices=list(range(8)),
        descriptors=np.eye(8, dtype=np.float64),
    )
    records = []
    labels = []
    rover_scores = []
    for query_idx in range(3, 8):
        retrieval = retrieve_historical_topk(query_idx, descriptors, top_k=2, recent_exclusion_keyframes=1)
        for candidate in retrieval.candidates:
            label = candidate.candidate_idx == 0 and query_idx >= 5
            loop_factor = np.linalg.inv(poses[query_idx]) @ poses[candidate.candidate_idx]
            pgo = run_full_prefix_pgo(
                prefix_indices=list(range(query_idx + 1)),
                odom_poses=poses[: query_idx + 1],
                loop_from_idx=query_idx,
                loop_to_idx=candidate.candidate_idx,
                loop_factor=loop_factor,
                noise=PgoNoise.default_for_tests(),
            )
            rmse = trajectory_deformation_rmse(poses[: query_idx + 1], pgo.optimized_poses)
            score = -rmse if pgo.converged else None
            records.append(
                {
                    "query_idx": query_idx,
                    "candidate_idx": candidate.candidate_idx,
                    "label": label,
                    "salad_score": candidate.score,
                    "trajectory_deformation_rmse": rmse,
                    "score_rover": score,
                    "pgo_converged": pgo.converged,
                }
            )
            labels.append(label)
            rover_scores.append(score)
    fixed_scores = assign_failure_worst_scores(rover_scores)
    metrics = {
        "da3_rover": {
            "AP": average_precision(labels, fixed_scores),
            "MR@100P": max_recall_at_100_precision(labels, fixed_scores),
        }
    }
    write_jsonl(run_root / "candidate_records.jsonl", records)
    write_json(run_root / "metrics.json", metrics)
    write_metrics_markdown(run_root / "metrics.md", metrics)
    return {"candidate_count": len(records), "metrics": metrics}
```

- [ ] **Step 3: Run tests in DA3 environment and checkpoint**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src /home/chenguyuan/anaconda3/envs/da3/bin/python -m pytest tests/robust_loop_verifier/test_pipeline.py -q
git status --short
```

Expected: PASS.

---

## Task 11: CLI Entry Points

**Files:**
- Create: `LoopAnything/src/robust_loop_verifier/cli.py`
- Modify: `LoopAnything/pyproject.toml`
- Create: `LoopAnything/tests/robust_loop_verifier/test_cli.py`

- [ ] **Step 1: Write CLI tests**

Create `tests/robust_loop_verifier/test_cli.py`:

```python
from typer.testing import CliRunner

from robust_loop_verifier.cli import app


def test_cli_help():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "preprocess-fusionportable" in result.stdout
    assert "run-mock" in result.stdout
```

- [ ] **Step 2: Implement CLI**

Create `src/robust_loop_verifier/cli.py`:

```python
from __future__ import annotations

from pathlib import Path

import typer

from robust_loop_verifier.fusionportable import preprocess_fusionportable_sequence
from robust_loop_verifier.pipeline import run_mock_sequence_evaluation
from robust_loop_verifier.schema import RobustLoopVerifierConfig


app = typer.Typer(no_args_is_help=True)


@app.command("preprocess-fusionportable")
def preprocess_fusionportable(
    config: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    raw_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    gt_trajectory_file: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    sequence_name: str = typer.Option(...),
    max_gt_delta_sec: float = typer.Option(0.05),
) -> None:
    cfg = RobustLoopVerifierConfig.from_yaml(config)
    out = preprocess_fusionportable_sequence(raw_dir, gt_trajectory_file, sequence_name, cfg, max_gt_delta_sec)
    typer.echo(str(out))


@app.command("run-mock")
def run_mock(output_root: Path = typer.Option(..., file_okay=False, dir_okay=True)) -> None:
    result = run_mock_sequence_evaluation(output_root)
    typer.echo(f"candidate_count={result['candidate_count']}")


if __name__ == "__main__":
    app()
```

Modify `pyproject.toml`:

```toml
[project.scripts]
da3 = "depth_anything_3.cli:app"
robust-loop-verifier = "robust_loop_verifier.cli:app"
```

- [ ] **Step 3: Run tests and help command**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/robust_loop_verifier/test_cli.py -q
PYTHONPATH=src python -m robust_loop_verifier.cli --help
git status --short
```

Expected: PASS and help text includes both commands.

---

## Task 12: Real SALAD And DA3 Backend Wiring

**Files:**
- Modify: `LoopAnything/src/robust_loop_verifier/retrieval.py`
- Modify: `LoopAnything/src/robust_loop_verifier/da3_runner.py`
- Create: `LoopAnything/tests/robust_loop_verifier/test_real_backend_config.py`

- [ ] **Step 1: Write backend config tests**

Create `tests/robust_loop_verifier/test_real_backend_config.py`:

```python
import numpy as np

from robust_loop_verifier.da3_runner import RealDa3RunnerConfig, convert_da3_extrinsics_to_c2w
from robust_loop_verifier.retrieval import SaladDescriptorBackendConfig


def test_real_da3_config_enforces_reference_strategy_and_resolution():
    config = RealDa3RunnerConfig(model_name="depth-anything/DA3-SMALL", process_res=504)
    assert config.ref_view_strategy == "first"
    assert not config.extrinsics_are_c2w


def test_convert_da3_3x4_w2c_to_c2w():
    extrinsics = np.eye(4, dtype=np.float64)[None, :3, :]
    extrinsics[0, 0, 3] = 3.0
    c2w = convert_da3_extrinsics_to_c2w(extrinsics, extrinsics_are_c2w=False)
    assert c2w.shape == (1, 4, 4)
    np.testing.assert_allclose(c2w[0, :3, 3], [-3.0, 0.0, 0.0])


def test_salad_backend_config_preserves_local_repo_and_checkpoint_paths(tmp_path):
    config = SaladDescriptorBackendConfig(
        salad_repo=tmp_path / "salad",
        checkpoint_path=tmp_path / "dino_salad.ckpt",
        device="cuda",
    )
    assert config.backbone == "dinov2_vitb14"
```

- [ ] **Step 2: Implement RealDa3RunnerConfig, real DA3 runner, and pose conversion guard**

Extend `da3_runner.py`:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robust_loop_verifier.geometry import invert_transform


@dataclass(frozen=True)
class RealDa3RunnerConfig:
    model_name: str
    device: str = "cuda"
    process_res: int = 504
    process_res_method: str = "upper_bound_resize"
    ref_view_strategy: str = "first"
    extrinsics_are_c2w: bool = False

    def __post_init__(self):
        if self.process_res < 224:
            raise ValueError("real DA3 geometry requires process_res >= 224")
        if self.ref_view_strategy != "first":
            raise ValueError('ref_view_strategy must be "first"')


def convert_da3_extrinsics_to_c2w(extrinsics: np.ndarray, extrinsics_are_c2w: bool) -> np.ndarray:
    poses = _as_4x4_batch(extrinsics)
    if extrinsics_are_c2w:
        return poses
    return np.stack([invert_transform(pose) for pose in poses], axis=0)


class RealDa3Runner:
    def __init__(self, config: RealDa3RunnerConfig, model: Any | None = None) -> None:
        self.config = config
        self._model = model

    def run_triplet(self, triplet: Da3Triplet) -> Da3TripletResult:
        import torch
        from PIL import Image

        model = self._load_model()
        images = [np.asarray(Image.open(path).convert("RGB")) for path in triplet.image_paths]
        images_cpu, _, _ = model.input_processor(
            images,
            process_res=self.config.process_res,
            process_res_method=self.config.process_res_method,
            num_workers=1,
            print_progress=False,
            sequential=True,
            desc=None,
        )
        images_gpu = images_cpu.view(1, len(images), *images_cpu.shape[1:]).to(self.config.device).float()
        with torch.inference_mode():
            output = model.forward(
                images_gpu,
                extrinsics=None,
                intrinsics=None,
                export_feat_layers=[],
                infer_gs=False,
                use_ray_pose=False,
                ref_view_strategy=self.config.ref_view_strategy,
            )
        extrinsics = _field(output, "extrinsics")
        c2w = convert_da3_extrinsics_to_c2w(_to_numpy(extrinsics), self.config.extrinsics_are_c2w)
        if c2w.shape[0] != len(triplet.image_paths):
            raise ValueError("DA3 prediction extrinsics count mismatch")
        if not np.isfinite(c2w).all():
            raise ValueError("DA3 prediction extrinsics contain non-finite values")
        return Da3TripletResult(
            keyframe_indices=triplet.keyframe_indices,
            view_roles=triplet.view_roles,
            predicted_c2w=c2w,
        )

    def _load_model(self):
        if self._model is None:
            from depth_anything_3.api import DepthAnything3

            self._model = DepthAnything3.from_pretrained(self.config.model_name).to(self.config.device)
            self._model.eval()
        return self._model


def _field(output: Any, name: str) -> Any:
    if isinstance(output, dict):
        return output[name]
    return getattr(output, name)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_4x4_batch(extrinsics: np.ndarray) -> np.ndarray:
    poses = np.asarray(extrinsics, dtype=np.float64)
    if poses.ndim == 4:
        if poses.shape[0] != 1:
            raise ValueError("DA3 batch size must be 1 for triplet inference")
        poses = poses[0]
    if poses.ndim != 3:
        raise ValueError(f"DA3 extrinsics must have shape Nx3x4 or Nx4x4, got {poses.shape}")
    if poses.shape[1:] == (4, 4):
        return poses
    if poses.shape[1:] == (3, 4):
        padded = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], poses.shape[0], axis=0)
        padded[:, :3, :4] = poses
        return padded
    raise ValueError(f"DA3 extrinsics must have shape Nx3x4 or Nx4x4, got {poses.shape}")
```

- [ ] **Step 3: Add SALAD descriptor backend**

Extend `retrieval.py` with a backend interface:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class SaladDescriptorBackendConfig:
    salad_repo: Path
    checkpoint_path: Path
    device: str = "cuda"
    backbone: str = "dinov2_vitb14"
    batch_size: int = 32


class DescriptorBackend:
    def compute(self, image_paths: list[str], keyframe_indices: list[int]) -> DescriptorSet:
        raise NotImplementedError


class SaladDescriptorBackend:
    def __init__(self, config: SaladDescriptorBackendConfig, model=None) -> None:
        self.config = config
        self._model = model

    def compute(self, image_paths: list[str], keyframe_indices: list[int]) -> DescriptorSet:
        import torch
        from PIL import Image
        from torchvision import transforms

        model = self._load_model()
        transform = transforms.Compose(
            [
                transforms.Resize((322, 322)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        descriptors = []
        with torch.inference_mode():
            for start in range(0, len(image_paths), self.config.batch_size):
                batch_paths = image_paths[start : start + self.config.batch_size]
                batch = torch.stack(
                    [transform(Image.open(path).convert("RGB")) for path in batch_paths], dim=0
                ).to(self.config.device)
                output = model(batch).detach().cpu().numpy().astype(np.float64)
                descriptors.append(output)
        matrix = np.concatenate(descriptors, axis=0)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms <= 0.0) or not np.isfinite(norms).all():
            raise ValueError("SALAD descriptor norm must be finite and positive")
        return DescriptorSet(keyframe_indices=list(keyframe_indices), descriptors=matrix / norms)

    def _load_model(self):
        import torch

        if self._model is not None:
            return self._model
        self._model = torch.hub.load(
            str(self.config.salad_repo),
            "dinov2_salad",
            source="local",
            backbone=self.config.backbone,
            pretrained=False,
        )
        state = torch.load(self.config.checkpoint_path, map_location="cpu")
        state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
        cleaned = {key.removeprefix("model."): value for key, value in state_dict.items()}
        missing, unexpected = self._model.load_state_dict(cleaned, strict=False)
        if missing or unexpected:
            raise ValueError(f"SALAD checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        self._model = self._model.eval().to(self.config.device)
        return self._model
```

- [ ] **Step 4: Run non-GPU tests**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/robust_loop_verifier/test_real_backend_config.py tests/robust_loop_verifier/test_retrieval.py tests/robust_loop_verifier/test_da3_runner.py -q
git status --short
```

Expected: PASS without loading real DA3 or SALAD checkpoints.

- [ ] **Step 5: Run mock backend smoke**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src /home/chenguyuan/anaconda3/envs/da3/bin/python -m robust_loop_verifier.cli run-mock --output-root /tmp/robust_loop_verifier_mock
```

Expected: command completes and writes `/tmp/robust_loop_verifier_mock/metrics.json`.

---

## Task 13: Real FusionPortableV2 Handheld Smoke Run

**Files:**
- Modify: `LoopAnything/src/robust_loop_verifier/pipeline.py`
- Modify: `LoopAnything/src/robust_loop_verifier/cli.py`
- Create: `LoopAnything/tests/robust_loop_verifier/test_manifest_contract.py`

- [ ] **Step 1: Add manifest contract test**

Create `tests/robust_loop_verifier/test_manifest_contract.py`:

```python
from pathlib import Path

from robust_loop_verifier.io import read_jsonl


def test_preprocessed_cache_contract(tmp_path: Path):
    sequence_dir = tmp_path / "FusionPortableV2" / "handheld" / "sequence"
    sequence_dir.mkdir(parents=True)
    (sequence_dir / "manifest.json").write_text('{"sequence_name":"sequence"}', encoding="utf-8")
    (sequence_dir / "keyframes.jsonl").write_text(
        '{"idx":0,"timestamp":0.0,"image_path":"images/000000.png","odom_pose":[],"gt_pose":[]}\n',
        encoding="utf-8",
    )
    (sequence_dir / "positives.jsonl").write_text('{"query_idx":0,"positive_indices":[]}\n', encoding="utf-8")
    assert (sequence_dir / "manifest.json").is_file()
    assert list(read_jsonl(sequence_dir / "keyframes.jsonl"))[0]["idx"] == 0
    assert list(read_jsonl(sequence_dir / "positives.jsonl"))[0]["positive_indices"] == []
```

- [ ] **Step 2: Add CLI run command for cached sequence**

Extend `cli.py`:

```python
@app.command("run-cache")
def run_cache(
    config: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    sequence_cache: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    output_root: Path = typer.Option(..., file_okay=False, dir_okay=True),
    query_limit: int = typer.Option(20),
    backend: str = typer.Option("real"),
) -> None:
    from robust_loop_verifier.pipeline import run_cached_sequence

    cfg = RobustLoopVerifierConfig.from_yaml(config)
    summary = run_cached_sequence(
        cfg,
        sequence_cache,
        output_root,
        query_limit=query_limit,
        backend=backend,
    )
    typer.echo(f"candidate_count={summary['candidate_count']}")
```

Implement `run_cached_sequence()` in `pipeline.py` with an explicit backend selector:

```python
def run_cached_sequence(
    config: RobustLoopVerifierConfig,
    sequence_cache: Path,
    output_root: Path,
    query_limit: int | None,
    backend: str,
) -> dict:
    if backend not in {"mock", "real"}:
        raise ValueError('backend must be "mock" or "real"')
    keyframes = list(read_jsonl(sequence_cache / "keyframes.jsonl"))
    positives_by_query = {
        int(row["query_idx"]): set(int(idx) for idx in row["positive_indices"])
        for row in read_jsonl(sequence_cache / "positives.jsonl")
    }
    selected_keyframes = keyframes if query_limit is None else keyframes[:query_limit]
    descriptor_backend, da3_runner = _make_backends(config, backend)
    descriptor_set = descriptor_backend.compute(
        [str(sequence_cache / str(row["image_path"])) for row in keyframes if row["image_path"]],
        [int(row["idx"]) for row in keyframes if row["image_path"]],
    )
    records = []
    method_scores = {"salad": [], "da3_sim3": [], "da3_rover": []}
    labels = []
    for query in selected_keyframes:
        query_idx = int(query["idx"])
        retrieval = retrieve_historical_topk(
            query_idx=query_idx,
            descriptors=descriptor_set,
            top_k=config.retrieval_top_k_main,
            recent_exclusion_keyframes=config.recent_exclusion_keyframes,
        )
        for candidate in retrieval.candidates:
            label = candidate.candidate_idx in positives_by_query.get(query_idx, set())
            candidate_record = _score_candidate(config, sequence_cache, keyframes, query_idx, candidate, da3_runner)
            records.append({**candidate_record, "label": label, "salad_score": candidate.score})
            labels.append(label)
            method_scores["salad"].append(candidate.score)
            method_scores["da3_sim3"].append(candidate_record.get("score_da3_sim3"))
            method_scores["da3_rover"].append(candidate_record.get("score_rover"))
    metrics = _compute_method_metrics(labels, method_scores)
    _write_run_artifacts(output_root, config, records, metrics)
    return {"candidate_count": len(records), "metrics": metrics}
```

The helper `_make_backends()` returns `MockDa3Runner` plus an in-memory descriptor backend for `backend="mock"`, and `RealDa3Runner` plus `SaladDescriptorBackend` for `backend="real"`. `_score_candidate()` is the single candidate path: support selection, DA3 triplet, candidate-support Sim3 factor, full-prefix PGO, trajectory deformation score. `_compute_method_metrics()` must call `assign_failure_worst_scores()` for every method before AP/MR@100P computation.

- [ ] **Step 3: Run unit tests**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/robust_loop_verifier -q
git status --short
```

Expected: PASS.

- [ ] **Step 4: Preprocess real handheld_escalator00**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26
bash LoopAnything/robust_loop_verification_scripts/generate_fusionportable_dataset_cache.sh handheld_escalator00
```

Expected: output path under `/data/datasets/FusionPortable/robust_loop_verifier_cache/FusionPortableV2/handheld/handheld_escalator00` with `manifest.json`, `keyframes.jsonl`, `positives.jsonl`, and `images/`.

For `handheld` and `legged`, the script uses:

```text
gt_trajectory_file=/data/datasets/FusionPortable/fusionportable_loop_dataset/<platform>/<sequence>/raw/trajectory_keyframes.txt
gt_label_source=aster_slam_trajectory_keyframes
```

External GT/reference trajectories remain available for other platforms, but
are not the default label source for `handheld` or `legged`.

- [ ] **Step 5: Run cached sequence smoke with query limit**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src /home/chenguyuan/anaconda3/envs/da3/bin/python -m robust_loop_verifier.cli run-cache \
  --config configs/robust_loop_verifier/fusionportablev2_handheld.yaml \
  --sequence-cache /data/datasets/FusionPortable/robust_loop_verifier_cache/FusionPortableV2/handheld/handheld_escalator00 \
  --output-root /tmp/robust_loop_verifier_handheld_escalator00_smoke \
  --query-limit 20 \
  --backend real
```

Expected: writes `candidate_records.jsonl`, `metrics.json`, `metrics.md`, `pr_curves/`, `visual_records/`, and `trajectory_plots/`.

---

## Task 14: Final Verification And Review

**Files:**
- Review all files created or modified by Tasks 1-13.

- [ ] **Step 1: Run full unit test suite for the new package**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src /home/chenguyuan/anaconda3/envs/da3/bin/python -m pytest tests/robust_loop_verifier -q
```

Expected: PASS.

- [ ] **Step 2: Compile new package**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
python -m py_compile src/robust_loop_verifier/*.py
```

Expected: no output and exit code 0.

- [ ] **Step 3: Check forbidden dependency boundary**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
rg -n "loop_policy|safe_loop_factor_v1|x_geom" src/robust_loop_verifier tests/robust_loop_verifier
```

Expected: no matches.

- [ ] **Step 4: Run real smoke commands**

Run the real preprocess and `run-cache --query-limit 20` commands from Task 13.

Expected:

- `positives.jsonl` exists and contains positives generated only from the
  selected GT/reference trajectory source, translation radius, configurable
  rotation threshold, and recent exclusion.
- `candidate_records.jsonl` includes a larger-is-better score for every retrieved candidate and every reported method.
- failed method scores are not excluded from `metrics.json`.
- `metrics.md` reports `SALAD score only`, `SALAD + DA3/Sim3 self-consistency score`, and `SALAD + DA3-ROVER full-prefix trajectory score`.

- [ ] **Step 5: Request review**

Ask for a subagent review with this scope:

```text
Review only src/robust_loop_verifier, tests/robust_loop_verifier,
configs/robust_loop_verifier, and pyproject.toml. Verify Stage 1 scope,
no legacy loop_policy dependency, GT label purity, failure-score handling,
and full-prefix GTSAM PGO semantics.
```

- [ ] **Step 6: Git checkpoint**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
git status --short
```

Expected: shows all implementation files. Do not commit unless the user explicitly requests a commit.

---

## Self-Review Checklist

- Spec coverage:
  - FusionPortableV2 preprocess: Task 3 and Task 13.
  - Explicit GT radius, rotation threshold, recent exclusion, and GT/reference
    trajectory-only labels: Task 1, Task 3, Task 13, and Task 14.
  - `handheld`/`legged` AsterSLAM keyframe-trajectory GT label source: Task 13
    and Convergence Review 2026-05-17.
  - Online-causal SALAD retrieval top10/top5/top20: Task 4 and Task 13.
  - Nearest-candidate support selection: Task 5.
  - DA3 `w2c -> c2w`, `ref_view_strategy="first"`, `process_res=504`: Task 6 and Task 12.
  - Candidate-support Sim3 metric loop factor: Task 6.
  - Full-prefix GTSAM PGO: Task 7.
  - ROVER larger-is-better canonical score: Task 7 and Task 8.
  - Residual-aware ROVER-style scoring is intentionally deferred to the
    improvement phase based on the 2026-05-17 runtime analysis.
  - AP/MR@100P and failure worst-score handling: Task 8.
  - Artifacts and visualizations: Task 9 and Task 13.
  - No AsterSLAM online port: plan contains no AsterSLAM source edits.
  - No legacy learned-policy priors: Task 1 and Task 14 enforce boundary.
- Placeholder scan:
  - No task uses open-ended implementation wording.
  - Real DA3 and SALAD GPU execution is isolated behind Task 12 and Task 13.
- Type consistency:
  - `RobustLoopVerifierConfig`, `PgoNoise`, `DescriptorSet`, `RetrievalRecord`, `SupportSelection`, `Da3TripletResult`, `Sim3LoopFactorResult`, and `PgoResult` are defined before use.
