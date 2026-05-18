# DA3-ROVER++ Stage A Support Uncertainty Implementation Plan

> **Status update (2026-05-18): PAUSED / mixed ablation result.**
>
> The implementation tasks below were executed far enough to run
> FusionPortableV2 handheld support-ensemble experiments. The result is mixed:
> `handheld_escalator00` strongly favors support-ensemble graph evidence, but
> the five-sequence handheld average still does not show a robust AP advantage
> over deformation-only ROVER-style scoring:
>
> ```text
> handheld_escalator00:
>   ROVER deformation only:                       AP=0.695, MR@100P=0.020
>   DA3-ROVER++ support ensemble graph evidence:  AP=0.906, MR@100P=0.159
>
> five handheld sequences:
>   ROVER deformation only:                       AP=0.853, MR@100P=0.324
>   DA3-ROVER++ support ensemble graph evidence:  AP=0.808, MR@100P=0.338
> ```
>
> Therefore this plan is not considered converged as the sole main paper method.
> The current Stage A code and scripts should be treated as experimental
> infrastructure and an important ablation/source of ideas, not as the complete
> active research direction. The active follow-up direction is now:
>
> ```text
> LoopAnything/docs/superpowers/specs/2026-05-18-self-calibrated-counterfactual-verifier-design.md
> ```
>
> Do not continue this plan task-by-task as the mainline unless explicitly
> reviving Stage A. New work should start from the self-calibrated
> counterfactual verifier spec, while preserving support-ensemble graph evidence
> as a useful ablation and possible diagnostic component.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Stage A Support-Ensemble DA3 Test-Time Uncertainty for the offline robust loop verifier.

**Architecture:** Extend the existing `robust_loop_verifier` package without importing or referencing legacy `loop_policy`. Stage A keeps each DA3 forward isolated as `[query, candidate, support]`, aggregates multiple support-derived loop factors into one robust SE(3) mean plus diagonal covariance, and scores the candidate with covariance-aware PGO graph evidence.

**Tech Stack:** Python 3.9+, NumPy, PyYAML, existing DA3 runner, Python GTSAM 4.1.1, pytest.

**Spec:** `LoopAnything/docs/superpowers/specs/2026-05-18-da3-rover-plus-plus-stage-a-support-uncertainty-design.md`

**Git Policy:** Do not commit during execution unless the user explicitly requests it. Each task ends with `git status --short`.

---

## Source Isolation Constraint

Do not reference, import, copy, or use:

```text
LoopAnything/src/loop_policy
LoopAnything/tests/loop_policy
LoopAnything/docs/superpowers/specs/legacy
LoopAnything/docs/superpowers/plans/legacy
docs/deferred
```

The only allowed continuity from previous work is the already implemented DA3 pose convention:

```text
DA3 prediction.extrinsics are native w2c and must be inverted to c2w.
DA3 ref_view_strategy must be first.
DA3 process_res=504 is the real-geometry default.
```

## DA3 Isolation Constraint

Every real DA3 forward must contain one geometric group only:

```text
[query, candidate, support]
```

Do not batch independent triplets or independent candidates in DA3 input. This is required because DA3's transformer performs cross-view attention over images in the forward pass.

## File Structure

- Modify `LoopAnything/src/robust_loop_verifier/schema.py`
  - Add optional Stage A config dataclasses and defaults.
- Modify `LoopAnything/configs/robust_loop_verifier/fusionportablev2_handheld.yaml`
  - Add explicit Stage A disabled/default-compatible section.
- Create `LoopAnything/configs/robust_loop_verifier/fusionportablev2_handheld_stage_a.yaml`
  - Stage A experiment config with `support_count=4`.
- Modify `LoopAnything/src/robust_loop_verifier/support.py`
  - Add multi-support selection while preserving existing single-support API.
- Modify `LoopAnything/src/robust_loop_verifier/geometry.py`
  - Add SO(3)/SE(3) log utilities and weighted SE(3) mean.
- Create `LoopAnything/src/robust_loop_verifier/support_ensemble.py`
  - Support weights, robust aggregation, diagonal covariance, graph-evidence helper.
- Modify `LoopAnything/src/robust_loop_verifier/pgo.py`
  - Add per-candidate loop noise and residual diagnostics.
- Modify `LoopAnything/src/robust_loop_verifier/pipeline.py`
  - Add optional Stage A path, candidate record fields, and metrics method.
- Modify `LoopAnything/src/robust_loop_verifier/score_sweep.py`
  - Include Stage A score when present.
- Create tests:
  - `LoopAnything/tests/robust_loop_verifier/test_stage_a_config.py`
  - `LoopAnything/tests/robust_loop_verifier/test_support_ensemble.py`
  - Extend `test_support.py`, `test_geometry.py`, `test_pgo.py`, `test_pipeline.py`, `test_score_sweep.py`

---

## Task 1: Stage A Config Schema

**Files:**
- Modify: `LoopAnything/src/robust_loop_verifier/schema.py`
- Modify: `LoopAnything/configs/robust_loop_verifier/fusionportablev2_handheld.yaml`
- Create: `LoopAnything/configs/robust_loop_verifier/fusionportablev2_handheld_stage_a.yaml`
- Create: `LoopAnything/tests/robust_loop_verifier/test_stage_a_config.py`

- [ ] **Step 1: Write failing config tests**

Create `tests/robust_loop_verifier/test_stage_a_config.py`:

```python
from pathlib import Path

import pytest

from robust_loop_verifier.schema import RobustLoopVerifierConfig


def _base_config(tmp_path: Path, **stage_a_overrides):
    stage_a = {
        "enabled": True,
        "support_count": 4,
        "sigma_rot_floor": 0.05,
        "sigma_trans_floor": 0.25,
        "covariance_scale": 1.0,
        "c_align": 1.0,
        "c_consensus": 2.0,
        "lambda_dir": 1.0,
        "robust_iterations": 3,
    }
    stage_a.update(stage_a_overrides)
    return {
        "dataset_name": "unit",
        "platform": "tiny",
        "input_root": str(tmp_path / "input"),
        "output_root": str(tmp_path / "cache"),
        "gt_root": str(tmp_path / "gt"),
        "positive_radius_m": 0.5,
        "positive_max_rotation_deg": 45.0,
        "recent_exclusion_keyframes": 2,
        "retrieval_top_k_main": 4,
        "retrieval_top_k_ablations": [1, 4],
        "support_window": 4,
        "support_count": 1,
        "min_support_baseline_m": 0.3,
        "pgo_noise": {
            "prior_sigmas": [0.01, 0.01, 0.01, 0.1, 0.1, 0.1],
            "odom_sigmas": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
            "loop_sigmas": [0.1, 0.1, 0.1, 1.0, 1.0, 1.0],
        },
        "da3": {"process_res": 504, "ref_view_strategy": "first"},
        "stage_a": stage_a,
    }


def test_stage_a_config_parses_explicit_values(tmp_path: Path):
    config = RobustLoopVerifierConfig.from_mapping(_base_config(tmp_path))

    assert config.stage_a.enabled is True
    assert config.stage_a.support_count == 4
    assert config.stage_a.sigma_rot_floor == 0.05
    assert config.stage_a.sigma_trans_floor == 0.25
    assert config.stage_a.covariance_scale == 1.0
    assert config.stage_a.c_align == 1.0
    assert config.stage_a.c_consensus == 2.0
    assert config.stage_a.lambda_dir == 1.0
    assert config.stage_a.robust_iterations == 3


def test_stage_a_config_defaults_to_disabled_when_missing(tmp_path: Path):
    data = _base_config(tmp_path)
    data.pop("stage_a")

    config = RobustLoopVerifierConfig.from_mapping(data)

    assert config.stage_a.enabled is False
    assert config.stage_a.support_count == 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("support_count", 0, "stage_a.support_count must be positive"),
        ("sigma_rot_floor", 0.0, "stage_a.sigma_rot_floor must be positive"),
        ("sigma_trans_floor", -1.0, "stage_a.sigma_trans_floor must be positive"),
        ("covariance_scale", 0.0, "stage_a.covariance_scale must be positive"),
        ("c_align", 0.0, "stage_a.c_align must be positive"),
        ("c_consensus", 0.0, "stage_a.c_consensus must be positive"),
        ("lambda_dir", -0.1, "stage_a.lambda_dir must be non-negative"),
        ("robust_iterations", 0, "stage_a.robust_iterations must be positive"),
    ],
)
def test_stage_a_config_rejects_invalid_values(
    tmp_path: Path, field: str, value: object, message: str
):
    with pytest.raises(ValueError, match=message):
        RobustLoopVerifierConfig.from_mapping(_base_config(tmp_path, **{field: value}))
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/robust_loop_verifier/test_stage_a_config.py -q
```

Expected: FAIL because `RobustLoopVerifierConfig.stage_a` does not exist.

- [ ] **Step 3: Implement Stage A dataclass**

In `src/robust_loop_verifier/schema.py`, add above `RobustLoopVerifierConfig`:

```python
@dataclass(frozen=True)
class StageAConfig:
    enabled: bool
    support_count: int
    sigma_rot_floor: float
    sigma_trans_floor: float
    covariance_scale: float
    c_align: float
    c_consensus: float
    lambda_dir: float
    robust_iterations: int

    @classmethod
    def disabled(cls) -> "StageAConfig":
        return cls(
            enabled=False,
            support_count=1,
            sigma_rot_floor=0.05,
            sigma_trans_floor=0.25,
            covariance_scale=1.0,
            c_align=1.0,
            c_consensus=2.0,
            lambda_dir=1.0,
            robust_iterations=3,
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "StageAConfig":
        if data is None:
            return cls.disabled()
        if not isinstance(data, Mapping):
            raise ValueError("stage_a must be a mapping")

        config = cls(
            enabled=bool(data.get("enabled", False)),
            support_count=int(data.get("support_count", 1)),
            sigma_rot_floor=float(data.get("sigma_rot_floor", 0.05)),
            sigma_trans_floor=float(data.get("sigma_trans_floor", 0.25)),
            covariance_scale=float(data.get("covariance_scale", 1.0)),
            c_align=float(data.get("c_align", 1.0)),
            c_consensus=float(data.get("c_consensus", 2.0)),
            lambda_dir=float(data.get("lambda_dir", 1.0)),
            robust_iterations=int(data.get("robust_iterations", 3)),
        )
        config._validate()
        return config

    def _validate(self) -> None:
        if self.support_count <= 0:
            raise ValueError("stage_a.support_count must be positive")
        if not np.isfinite(self.sigma_rot_floor) or self.sigma_rot_floor <= 0.0:
            raise ValueError("stage_a.sigma_rot_floor must be positive")
        if not np.isfinite(self.sigma_trans_floor) or self.sigma_trans_floor <= 0.0:
            raise ValueError("stage_a.sigma_trans_floor must be positive")
        if not np.isfinite(self.covariance_scale) or self.covariance_scale <= 0.0:
            raise ValueError("stage_a.covariance_scale must be positive")
        if not np.isfinite(self.c_align) or self.c_align <= 0.0:
            raise ValueError("stage_a.c_align must be positive")
        if not np.isfinite(self.c_consensus) or self.c_consensus <= 0.0:
            raise ValueError("stage_a.c_consensus must be positive")
        if not np.isfinite(self.lambda_dir) or self.lambda_dir < 0.0:
            raise ValueError("stage_a.lambda_dir must be non-negative")
        if self.robust_iterations <= 0:
            raise ValueError("stage_a.robust_iterations must be positive")
```

Add `stage_a: StageAConfig` to `RobustLoopVerifierConfig`, and in `from_mapping` pass:

```python
stage_a=StageAConfig.from_mapping(data.get("stage_a")),
```

- [ ] **Step 4: Update configs**

Append to `configs/robust_loop_verifier/fusionportablev2_handheld.yaml`:

```yaml
stage_a:
  enabled: false
  support_count: 1
  sigma_rot_floor: 0.05
  sigma_trans_floor: 0.25
  covariance_scale: 1.0
  c_align: 1.0
  c_consensus: 2.0
  lambda_dir: 1.0
  robust_iterations: 3
```

Create `configs/robust_loop_verifier/fusionportablev2_handheld_stage_a.yaml` with the same base fields as `fusionportablev2_handheld.yaml`, but:

```yaml
retrieval_top_k_main: 10
support_count: 1
stage_a:
  enabled: true
  support_count: 4
  sigma_rot_floor: 0.05
  sigma_trans_floor: 0.25
  covariance_scale: 1.0
  c_align: 1.0
  c_consensus: 2.0
  lambda_dir: 1.0
  robust_iterations: 3
```

Keep `retrieval_top_k_main: 10` for Stage A because candidate-level reranking is Stage B. Stage A evaluates candidate measurements from the existing retrieval set.

- [ ] **Step 5: Verify tests**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/robust_loop_verifier/test_stage_a_config.py tests/robust_loop_verifier/test_schema_io.py -q
```

Expected: PASS.

- [ ] **Step 6: Check status**

Run:

```bash
git status --short
```

Expected: only Stage A config/schema/test files are changed or added.

---

## Task 2: Multi-Support Selection

**Files:**
- Modify: `LoopAnything/src/robust_loop_verifier/support.py`
- Modify: `LoopAnything/tests/robust_loop_verifier/test_support.py`

- [ ] **Step 1: Add failing multi-support tests**

Append to `tests/robust_loop_verifier/test_support.py`:

```python
from robust_loop_verifier.support import select_supports


def test_select_supports_returns_nearest_candidate_neighbors_in_order():
    poses = {idx: np.eye(4, dtype=np.float64) for idx in range(20)}
    for idx, pose in poses.items():
        pose[0, 3] = float(idx) * 0.5

    result = select_supports(
        query_idx=15,
        candidate_idx=5,
        available_indices=list(range(20)),
        image_indices=list(range(20)),
        camera_poses=poses,
        support_window=4,
        recent_exclusion_keyframes=2,
        min_support_baseline_m=0.3,
        support_count=4,
    )

    assert result.query_idx == 15
    assert result.candidate_idx == 5
    assert [support.support_idx for support in result.supports] == [4, 6, 3, 7]
    assert result.rejection_reason is None


def test_select_supports_reports_no_valid_supports():
    poses = {0: np.eye(4, dtype=np.float64), 10: np.eye(4, dtype=np.float64)}

    result = select_supports(
        query_idx=10,
        candidate_idx=0,
        available_indices=[0, 10],
        image_indices=[0, 10],
        camera_poses=poses,
        support_window=1,
        recent_exclusion_keyframes=2,
        min_support_baseline_m=0.3,
        support_count=4,
    )

    assert result.supports == []
    assert result.rejection_reason == "no_valid_support"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/robust_loop_verifier/test_support.py -q
```

Expected: FAIL because `select_supports` does not exist.

- [ ] **Step 3: Implement multi-support dataclasses and function**

In `src/robust_loop_verifier/support.py`, add:

```python
@dataclass(frozen=True)
class SupportCandidate:
    support_idx: int
    support_baseline_m: float


@dataclass(frozen=True)
class MultiSupportSelection:
    query_idx: int
    candidate_idx: int
    supports: list[SupportCandidate]
    rejection_reason: str | None = None
```

Add:

```python
def select_supports(
    query_idx: int,
    candidate_idx: int,
    available_indices,
    image_indices,
    camera_poses,
    support_window: int,
    recent_exclusion_keyframes: int,
    min_support_baseline_m: float,
    support_count: int,
) -> MultiSupportSelection:
    if support_count <= 0:
        raise ValueError("support_count must be positive")
    if candidate_idx not in camera_poses:
        return MultiSupportSelection(query_idx, candidate_idx, [], "missing_candidate_pose")

    image_index_set = set(image_indices)
    try:
        candidate_translation = _pose_translation(camera_poses[candidate_idx])
    except ValueError:
        return MultiSupportSelection(query_idx, candidate_idx, [], "invalid_candidate_pose")

    valid_supports: list[SupportCandidate] = []
    for support_idx in available_indices:
        if support_idx == candidate_idx:
            continue
        if abs(support_idx - candidate_idx) > support_window:
            continue
        if abs(query_idx - support_idx) <= recent_exclusion_keyframes:
            continue
        if support_idx not in image_index_set:
            continue
        if support_idx not in camera_poses:
            continue
        try:
            support_translation = _pose_translation(camera_poses[support_idx])
        except ValueError:
            continue
        baseline_m = float(np.linalg.norm(candidate_translation - support_translation))
        if baseline_m < min_support_baseline_m:
            continue
        valid_supports.append(SupportCandidate(support_idx, baseline_m))

    valid_supports.sort(key=lambda support: (abs(support.support_idx - candidate_idx), support.support_idx))
    selected = valid_supports[:support_count]
    if not selected:
        return MultiSupportSelection(query_idx, candidate_idx, [], "no_valid_support")
    return MultiSupportSelection(query_idx, candidate_idx, selected, None)
```

Update existing `select_support` to call `select_supports(..., support_count=1)` and convert the first result into `SupportSelection`. Preserve all current rejection reasons.

- [ ] **Step 4: Verify support tests**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/robust_loop_verifier/test_support.py -q
```

Expected: PASS.

- [ ] **Step 5: Check status**

Run:

```bash
git status --short
```

Expected: support module and tests changed.

---

## Task 3: SE(3) Log And Robust Mean Utilities

**Files:**
- Modify: `LoopAnything/src/robust_loop_verifier/geometry.py`
- Modify: `LoopAnything/tests/robust_loop_verifier/test_geometry.py`

- [ ] **Step 1: Add failing geometry tests**

Append to `tests/robust_loop_verifier/test_geometry.py`:

```python
from robust_loop_verifier.geometry import se3_log, weighted_se3_mean


def test_se3_log_identity_is_zero():
    np.testing.assert_allclose(se3_log(np.eye(4)), np.zeros(6), atol=1e-12)


def test_se3_log_translation_uses_last_three_components():
    transform = make_transform(np.eye(3), [1.0, -2.0, 3.0])

    residual = se3_log(transform)

    np.testing.assert_allclose(residual[:3], np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(residual[3:], [1.0, -2.0, 3.0], atol=1e-12)


def test_weighted_se3_mean_translation_matches_weighted_average_for_identity_rotations():
    poses = [
        make_transform(np.eye(3), [0.0, 0.0, 0.0]),
        make_transform(np.eye(3), [2.0, 0.0, 0.0]),
        make_transform(np.eye(3), [10.0, 0.0, 0.0]),
    ]

    mean = weighted_se3_mean(poses, weights=[1.0, 1.0, 0.0], iterations=3)

    np.testing.assert_allclose(mean[:3, :3], np.eye(3), atol=1e-9)
    np.testing.assert_allclose(mean[:3, 3], [1.0, 0.0, 0.0], atol=1e-9)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/robust_loop_verifier/test_geometry.py -q
```

Expected: FAIL because `se3_log` and `weighted_se3_mean` do not exist.

- [ ] **Step 3: Implement SE(3) utilities**

Add to `src/robust_loop_verifier/geometry.py`:

```python
def so3_log(rotation) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("Rotation must have shape (3, 3)")
    angle_cos = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(angle_cos))
    if angle < 1e-12:
        return np.zeros(3, dtype=np.float64)
    skew = (rotation - rotation.T) * (0.5 * angle / np.sin(angle))
    return np.array([skew[2, 1], skew[0, 2], skew[1, 0]], dtype=np.float64)


def se3_log(transform) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    invert_transform(transform)
    residual = np.zeros(6, dtype=np.float64)
    residual[:3] = so3_log(transform[:3, :3])
    residual[3:] = transform[:3, 3]
    return residual


def _project_rotation(rotation: np.ndarray) -> np.ndarray:
    u_matrix, _, vt_matrix = np.linalg.svd(rotation)
    projected = u_matrix @ vt_matrix
    if np.linalg.det(projected) < 0.0:
        u_matrix[:, -1] *= -1.0
        projected = u_matrix @ vt_matrix
    return projected


def weighted_se3_mean(poses, weights, iterations: int = 3) -> np.ndarray:
    pose_list = [np.asarray(pose, dtype=np.float64) for pose in poses]
    if not pose_list:
        raise ValueError("poses must not be empty")
    weight_array = np.asarray(weights, dtype=np.float64)
    if weight_array.shape != (len(pose_list),):
        raise ValueError("weights must match poses")
    if not np.all(np.isfinite(weight_array)) or np.any(weight_array < 0.0):
        raise ValueError("weights must be finite and non-negative")
    weight_sum = float(np.sum(weight_array))
    if weight_sum <= 0.0:
        raise ValueError("at least one weight must be positive")
    normalized = weight_array / weight_sum

    for pose in pose_list:
        invert_transform(pose)

    mean = np.eye(4, dtype=np.float64)
    mean[:3, :3] = _project_rotation(
        sum(weight * pose[:3, :3] for weight, pose in zip(normalized, pose_list))
    )
    mean[:3, 3] = sum(weight * pose[:3, 3] for weight, pose in zip(normalized, pose_list))

    # The current loop-factor ensemble uses small support perturbations. Reprojecting
    # the weighted rotation is stable and avoids introducing a SciPy dependency.
    for _ in range(max(1, int(iterations)) - 1):
        mean[:3, :3] = _project_rotation(
            sum(weight * pose[:3, :3] for weight, pose in zip(normalized, pose_list))
        )
        mean[:3, 3] = sum(weight * pose[:3, 3] for weight, pose in zip(normalized, pose_list))
    return mean
```

This mean is intentionally simple and deterministic for Stage A. The plan uses residuals in `se3_log(mean^-1 * T_i)` for covariance and consensus.

- [ ] **Step 4: Verify geometry tests**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/robust_loop_verifier/test_geometry.py -q
```

Expected: PASS.

- [ ] **Step 5: Check status**

Run:

```bash
git status --short
```

Expected: geometry module and tests changed.

---

## Task 4: Support Ensemble Aggregation

**Files:**
- Create: `LoopAnything/src/robust_loop_verifier/support_ensemble.py`
- Create: `LoopAnything/tests/robust_loop_verifier/test_support_ensemble.py`

- [ ] **Step 1: Write failing support ensemble tests**

Create `tests/robust_loop_verifier/test_support_ensemble.py`:

```python
import math

import numpy as np

from robust_loop_verifier.geometry import make_transform
from robust_loop_verifier.support_ensemble import (
    SupportEnsembleConfig,
    SupportLoopFactor,
    aggregate_support_loop_factors,
    cauchy_weight,
    graph_evidence_nll,
    huber_weight,
)


def _factor(x: float) -> np.ndarray:
    return make_transform(np.eye(3), [x, 0.0, 0.0])


def test_robust_weight_functions_are_bounded():
    assert cauchy_weight(0.0) == 1.0
    assert 0.0 < cauchy_weight(10.0) < 0.02
    assert huber_weight(0.5) == 1.0
    assert math.isclose(huber_weight(4.0), 0.25)


def test_aggregate_support_loop_factors_downweights_outlier():
    config = SupportEnsembleConfig(
        sigma_rot_floor=0.05,
        sigma_trans_floor=0.25,
        covariance_scale=1.0,
        c_align=1.0,
        c_consensus=1.0,
        lambda_dir=1.0,
        robust_iterations=3,
    )
    factors = [
        SupportLoopFactor(1, _factor(1.00), 0.01, 0.0, 1.0),
        SupportLoopFactor(2, _factor(1.05), 0.01, 0.0, 1.0),
        SupportLoopFactor(3, _factor(0.95), 0.01, 0.0, 1.0),
        SupportLoopFactor(4, _factor(5.00), 0.01, 0.0, 1.0),
    ]

    result = aggregate_support_loop_factors(factors, config)

    assert result.valid
    assert result.effective_support_count > 2.0
    assert result.support_weights[4] < 0.2
    np.testing.assert_allclose(result.loop_factor_mean[:3, 3], [1.0, 0.0, 0.0], atol=0.15)
    assert result.loop_sigmas[0] >= config.sigma_rot_floor
    assert result.loop_sigmas[3] >= config.sigma_trans_floor
    assert result.uncertainty_logdet_penalty >= 0.0


def test_aggregate_single_support_returns_floor_covariance():
    config = SupportEnsembleConfig.default()
    result = aggregate_support_loop_factors(
        [SupportLoopFactor(7, _factor(2.0), 0.0, 0.0, 1.0)],
        config,
    )

    assert result.valid
    assert result.effective_support_count == 1.0
    np.testing.assert_allclose(result.loop_factor_mean[:3, 3], [2.0, 0.0, 0.0])
    np.testing.assert_allclose(result.loop_sigmas[:3], [config.sigma_rot_floor] * 3)
    np.testing.assert_allclose(result.loop_sigmas[3:], [config.sigma_trans_floor] * 3)


def test_graph_evidence_nll_adds_uncertainty_penalty():
    assert graph_evidence_nll(
        loop_chi2_after=2.0,
        odom_strain_chi2_after=4.0,
        uncertainty_logdet_penalty=6.0,
    ) == 6.0
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/robust_loop_verifier/test_support_ensemble.py -q
```

Expected: FAIL because `support_ensemble.py` does not exist.

- [ ] **Step 3: Implement support ensemble module**

Create `src/robust_loop_verifier/support_ensemble.py` with these public types and functions:

```python
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from robust_loop_verifier.geometry import invert_transform, se3_log, weighted_se3_mean


@dataclass(frozen=True)
class SupportEnsembleConfig:
    sigma_rot_floor: float
    sigma_trans_floor: float
    covariance_scale: float
    c_align: float
    c_consensus: float
    lambda_dir: float
    robust_iterations: int

    @classmethod
    def default(cls) -> "SupportEnsembleConfig":
        return cls(0.05, 0.25, 1.0, 1.0, 2.0, 1.0, 3)


@dataclass(frozen=True)
class SupportLoopFactor:
    support_idx: int
    loop_factor: np.ndarray
    support_alignment_residual_m: float
    direction_error_deg: float
    candidate_support_baseline_m: float


@dataclass(frozen=True)
class SupportEnsembleResult:
    valid: bool
    loop_factor_mean: np.ndarray | None
    loop_sigmas: tuple[float, float, float, float, float, float] | None
    support_weights: Mapping[int, float]
    support_residual_norms: Mapping[int, float]
    effective_support_count: float
    sigma_rot: float | None
    sigma_trans: float | None
    uncertainty_logdet_penalty: float | None
    rejection_reason: str | None = None
```

Implement:

```python
def cauchy_weight(normalized_error: float) -> float:
    value = float(normalized_error)
    if not math.isfinite(value):
        return 0.0
    return float(1.0 / (1.0 + value * value))


def huber_weight(normalized_error: float) -> float:
    value = abs(float(normalized_error))
    if not math.isfinite(value):
        return 0.0
    if value <= 1.0:
        return 1.0
    return float(1.0 / value)
```

Implement `aggregate_support_loop_factors(factors, config)` using:

```text
1. Validate every loop factor is SE(3).
2. If no factors, return invalid with rejection_reason="no_valid_support_loop_factors".
3. Compute alignment weights:
   e_align = residual / max(baseline, 1e-9) + lambda_dir * 2*sin(direction_rad/2)
   w_align = cauchy_weight(e_align / c_align)
4. Initialize mean with weighted_se3_mean using alignment weights.
5. Repeat robust_iterations times:
   residual_i = se3_log(inv(mean) @ loop_factor_i)
   normalized = sqrt(sum((rot/sigma_rot_floor)^2) + sum((trans/sigma_trans_floor)^2))
   w_consensus = huber_weight(normalized / c_consensus)
   w_i = w_align_i * w_consensus
   mean = weighted_se3_mean(factors, w_i)
6. Compute weighted residual RMS:
   sigma_rot = sqrt(mean(weighted rot residual squared)) * covariance_scale + sigma_rot_floor
   sigma_trans = sqrt(mean(weighted trans residual squared)) * covariance_scale + sigma_trans_floor
7. loop_sigmas = (sigma_rot, sigma_rot, sigma_rot, sigma_trans, sigma_trans, sigma_trans)
8. uncertainty_logdet_penalty =
   3 * log((sigma_rot^2)/(sigma_rot_floor^2)) +
   3 * log((sigma_trans^2)/(sigma_trans_floor^2))
```

When all final weights are zero, fall back to uniform weights and keep the same residual/covariance computation.

Implement:

```python
def graph_evidence_nll(
    loop_chi2_after: float,
    odom_strain_chi2_after: float,
    uncertainty_logdet_penalty: float,
) -> float:
    values = [loop_chi2_after, odom_strain_chi2_after, uncertainty_logdet_penalty]
    if not all(math.isfinite(float(value)) for value in values):
        return float("inf")
    return float(0.5 * sum(values))
```

- [ ] **Step 4: Verify support ensemble tests**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/robust_loop_verifier/test_support_ensemble.py -q
```

Expected: PASS.

- [ ] **Step 5: Run geometry and support regression tests**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/robust_loop_verifier/test_geometry.py tests/robust_loop_verifier/test_support.py tests/robust_loop_verifier/test_support_ensemble.py -q
```

Expected: PASS.

- [ ] **Step 6: Check status**

Run:

```bash
git status --short
```

Expected: new support ensemble module and tests are staged-ready changes.

---

## Task 5: PGO Residual Diagnostics And Per-Candidate Loop Noise

**Files:**
- Modify: `LoopAnything/src/robust_loop_verifier/pgo.py`
- Modify: `LoopAnything/tests/robust_loop_verifier/test_pgo.py`

- [ ] **Step 1: Add failing PGO diagnostics tests**

Append to `tests/robust_loop_verifier/test_pgo.py`:

```python
def test_full_prefix_pgo_reports_loop_and_odom_chi2_diagnostics():
    _require_gtsam()
    poses = _poses(4)
    loop_factor = np.linalg.inv(poses[3]) @ poses[0]

    result = run_full_prefix_pgo(
        prefix_indices=list(range(4)),
        odom_poses=poses,
        loop_from_idx=3,
        loop_to_idx=0,
        loop_factor=loop_factor,
        noise=PgoNoise.default_for_tests(),
    )

    assert result.converged
    assert result.loop_chi2_after is not None
    assert result.odom_chi2_before is not None
    assert result.odom_chi2_after is not None
    assert result.odom_strain_chi2_after is not None
    assert result.loop_chi2_after < 1e-6
    assert result.odom_strain_chi2_after < 1e-6


def test_full_prefix_pgo_accepts_loop_sigmas_override():
    _require_gtsam()
    poses = _poses(4)
    bad_loop = make_transform(np.eye(3), [-20.0, 0.0, 0.0])

    tight = run_full_prefix_pgo(
        prefix_indices=list(range(4)),
        odom_poses=poses,
        loop_from_idx=3,
        loop_to_idx=0,
        loop_factor=bad_loop,
        noise=PgoNoise.default_for_tests(),
        loop_sigmas_override=(0.01, 0.01, 0.01, 0.01, 0.01, 0.01),
    )
    loose = run_full_prefix_pgo(
        prefix_indices=list(range(4)),
        odom_poses=poses,
        loop_from_idx=3,
        loop_to_idx=0,
        loop_factor=bad_loop,
        noise=PgoNoise.default_for_tests(),
        loop_sigmas_override=(10.0, 10.0, 10.0, 10.0, 10.0, 10.0),
    )

    assert tight.converged
    assert loose.converged
    assert tight.loop_chi2_after is not None
    assert loose.loop_chi2_after is not None
    assert loose.loop_chi2_after < tight.loop_chi2_after
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/robust_loop_verifier/test_pgo.py -q
```

Expected: FAIL because diagnostics fields and `loop_sigmas_override` do not exist.

- [ ] **Step 3: Extend `PgoResult`**

In `src/robust_loop_verifier/pgo.py`, update `PgoResult`:

```python
@dataclass(frozen=True)
class PgoResult:
    converged: bool
    optimized_poses: list[np.ndarray]
    error_before: float
    error_after: float
    failure_reason: str | None
    loop_chi2_after: float | None = None
    odom_chi2_before: float | None = None
    odom_chi2_after: float | None = None
    odom_strain_chi2_after: float | None = None
```

- [ ] **Step 4: Add loop sigmas override**

Update `run_full_prefix_pgo` signature:

```python
def run_full_prefix_pgo(
    prefix_indices: Sequence[int],
    odom_poses: Sequence[np.ndarray],
    loop_from_idx: int,
    loop_to_idx: int,
    loop_factor: np.ndarray,
    noise: PgoNoise,
    loop_sigmas_override: Sequence[float] | None = None,
) -> PgoResult:
```

At graph construction:

```python
loop_sigmas = (
    np.asarray(loop_sigmas_override, dtype=np.float64)
    if loop_sigmas_override is not None
    else np.asarray(noise.loop_sigmas, dtype=np.float64)
)
loop_noise = gtsam.noiseModel.Diagonal.Sigmas(loop_sigmas)
```

Extend `_validate_inputs` to validate `loop_sigmas_override` when provided by adding a helper:

```python
def _validate_sigmas(name: str, sigmas) -> str | None:
    sigmas = np.asarray(sigmas, dtype=np.float64)
    if sigmas.shape != (6,):
        return f"{name} must contain exactly six values"
    if not np.all(np.isfinite(sigmas)):
        return f"{name} must contain only finite values"
    if not np.all(sigmas > 0.0):
        return f"{name} must contain only positive values"
    return None
```

- [ ] **Step 5: Compute residual diagnostics**

Add helper:

```python
def _between_chi2(left: np.ndarray, right: np.ndarray, measured: np.ndarray, sigmas) -> float:
    from robust_loop_verifier.geometry import se3_log

    predicted = pose_between(left, right)
    residual_transform = invert_transform(measured) @ predicted
    residual = se3_log(residual_transform)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    whitened = residual / sigmas
    return float(np.dot(whitened, whitened))
```

Add helper:

```python
def _odom_chi2(poses: Sequence[np.ndarray], odom_measurements: Sequence[np.ndarray], sigmas) -> float:
    total = 0.0
    for position, measurement in enumerate(odom_measurements):
        total += _between_chi2(poses[position], poses[position + 1], measurement, sigmas)
    return float(total)
```

Before adding factors, build:

```python
odom_measurements = [
    pose_between(original_poses[position], original_poses[position + 1])
    for position in range(len(original_poses) - 1)
]
```

After optimization:

```python
odom_chi2_before = _odom_chi2(original_poses, odom_measurements, noise.odom_sigmas)
odom_chi2_after = _odom_chi2(optimized_poses, odom_measurements, noise.odom_sigmas)
loop_chi2_after = _between_chi2(
    optimized_poses[prefix_indices.index(loop_from_idx)],
    optimized_poses[prefix_indices.index(loop_to_idx)],
    loop_factor,
    loop_sigmas,
)
odom_strain_chi2_after = max(0.0, odom_chi2_after - odom_chi2_before)
```

Return these fields in `PgoResult`.

- [ ] **Step 6: Verify PGO tests**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/robust_loop_verifier/test_pgo.py -q
```

Expected: PASS.

- [ ] **Step 7: Check status**

Run:

```bash
git status --short
```

Expected: PGO module and tests changed.

---

## Task 6: Stage A Pipeline Integration With Mock Backend

**Files:**
- Modify: `LoopAnything/src/robust_loop_verifier/pipeline.py`
- Modify: `LoopAnything/tests/robust_loop_verifier/test_pipeline.py`

- [ ] **Step 1: Add failing pipeline tests**

Append to `tests/robust_loop_verifier/test_pipeline.py`:

```python
def test_stage_a_mock_pipeline_emits_support_ensemble_fields(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    _write_tiny_sequence_cache(
        cache_dir,
        keyframe_count=8,
        positives_by_query={7: [0]},
    )
    config = _cached_config(
        tmp_path,
        retrieval_top_k_main=1,
        support_window=4,
        recent_exclusion_keyframes=1,
        stage_a={
            "enabled": True,
            "support_count": 4,
            "sigma_rot_floor": 0.05,
            "sigma_trans_floor": 0.25,
            "covariance_scale": 1.0,
            "c_align": 1.0,
            "c_consensus": 2.0,
            "lambda_dir": 1.0,
            "robust_iterations": 3,
        },
    )
    run_root = tmp_path / "run"

    result = run_cached_sequence(
        sequence_cache=cache_dir,
        config=config,
        output_root=run_root,
        backend="mock",
    )

    records = _read_candidate_records(run_root)
    assert records
    scored = [record for record in records if record["score_stage_a"] is not None]
    assert scored
    first = scored[0]
    assert first["stage_a_enabled"] is True
    assert first["stage_a_support_count_requested"] == 4
    assert first["stage_a_support_count_used"] >= 1
    assert isinstance(first["stage_a_supports"], list)
    assert len(first["stage_a_loop_sigmas"]) == 6
    assert first["stage_a_effective_support_count"] >= 1.0
    assert first["stage_a_graph_evidence_nll"] >= 0.0
    assert "da3_rover_plus_plus_stage_a" in result["metrics"]


def test_stage_a_pipeline_preserves_single_support_metrics_when_disabled(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    _write_tiny_sequence_cache(cache_dir, keyframe_count=8, positives_by_query={7: [0]})
    config = _cached_config(tmp_path, retrieval_top_k_main=1)
    run_root = tmp_path / "run"

    run_cached_sequence(
        sequence_cache=cache_dir,
        config=config,
        output_root=run_root,
        backend="mock",
    )

    records = _read_candidate_records(run_root)
    assert records
    assert all(record["stage_a_enabled"] is False for record in records)
    assert all(record["score_stage_a"] is None for record in records)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/robust_loop_verifier/test_pipeline.py -q
```

Expected: FAIL because Stage A fields are not emitted.

- [ ] **Step 3: Add Stage A record defaults**

In `_score_candidate`, extend the initial record dictionary:

```python
"stage_a_enabled": bool(config.stage_a.enabled),
"stage_a_support_count_requested": int(config.stage_a.support_count),
"stage_a_support_count_used": 0,
"stage_a_supports": [],
"stage_a_effective_support_count": None,
"stage_a_loop_sigmas": None,
"stage_a_sigma_rot": None,
"stage_a_sigma_trans": None,
"stage_a_uncertainty_logdet_penalty": None,
"stage_a_loop_chi2_after": None,
"stage_a_odom_strain_chi2_after": None,
"stage_a_graph_evidence_nll": None,
"score_stage_a": None,
```

- [ ] **Step 4: Keep old path when Stage A disabled**

The existing single-support code path must run unchanged when:

```python
if not config.stage_a.enabled:
    ...
```

This preserves current metrics and real runs.

- [ ] **Step 5: Add Stage A scoring path**

Create helper in `pipeline.py`:

```python
def _score_candidate_stage_a(
    record,
    config,
    query_idx,
    candidate_idx,
    cache_order,
    image_by_idx,
    odom_by_idx,
    da3_runner,
    failure_reasons,
):
    ...
```

Behavior:

```text
1. Call select_supports(..., support_count=config.stage_a.support_count).
2. For each selected support:
   a. Build DA3 triplet [query, candidate, support].
   b. Call da3_runner.run_triplet(triplet). This call is inside the support loop.
   c. Run align_triplet_to_candidate_support.
   d. If valid, create SupportLoopFactor.
3. If no valid support loop factors, append failure "stage_a: no_valid_support_loop_factors" and return record.
4. Aggregate with aggregate_support_loop_factors.
5. Run run_full_prefix_pgo(..., loop_sigmas_override=ensemble.loop_sigmas).
6. Compute graph_evidence_nll from PGO diagnostics and ensemble uncertainty.
7. Set score_stage_a = -stage_a_graph_evidence_nll.
8. Also set score_rover from trajectory_deformation_rmse for backward compatibility.
```

Use `SupportEnsembleConfig` constructed from `config.stage_a`:

```python
SupportEnsembleConfig(
    sigma_rot_floor=config.stage_a.sigma_rot_floor,
    sigma_trans_floor=config.stage_a.sigma_trans_floor,
    covariance_scale=config.stage_a.covariance_scale,
    c_align=config.stage_a.c_align,
    c_consensus=config.stage_a.c_consensus,
    lambda_dir=config.stage_a.lambda_dir,
    robust_iterations=config.stage_a.robust_iterations,
)
```

Each `stage_a_supports` entry should be JSON-compatible:

```python
{
    "support_idx": support_idx,
    "support_baseline_m": baseline_m,
    "sim3_scale": sim3_scale,
    "support_alignment_residual_m": residual_m,
    "direction_error_deg": direction_error_deg,
    "weight": ensemble.support_weights.get(support_idx),
    "residual_norm": ensemble.support_residual_norms.get(support_idx),
}
```

- [ ] **Step 6: Add Stage A metrics**

Add constant:

```python
METHOD_STAGE_A = "DA3-ROVER++ Stage A graph evidence"
```

In method metrics, include:

```python
METHOD_STAGE_A: [record["score_stage_a"] for record in records]
```

Only include this method when at least one record has `stage_a_enabled` true, to avoid adding an all-failure metric to disabled runs.

In returned metrics JSON, use key:

```text
da3_rover_plus_plus_stage_a
```

- [ ] **Step 7: Verify pipeline tests**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/robust_loop_verifier/test_pipeline.py -q
```

Expected: PASS.

- [ ] **Step 8: Check status**

Run:

```bash
git status --short
```

Expected: pipeline and tests changed.

---

## Task 7: Score Sweep And Candidate Record Compatibility

**Files:**
- Modify: `LoopAnything/src/robust_loop_verifier/score_sweep.py`
- Modify: `LoopAnything/tests/robust_loop_verifier/test_score_sweep.py`

- [ ] **Step 1: Add failing score sweep test**

Append to `tests/robust_loop_verifier/test_score_sweep.py`:

```python
def test_score_sweep_reports_stage_a_score_when_present():
    records = [
        {"label": True, "score_salad": 0.1, "trajectory_deformation_rmse": 0.2, "pgo_error_after": 1.0, "score_stage_a": -1.0},
        {"label": False, "score_salad": 0.9, "trajectory_deformation_rmse": 1.0, "pgo_error_after": 10.0, "score_stage_a": -5.0},
    ]

    result = compute_score_sweep(records)

    names = {row["name"] for row in result["scores"]}
    assert "DA3-ROVER++ Stage A graph evidence" in names
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/robust_loop_verifier/test_score_sweep.py -q
```

Expected: FAIL because score sweep does not report Stage A.

- [ ] **Step 3: Add Stage A score sweep row**

In `score_sweep.py`, add near the baseline rows:

```python
stage_a_scores = [_finite_float_or_none(record.get("score_stage_a")) for record in records]
if any(score is not None for score in stage_a_scores):
    score_rows.append(
        _metrics_row(
            "DA3-ROVER++ Stage A graph evidence",
            labels,
            stage_a_scores,
        )
    )
```

- [ ] **Step 4: Verify candidate records preserve nested Stage A supports**

The pipeline writes `candidate_records.jsonl` through its existing JSONL writer. Task 6 already adds this assertion in `test_stage_a_mock_pipeline_emits_support_ensemble_fields`:

```python
assert isinstance(first["stage_a_supports"], list)
assert len(first["stage_a_loop_sigmas"]) == 6
```

No `artifacts.py` change is required for Stage A because `artifacts.py` only writes metrics Markdown, metrics JSON, and triplet image visualizations; candidate records are written by `pipeline.py`.

- [ ] **Step 5: Verify score sweep and pipeline candidate-record tests**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/robust_loop_verifier/test_score_sweep.py tests/robust_loop_verifier/test_pipeline.py::test_stage_a_mock_pipeline_emits_support_ensemble_fields -q
```

Expected: PASS.

- [ ] **Step 6: Check status**

Run:

```bash
git status --short
```

Expected: score sweep and pipeline test files changed.

---

## Task 8: Real Backend Guardrails For DA3 Isolation

**Files:**
- Modify: `LoopAnything/src/robust_loop_verifier/pipeline.py`
- Modify: `LoopAnything/tests/robust_loop_verifier/test_pipeline.py`

- [ ] **Step 1: Add failing DA3 isolation test**

Append to `tests/robust_loop_verifier/test_pipeline.py`:

```python
def test_stage_a_calls_da3_once_per_support_triplet(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    _write_tiny_sequence_cache(cache_dir, keyframe_count=8, positives_by_query={7: [0]})
    config = _cached_config(
        tmp_path,
        retrieval_top_k_main=1,
        stage_a={
            "enabled": True,
            "support_count": 3,
            "sigma_rot_floor": 0.05,
            "sigma_trans_floor": 0.25,
            "covariance_scale": 1.0,
            "c_align": 1.0,
            "c_consensus": 2.0,
            "lambda_dir": 1.0,
            "robust_iterations": 3,
        },
    )
    calls = []

    class RecordingRunner:
        def run_triplet(self, triplet):
            calls.append(tuple(triplet.view_roles))
            from robust_loop_verifier.da3_runner import MockDa3Runner

            return MockDa3Runner().run_triplet(triplet)

    run_cached_sequence(
        sequence_cache=cache_dir,
        config=config,
        output_root=tmp_path / "run",
        backend=RecordingRunner(),
    )

    assert calls
    assert all(call == ("query", "candidate", "support") for call in calls)
```

If `run_cached_sequence` currently accepts only string backends, add a small test-only path that accepts an object with `run_triplet`. Keep string backend behavior unchanged.

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/robust_loop_verifier/test_pipeline.py::test_stage_a_calls_da3_once_per_support_triplet -q
```

Expected: FAIL until backend object injection or Stage A support loop is implemented.

- [ ] **Step 3: Enforce triplet-only Stage A calls**

In the Stage A support loop, keep this invariant:

```python
triplet = build_da3_triplet(
    image_by_idx[query_idx],
    image_by_idx[candidate_idx],
    image_by_idx[support.support_idx],
    query_idx=query_idx,
    candidate_idx=candidate_idx,
    support_idx=support.support_idx,
)
da3_result = da3_runner.run_triplet(triplet)
```

Do not add any list of triplets to `run_triplet`. Do not modify `RealDa3Runner` to accept batched independent triplets.

- [ ] **Step 4: Verify isolation test**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/robust_loop_verifier/test_pipeline.py::test_stage_a_calls_da3_once_per_support_triplet -q
```

Expected: PASS.

- [ ] **Step 5: Check status**

Run:

```bash
git status --short
```

Expected: pipeline tests and pipeline changed.

---

## Task 9: Stage A Scripts And Runtime Commands

**Files:**
- Create: `LoopAnything/robust_loop_verification_scripts/run_stage_a_support_uncertainty.sh`
- Create: `LoopAnything/robust_loop_verification_scripts/run_stage_a_support_uncertainty_sweep.sh`
- Create: `LoopAnything/tests/robust_loop_verifier/test_stage_a_scripts.py`

- [ ] **Step 1: Write failing script smoke tests**

Create `tests/robust_loop_verifier/test_stage_a_scripts.py`:

```python
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def test_stage_a_run_script_has_help():
    script = ROOT / "robust_loop_verification_scripts" / "run_stage_a_support_uncertainty.sh"
    result = subprocess.run(["bash", str(script), "--help"], text=True, capture_output=True)
    assert result.returncode == 0
    assert "run_stage_a_support_uncertainty.sh" in result.stdout
    assert "SEQUENCE_NAME" in result.stdout


def test_stage_a_sweep_script_has_help():
    script = ROOT / "robust_loop_verification_scripts" / "run_stage_a_support_uncertainty_sweep.sh"
    result = subprocess.run(["bash", str(script), "--help"], text=True, capture_output=True)
    assert result.returncode == 0
    assert "run_stage_a_support_uncertainty_sweep.sh" in result.stdout
    assert "RUN_ROOT" in result.stdout
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/robust_loop_verifier/test_stage_a_scripts.py -q
```

Expected: FAIL because scripts do not exist.

- [ ] **Step 3: Create Stage A run script**

Create `robust_loop_verification_scripts/run_stage_a_support_uncertainty.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
run_stage_a_support_uncertainty.sh

Runs the DA3-ROVER++ Stage A support-uncertainty pipeline for one cached sequence.

Environment:
  SEQUENCE_NAME   Sequence name, default handheld_escalator00.
  PLATFORM        Platform, default handheld.
  DATASET_NAME    Dataset name, default FusionPortableV2.
  CACHE_ROOT      VPR cache root, default /data/datasets/FusionPortable/robust_loop_verifier_cache.
  CONFIG          Stage A config path.
  OUTPUT_ROOT     Output run directory. If unset, timestamped workspace path is used.
  PYTHON_BIN      Python interpreter, default /home/chenguyuan/anaconda3/envs/da3/bin/python.
EOF
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SEQUENCE_NAME="${SEQUENCE_NAME:-handheld_escalator00}"
PLATFORM="${PLATFORM:-handheld}"
DATASET_NAME="${DATASET_NAME:-FusionPortableV2}"
CACHE_ROOT="${CACHE_ROOT:-/data/datasets/FusionPortable/robust_loop_verifier_cache}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/robust_loop_verifier/fusionportablev2_handheld_stage_a.yaml}"
PYTHON_BIN="${PYTHON_BIN:-/home/chenguyuan/anaconda3/envs/da3/bin/python}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/workspace/robust_loop_verifier_runs/${DATASET_NAME}/${PLATFORM}/${SEQUENCE_NAME}/${RUN_ID}_stage_a}"
SEQUENCE_CACHE="${SEQUENCE_CACHE:-${CACHE_ROOT}/${DATASET_NAME}/${PLATFORM}/${SEQUENCE_NAME}}"

cd "${REPO_ROOT}"

echo "Running Stage A support uncertainty"
echo "  sequence_cache: ${SEQUENCE_CACHE}"
echo "  config: ${CONFIG}"
echo "  output_root: ${OUTPUT_ROOT}"

PYTHONPATH=src "${PYTHON_BIN}" -m robust_loop_verifier.cli run-cache \
  --sequence-cache "${SEQUENCE_CACHE}" \
  --config "${CONFIG}" \
  --output-root "${OUTPUT_ROOT}" \
  --backend real

echo "metrics_json=${OUTPUT_ROOT}/metrics.json"
echo "candidate_records=${OUTPUT_ROOT}/candidate_records.jsonl"
```

- [ ] **Step 4: Create Stage A sweep script**

Create `robust_loop_verification_scripts/run_stage_a_support_uncertainty_sweep.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
run_stage_a_support_uncertainty_sweep.sh

Runs score sweep over a Stage A run root.

Environment:
  RUN_ROOT     Existing robust loop verifier run root.
  OUTPUT_ROOT  Sweep output directory, default ${RUN_ROOT}/stage_a_score_sweep.
  PYTHON_BIN   Python interpreter, default /home/chenguyuan/anaconda3/envs/da3/bin/python.
EOF
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUN_ROOT}/stage_a_score_sweep}"
PYTHON_BIN="${PYTHON_BIN:-/home/chenguyuan/anaconda3/envs/da3/bin/python}"

cd "${REPO_ROOT}"

PYTHONPATH=src "${PYTHON_BIN}" -m robust_loop_verifier.cli sweep-scores \
  --candidate-records "${RUN_ROOT}/candidate_records.jsonl" \
  --output-root "${OUTPUT_ROOT}"

echo "score_sweep_json=${OUTPUT_ROOT}/score_sweep.json"
echo "score_sweep_md=${OUTPUT_ROOT}/score_sweep.md"
```

Run:

```bash
chmod +x robust_loop_verification_scripts/run_stage_a_support_uncertainty.sh robust_loop_verification_scripts/run_stage_a_support_uncertainty_sweep.sh
```

- [ ] **Step 5: Verify script tests**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/robust_loop_verifier/test_stage_a_scripts.py -q
```

Expected: PASS.

- [ ] **Step 6: Check status**

Run:

```bash
git status --short
```

Expected: new scripts and tests changed.

---

## Task 10: Full Test Pass And Runtime Smoke

**Files:**
- No new source files unless previous tasks expose a small integration issue.

- [ ] **Step 1: Run focused Stage A test suite**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest \
  tests/robust_loop_verifier/test_stage_a_config.py \
  tests/robust_loop_verifier/test_support.py \
  tests/robust_loop_verifier/test_geometry.py \
  tests/robust_loop_verifier/test_support_ensemble.py \
  tests/robust_loop_verifier/test_pgo.py \
  tests/robust_loop_verifier/test_pipeline.py \
  tests/robust_loop_verifier/test_score_sweep.py \
  tests/robust_loop_verifier/test_stage_a_scripts.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run full robust verifier tests**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/robust_loop_verifier -q
```

Expected: PASS.

- [ ] **Step 3: Run formatting check**

Run:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
python -m black --check src/robust_loop_verifier tests/robust_loop_verifier
```

Expected: PASS. If it fails, run:

```bash
python -m black src/robust_loop_verifier tests/robust_loop_verifier
```

Then rerun the `--check` command.

- [ ] **Step 4: Run mock CLI smoke**

Stage A mock behavior is covered by `test_stage_a_mock_pipeline_emits_support_ensemble_fields`. This command verifies that the package CLI still runs after the Stage A changes:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
OUTPUT_ROOT="$(mktemp -d /tmp/robust_loop_verifier_cli_mock.XXXXXX)"
PYTHONPATH=src python -m robust_loop_verifier.cli run-mock \
  --output-root "${OUTPUT_ROOT}"
test -f "${OUTPUT_ROOT}/candidate_records.jsonl"
test -f "${OUTPUT_ROOT}/metrics.json"
```

Expected: command exits 0 and both `test -f` checks pass for:

```text
${OUTPUT_ROOT}/candidate_records.jsonl
${OUTPUT_ROOT}/metrics.json
```

- [ ] **Step 5: Real handheld runtime command for user validation**

Do not run this automatically during plan execution unless the user requests GPU runtime testing. Provide this exact command:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
SEQUENCE_NAME=handheld_escalator00 \
PLATFORM=handheld \
PYTHON_BIN=/home/chenguyuan/anaconda3/envs/da3/bin/python \
robust_loop_verification_scripts/run_stage_a_support_uncertainty.sh
```

Expected outputs:

```text
workspace/robust_loop_verifier_runs/FusionPortableV2/handheld/handheld_escalator00/<timestamp>_stage_a/metrics.json
workspace/robust_loop_verifier_runs/FusionPortableV2/handheld/handheld_escalator00/<timestamp>_stage_a/candidate_records.jsonl
```

- [ ] **Step 6: Check final git status**

Run:

```bash
git status --short
```

Expected: source, tests, config, script, spec, roadmap, and this plan are modified or added. No dataset, checkpoint, or workspace generated output should be tracked.

---

## Execution Review Checklist

Before handing this implementation to the user:

- [ ] Confirm no file under `src/robust_loop_verifier` imports `loop_policy`:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
rg -n "loop_policy|safe_loop_factor|learned_policy|x_geom" src/robust_loop_verifier tests/robust_loop_verifier
```

Expected: no matches except this string in tests if the package-boundary test intentionally checks for forbidden strings.

- [ ] Confirm DA3 isolation is preserved:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/robust_loop_verifier/test_pipeline.py::test_stage_a_calls_da3_once_per_support_triplet -q
```

Expected: PASS.

- [ ] Confirm Stage A score is present in score sweep when records include `score_stage_a`:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/robust_loop_verifier/test_score_sweep.py::test_score_sweep_reports_stage_a_score_when_present -q
```

Expected: PASS.

- [ ] Confirm full robust verifier tests pass:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything
PYTHONPATH=src pytest tests/robust_loop_verifier -q
```

Expected: PASS.
