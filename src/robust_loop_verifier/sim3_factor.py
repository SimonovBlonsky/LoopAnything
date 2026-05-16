from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from robust_loop_verifier.geometry import invert_transform, pose_between


_BASELINE_EPS = 1e-9


@dataclass(frozen=True)
class Sim3LoopFactorResult:
    valid: bool
    loop_factor: Optional[np.ndarray]
    sim3_scale: Optional[float]
    support_alignment_residual_m: Optional[float]
    direction_error_deg: Optional[float]
    rejection_reason: Optional[str]


def align_triplet_to_candidate_support(
    da3_predicted_c2w,
    odom_candidate_c2w,
    odom_support_c2w,
    baseline_epsilon: float = _BASELINE_EPS,
) -> Sim3LoopFactorResult:
    try:
        da3_c2w = _as_da3_triplet_c2w(da3_predicted_c2w)
        odom_candidate = _as_valid_pose(odom_candidate_c2w)
        odom_support = _as_valid_pose(odom_support_c2w)
    except (TypeError, ValueError):
        return _invalid("invalid_pose")

    da3_candidate = da3_c2w[1]
    da3_support = da3_c2w[2]
    da3_baseline = da3_support[:3, 3] - da3_candidate[:3, 3]
    odom_baseline = odom_support[:3, 3] - odom_candidate[:3, 3]

    da3_baseline_norm = float(np.linalg.norm(da3_baseline))
    if not np.isfinite(da3_baseline_norm) or da3_baseline_norm <= baseline_epsilon:
        return _invalid("invalid_da3_support_baseline")

    odom_baseline_norm = float(np.linalg.norm(odom_baseline))
    if not np.isfinite(odom_baseline_norm) or odom_baseline_norm <= baseline_epsilon:
        return _invalid("invalid_odom_support_baseline")

    sim3_scale = odom_baseline_norm / da3_baseline_norm
    if not np.isfinite(sim3_scale) or sim3_scale <= 0.0:
        return _invalid("invalid_sim3_scale")

    alignment_rotation = odom_candidate[:3, :3] @ da3_candidate[:3, :3].T
    translation = odom_candidate[:3, 3] - sim3_scale * (
        alignment_rotation @ da3_candidate[:3, 3]
    )

    aligned_c2w = np.empty_like(da3_c2w)
    for pose_index, pose in enumerate(da3_c2w):
        aligned_c2w[pose_index] = np.eye(4, dtype=np.float64)
        aligned_c2w[pose_index][:3, :3] = alignment_rotation @ pose[:3, :3]
        aligned_c2w[pose_index][:3, 3] = sim3_scale * (
            alignment_rotation @ pose[:3, 3]
        ) + translation

    support_alignment_residual_m = float(
        np.linalg.norm(aligned_c2w[2][:3, 3] - odom_support[:3, 3])
    )
    aligned_baseline = aligned_c2w[2][:3, 3] - aligned_c2w[1][:3, 3]
    direction_error_deg = _direction_error_deg(aligned_baseline, odom_baseline)
    loop_factor = pose_between(aligned_c2w[0], aligned_c2w[1])

    return Sim3LoopFactorResult(
        valid=True,
        loop_factor=loop_factor,
        sim3_scale=float(sim3_scale),
        support_alignment_residual_m=support_alignment_residual_m,
        direction_error_deg=direction_error_deg,
        rejection_reason=None,
    )


def _invalid(reason: str) -> Sim3LoopFactorResult:
    return Sim3LoopFactorResult(
        valid=False,
        loop_factor=None,
        sim3_scale=None,
        support_alignment_residual_m=None,
        direction_error_deg=None,
        rejection_reason=reason,
    )


def _as_da3_triplet_c2w(poses) -> np.ndarray:
    c2w = np.asarray(poses, dtype=np.float64)
    if c2w.shape != (3, 4, 4):
        raise ValueError("DA3 predicted c2w poses must have shape (3, 4, 4)")
    for pose in c2w:
        _as_valid_pose(pose)
    return c2w


def _as_valid_pose(pose) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    invert_transform(pose)
    return pose


def _direction_error_deg(aligned_baseline: np.ndarray, odom_baseline: np.ndarray) -> float:
    aligned_norm = np.linalg.norm(aligned_baseline)
    odom_norm = np.linalg.norm(odom_baseline)
    if aligned_norm <= 0.0 or odom_norm <= 0.0:
        return float("nan")
    aligned_direction = aligned_baseline / aligned_norm
    odom_direction = odom_baseline / odom_norm
    dot = float(np.clip(np.dot(aligned_direction, odom_direction), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))
