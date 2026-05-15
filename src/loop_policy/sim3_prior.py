from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from loop_policy.geometry import PoseResidual, camera_center_baseline, pose_residual


@dataclass(frozen=True)
class Sim3PriorConfig:
    min_da3_support_baseline_m: float = 1e-6
    min_sim3_scale: float = 0.05
    max_sim3_scale: float = 20.0
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


def _unit_direction(candidate_pose: np.ndarray, support_pose: np.ndarray) -> Optional[np.ndarray]:
    offset = support_pose[:3, 3] - candidate_pose[:3, 3]
    norm = float(np.linalg.norm(offset))
    if norm <= 1e-12:
        return None
    return offset / norm


def _direction_error_deg(da3_direction: np.ndarray, odom_direction: np.ndarray) -> float:
    cos_angle = float(np.clip(np.dot(da3_direction, odom_direction), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def _rejected_result(reason: str) -> Sim3PriorResult:
    return Sim3PriorResult(
        accepted=False,
        rejection_reason=reason,
        sim3_scale=1.0,
        abs_log_sim3_scale=0.0,
        support_align_rmse=0.0,
        direction_error_deg=0.0,
        aligned_loop=PoseResidual(rotation_deg=0.0, translation_norm=0.0),
        aligned_vs_odom=PoseResidual(rotation_deg=0.0, translation_norm=0.0),
        aligned_query_pose=np.eye(4, dtype=np.float64),
    )


def _poses_are_finite(poses: Sequence[np.ndarray]) -> bool:
    return all(np.isfinite(np.asarray(pose, dtype=np.float64)).all() for pose in poses)


def align_da3_poses_with_candidate_support_prior(
    da3_query_pose: np.ndarray,
    da3_candidate_pose: np.ndarray,
    da3_support_poses: Sequence[np.ndarray],
    odom_query_pose: np.ndarray,
    odom_candidate_pose: np.ndarray,
    odom_support_poses: Sequence[np.ndarray],
    config: Sim3PriorConfig,
) -> Sim3PriorResult:
    if len(da3_support_poses) != len(odom_support_poses) or len(da3_support_poses) == 0:
        raise ValueError("DA3 and odom supports must have the same positive length")
    if not _poses_are_finite(
        [
            da3_query_pose,
            da3_candidate_pose,
            *da3_support_poses,
            odom_query_pose,
            odom_candidate_pose,
            *odom_support_poses,
        ]
    ):
        return _rejected_result("sim3_alignment_failed")

    da3_baselines = np.array(
        [camera_center_baseline(da3_candidate_pose, support) for support in da3_support_poses],
        dtype=np.float64,
    )
    odom_baselines = np.array(
        [camera_center_baseline(odom_candidate_pose, support) for support in odom_support_poses],
        dtype=np.float64,
    )

    rejection_reason = None
    if np.any(da3_baselines <= config.min_da3_support_baseline_m):
        scale = 0.0
        rejection_reason = "invalid_support_baseline"
    else:
        scale = float(np.median(odom_baselines / da3_baselines))
        if not np.isfinite(scale) or scale <= 0.0:
            rejection_reason = "invalid_sim3_scale"

    r_align = odom_candidate_pose[:3, :3] @ da3_candidate_pose[:3, :3].T

    aligned_query_pose = np.array(odom_candidate_pose, dtype=np.float64, copy=True)
    aligned_query_pose[:3, :3] = r_align @ da3_query_pose[:3, :3]
    aligned_query_pose[:3, 3] = odom_candidate_pose[:3, 3] + scale * (
        r_align @ (da3_query_pose[:3, 3] - da3_candidate_pose[:3, 3])
    )

    support_errors = []
    direction_errors = []
    for da3_support_pose, odom_support_pose in zip(da3_support_poses, odom_support_poses):
        aligned_support_position = odom_candidate_pose[:3, 3] + scale * (
            r_align @ (da3_support_pose[:3, 3] - da3_candidate_pose[:3, 3])
        )
        support_errors.append(
            float(np.linalg.norm(aligned_support_position - odom_support_pose[:3, 3]))
        )

        da3_direction = _unit_direction(da3_candidate_pose, da3_support_pose)
        odom_direction = _unit_direction(odom_candidate_pose, odom_support_pose)
        if da3_direction is None or odom_direction is None:
            direction_errors.append(180.0)
        else:
            direction_errors.append(_direction_error_deg(r_align @ da3_direction, odom_direction))

    support_align_rmse = float(np.sqrt(np.mean(np.square(support_errors))))
    direction_error_deg = float(max(direction_errors))
    abs_log_sim3_scale = float(abs(np.log(scale))) if scale > 0.0 else float("inf")

    if rejection_reason is None and support_align_rmse > config.max_support_align_rmse_m:
        rejection_reason = "support_align_rmse_too_large"
    if rejection_reason is None and direction_error_deg > config.max_direction_error_deg:
        rejection_reason = "direction_error_too_large"

    return Sim3PriorResult(
        accepted=rejection_reason is None,
        rejection_reason=rejection_reason,
        sim3_scale=scale,
        abs_log_sim3_scale=abs_log_sim3_scale,
        support_align_rmse=support_align_rmse,
        direction_error_deg=direction_error_deg,
        aligned_loop=pose_residual(odom_candidate_pose, aligned_query_pose),
        aligned_vs_odom=pose_residual(odom_query_pose, aligned_query_pose),
        aligned_query_pose=aligned_query_pose,
    )
