from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SupportSelection:
    query_idx: int
    candidate_idx: int
    support_idx: int | None
    support_baseline_m: float | None
    rejection_reason: str | None = None


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


def _pose_translation(pose) -> np.ndarray:
    pose_array = np.asarray(pose, dtype=np.float64)
    if pose_array.shape != (4, 4):
        raise ValueError("Pose must have shape (4, 4)")
    if not np.all(np.isfinite(pose_array)):
        raise ValueError("Pose must contain only finite values")
    return pose_array[:3, 3]


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
        return MultiSupportSelection(
            query_idx=query_idx,
            candidate_idx=candidate_idx,
            supports=[],
            rejection_reason="missing_candidate_pose",
        )

    image_index_set = set(image_indices)
    try:
        candidate_translation = _pose_translation(camera_poses[candidate_idx])
    except ValueError:
        return MultiSupportSelection(
            query_idx=query_idx,
            candidate_idx=candidate_idx,
            supports=[],
            rejection_reason="invalid_candidate_pose",
        )
    valid_supports = []

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

        valid_supports.append(
            SupportCandidate(support_idx=support_idx, support_baseline_m=baseline_m)
        )

    if not valid_supports:
        return MultiSupportSelection(
            query_idx=query_idx,
            candidate_idx=candidate_idx,
            supports=[],
            rejection_reason="no_valid_support",
        )

    ordered_supports = sorted(
        valid_supports,
        key=lambda support: (abs(support.support_idx - candidate_idx), support.support_idx),
    )
    return MultiSupportSelection(
        query_idx=query_idx,
        candidate_idx=candidate_idx,
        supports=ordered_supports[:support_count],
    )


def select_support(
    query_idx: int,
    candidate_idx: int,
    available_indices,
    image_indices,
    camera_poses,
    support_window: int,
    recent_exclusion_keyframes: int,
    min_support_baseline_m: float,
) -> SupportSelection:
    result = select_supports(
        query_idx=query_idx,
        candidate_idx=candidate_idx,
        available_indices=available_indices,
        image_indices=image_indices,
        camera_poses=camera_poses,
        support_window=support_window,
        recent_exclusion_keyframes=recent_exclusion_keyframes,
        min_support_baseline_m=min_support_baseline_m,
        support_count=1,
    )
    if not result.supports:
        return SupportSelection(
            query_idx=result.query_idx,
            candidate_idx=result.candidate_idx,
            support_idx=None,
            support_baseline_m=None,
            rejection_reason=result.rejection_reason,
        )

    support = result.supports[0]
    return SupportSelection(
        query_idx=result.query_idx,
        candidate_idx=result.candidate_idx,
        support_idx=support.support_idx,
        support_baseline_m=support.support_baseline_m,
    )
