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
    if candidate.keyframe_idx not in camera_poses_by_idx:
        raise ValueError(f"missing candidate pose for keyframe_idx={candidate.keyframe_idx}")

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
        if not (
            candidate.keyframe_idx - support_window
            <= support.keyframe_idx
            <= candidate.keyframe_idx + support_window
        ):
            continue
        if abs(query.keyframe_idx - support.keyframe_idx) <= exclude_recent_keyframes:
            continue
        if support.keyframe_idx not in camera_poses_by_idx:
            continue
        baseline = camera_center_baseline(
            candidate_pose,
            camera_poses_by_idx[support.keyframe_idx],
        )
        if baseline < min_support_baseline_m:
            continue
        scored.append((baseline, support.keyframe_idx, support.timestamp))

    scored.sort(key=lambda item: (abs(item[1] - candidate.keyframe_idx), item[1]))
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
