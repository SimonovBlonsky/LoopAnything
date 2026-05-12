import numpy as np
import pytest

from loop_policy.geometry import make_transform
from loop_policy.schema import KeyframeRecord, LoopPolicyDatasetConfig
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
    assert decision.selected_support_indices == [0, 1]
    assert decision.selected_support_timestamps == [100.0, 101.0]
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


def test_select_supports_recent_filter_is_query_relative():
    keyframes = _keyframes()
    query = keyframes[7]
    candidate = keyframes[3]

    decision = select_supports(
        sequence="handheld_room01",
        query=query,
        candidate=candidate,
        keyframes=keyframes,
        camera_poses_by_idx=_camera_poses(),
        support_window=1,
        support_count=2,
        exclude_recent_keyframes=1,
        min_support_baseline_m=0.5,
    )

    assert decision.selected_support_indices == [2, 4]


def test_loop_policy_config_default_support_baseline_matches_aster_slam():
    config = LoopPolicyDatasetConfig(
        dataset_root="/dataset",
        output_root="/output",
        sequences=("handheld_room01",),
    )

    assert config.support_window == 20
    assert config.min_support_baseline_m == 0.3


def test_select_supports_raises_domain_error_for_missing_candidate_pose():
    keyframes = _keyframes()
    query = keyframes[7]
    candidate = keyframes[3]
    camera_poses = _camera_poses()
    del camera_poses[candidate.keyframe_idx]

    with pytest.raises(ValueError, match="missing candidate pose for keyframe_idx=3"):
        select_supports(
            sequence="handheld_room01",
            query=query,
            candidate=candidate,
            keyframes=keyframes,
            camera_poses_by_idx=camera_poses,
            support_window=3,
            support_count=2,
            exclude_recent_keyframes=1,
            min_support_baseline_m=0.5,
        )


def test_select_supports_filters_invalid_snapshot_and_support_pose_entries():
    keyframes = [
        KeyframeRecord(keyframe_idx=0, timestamp=100.0, image_path="0.jpg"),
        KeyframeRecord(keyframe_idx=1, timestamp=101.0, image_path="1.jpg"),
        KeyframeRecord(keyframe_idx=2, timestamp=102.0, image_path=None),
        KeyframeRecord(keyframe_idx=3, timestamp=103.0, image_path="3.jpg"),
        KeyframeRecord(keyframe_idx=4, timestamp=104.0, image_path="4.jpg"),
        KeyframeRecord(keyframe_idx=5, timestamp=105.0, image_path="5.jpg"),
        KeyframeRecord(keyframe_idx=6, timestamp=106.0, image_path="6.jpg"),
        KeyframeRecord(keyframe_idx=7, timestamp=107.0, image_path="7.jpg"),
        KeyframeRecord(keyframe_idx=8, timestamp=108.0, image_path="8.jpg"),
    ]
    camera_poses = {
        0: make_transform(np.eye(3), np.array([0.0, 0.0, 0.0])),
        3: make_transform(np.eye(3), np.array([3.0, 0.0, 0.0])),
        5: make_transform(np.eye(3), np.array([5.0, 0.0, 0.0])),
        8: make_transform(np.eye(3), np.array([8.0, 0.0, 0.0])),
    }

    decision = select_supports(
        sequence="handheld_room01",
        query=keyframes[7],
        candidate=keyframes[3],
        keyframes=keyframes,
        camera_poses_by_idx=camera_poses,
        support_window=5,
        support_count=3,
        exclude_recent_keyframes=0,
        min_support_baseline_m=0.5,
    )

    assert decision.selected_support_indices == [0, 5]
    assert decision.selected_support_timestamps == [100.0, 105.0]
    assert decision.selected_support_baselines == [3.0, 2.0]
    assert decision.support_snapshot_max_idx == 6
    assert decision.support_snapshot_max_timestamp == 106.0
