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

    assert result.query_idx == 9
    assert result.candidate_idx == 3
    assert result.support_idx == 2
    assert result.support_baseline_m == 1.0
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

    assert result.query_idx == 5
    assert result.candidate_idx == 3
    assert result.support_idx is None
    assert result.support_baseline_m is None
    assert result.rejection_reason == "no_valid_support"


def test_support_rejects_when_candidate_pose_is_missing():
    poses = _poses()
    poses.pop(3)

    result = select_support(
        query_idx=9,
        candidate_idx=3,
        available_indices=[2, 4],
        image_indices={2, 4},
        camera_poses=poses,
        support_window=2,
        recent_exclusion_keyframes=2,
        min_support_baseline_m=0.3,
    )

    assert result.query_idx == 9
    assert result.candidate_idx == 3
    assert result.support_idx is None
    assert result.support_baseline_m is None
    assert result.rejection_reason == "missing_candidate_pose"


def test_support_rejects_when_candidate_pose_is_invalid():
    poses = _poses()
    poses[3] = make_transform(np.eye(3), [np.nan, 0.0, 0.0])

    result = select_support(
        query_idx=9,
        candidate_idx=3,
        available_indices=[2, 4],
        image_indices={2, 4},
        camera_poses=poses,
        support_window=2,
        recent_exclusion_keyframes=2,
        min_support_baseline_m=0.3,
    )

    assert result.query_idx == 9
    assert result.candidate_idx == 3
    assert result.support_idx is None
    assert result.support_baseline_m is None
    assert result.rejection_reason == "invalid_candidate_pose"


def test_support_rejects_when_candidate_pose_is_malformed():
    poses = _poses()
    poses[3] = np.eye(3)

    result = select_support(
        query_idx=9,
        candidate_idx=3,
        available_indices=[2, 4],
        image_indices={2, 4},
        camera_poses=poses,
        support_window=2,
        recent_exclusion_keyframes=2,
        min_support_baseline_m=0.3,
    )

    assert result.query_idx == 9
    assert result.candidate_idx == 3
    assert result.support_idx is None
    assert result.support_baseline_m is None
    assert result.rejection_reason == "invalid_candidate_pose"


def test_support_requires_image_pose_and_minimum_baseline():
    result = select_support(
        query_idx=9,
        candidate_idx=3,
        available_indices=[2, 4, 5],
        image_indices={2, 5},
        camera_poses={
            2: make_transform(np.eye(3), [2.9, 0.0, 0.0]),
            3: make_transform(np.eye(3), [3.0, 0.0, 0.0]),
            4: make_transform(np.eye(3), [4.0, 0.0, 0.0]),
            5: make_transform(np.eye(3), [5.0, 0.0, 0.0]),
        },
        support_window=4,
        recent_exclusion_keyframes=2,
        min_support_baseline_m=0.3,
    )

    assert result.query_idx == 9
    assert result.candidate_idx == 3
    assert result.support_idx == 5
    assert result.support_baseline_m == 2.0
    assert result.rejection_reason is None


def test_support_skips_missing_support_pose_and_selects_next_valid():
    poses = _poses()
    poses.pop(2)

    result = select_support(
        query_idx=9,
        candidate_idx=3,
        available_indices=[2, 4],
        image_indices={2, 4},
        camera_poses=poses,
        support_window=2,
        recent_exclusion_keyframes=2,
        min_support_baseline_m=0.3,
    )

    assert result.query_idx == 9
    assert result.candidate_idx == 3
    assert result.support_idx == 4
    assert result.support_baseline_m == 1.0
    assert result.rejection_reason is None


def test_support_skips_invalid_support_pose_and_selects_next_valid():
    poses = _poses()
    poses[2] = make_transform(np.eye(3), [np.nan, 0.0, 0.0])

    result = select_support(
        query_idx=9,
        candidate_idx=3,
        available_indices=[2, 4],
        image_indices={2, 4},
        camera_poses=poses,
        support_window=2,
        recent_exclusion_keyframes=2,
        min_support_baseline_m=0.3,
    )

    assert result.query_idx == 9
    assert result.candidate_idx == 3
    assert result.support_idx == 4
    assert result.support_baseline_m == 1.0
    assert result.rejection_reason is None


def test_support_window_filters_otherwise_valid_support():
    result = select_support(
        query_idx=9,
        candidate_idx=3,
        available_indices=[0, 4],
        image_indices={0, 4},
        camera_poses=_poses(),
        support_window=1,
        recent_exclusion_keyframes=2,
        min_support_baseline_m=0.3,
    )

    assert result.query_idx == 9
    assert result.candidate_idx == 3
    assert result.support_idx == 4
    assert result.support_baseline_m == 1.0
    assert result.rejection_reason is None


def test_support_tie_breaks_by_smaller_index():
    result = select_support(
        query_idx=9,
        candidate_idx=3,
        available_indices=[2, 4],
        image_indices={2, 4},
        camera_poses=_poses(),
        support_window=2,
        recent_exclusion_keyframes=2,
        min_support_baseline_m=0.3,
    )

    assert result.query_idx == 9
    assert result.candidate_idx == 3
    assert result.support_idx == 2
    assert result.support_baseline_m == 1.0
    assert result.rejection_reason is None
