from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'src'
for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from da3_streaming.loop_utils.training_free_gate import (
    combine_loop_scores,
    estimate_overlap_score,
    relative_pose_from_extrinsics,
)


def _one_hot_depth(height: int, width: int, row: int, col: int, depth: float = 1.0) -> np.ndarray:
    values = np.zeros((height, width), dtype=np.float32)
    values[row, col] = depth
    return values


def test_relative_pose_from_extrinsics_matches_manual_transform():
    a = np.eye(4, dtype=np.float32)
    b = np.eye(4, dtype=np.float32)
    b[0, 3] = 2.0
    rel = relative_pose_from_extrinsics(a, b)
    assert np.allclose(rel[:3, 3], np.array([-2.0, 0.0, 0.0]), atol=1e-6)


def test_relative_pose_from_extrinsics_supports_3x4_inputs():
    a = np.eye(4, dtype=np.float32)[:3, :]
    b = np.eye(4, dtype=np.float32)[:3, :]
    b[0, 3] = 2.0
    rel = relative_pose_from_extrinsics(a, b)
    expected = np.eye(4, dtype=np.float32)
    expected[0, 3] = -2.0
    assert np.allclose(rel, expected, atol=1e-6)


def test_estimate_overlap_score_is_high_for_identity_pose():
    depth_a = _one_hot_depth(2, 2, 0, 0)
    depth_b = _one_hot_depth(2, 2, 0, 0)
    intr = np.eye(3, dtype=np.float32)
    score = estimate_overlap_score(depth_a, depth_b, intr, intr, np.eye(4, dtype=np.float32), stride=1)
    assert score > 0.95


def test_estimate_overlap_score_is_low_for_wrong_pose():
    depth_a = _one_hot_depth(2, 2, 0, 1)
    depth_b = _one_hot_depth(2, 2, 0, 0)
    intr = np.eye(3, dtype=np.float32)
    rel_pose = np.eye(4, dtype=np.float32)
    score = estimate_overlap_score(depth_a, depth_b, intr, intr, rel_pose, stride=1)
    assert score < 0.1


def test_estimate_overlap_score_rejects_nonpositive_stride():
    depth = _one_hot_depth(2, 2, 0, 0)
    intr = np.eye(3, dtype=np.float32)
    try:
        estimate_overlap_score(depth, depth, intr, intr, np.eye(4, dtype=np.float32), stride=0)
    except ValueError as exc:
        assert 'stride must be positive' in str(exc)
    else:
        raise AssertionError('expected ValueError for stride <= 0')


def test_estimate_overlap_score_accepts_relative_pose_from_extrinsics():
    depth_a = _one_hot_depth(2, 2, 0, 1)
    depth_b = _one_hot_depth(2, 2, 0, 0)
    intr = np.eye(3, dtype=np.float32)
    extrinsics_a = np.eye(4, dtype=np.float32)
    extrinsics_b = np.eye(4, dtype=np.float32)
    extrinsics_b[0, 3] = -1.0
    rel_pose = relative_pose_from_extrinsics(extrinsics_a, extrinsics_b)
    score = estimate_overlap_score(depth_a, depth_b, intr, intr, rel_pose, stride=1)
    assert score > 0.95


def test_combine_loop_scores_matches_default_weights_exact_value():
    score = combine_loop_scores(desc_sim=0.75, pair_sim=0.5, overlap=0.25)
    assert score == 0.5625


def test_combine_loop_scores_is_monotonic_in_overlap():
    low = combine_loop_scores(desc_sim=0.9, pair_sim=0.8, overlap=0.1)
    high = combine_loop_scores(desc_sim=0.9, pair_sim=0.8, overlap=0.8)
    assert high > low
