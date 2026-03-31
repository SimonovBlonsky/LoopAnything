import pytest
import numpy as np


def test_pose_distance():
    from depth_anything_3.data.unified_visloc_dataset import compute_pose_distance
    pose_a = np.eye(4, dtype=np.float32)
    pose_b = np.eye(4, dtype=np.float32)
    pose_b[:3, 3] = [1.0, 0.0, 0.0]  # 1m translation
    dist = compute_pose_distance(pose_a, pose_b, alpha=1.0)
    assert abs(dist - 1.0) < 1e-5, f"Expected ~1.0, got {dist}"


def test_pose_distance_rotation():
    from depth_anything_3.data.unified_visloc_dataset import compute_pose_distance
    pose_a = np.eye(4, dtype=np.float32)
    pose_b = np.eye(4, dtype=np.float32)
    # 90 degree rotation around Z axis
    pose_b[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    dist = compute_pose_distance(pose_a, pose_b, alpha=0.0)
    assert abs(dist - 90.0) < 1.0, f"Expected ~90.0, got {dist}"


def test_sample_candidates():
    from depth_anything_3.data.unified_visloc_dataset import sample_candidates
    np.random.seed(42)
    # Create 20 entries with varying poses
    entries = []
    for i in range(20):
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = [i * 0.5, 0, 0]
        entries.append({"image_path": f"img_{i}.png", "pose": pose})
    query_pose = entries[0]["pose"]
    candidates = sample_candidates(
        query_idx=0,
        all_entries=entries,
        num_candidates=10,
        pos_threshold=2.0,
        neg_threshold=5.0,
        pos_ratio=0.7,
        distance_alpha=1.0,
    )
    assert len(candidates) == 10
    # Each candidate should be an index into all_entries
    for idx in candidates:
        assert 0 <= idx < 20
        assert idx != 0  # query should not be in candidates
