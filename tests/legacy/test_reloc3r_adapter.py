"""Tests for the reloc3r data adapter (DummyStereoDataset path only, no real data)."""

import numpy as np
import torch
import pytest

from depth_anything_3.data.reloc3r_adapter import (
    DummyStereoDataset,
    _compute_relative_pose,
    _imagenet_normalize,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


@pytest.fixture
def dummy_dataset():
    return DummyStereoDataset(length=16)


class TestDummyDataset:
    def test_length(self, dummy_dataset):
        assert len(dummy_dataset) == 16

    def test_output_keys(self, dummy_dataset):
        sample = dummy_dataset[0]
        expected_keys = {"img1", "img2", "c2w_1", "c2w_2", "rel_pose_2to1", "rel_pose_1to2"}
        assert set(sample.keys()) == expected_keys

    def test_image_shape(self, dummy_dataset):
        sample = dummy_dataset[0]
        assert sample["img1"].shape == (3, 504, 504)
        assert sample["img2"].shape == (3, 504, 504)

    def test_image_dtype(self, dummy_dataset):
        sample = dummy_dataset[0]
        assert sample["img1"].dtype == torch.float32

    def test_pose_shape(self, dummy_dataset):
        sample = dummy_dataset[0]
        assert sample["c2w_1"].shape == (4, 4)
        assert sample["rel_pose_2to1"].shape == (4, 4)
        assert sample["rel_pose_1to2"].shape == (4, 4)

    def test_pose_dtype(self, dummy_dataset):
        sample = dummy_dataset[0]
        assert sample["rel_pose_2to1"].dtype == torch.float32

    def test_reproducibility(self, dummy_dataset):
        s1 = dummy_dataset[5]
        s2 = dummy_dataset[5]
        assert torch.allclose(s1["img1"], s2["img1"])
        assert torch.allclose(s1["rel_pose_2to1"], s2["rel_pose_2to1"])

    def test_different_indices_differ(self, dummy_dataset):
        s0 = dummy_dataset[0]
        s1 = dummy_dataset[1]
        assert not torch.allclose(s0["rel_pose_2to1"], s1["rel_pose_2to1"])


class TestRelativePose:
    def test_inverse_consistency(self, dummy_dataset):
        """rel_2to1 @ rel_1to2 should be identity."""
        sample = dummy_dataset[0]
        rel_21 = sample["rel_pose_2to1"].double()
        rel_12 = sample["rel_pose_1to2"].double()
        product = rel_21 @ rel_12
        eye = torch.eye(4, dtype=torch.float64)
        assert torch.allclose(product, eye, atol=1e-6), f"Product deviates from I: {(product - eye).abs().max()}"

    def test_relative_pose_matches_c2w(self, dummy_dataset):
        """rel_2to1 = inv(c2w_1) @ c2w_2."""
        sample = dummy_dataset[0]
        c2w_1 = sample["c2w_1"].double()
        c2w_2 = sample["c2w_2"].double()
        rel_21_expected = torch.linalg.inv(c2w_1) @ c2w_2
        rel_21_actual = sample["rel_pose_2to1"].double()
        assert torch.allclose(rel_21_actual, rel_21_expected, atol=1e-6)

    def test_rotation_is_so3(self, dummy_dataset):
        sample = dummy_dataset[0]
        R = sample["rel_pose_2to1"][:3, :3].double()
        det = torch.det(R)
        assert abs(det - 1.0) < 1e-6
        eye = torch.eye(3, dtype=torch.float64)
        assert torch.allclose(R.T @ R, eye, atol=1e-6)

    def test_last_row(self, dummy_dataset):
        sample = dummy_dataset[0]
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0])
        assert torch.allclose(sample["rel_pose_2to1"][3, :], expected, atol=1e-6)


class TestImageNetNormalization:
    def test_normalize_range(self):
        """After ImageNet normalization, values should be roughly in [-2.2, 2.7]."""
        img = torch.rand(3, 64, 64)  # [0, 1]
        normed = _imagenet_normalize(img)
        # Just check it's not in [0,1] anymore
        assert normed.min() < 0.0
        assert normed.max() > 1.0

    def test_denormalize_roundtrip(self):
        img = torch.rand(3, 64, 64)
        normed = _imagenet_normalize(img)
        recovered = normed * IMAGENET_STD + IMAGENET_MEAN
        assert torch.allclose(img, recovered, atol=1e-6)


class TestComputeRelativePose:
    def test_identity_when_same_pose(self):
        pose = np.eye(4)
        rel = _compute_relative_pose(pose, pose)
        np.testing.assert_allclose(rel, np.eye(4), atol=1e-10)

    def test_pure_translation(self):
        c2w_a = np.eye(4)
        c2w_b = np.eye(4)
        c2w_b[0, 3] = 1.0  # translate B by +1 along x
        rel_a_to_b = _compute_relative_pose(c2w_a, c2w_b)
        # In B's frame, A is at -1 along x
        np.testing.assert_allclose(rel_a_to_b[:3, 3], [-1.0, 0.0, 0.0], atol=1e-10)
