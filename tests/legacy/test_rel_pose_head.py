"""Tests for RelPoseHead."""

import torch
import pytest

from depth_anything_3.model.rel_pose_head import RelPoseHead


@pytest.fixture
def head():
    return RelPoseHead(token_dim=2048).eval()


def _random_tokens(batch: int, dim: int = 2048):
    return torch.randn(batch, dim)


class TestRelPoseHeadOutput:
    def test_output_shape(self, head):
        cam_a = _random_tokens(4)
        cam_b = _random_tokens(4)
        pose = head(cam_a, cam_b)
        assert pose.shape == (4, 4, 4)

    def test_rotation_is_so3(self, head):
        """R must satisfy det(R)=1 and R^T @ R = I."""
        cam_a = _random_tokens(8)
        cam_b = _random_tokens(8)
        pose = head(cam_a, cam_b)
        R = pose[:, :3, :3].double()

        det = torch.det(R)
        assert torch.allclose(det, torch.ones_like(det), atol=1e-4), f"det(R) = {det}"

        eye = torch.eye(3, dtype=torch.float64).unsqueeze(0).expand_as(R)
        RtR = torch.bmm(R.transpose(1, 2), R)
        assert torch.allclose(RtR, eye, atol=1e-4), f"R^T @ R != I, max err = {(RtR - eye).abs().max()}"

    def test_last_row(self, head):
        """Last row of 4x4 pose must be [0, 0, 0, 1]."""
        pose = head(_random_tokens(2), _random_tokens(2))
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0])
        assert torch.allclose(pose[:, 3, :], expected.unsqueeze(0).expand(2, -1), atol=1e-6)

    def test_batch_size_one(self, head):
        pose = head(_random_tokens(1), _random_tokens(1))
        assert pose.shape == (1, 4, 4)

    def test_different_token_dim(self):
        head_small = RelPoseHead(token_dim=1024).eval()
        pose = head_small(_random_tokens(2, 1024), _random_tokens(2, 1024))
        assert pose.shape == (2, 4, 4)


class TestRelPoseHeadGradient:
    def test_backward_pass(self):
        head = RelPoseHead(token_dim=2048)
        head.train()
        cam_a = torch.randn(4, 2048, requires_grad=False)
        cam_b = torch.randn(4, 2048, requires_grad=False)
        pose = head(cam_a, cam_b)
        loss = pose.sum()
        loss.backward()

        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
        assert has_grad, "No gradients flowed to head parameters"

    def test_no_nan_in_gradients(self):
        head = RelPoseHead(token_dim=2048)
        head.train()
        cam_a = torch.randn(8, 2048)
        cam_b = torch.randn(8, 2048)
        pose = head(cam_a, cam_b)
        loss = pose.sum()
        loss.backward()

        for name, p in head.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"


class TestRelPoseHeadSymmetry:
    def test_asymmetric_by_default(self, head):
        """head(a, b) != head(b, a) in general (not symmetric)."""
        cam_a = _random_tokens(4)
        cam_b = _random_tokens(4)
        pose_ab = head(cam_a, cam_b)
        pose_ba = head(cam_b, cam_a)
        assert not torch.allclose(pose_ab, pose_ba, atol=1e-3), "Outputs should differ for swapped inputs"
