import pytest
import torch


def test_unidirectional_fusion_shapes():
    from depth_anything_3.model.cross_view_fusion import CrossViewFusion
    fusion = CrossViewFusion(embed_dim=768, num_heads=8, num_layers=2, bidirectional=False)
    B, K, N_tokens, C = 2, 5, 1296, 768
    query_patch = torch.randn(B, N_tokens, C)
    query_cam = torch.randn(B, C)
    cand_patch = torch.randn(B, K, N_tokens, C)
    cand_cam = torch.randn(B, K, C)
    weights = torch.softmax(torch.randn(B, K), dim=1)
    enhanced_patch, enhanced_cam = fusion(query_patch, query_cam, cand_patch, cand_cam, weights)
    assert enhanced_patch.shape == (B, N_tokens, C)
    assert enhanced_cam.shape == (B, C)


def test_bidirectional_fusion_shapes():
    from depth_anything_3.model.cross_view_fusion import CrossViewFusion
    fusion = CrossViewFusion(embed_dim=768, num_heads=8, num_layers=2, bidirectional=True)
    B, K, N_tokens, C = 2, 3, 256, 768
    query_patch = torch.randn(B, N_tokens, C)
    query_cam = torch.randn(B, C)
    cand_patch = torch.randn(B, K, N_tokens, C)
    cand_cam = torch.randn(B, K, C)
    weights = torch.softmax(torch.randn(B, K), dim=1)
    enhanced_patch, enhanced_cam = fusion(query_patch, query_cam, cand_patch, cand_cam, weights)
    assert enhanced_patch.shape == (B, N_tokens, C)
    assert enhanced_cam.shape == (B, C)


def test_fusion_gradient_flows():
    from depth_anything_3.model.cross_view_fusion import CrossViewFusion
    fusion = CrossViewFusion(embed_dim=64, num_heads=4, num_layers=1, bidirectional=False)
    B, K, N_tokens, C = 1, 3, 16, 64
    query_patch = torch.randn(B, N_tokens, C, requires_grad=True)
    query_cam = torch.randn(B, C, requires_grad=True)
    cand_patch = torch.randn(B, K, N_tokens, C)
    cand_cam = torch.randn(B, K, C)
    weights = torch.softmax(torch.randn(B, K), dim=1)
    enhanced_patch, enhanced_cam = fusion(query_patch, query_cam, cand_patch, cand_cam, weights)
    loss = enhanced_patch.sum() + enhanced_cam.sum()
    loss.backward()
    assert query_patch.grad is not None
    assert query_cam.grad is not None


def test_fusion_with_hard_mask():
    """Verify fusion works with 0/1 hard mask weights (inference mode)."""
    from depth_anything_3.model.cross_view_fusion import CrossViewFusion
    fusion = CrossViewFusion(embed_dim=64, num_heads=4, num_layers=1)
    B, K, N_tokens, C = 1, 5, 16, 64
    query_patch = torch.randn(B, N_tokens, C)
    query_cam = torch.randn(B, C)
    cand_patch = torch.randn(B, K, N_tokens, C)
    cand_cam = torch.randn(B, K, C)
    weights = torch.zeros(B, K)
    weights[0, [0, 2, 4]] = 1.0
    enhanced_patch, enhanced_cam = fusion(query_patch, query_cam, cand_patch, cand_cam, weights)
    assert enhanced_patch.shape == (B, N_tokens, C)
    assert enhanced_cam.shape == (B, C)


def test_build_cross_view_fusion():
    from depth_anything_3.model.cross_view_fusion import build_cross_view_fusion
    config = {"embed_dim": 768, "num_heads": 8, "num_layers": 2, "bidirectional": False, "dropout": 0.0}
    fusion = build_cross_view_fusion(config)
    assert isinstance(fusion, torch.nn.Module)
