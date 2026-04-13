import torch
import torch.nn as nn
from unittest.mock import MagicMock
from types import SimpleNamespace

from addict import Dict


def test_unified_pipeline_forward_uses_image_contract_and_pose_top_m():
    from depth_anything_3.model.unified_pipeline import UnifiedPipeline

    B, K, H, W = 2, 4, 8, 8

    pipeline = UnifiedPipeline.__new__(UnifiedPipeline)
    nn.Module.__init__(pipeline)
    pipeline.pose_top_m = 2
    pipeline._run_backbone_single = MagicMock(return_value=(None, None, H, W))
    pipeline._run_vpr_branch = MagicMock(
        side_effect=[
            torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
            torch.tensor([[1.00, 0.0, 0.0, 0.0], [1.00, 0.0, 0.0, 0.0]]),
            torch.tensor([[0.00, 1.0, 0.0, 0.0], [0.00, 1.0, 0.0, 0.0]]),
            torch.tensor([[0.80, 0.2, 0.0, 0.0], [0.20, 0.8, 0.0, 0.0]]),
            torch.tensor([[-1.0, 0.0, 0.0, 0.0], [0.00, -1.0, 0.0, 0.0]]),
        ],
    )
    pipeline._run_backbone_multiview = MagicMock(return_value=(None, H, W))
    pipeline._run_pose_cam_dec = MagicMock(
        return_value=Dict(pose_enc=torch.randn(B, 1 + pipeline.pose_top_m, 9)),
    )

    query_image = torch.randn(B, 1, 3, H, W)
    candidate_images = torch.zeros(B, K, 3, H, W)
    for b in range(B):
        for k in range(K):
            candidate_images[b, k].fill_(float(10 * b + k))

    output = pipeline.forward(query_image, candidate_images)

    expected = torch.tensor([[0, 2], [1, 2]])
    assert torch.equal(output.selected_indices.cpu(), expected)
    assert "pose_enc" in output
    assert "query_descriptor" in output
    multi_view_input = pipeline._run_backbone_multiview.call_args[0][0]
    assert multi_view_input.shape[1] == 1 + pipeline.pose_top_m
    assert torch.equal(multi_view_input[:, :1], query_image)
    batch_idx = torch.arange(B).unsqueeze(1)
    expected_selected = candidate_images[batch_idx, expected]
    assert torch.equal(multi_view_input[:, 1:], expected_selected)


def test_unified_pipeline_pose_only():
    from depth_anything_3.model.unified_pipeline import UnifiedPipeline

    B, M, H, W = 1, 3, 8, 8

    pipeline = UnifiedPipeline.__new__(UnifiedPipeline)
    nn.Module.__init__(pipeline)
    pipeline._run_backbone_multiview = MagicMock(return_value=(None, H, W))
    pipeline._run_pose_cam_dec = MagicMock(
        return_value=Dict(pose_enc=torch.randn(B, 1 + M, 9)),
    )

    query_image = torch.randn(B, 1, 3, H, W)
    candidate_images = torch.randn(B, M, 3, H, W)

    output = pipeline.pose_only(query_image, candidate_images, pose_path="cam_dec")
    assert "pose_enc" in output
    multi_view_input = pipeline._run_backbone_multiview.call_args[0][0]
    assert torch.equal(multi_view_input[:, :1], query_image)
    assert torch.equal(multi_view_input[:, 1:], candidate_images)


def test_unified_pipeline_extract_database_features_returns_descriptors_only():
    from depth_anything_3.model.unified_pipeline import UnifiedPipeline

    B, H, W = 2, 8, 8
    expected = torch.randn(B, 16)

    pipeline = UnifiedPipeline.__new__(UnifiedPipeline)
    nn.Module.__init__(pipeline)
    pipeline._run_backbone_single = MagicMock(return_value=(None, None, H, W))
    pipeline._run_vpr_branch = MagicMock(return_value=expected)

    images = torch.randn(B, 1, 3, H, W)
    descriptors = pipeline.extract_database_features(images)
    assert torch.equal(descriptors, expected)
    assert isinstance(descriptors, torch.Tensor)


def test_build_unified_pipeline_uses_v11_constructor_without_cross_view_fusion(monkeypatch):
    from depth_anything_3.model import unified_pipeline_helper as helper

    captured = {}

    class FakeUnifiedPipeline(nn.Module):
        def __init__(
            self,
            da3_backbone,
            feature_adapter,
            aggregator,
            retrieval_strategy,
            da3_head,
            cam_dec,
            aux_layer=5,
            pose_top_m=3,
            rel_pose_head=None,
        ):
            super().__init__()
            captured["aux_layer"] = aux_layer
            captured["pose_top_m"] = pose_top_m
            self.da3_backbone = da3_backbone
            self.feature_adapter = feature_adapter
            self.aggregator = aggregator
            self.da3_head = da3_head
            self.cam_dec = cam_dec

    fake_wrapper = SimpleNamespace(
        model=SimpleNamespace(
            backbone=nn.Identity(),
            head=nn.Identity(),
            cam_dec=nn.Identity(),
        ),
    )

    monkeypatch.setattr(helper.DepthAnything3, "from_pretrained", lambda _: fake_wrapper)
    monkeypatch.setattr(helper, "build_aggregator", lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(helper, "build_feature_adapter", lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(helper, "build_retrieval_strategy", lambda cfg: object())
    monkeypatch.setattr(helper, "UnifiedPipeline", FakeUnifiedPipeline)

    cfg = {
        "model": {
            "da3_model_name_or_path": "dummy",
            "agg_arch": "gem",
            "feature_adapter_arch": "identity",
            "retrieval": {"strategy": "soft_attention", "top_k": 10, "temperature": 1.0},
            "cross_view_fusion": {"embed_dim": 1536, "num_heads": 8, "num_layers": 2},
            "pose_top_m": 5,
            "freeze": {"backbone": False, "vpr": False, "fusion": False, "head": False},
        },
    }

    helper.build_unified_pipeline(cfg)
    assert captured["pose_top_m"] == 5


def test_apply_freeze_supports_legacy_fusion_flag_without_cross_view_module():
    from depth_anything_3.model import unified_pipeline_helper as helper

    pipeline = SimpleNamespace(
        da3_backbone=nn.Linear(2, 2),
        feature_adapter=nn.Linear(2, 2),
        aggregator=nn.Linear(2, 2),
        da3_head=nn.Linear(2, 2),
        cam_dec=nn.Linear(2, 2),
    )

    helper._apply_freeze(
        pipeline,
        {"backbone": True, "vpr": True, "fusion": True, "head": False},
    )

    assert all(not p.requires_grad for p in pipeline.da3_backbone.parameters())
    assert all(not p.requires_grad for p in pipeline.feature_adapter.parameters())
    assert all(not p.requires_grad for p in pipeline.aggregator.parameters())
