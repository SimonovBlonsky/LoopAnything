import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock


def _make_mock_backbone(B=1, N=1, P=1296, C_final=1536, C_aux=768):
    """Create a mock backbone that returns realistic feature shapes."""
    mock = MagicMock(spec=nn.Module)
    mock.parameters = MagicMock(return_value=iter([torch.randn(1)]))
    mock.named_parameters = MagicMock(return_value=iter([("w", torch.randn(1))]))

    feats = [
        (torch.randn(B, N, P, C_final), torch.randn(B, 1, C_final))
        for _ in range(4)
    ]
    aux_feats = [torch.randn(B, N, P, C_aux)]

    def backbone_forward(x, cam_token=None, export_feat_layers=None, ref_view_strategy=None):
        return feats, aux_feats

    mock.side_effect = backbone_forward
    mock.__call__ = backbone_forward
    return mock, feats, aux_feats


def test_unified_pipeline_forward_smoke():
    """Smoke test: verify forward runs without error and returns expected keys."""
    from depth_anything_3.model.unified_pipeline import UnifiedPipeline
    from depth_anything_3.model.retrieval_strategy import SoftAttentionRetrieval
    from depth_anything_3.model.cross_view_fusion import CrossViewFusion
    from depth_anything_3.model.vpr_feature_adapter import PatchOnlyFeatureAdapter

    B, K, P, C_final, C_aux = 1, 5, 1296, 1536, 768

    # Build with mock backbone (avoid loading real DA3 weights)
    pipeline = UnifiedPipeline.__new__(UnifiedPipeline)
    nn.Module.__init__(pipeline)

    mock_backbone, feats, aux_feats = _make_mock_backbone(B=B, P=P, C_final=C_final, C_aux=C_aux)
    pipeline.da3_backbone = mock_backbone
    pipeline.aux_layer = 5
    pipeline.PATCH_SIZE = 14
    pipeline.feature_adapter = PatchOnlyFeatureAdapter(channels=C_aux)
    pipeline.aggregator = MagicMock()
    pipeline.aggregator.return_value = torch.randn(B, 8192)
    pipeline.retrieval_strategy = SoftAttentionRetrieval(temperature=1.0, top_k=3)
    pipeline.cross_view_fusion = CrossViewFusion(embed_dim=C_final, num_heads=8, num_layers=1)

    # Mock head and cam_dec
    pipeline.da3_head = MagicMock()
    pipeline.da3_head.return_value = {"depth": torch.randn(B, 1, 504, 504)}
    pipeline.cam_dec = MagicMock()
    pipeline.cam_dec.return_value = torch.randn(B, 1, 9)

    # Inputs
    query = torch.randn(B, 1, 3, 504, 504)
    candidate_descriptors = torch.randn(K, 8192)
    candidate_patch_tokens = torch.randn(B, K, P, C_final)
    candidate_camera_tokens = torch.randn(B, K, C_final)

    pipeline.train()
    output = pipeline(query, candidate_patch_tokens, candidate_camera_tokens, candidate_descriptors)
    assert "pose_enc" in output
    assert "query_descriptor" in output


def test_unified_pipeline_retrieval_only():
    from depth_anything_3.model.unified_pipeline import UnifiedPipeline
    from depth_anything_3.model.vpr_feature_adapter import PatchOnlyFeatureAdapter

    B, P, C_aux = 1, 1296, 768

    pipeline = UnifiedPipeline.__new__(UnifiedPipeline)
    nn.Module.__init__(pipeline)

    mock_backbone, feats, aux_feats = _make_mock_backbone(B=B, P=P, C_aux=C_aux)
    pipeline.da3_backbone = mock_backbone
    pipeline.aux_layer = 5
    pipeline.PATCH_SIZE = 14
    pipeline.feature_adapter = PatchOnlyFeatureAdapter(channels=C_aux)
    pipeline.aggregator = MagicMock()
    pipeline.aggregator.return_value = torch.randn(B, 8192)

    images = torch.randn(B, 1, 3, 504, 504)
    descriptor = pipeline.retrieval_only(images)
    assert descriptor.shape == (B, 8192)


def test_unified_pipeline_pose_only():
    from depth_anything_3.model.unified_pipeline import UnifiedPipeline
    from depth_anything_3.model.cross_view_fusion import CrossViewFusion

    B, K, P, C_final = 1, 3, 1296, 1536

    pipeline = UnifiedPipeline.__new__(UnifiedPipeline)
    nn.Module.__init__(pipeline)

    mock_backbone, feats, aux_feats = _make_mock_backbone(B=B, P=P, C_final=C_final)
    pipeline.da3_backbone = mock_backbone
    pipeline.aux_layer = 5
    pipeline.PATCH_SIZE = 14
    pipeline.cross_view_fusion = CrossViewFusion(embed_dim=C_final, num_heads=8, num_layers=1)
    pipeline.da3_head = MagicMock()
    pipeline.da3_head.return_value = {"depth": torch.randn(B, 1, 504, 504)}
    pipeline.cam_dec = MagicMock()
    pipeline.cam_dec.return_value = torch.randn(B, 1, 9)

    query_image = torch.randn(B, 1, 3, 504, 504)
    candidate_images = torch.randn(B, K, 3, 504, 504)

    output = pipeline.pose_only(query_image, candidate_images)
    assert "pose_enc" in output
