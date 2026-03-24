import torch
import pytest

from depth_anything_3.model.vpr_encoder_adapter import DA3EncoderAdapter


class StubBackbone(torch.nn.Module):
    def __init__(self, feat):
        super().__init__()
        self.feat = feat

    def forward(self, x, **kwargs):
        camera_tokens = self.feat[:, :, 0]
        return ((self.feat, camera_tokens),), []


class StubDA3Net(torch.nn.Module):
    def __init__(self, feat):
        super().__init__()
        self.backbone = StubBackbone(feat)


def test_adapter_normalizes_4d_input_to_feature_dict():
    feat = torch.randn(2, 1, 6, 8)
    model = StubDA3Net(feat)
    adapter = DA3EncoderAdapter(model, patch_size=2)
    out = adapter(torch.randn(2, 3, 4, 6))
    assert out["patch_tokens"].shape == (2, 6, 8)
    assert out["feature_map"].shape == (2, 8, 2, 3)
    assert out["global_token"].shape == (2, 8)
    assert out["spatial_shape"] == (2, 3)


def test_adapter_accepts_single_view_5d_input():
    feat = torch.randn(1, 1, 2, 16)
    adapter = DA3EncoderAdapter(StubDA3Net(feat), patch_size=2)
    out = adapter(torch.randn(1, 1, 3, 2, 4))
    assert out["feature_map"].shape == (1, 16, 1, 2)


def test_adapter_rejects_non_single_view_batches():
    feat = torch.randn(1, 2, 4, 16)
    adapter = DA3EncoderAdapter(StubDA3Net(feat), patch_size=2)
    with pytest.raises(ValueError, match="single-view"):
        adapter(torch.randn(1, 2, 3, 2, 4))


def test_adapter_rejects_non_factorable_patch_count():
    feat = torch.randn(1, 1, 5, 16)
    adapter = DA3EncoderAdapter(StubDA3Net(feat), patch_size=2)
    with pytest.raises(ValueError, match="reshape"):
        adapter(torch.randn(1, 3, 2, 4))
