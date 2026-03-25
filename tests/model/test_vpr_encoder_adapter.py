import pytest
import torch

from depth_anything_3.model.vpr_encoder_adapter import DA3EncoderAdapter


class StubBackbone(torch.nn.Module):
    def __init__(self, feat, camera_feat=None, aux_feats=None):
        super().__init__()
        self.feat = feat
        self.camera_feat = camera_feat
        self.aux_feats = [] if aux_feats is None else aux_feats
        self.calls = []

    def forward(self, x, **kwargs):
        self.calls.append((x, kwargs))
        camera_tokens = self.camera_feat if self.camera_feat is not None else self.feat[:, :, 0]
        return ((self.feat, camera_tokens),), list(self.aux_feats)


class StubMalformedBackbone(torch.nn.Module):
    def __init__(self, feat):
        super().__init__()
        self.feat = feat

    def forward(self, x, **kwargs):
        return ((self.feat,),), []


class StubDA3Net(torch.nn.Module):
    def __init__(self, feat, camera_feat=None, aux_feats=None):
        super().__init__()
        self.backbone = StubBackbone(feat, camera_feat=camera_feat, aux_feats=aux_feats)


class StubMalformedDA3Net(torch.nn.Module):
    def __init__(self, feat):
        super().__init__()
        self.backbone = StubMalformedBackbone(feat)


class StubDA3Wrapper:
    def __init__(self, model):
        self.model = model


def test_adapter_normalizes_4d_input_to_feature_dict():
    feat = torch.randn(2, 1, 6, 8)
    model = StubDA3Net(feat)
    adapter = DA3EncoderAdapter(model, patch_size=2)

    out = adapter(torch.randn(2, 3, 4, 6))

    called_x, called_kwargs = model.backbone.calls[0]
    assert called_x.shape == (2, 1, 3, 4, 6)
    assert called_kwargs == {
        "cam_token": None,
        "export_feat_layers": [],
        "ref_view_strategy": "saddle_balanced",
    }
    assert out["patch_tokens"].shape == (2, 6, 8)
    assert out["feature_map"].shape == (2, 8, 2, 3)
    assert out["global_token"].shape == (2, 8)
    assert out["spatial_shape"] == (2, 3)


def test_adapter_accepts_single_view_5d_input():
    feat = torch.randn(1, 1, 2, 16)
    adapter = DA3EncoderAdapter(StubDA3Net(feat), patch_size=2)

    out = adapter(torch.randn(1, 1, 3, 2, 4))

    assert out["feature_map"].shape == (1, 16, 1, 2)


def test_adapter_unwraps_model_attribute():
    feat = torch.randn(1, 1, 2, 16)
    model = StubDA3Wrapper(StubDA3Net(feat))
    adapter = DA3EncoderAdapter(model, patch_size=2)

    adapter(torch.randn(1, 3, 2, 4))

    called_x, called_kwargs = model.model.backbone.calls[0]
    assert called_x.shape == (1, 1, 3, 2, 4)
    assert called_kwargs["ref_view_strategy"] == "saddle_balanced"


def test_adapter_uses_requested_aux_layer_as_feature_source():
    feat = torch.randn(1, 1, 4, 16)
    aux_feat = torch.randn(1, 1, 4, 8)
    model = StubDA3Net(feat, aux_feats=[aux_feat])
    adapter = DA3EncoderAdapter(model, patch_size=2, feature_source="aux", aux_layer=3)

    out = adapter(torch.randn(1, 3, 4, 4))

    called_x, called_kwargs = model.backbone.calls[0]
    assert called_x.shape == (1, 1, 3, 4, 4)
    assert called_kwargs["export_feat_layers"] == [3]
    assert out["patch_tokens"].shape == (1, 4, 8)
    assert out["feature_map"].shape == (1, 8, 2, 2)
    assert torch.allclose(out["global_token"], aux_feat[:, 0].mean(dim=1))


def test_adapter_rejects_invalid_feature_source():
    feat = torch.randn(1, 1, 4, 16)
    adapter = DA3EncoderAdapter(StubDA3Net(feat), patch_size=2, feature_source="bad")

    with pytest.raises(ValueError, match="Unsupported feature_source"):
        adapter(torch.randn(1, 3, 4, 4))


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


def test_adapter_rejects_non_finite_feature_map():
    feat = torch.randn(1, 1, 4, 16)
    feat[0, 0, 1, 0] = float("inf")
    adapter = DA3EncoderAdapter(StubDA3Net(feat), patch_size=2)

    with pytest.raises(ValueError, match="feature_map contains non-finite values"):
        adapter(torch.randn(1, 3, 4, 4))


def test_adapter_rejects_non_finite_global_token():
    feat = torch.randn(1, 1, 4, 16)
    camera_feat = torch.randn(1, 1, 16)
    camera_feat[0, 0, 0] = float("nan")
    adapter = DA3EncoderAdapter(StubDA3Net(feat, camera_feat=camera_feat), patch_size=2)

    with pytest.raises(ValueError, match="global_token contains non-finite values"):
        adapter(torch.randn(1, 3, 4, 4))


def test_adapter_rejects_missing_aux_features():
    feat = torch.randn(1, 1, 4, 16)
    adapter = DA3EncoderAdapter(StubDA3Net(feat, aux_feats=[]), patch_size=2, feature_source="aux", aux_layer=3)

    with pytest.raises(ValueError, match="No auxiliary features"):
        adapter(torch.randn(1, 3, 4, 4))


def test_adapter_rejects_malformed_backbone_output():
    feat = torch.randn(1, 1, 4, 16)
    adapter = DA3EncoderAdapter(StubMalformedDA3Net(feat), patch_size=2)

    with pytest.raises(ValueError, match=r"expected \(patch_tokens, camera_tokens\) from backbone"):
        adapter(torch.randn(1, 3, 4, 4))


def test_adapter_handles_non_contiguous_patch_tokens():
    feat = torch.randn(1, 1, 8, 16)[:, :, ::2]
    adapter = DA3EncoderAdapter(StubDA3Net(feat), patch_size=2)

    out = adapter(torch.randn(1, 3, 4, 4))

    assert not feat[:, 0].is_contiguous()
    assert out["patch_tokens"].shape == (1, 4, 16)
    assert out["feature_map"].shape == (1, 16, 2, 2)
