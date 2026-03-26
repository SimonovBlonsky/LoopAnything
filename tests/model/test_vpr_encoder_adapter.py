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


@pytest.mark.parametrize(
    "layer_combine, layer_weights, expected_patch_tokens",
    [
        ("avg", None, lambda a, b: (a + b) / 2),
        ("sum", None, lambda a, b: a + b),
        ("weighted_avg", [1.0, 3.0], lambda a, b: (a + 3.0 * b) / 4.0),
    ],
)
def test_adapter_fuses_multiple_aux_layers_before_reshaping(layer_combine, layer_weights, expected_patch_tokens):
    feat = torch.randn(1, 1, 4, 16)
    aux_feat_1 = torch.tensor(
        [[[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]]],
        dtype=torch.float32,
    )
    aux_feat_2 = torch.tensor(
        [[[[5.0, 50.0], [6.0, 60.0], [7.0, 70.0], [8.0, 80.0]]]],
        dtype=torch.float32,
    )
    model = StubDA3Net(feat, aux_feats=[aux_feat_1, aux_feat_2])
    adapter = DA3EncoderAdapter(
        model,
        patch_size=2,
        feature_source="aux",
        aux_layers=[3, 6],
        layer_combine=layer_combine,
        layer_weights=layer_weights,
    )

    out = adapter(torch.randn(1, 3, 4, 4))

    called_x, called_kwargs = model.backbone.calls[0]
    assert called_x.shape == (1, 1, 3, 4, 4)
    assert called_kwargs["export_feat_layers"] == [3, 6]
    expected = expected_patch_tokens(aux_feat_1[:, 0], aux_feat_2[:, 0])
    assert out["patch_tokens"].shape == (1, 4, 2)
    assert torch.allclose(out["patch_tokens"], expected)
    assert torch.allclose(out["global_token"], expected.mean(dim=1))
    assert out["feature_map"].shape == (1, 2, 2, 2)


def test_adapter_applies_layer_scale_after_aux_fusion():
    feat = torch.randn(1, 1, 4, 16)
    aux_feat_1 = torch.tensor(
        [[[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]]],
        dtype=torch.float32,
    )
    aux_feat_2 = torch.tensor(
        [[[[5.0, 50.0], [6.0, 60.0], [7.0, 70.0], [8.0, 80.0]]]],
        dtype=torch.float32,
    )
    adapter = DA3EncoderAdapter(
        StubDA3Net(feat, aux_feats=[aux_feat_1, aux_feat_2]),
        patch_size=2,
        feature_source="aux",
        aux_layers=[3, 6],
        layer_combine="avg",
        layer_scale=4.0,
    )

    out = adapter(torch.randn(1, 3, 4, 4))

    expected_unscaled = (aux_feat_1[:, 0] + aux_feat_2[:, 0]) / 2
    expected_scaled = expected_unscaled * 4.0
    assert torch.allclose(out["patch_tokens"], expected_scaled)
    assert torch.allclose(out["global_token"], expected_scaled.mean(dim=1))


def test_adapter_applies_post_fusion_token_l2_norm():
    feat = torch.randn(1, 1, 4, 16)
    aux_feat_1 = torch.tensor(
        [[[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]]],
        dtype=torch.float32,
    )
    aux_feat_2 = torch.tensor(
        [[[[5.0, 50.0], [6.0, 60.0], [7.0, 70.0], [8.0, 80.0]]]],
        dtype=torch.float32,
    )
    adapter = DA3EncoderAdapter(
        StubDA3Net(feat, aux_feats=[aux_feat_1, aux_feat_2]),
        patch_size=2,
        feature_source="aux",
        aux_layers=[3, 6],
        layer_combine="avg",
        post_fusion_norm="token_l2",
    )

    out = adapter(torch.randn(1, 3, 4, 4))

    expected_unscaled = (aux_feat_1[:, 0] + aux_feat_2[:, 0]) / 2
    expected_normed = torch.nn.functional.normalize(expected_unscaled, p=2, dim=-1)
    assert torch.allclose(out["patch_tokens"], expected_normed)
    assert torch.allclose(out["patch_tokens"].norm(dim=-1), torch.ones_like(out["patch_tokens"].norm(dim=-1)))



def test_adapter_applies_post_fusion_feature_layernorm():
    feat = torch.randn(1, 1, 4, 16)
    aux_feat_1 = torch.tensor(
        [[[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]]],
        dtype=torch.float32,
    )
    aux_feat_2 = torch.tensor(
        [[[[5.0, 50.0], [6.0, 60.0], [7.0, 70.0], [8.0, 80.0]]]],
        dtype=torch.float32,
    )
    adapter = DA3EncoderAdapter(
        StubDA3Net(feat, aux_feats=[aux_feat_1, aux_feat_2]),
        patch_size=2,
        feature_source="aux",
        aux_layers=[3, 6],
        layer_combine="avg",
        post_fusion_norm="feature_layernorm",
    )

    out = adapter(torch.randn(1, 3, 4, 4))

    expected_unscaled = (aux_feat_1[:, 0] + aux_feat_2[:, 0]) / 2
    expected_normed = torch.nn.functional.layer_norm(expected_unscaled, normalized_shape=(expected_unscaled.shape[-1],))
    assert torch.allclose(out["patch_tokens"], expected_normed)



def test_adapter_rejects_invalid_feature_source():
    feat = torch.randn(1, 1, 4, 16)
    adapter = DA3EncoderAdapter(StubDA3Net(feat), patch_size=2, feature_source="bad")

    with pytest.raises(ValueError, match="Unsupported feature_source"):
        adapter(torch.randn(1, 3, 4, 4))


def test_adapter_rejects_final_with_explicit_aux_fusion_args():
    feat = torch.randn(1, 1, 4, 16)
    adapter = DA3EncoderAdapter(StubDA3Net(feat), patch_size=2, feature_source="final", aux_layers=[3, 6])

    with pytest.raises(ValueError, match="explicit AUX fusion args"):
        adapter(torch.randn(1, 3, 4, 4))


def test_adapter_accepts_final_mode_with_mixed_case_single_layer_combine():
    feat = torch.randn(1, 1, 4, 16)
    adapter = DA3EncoderAdapter(StubDA3Net(feat), patch_size=2, feature_source="final", layer_combine="Single")

    out = adapter(torch.randn(1, 3, 4, 4))

    called_x, called_kwargs = adapter.da3_model.backbone.calls[0]
    assert called_x.shape == (1, 1, 3, 4, 4)
    assert called_kwargs["export_feat_layers"] == []
    assert out["patch_tokens"].shape == (1, 4, 16)
    assert out["global_token"].shape == (1, 16)


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


def test_adapter_rejects_malformed_final_patch_tokens():
    feat = torch.randn(1, 4, 16)
    adapter = DA3EncoderAdapter(StubDA3Net(feat), patch_size=2)

    with pytest.raises(ValueError, match="final patch_tokens with shape"):
        adapter(torch.randn(1, 3, 4, 4))


def test_adapter_rejects_malformed_final_camera_tokens():
    feat = torch.randn(1, 1, 4, 16)
    camera_feat = torch.randn(1, 16)
    adapter = DA3EncoderAdapter(StubDA3Net(feat, camera_feat=camera_feat), patch_size=2)

    with pytest.raises(ValueError, match="final camera_tokens with shape"):
        adapter(torch.randn(1, 3, 4, 4))


def test_adapter_rejects_missing_aux_features():
    feat = torch.randn(1, 1, 4, 16)
    adapter = DA3EncoderAdapter(StubDA3Net(feat, aux_feats=[]), patch_size=2, feature_source="aux", aux_layer=3)

    with pytest.raises(ValueError, match="No auxiliary features"):
        adapter(torch.randn(1, 3, 4, 4))


def test_adapter_rejects_malformed_aux_feature_tensors():
    feat = torch.randn(1, 1, 4, 16)
    aux_feat = torch.randn(1, 2, 4, 8)
    adapter = DA3EncoderAdapter(StubDA3Net(feat, aux_feats=[aux_feat]), patch_size=2, feature_source="aux", aux_layer=3)

    with pytest.raises(ValueError, match="Expected auxiliary patch tokens"):
        adapter(torch.randn(1, 3, 4, 4))


def test_adapter_rejects_malformed_incompatible_aux_feature_shapes():
    feat = torch.randn(1, 1, 4, 16)
    aux_feat_1 = torch.randn(1, 1, 4, 8)
    aux_feat_2 = torch.randn(1, 1, 5, 8)
    adapter = DA3EncoderAdapter(
        StubDA3Net(feat, aux_feats=[aux_feat_1, aux_feat_2]),
        patch_size=2,
        feature_source="aux",
        aux_layers=[3, 6],
        layer_combine="avg",
    )

    with pytest.raises(ValueError, match="Malformed/incompatible AUX features"):
        adapter(torch.randn(1, 3, 4, 4))


def test_adapter_rejects_single_combine_with_multiple_aux_layers():
    feat = torch.randn(1, 1, 4, 16)
    aux_feat = torch.randn(1, 1, 4, 8)
    adapter = DA3EncoderAdapter(
        StubDA3Net(feat, aux_feats=[aux_feat, aux_feat]),
        patch_size=2,
        feature_source="aux",
        aux_layers=[3, 6],
    )

    with pytest.raises(ValueError, match="single layer_combine requires exactly one aux layer"):
        adapter(torch.randn(1, 3, 4, 4))


def test_adapter_rejects_avg_with_single_aux_layer():
    feat = torch.randn(1, 1, 4, 16)
    aux_feat = torch.randn(1, 1, 4, 8)
    adapter = DA3EncoderAdapter(
        StubDA3Net(feat, aux_feats=[aux_feat]),
        patch_size=2,
        feature_source="aux",
        layer_combine="avg",
    )

    with pytest.raises(ValueError, match="avg layer_combine requires multiple aux layers"):
        adapter(torch.randn(1, 3, 4, 4))


def test_adapter_rejects_weighted_avg_without_weights():
    feat = torch.randn(1, 1, 4, 16)
    aux_feat = torch.randn(1, 1, 4, 8)
    adapter = DA3EncoderAdapter(
        StubDA3Net(feat, aux_feats=[aux_feat, aux_feat]),
        patch_size=2,
        feature_source="aux",
        aux_layers=[3, 6],
        layer_combine="weighted_avg",
    )

    with pytest.raises(ValueError, match="weighted_avg requires layer_weights to match aux_layers"):
        adapter(torch.randn(1, 3, 4, 4))


def test_adapter_rejects_weighted_avg_with_mismatched_weights():
    feat = torch.randn(1, 1, 4, 16)
    aux_feat = torch.randn(1, 1, 4, 8)
    adapter = DA3EncoderAdapter(
        StubDA3Net(feat, aux_feats=[aux_feat, aux_feat]),
        patch_size=2,
        feature_source="aux",
        aux_layers=[3, 6],
        layer_combine="weighted_avg",
        layer_weights=[1.0],
    )

    with pytest.raises(ValueError, match="weighted_avg requires layer_weights to match aux_layers"):
        adapter(torch.randn(1, 3, 4, 4))


@pytest.mark.parametrize("layer_weights", [[1.0, float("nan")], [1.0, float("inf")]])
def test_adapter_rejects_non_finite_weighted_avg_weights(layer_weights):
    feat = torch.randn(1, 1, 4, 16)
    aux_feat = torch.randn(1, 1, 4, 8)
    adapter = DA3EncoderAdapter(
        StubDA3Net(feat, aux_feats=[aux_feat, aux_feat]),
        patch_size=2,
        feature_source="aux",
        aux_layers=[3, 6],
        layer_combine="weighted_avg",
        layer_weights=layer_weights,
    )

    with pytest.raises(ValueError, match="finite layer_weights"):
        adapter(torch.randn(1, 3, 4, 4))


def test_adapter_rejects_weighted_avg_with_zero_total_weight():
    feat = torch.randn(1, 1, 4, 16)
    aux_feat = torch.randn(1, 1, 4, 8)
    adapter = DA3EncoderAdapter(
        StubDA3Net(feat, aux_feats=[aux_feat, aux_feat]),
        patch_size=2,
        feature_source="aux",
        aux_layers=[3, 6],
        layer_combine="weighted_avg",
        layer_weights=[1.0, -1.0],
    )

    with pytest.raises(ValueError, match="non-zero total weight"):
        adapter(torch.randn(1, 3, 4, 4))


def test_adapter_rejects_negative_aux_indices():
    feat = torch.randn(1, 1, 4, 16)
    aux_feat = torch.randn(1, 1, 4, 8)
    adapter = DA3EncoderAdapter(
        StubDA3Net(feat, aux_feats=[aux_feat]),
        patch_size=2,
        feature_source="aux",
        aux_layers=[-1],
    )

    with pytest.raises(ValueError, match="negative indices"):
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
