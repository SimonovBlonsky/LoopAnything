import pytest
import torch

from depth_anything_3.model.vpr_feature_adapter import (
    DualBranchFeatureAdapter,
    IdentityFeatureAdapter,
    PatchOnlyFeatureAdapter,
    count_trainable_parameters,
)


def make_feature_dict(batch_size=2, channels=768, height=4, width=4):
    patch_tokens = torch.randn(batch_size, height * width, channels)
    feature_map = patch_tokens.view(batch_size, height, width, channels).permute(0, 3, 1, 2).contiguous()
    global_token = torch.randn(batch_size, channels)
    return {
        "patch_tokens": patch_tokens,
        "feature_map": feature_map,
        "global_token": global_token,
        "spatial_shape": (height, width),
    }


def flatten_feature_map(feature_map):
    return feature_map.permute(0, 2, 3, 1).reshape(feature_map.shape[0], -1, feature_map.shape[1])


def test_identity_feature_adapter_returns_same_feature_dict():
    features = make_feature_dict()
    adapter = IdentityFeatureAdapter()

    out = adapter(features)

    assert out is features
    assert set(out) == {"patch_tokens", "feature_map", "global_token", "spatial_shape"}
    assert out["spatial_shape"] == features["spatial_shape"]


def test_patch_only_adapter_updates_feature_map_but_not_global_token():
    features = make_feature_dict()
    adapter = PatchOnlyFeatureAdapter(channels=768, bottleneck=640)

    with torch.no_grad():
        adapter.local_branch.expand.weight.fill_(0.01)
        adapter.local_branch.expand.bias.zero_()

    out = adapter(features)

    assert set(out) == set(features)
    assert out["spatial_shape"] == features["spatial_shape"]
    assert torch.allclose(out["global_token"], features["global_token"])
    assert out["feature_map"].shape == features["feature_map"].shape
    assert not torch.allclose(out["feature_map"], features["feature_map"])
    assert torch.allclose(out["patch_tokens"], flatten_feature_map(out["feature_map"]))

    dual = DualBranchFeatureAdapter(channels=768, local_bottleneck=256, global_hidden_dim=256)
    patch_params = count_trainable_parameters(adapter)
    dual_params = count_trainable_parameters(dual)
    assert patch_params == pytest.approx(dual_params, rel=0.10)


def test_dual_branch_adapter_zero_init_starts_as_identity():
    features = make_feature_dict()
    adapter = DualBranchFeatureAdapter(channels=768, local_bottleneck=256, global_hidden_dim=256)

    out = adapter(features)

    assert set(out) == set(features)
    assert out["spatial_shape"] == features["spatial_shape"]
    assert torch.allclose(out["feature_map"], features["feature_map"])
    assert torch.allclose(out["patch_tokens"], features["patch_tokens"])
    assert torch.allclose(out["global_token"], features["global_token"])


def test_dual_branch_adapter_returns_finite_outputs_and_regenerates_patch_tokens():
    features = make_feature_dict()
    adapter = DualBranchFeatureAdapter(channels=768, local_bottleneck=256, global_hidden_dim=256)

    with torch.no_grad():
        adapter.local_branch.expand.weight.fill_(0.01)
        adapter.local_branch.expand.bias.zero_()
        adapter.global_branch.expand.weight.fill_(0.01)
        adapter.global_branch.expand.bias.zero_()

    out = adapter(features)

    assert set(out) == set(features)
    assert out["spatial_shape"] == features["spatial_shape"]
    assert torch.isfinite(out["feature_map"]).all()
    assert torch.isfinite(out["patch_tokens"]).all()
    assert torch.isfinite(out["global_token"]).all()
    assert out["global_token"].shape == features["global_token"].shape
    assert out["feature_map"].shape == features["feature_map"].shape
    assert torch.allclose(out["patch_tokens"], flatten_feature_map(out["feature_map"]))
    assert not torch.allclose(out["feature_map"], features["feature_map"])
    assert not torch.allclose(out["global_token"], features["global_token"])


@pytest.mark.parametrize(
    "adapter, bad_features",
    [
        (
            PatchOnlyFeatureAdapter(channels=768, bottleneck=640),
            {
                "patch_tokens": torch.randn(2, 17, 768),
                "feature_map": torch.randn(2, 768, 4, 4),
                "global_token": torch.randn(2, 768),
                "spatial_shape": (4, 4),
            },
        ),
        (
            DualBranchFeatureAdapter(channels=768, local_bottleneck=256, global_hidden_dim=256),
            {
                "patch_tokens": torch.randn(2, 16, 768),
                "feature_map": torch.randn(2, 768, 4, 4),
                "global_token": torch.full((2, 768), float("nan")),
                "spatial_shape": (4, 4),
            },
        ),
    ],
)
def test_feature_adapters_raise_value_error_for_shape_mismatch_or_non_finite_inputs(adapter, bad_features):
    with pytest.raises(ValueError):
        adapter(bad_features)
