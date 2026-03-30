import torch

from depth_anything_3.model.vpr_model import VPRModel
from depth_anything_3.model.vpr_feature_adapter import PatchOnlyFeatureAdapter


class StubEncoder(torch.nn.Module):
    def forward(self, x, **kwargs):
        batch_size = x.shape[0]
        patch_tokens = torch.arange(batch_size * 4 * 8, dtype=torch.float32).reshape(batch_size, 4, 8)
        feature_map = torch.arange(batch_size * 8 * 2 * 2, dtype=torch.float32).reshape(batch_size, 8, 2, 2)
        global_token = torch.arange(batch_size * 8, dtype=torch.float32).reshape(batch_size, 8) + 1000
        return {
            "patch_tokens": patch_tokens,
            "feature_map": feature_map,
            "global_token": global_token,
            "spatial_shape": (2, 2),
        }


class RecordingAggregator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_input = None

    def forward(self, x):
        self.last_input = x
        if isinstance(x, tuple):
            feature_map, global_token = x
            return torch.cat([global_token, feature_map.flatten(1)], dim=1)
        return x.flatten(1)


class RecordingFeatureAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_features = None

    def forward(self, features):
        self.seen_features = features
        return {
            "patch_tokens": features["patch_tokens"] + 3,
            "feature_map": features["feature_map"] + 5,
            "global_token": features["global_token"] + 7,
            "spatial_shape": features["spatial_shape"],
        }


def test_salad_aggregator_receives_feature_map_and_real_global_token():
    aggregator = RecordingAggregator()
    model = VPRModel(encoder=StubEncoder(), aggregator=aggregator, agg_arch="SALAD")
    input_tensor = torch.randn(2, 3, 28, 28)
    expected_features = model.encoder(input_tensor)
    descriptor = model(input_tensor)
    assert isinstance(aggregator.last_input, tuple)
    feature_map, global_token = aggregator.last_input
    assert torch.equal(feature_map, expected_features["feature_map"])
    assert torch.equal(global_token, expected_features["global_token"])
    assert descriptor.shape[0] == 2


def test_non_salad_aggregator_receives_feature_map_only():
    aggregator = RecordingAggregator()
    model = VPRModel(encoder=StubEncoder(), aggregator=aggregator, agg_arch="GeM")
    model(torch.randn(2, 3, 28, 28))
    assert isinstance(aggregator.last_input, torch.Tensor)
    assert aggregator.last_input.shape == (2, 8, 2, 2)


def test_vpr_model_routes_features_through_feature_adapter_before_aggregation():
    aggregator = RecordingAggregator()
    adapter = RecordingFeatureAdapter()
    model = VPRModel(encoder=StubEncoder(), aggregator=aggregator, agg_arch="GeM", feature_adapter=adapter)

    input_tensor = torch.randn(2, 3, 28, 28)
    original_features = model.encoder(input_tensor)

    model(input_tensor)

    assert adapter.seen_features is not None
    assert torch.equal(adapter.seen_features["feature_map"], original_features["feature_map"])
    assert torch.equal(adapter.seen_features["global_token"], original_features["global_token"])
    assert isinstance(aggregator.last_input, torch.Tensor)
    assert torch.equal(aggregator.last_input, original_features["feature_map"] + 5)


def test_salad_routing_uses_adapter_mutated_feature_map_and_global_token():
    aggregator = RecordingAggregator()
    adapter = RecordingFeatureAdapter()
    model = VPRModel(encoder=StubEncoder(), aggregator=aggregator, agg_arch="SALAD", feature_adapter=adapter)

    input_tensor = torch.randn(2, 3, 28, 28)
    original_features = model.encoder(input_tensor)

    model(input_tensor)

    assert isinstance(aggregator.last_input, tuple)
    feature_map, global_token = aggregator.last_input
    assert torch.equal(feature_map, original_features["feature_map"] + 5)
    assert torch.equal(global_token, original_features["global_token"] + 7)


def test_return_features_returns_adapted_feature_dict():
    aggregator = RecordingAggregator()
    adapter = PatchOnlyFeatureAdapter(channels=8, bottleneck=4)
    model = VPRModel(encoder=StubEncoder(), aggregator=aggregator, agg_arch="ConvAP", feature_adapter=adapter)

    with torch.no_grad():
        adapter.local_branch.expand.weight.fill_(0.01)
        adapter.local_branch.expand.bias.zero_()

    input_tensor = torch.randn(1, 3, 28, 28)
    original_features = model.encoder(input_tensor)
    descriptor, features = model(input_tensor, return_features=True)

    assert set(features) == {"patch_tokens", "feature_map", "global_token", "spatial_shape"}
    assert torch.allclose(features["global_token"], original_features["global_token"])
    assert not torch.allclose(features["feature_map"], original_features["feature_map"])
    assert torch.allclose(features["patch_tokens"], features["feature_map"].permute(0, 2, 3, 1).reshape(1, -1, 8))
    assert descriptor.shape[0] == 1


class NonFiniteAggregator(torch.nn.Module):
    def forward(self, x):
        return torch.full((x.shape[0], 4), float("inf"))


def test_non_finite_descriptor_raises_value_error():
    model = VPRModel(encoder=StubEncoder(), aggregator=NonFiniteAggregator(), agg_arch="GeM")

    try:
        model(torch.randn(1, 3, 28, 28))
    except ValueError as exc:
        assert str(exc) == "Non-finite descriptor produced by VPRModel"
    else:
        raise AssertionError("Expected ValueError for non-finite descriptor")
