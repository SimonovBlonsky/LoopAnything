import torch

from depth_anything_3.model.vpr_model import VPRModel


class StubEncoder(torch.nn.Module):
    def forward(self, x, **kwargs):
        return {
            "patch_tokens": torch.randn(x.shape[0], 4, 8),
            "feature_map": torch.randn(x.shape[0], 8, 2, 2),
            "global_token": torch.randn(x.shape[0], 8),
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


def test_salad_aggregator_receives_tuple():
    aggregator = RecordingAggregator()
    model = VPRModel(encoder=StubEncoder(), aggregator=aggregator, agg_arch="SALAD")
    descriptor = model(torch.randn(2, 3, 28, 28))
    assert isinstance(aggregator.last_input, tuple)
    assert descriptor.shape[0] == 2


def test_non_salad_aggregator_receives_feature_map_only():
    aggregator = RecordingAggregator()
    model = VPRModel(encoder=StubEncoder(), aggregator=aggregator, agg_arch="GeM")
    model(torch.randn(2, 3, 28, 28))
    assert isinstance(aggregator.last_input, torch.Tensor)
    assert aggregator.last_input.shape == (2, 8, 2, 2)


def test_return_features_returns_descriptor_and_feature_dict():
    aggregator = RecordingAggregator()
    model = VPRModel(encoder=StubEncoder(), aggregator=aggregator, agg_arch="ConvAP")
    descriptor, features = model(torch.randn(1, 3, 28, 28), return_features=True)
    assert "patch_tokens" in features
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
