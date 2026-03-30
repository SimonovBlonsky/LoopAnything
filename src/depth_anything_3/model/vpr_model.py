import torch
import torch.nn as nn

from depth_anything_3.model.vpr_feature_adapter import IdentityFeatureAdapter


class VPRModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        aggregator: nn.Module,
        agg_arch: str,
        feature_adapter: nn.Module | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.aggregator = aggregator
        self.agg_arch = agg_arch.lower()
        self.feature_adapter = feature_adapter if feature_adapter is not None else IdentityFeatureAdapter()

    def extract_features(self, x, **kwargs):
        return self.encoder(x, **kwargs)

    def adapt_features(self, features):
        return self.feature_adapter(features)

    def aggregate(self, features):
        if self.agg_arch == "salad":
            return self.aggregator((features["feature_map"], features["global_token"]))
        return self.aggregator(features["feature_map"])

    def forward(self, x, return_features=False, **kwargs):
        features = self.extract_features(x, **kwargs)
        features = self.adapt_features(features)
        descriptor = self.aggregate(features)
        if not torch.isfinite(descriptor).all():
            raise ValueError("Non-finite descriptor produced by VPRModel")
        if return_features:
            return descriptor, features
        return descriptor
