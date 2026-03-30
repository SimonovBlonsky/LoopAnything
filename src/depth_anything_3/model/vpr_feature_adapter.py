from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def _is_finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().item())


def _zero_init_linear(module: nn.Linear) -> None:
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def _validate_feature_dict(features, channels: int | None = None):
    if not isinstance(features, Mapping):
        raise ValueError("Expected a feature dict")

    required_keys = {"patch_tokens", "feature_map", "global_token", "spatial_shape"}
    if set(features) != required_keys:
        raise ValueError("Feature dict schema must contain patch_tokens, feature_map, global_token, and spatial_shape")

    patch_tokens = features["patch_tokens"]
    feature_map = features["feature_map"]
    global_token = features["global_token"]
    spatial_shape = features["spatial_shape"]

    if not torch.is_tensor(patch_tokens) or not torch.is_tensor(feature_map) or not torch.is_tensor(global_token):
        raise ValueError("Feature dict tensors must be torch.Tensor instances")
    if patch_tokens.ndim != 3:
        raise ValueError("patch_tokens must have shape [B, N, C]")
    if feature_map.ndim != 4:
        raise ValueError("feature_map must have shape [B, C, H, W]")
    if global_token.ndim != 2:
        raise ValueError("global_token must have shape [B, C]")
    if feature_map.shape[0] != patch_tokens.shape[0] or feature_map.shape[0] != global_token.shape[0]:
        raise ValueError("Feature batch dimensions must match")
    inferred_channels = feature_map.shape[1]
    if channels is None:
        channels = inferred_channels
    if feature_map.shape[1] != channels or patch_tokens.shape[2] != channels or global_token.shape[1] != channels:
        raise ValueError("Feature channel dimensions must match the adapter channels")
    if not isinstance(spatial_shape, (tuple, list)) or len(spatial_shape) != 2:
        raise ValueError("spatial_shape must be a length-2 tuple or list")

    height, width = spatial_shape
    if feature_map.shape[2:] != (height, width):
        raise ValueError("feature_map shape does not match spatial_shape")
    if patch_tokens.shape[1] != height * width:
        raise ValueError("patch_tokens shape does not match spatial_shape")

    if not _is_finite(patch_tokens):
        raise ValueError("patch_tokens contains non-finite values")
    if not _is_finite(feature_map):
        raise ValueError("feature_map contains non-finite values")
    if not _is_finite(global_token):
        raise ValueError("global_token contains non-finite values")

    return patch_tokens, feature_map, global_token, spatial_shape


class _LocalFeatureBranch(nn.Module):
    def __init__(self, channels: int, bottleneck: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.reduce = nn.Linear(channels, bottleneck)
        self.activation = nn.GELU()
        self.depthwise = nn.Conv2d(bottleneck, bottleneck, kernel_size=3, padding=1, groups=bottleneck)
        self.expand = nn.Linear(bottleneck, channels)
        _zero_init_linear(self.expand)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        residual = feature_map
        x = self.norm(feature_map.permute(0, 2, 3, 1).contiguous())
        x = self.reduce(x)
        x = self.activation(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.depthwise(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.expand(x).permute(0, 3, 1, 2).contiguous()
        return residual + x


class _GlobalFeatureBranch(nn.Module):
    def __init__(self, channels: int, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels * 2)
        self.reduce = nn.Linear(channels * 2, hidden_dim)
        self.activation = nn.GELU()
        self.expand = nn.Linear(hidden_dim, channels)
        _zero_init_linear(self.expand)

    def forward(self, global_token: torch.Tensor, feature_map: torch.Tensor) -> torch.Tensor:
        pooled_feature = feature_map.mean(dim=(2, 3))
        x = torch.cat([global_token, pooled_feature], dim=-1)
        x = self.norm(x)
        x = self.reduce(x)
        x = self.activation(x)
        x = self.expand(x)
        return global_token + x


class IdentityFeatureAdapter(nn.Module):
    def forward(self, features):
        _validate_feature_dict(features)
        return features


class PatchOnlyFeatureAdapter(nn.Module):
    def __init__(self, channels: int = 768, bottleneck: int = 640):
        super().__init__()
        self.channels = channels
        self.bottleneck = bottleneck
        self.local_branch = _LocalFeatureBranch(channels, bottleneck)

    def forward(self, features):
        _, feature_map, global_token, spatial_shape = _validate_feature_dict(features, self.channels)
        adapted_feature_map = self.local_branch(feature_map)
        patch_tokens = adapted_feature_map.permute(0, 2, 3, 1).reshape(
            adapted_feature_map.shape[0], -1, adapted_feature_map.shape[1]
        )
        if not _is_finite(adapted_feature_map):
            raise ValueError("feature_map contains non-finite values")
        if not _is_finite(patch_tokens):
            raise ValueError("patch_tokens contains non-finite values")
        return {
            "patch_tokens": patch_tokens,
            "feature_map": adapted_feature_map,
            "global_token": global_token,
            "spatial_shape": spatial_shape,
        }


class DualBranchFeatureAdapter(nn.Module):
    def __init__(self, channels: int = 768, local_bottleneck: int = 256, global_hidden_dim: int = 256):
        super().__init__()
        self.channels = channels
        self.local_bottleneck = local_bottleneck
        self.global_hidden_dim = global_hidden_dim
        self.local_branch = _LocalFeatureBranch(channels, local_bottleneck)
        self.global_branch = _GlobalFeatureBranch(channels, global_hidden_dim)

    def forward(self, features):
        _, feature_map, global_token, spatial_shape = _validate_feature_dict(features, self.channels)
        adapted_feature_map = self.local_branch(feature_map)
        adapted_global_token = self.global_branch(global_token, adapted_feature_map)
        patch_tokens = adapted_feature_map.permute(0, 2, 3, 1).reshape(
            adapted_feature_map.shape[0], -1, adapted_feature_map.shape[1]
        )

        if not _is_finite(adapted_feature_map):
            raise ValueError("feature_map contains non-finite values")
        if not _is_finite(adapted_global_token):
            raise ValueError("global_token contains non-finite values")
        if not _is_finite(patch_tokens):
            raise ValueError("patch_tokens contains non-finite values")

        return {
            "patch_tokens": patch_tokens,
            "feature_map": adapted_feature_map,
            "global_token": adapted_global_token,
            "spatial_shape": spatial_shape,
        }
