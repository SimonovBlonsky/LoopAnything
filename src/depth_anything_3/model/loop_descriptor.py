"""Training-free loop descriptor helpers built from DA3 tokens."""

from __future__ import annotations

from typing import Literal, Optional, Sequence, Union

import torch
import torch.nn.functional as F

PoolingMode = Literal["mean", "gem"]


def _normalize_weights(weights: torch.Tensor, eps: float) -> torch.Tensor:
    if weights.ndim == 3 and weights.shape[-1] == 1:
        weights = weights.squeeze(-1)
    return weights / (weights.sum(dim=1, keepdim=True) + eps)


def confidence_weighted_pool(
    patch_tokens: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    if weights is None:
        return patch_tokens.mean(dim=1)
    normalized_weights = _normalize_weights(weights, eps)
    return torch.einsum("bnc,bn->bc", patch_tokens, normalized_weights)


def generalized_mean_pool(
    patch_tokens: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    p: float = 3.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    tokens = patch_tokens.abs().clamp_min(eps)
    if weights is None:
        pooled = tokens.pow(p).mean(dim=1)
    else:
        normalized_weights = _normalize_weights(weights, eps)
        pooled = torch.einsum("bnc,bn->bc", tokens.pow(p), normalized_weights)
    return pooled.pow(1.0 / p)


def confidence_map_to_token_weights(
    confidence_map: torch.Tensor,
    token_hw: Optional[Union[int, Sequence[int]]] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    if confidence_map.ndim == 4 and confidence_map.shape[1] == 1:
        confidence_map = confidence_map.squeeze(1)
    if confidence_map.ndim == 2:
        weights = confidence_map
    elif confidence_map.ndim == 3:
        if token_hw is None:
            weights = confidence_map.flatten(1)
        else:
            if isinstance(token_hw, int):
                token_hw = (token_hw, token_hw)
            weights = F.adaptive_avg_pool2d(confidence_map.unsqueeze(1), tuple(token_hw)).flatten(1)
    else:
        raise ValueError("confidence_map must have shape [B, N], [B, H, W], or [B, 1, H, W]")
    return weights.clamp_min(0.0)


def build_loop_descriptor(
    camera_tokens: torch.Tensor,
    patch_tokens: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    pooling: PoolingMode = "mean",
    gem_p: float = 3.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    camera_tokens = F.normalize(camera_tokens, dim=-1, eps=eps)
    if pooling == "mean":
        pooled = confidence_weighted_pool(patch_tokens, weights=weights, eps=eps)
    elif pooling == "gem":
        pooled = generalized_mean_pool(patch_tokens, weights=weights, p=gem_p, eps=eps)
    else:
        raise ValueError(f"Unsupported pooling mode: {pooling}")
    pooled = F.normalize(pooled, dim=-1, eps=eps)
    return F.normalize(torch.cat([camera_tokens, pooled], dim=-1), dim=-1, eps=eps)


__all__ = [
    "build_loop_descriptor",
    "confidence_map_to_token_weights",
    "confidence_weighted_pool",
    "generalized_mean_pool",
]
