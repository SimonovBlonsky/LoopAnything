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


def _flatten_patch_tokens(
    patch_tokens: torch.Tensor,
) -> tuple[torch.Tensor, Optional[tuple[int, int]]]:
    if patch_tokens.ndim == 3:
        return patch_tokens, None
    if patch_tokens.ndim == 4:
        batch, views, tokens, channels = patch_tokens.shape
        return patch_tokens.reshape(batch * views, tokens, channels), (batch, views)
    raise ValueError("patch_tokens must have shape [B, N, C] or [B, S, N, C]")


def _prepare_weights(
    weights: Optional[torch.Tensor],
    patch_tokens: torch.Tensor,
) -> Optional[torch.Tensor]:
    if weights is None:
        return None
    if weights.ndim == patch_tokens.ndim and weights.shape[-1] == 1:
        weights = weights.squeeze(-1)
    if patch_tokens.ndim == 3:
        if weights.ndim != 2 or weights.shape != patch_tokens.shape[:2]:
            raise ValueError("weights must have shape [B, N] for patch_tokens shaped [B, N, C]")
        return weights
    if weights.ndim != 3 or weights.shape != patch_tokens.shape[:3]:
        raise ValueError("weights must have shape [B, S, N] for patch_tokens shaped [B, S, N, C]")
    batch, views, tokens = weights.shape
    return weights.reshape(batch * views, tokens)


def _restore_view_shape(
    tensor: torch.Tensor,
    view_shape: Optional[tuple[int, int]],
) -> torch.Tensor:
    if view_shape is None:
        return tensor
    return tensor.reshape(*view_shape, tensor.shape[-1])


def confidence_weighted_pool(
    patch_tokens: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    flat_tokens, view_shape = _flatten_patch_tokens(patch_tokens)
    flat_weights = _prepare_weights(weights, patch_tokens)
    if flat_weights is None:
        pooled = flat_tokens.mean(dim=1)
    else:
        normalized_weights = _normalize_weights(flat_weights, eps)
        pooled = torch.einsum("bnc,bn->bc", flat_tokens, normalized_weights)
    return _restore_view_shape(pooled, view_shape)


def generalized_mean_pool(
    patch_tokens: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    p: float = 3.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    flat_tokens, view_shape = _flatten_patch_tokens(patch_tokens)
    flat_weights = _prepare_weights(weights, patch_tokens)
    signed_tokens = torch.sign(flat_tokens) * flat_tokens.abs().pow(p)
    if flat_weights is None:
        pooled = signed_tokens.mean(dim=1)
    else:
        normalized_weights = _normalize_weights(flat_weights, eps)
        pooled = torch.einsum("bnc,bn->bc", signed_tokens, normalized_weights)
    pooled = torch.sign(pooled) * pooled.abs().pow(1.0 / p)
    return _restore_view_shape(pooled, view_shape)


def confidence_map_to_token_weights(
    confidence_map: torch.Tensor,
    token_hw: Optional[Union[int, Sequence[int]]] = None,
) -> torch.Tensor:
    if confidence_map.ndim == 5 and confidence_map.shape[2] == 1:
        confidence_map = confidence_map.squeeze(2)
    if confidence_map.ndim == 4:
        batch, views, height, width = confidence_map.shape
        if token_hw is None:
            return confidence_map.flatten(2).clamp_min(0.0)
        if isinstance(token_hw, int):
            token_hw = (token_hw, token_hw)
        pooled = F.adaptive_avg_pool2d(
            confidence_map.reshape(batch * views, 1, height, width), tuple(token_hw)
        )
        return pooled.flatten(1).reshape(batch, views, -1).clamp_min(0.0)
    if confidence_map.ndim == 3:
        if token_hw is None:
            return confidence_map.flatten(1).clamp_min(0.0)
        if isinstance(token_hw, int):
            token_hw = (token_hw, token_hw)
        return F.adaptive_avg_pool2d(confidence_map.unsqueeze(1), tuple(token_hw)).flatten(1).clamp_min(0.0)
    if confidence_map.ndim == 2:
        return confidence_map.clamp_min(0.0)
    raise ValueError(
        "confidence_map must have shape [B, N], [B, H, W], [B, S, H, W], [B, 1, H, W], or [B, S, 1, H, W]"
    )


def build_loop_descriptor(
    camera_tokens: torch.Tensor,
    patch_tokens: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    pooling: PoolingMode = "mean",
    gem_p: float = 3.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    if camera_tokens.ndim not in (2, 3):
        raise ValueError("camera_tokens must have shape [B, C] or [B, S, C]")
    if patch_tokens.ndim not in (3, 4):
        raise ValueError("patch_tokens must have shape [B, N, C] or [B, S, N, C]")
    if camera_tokens.ndim != patch_tokens.ndim - 1:
        raise ValueError(
            "camera_tokens and patch_tokens must either both omit the view dimension "
            "([B, C] with [B, N, C]) or both include it ([B, S, C] with [B, S, N, C])"
        )

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
