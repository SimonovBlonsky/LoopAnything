from __future__ import annotations

import torch


def extract_aux_patch_tokens(aux_feats: list[torch.Tensor]) -> torch.Tensor:
    """Extract patch tokens from backbone auxiliary features.

    Args:
        aux_feats: list of tensors from backbone export_feat_layers,
                   each [B, N, P, C] where N=1 for single-view.

    Returns:
        patch_tokens: [B, P, C] (squeezed from N=1 dimension)
    """
    if len(aux_feats) != 1:
        raise ValueError(f"Expected exactly 1 aux feature layer, got {len(aux_feats)}")
    aux_feat = aux_feats[0]
    if aux_feat.ndim != 4 or aux_feat.shape[1] != 1:
        raise ValueError(f"Expected aux feature shape [B, 1, P, C], got {aux_feat.shape}")
    return aux_feat[:, 0]  # [B, P, C]


def patch_tokens_to_feature_dict(
    patch_tokens: torch.Tensor,
    image_h: int,
    image_w: int,
    patch_size: int = 14,
) -> dict[str, torch.Tensor | tuple[int, int]]:
    """Convert patch tokens to the feature dict expected by vpr_feature_adapter and SALAD.

    Args:
        patch_tokens: [B, P, C] where P = (H/patch_size) * (W/patch_size)
        image_h: input image height in pixels
        image_w: input image width in pixels
        patch_size: ViT patch size (default 14)

    Returns:
        dict with keys: patch_tokens, feature_map, global_token, spatial_shape
    """
    hp = image_h // patch_size
    wp = image_w // patch_size
    B, P, C = patch_tokens.shape
    if P != hp * wp:
        raise ValueError(f"Patch count mismatch: {P} != {hp}*{wp}")
    feature_map = patch_tokens.reshape(B, hp, wp, C).permute(0, 3, 1, 2)  # [B, C, hp, wp]
    global_token = patch_tokens.mean(dim=1)  # [B, C]
    return {
        "patch_tokens": patch_tokens,
        "feature_map": feature_map,
        "global_token": global_token,
        "spatial_shape": (hp, wp),
    }


def extract_final_layer_features(
    feats: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract patch tokens and camera token from the backbone final layer output.

    Args:
        feats: list of (patch_tokens [B, N, P, C], camera_tokens [B, 1, C]) from backbone.
               Uses the last element (final layer).

    Returns:
        patch_tokens: [B, P, C] (squeezed from N=1)
        camera_token: [B, C] (squeezed from dim=1)
    """
    final_patch, final_cam = feats[-1]
    if final_patch.ndim == 4 and final_patch.shape[1] == 1:
        final_patch = final_patch[:, 0]  # [B, P, C]
    elif final_patch.ndim != 3:
        raise ValueError(f"Unexpected final patch tokens shape: {final_patch.shape}")
    if final_cam.ndim == 3 and final_cam.shape[1] == 1:
        final_cam = final_cam[:, 0]  # [B, C]
    elif final_cam.ndim != 2:
        raise ValueError(f"Unexpected final camera tokens shape: {final_cam.shape}")
    return final_patch, final_cam
