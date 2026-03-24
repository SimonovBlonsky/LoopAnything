from __future__ import annotations

import torch
import torch.nn as nn


class DA3EncoderAdapter(nn.Module):
    def __init__(self, da3_model, feat_layer=-1, ref_view_strategy="saddle_balanced", patch_size=14):
        super().__init__()
        self.da3_model = da3_model
        self.feat_layer = feat_layer
        self.ref_view_strategy = ref_view_strategy
        self.patch_size = patch_size

    def _unwrap_da3_net(self):
        return self.da3_model.model if hasattr(self.da3_model, "model") else self.da3_model

    def _normalize_input(self, x):
        if x.ndim == 4:
            return x[:, None]
        if x.ndim == 5 and x.shape[1] == 1:
            return x
        raise ValueError("DA3EncoderAdapter only supports single-view retrieval inputs")

    def forward(self, x):
        x = self._normalize_input(x)
        da3_net = self._unwrap_da3_net()
        feats, _ = da3_net.backbone(
            x,
            cam_token=None,
            export_feat_layers=[],
            ref_view_strategy=self.ref_view_strategy,
        )
        patch_tokens, camera_tokens = feats[self.feat_layer]
        patch_tokens = patch_tokens[:, 0]
        global_token = camera_tokens[:, 0]
        hp, wp = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        if patch_tokens.shape[1] != hp * wp:
            raise ValueError("Cannot reshape patch tokens into a valid spatial map")
        feature_map = patch_tokens.view(x.shape[0], hp, wp, patch_tokens.shape[-1]).permute(0, 3, 1, 2)
        if not torch.isfinite(feature_map).all():
            raise ValueError("feature_map contains non-finite values")
        if not torch.isfinite(global_token).all():
            raise ValueError("global_token contains non-finite values")
        return {
            "patch_tokens": patch_tokens,
            "feature_map": feature_map,
            "global_token": global_token,
            "spatial_shape": (hp, wp),
        }
