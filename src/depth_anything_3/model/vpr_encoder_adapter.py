from __future__ import annotations

import torch
import torch.nn as nn


class DA3EncoderAdapter(nn.Module):
    def __init__(
        self,
        da3_model,
        feat_layer=-1,
        ref_view_strategy="saddle_balanced",
        patch_size=14,
        feature_source="final",
        aux_layer=3,
    ):
        super().__init__()
        self.da3_model = da3_model
        self.feat_layer = feat_layer
        self.ref_view_strategy = ref_view_strategy
        self.patch_size = patch_size
        self.feature_source = feature_source
        self.aux_layer = aux_layer

    def _unwrap_da3_net(self):
        return self.da3_model.model if hasattr(self.da3_model, "model") else self.da3_model

    def _normalize_input(self, x):
        if x.ndim == 4:
            return x[:, None]
        if x.ndim == 5 and x.shape[1] == 1:
            return x
        raise ValueError("DA3EncoderAdapter only supports single-view retrieval inputs")

    def _patch_tokens_to_feature_map(self, patch_tokens, x):
        hp, wp = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        if patch_tokens.shape[1] != hp * wp:
            raise ValueError("Cannot reshape patch tokens into a valid spatial map")
        feature_map = patch_tokens.reshape(x.shape[0], hp, wp, patch_tokens.shape[-1]).permute(0, 3, 1, 2)
        return feature_map, (hp, wp)

    def _extract_final_features(self, feats):
        selected_feat = feats[self.feat_layer]
        if not isinstance(selected_feat, (tuple, list)) or len(selected_feat) != 2:
            raise ValueError("DA3EncoderAdapter expected (patch_tokens, camera_tokens) from backbone")
        patch_tokens, camera_tokens = selected_feat
        patch_tokens = patch_tokens[:, 0]
        global_token = camera_tokens[:, 0]
        return patch_tokens, global_token

    def _extract_aux_features(self, aux_feats):
        if self.aux_layer is None or self.aux_layer < 0:
            raise ValueError("aux_layer must be a non-negative integer when feature_source='aux'")
        if len(aux_feats) == 0:
            raise ValueError("No auxiliary features were returned by the backbone")
        if len(aux_feats) != 1:
            raise ValueError("Expected exactly one auxiliary feature map for aux mode")

        patch_tokens = aux_feats[0]
        if patch_tokens.ndim != 4 or patch_tokens.shape[1] != 1:
            raise ValueError("Expected auxiliary patch tokens with shape [B, 1, N, C]")
        patch_tokens = patch_tokens[:, 0]
        global_token = patch_tokens.mean(dim=1)
        return patch_tokens, global_token

    def forward(self, x):
        x = self._normalize_input(x)
        da3_net = self._unwrap_da3_net()

        feature_source = str(self.feature_source).lower()
        if feature_source == "final":
            export_feat_layers = []
        elif feature_source == "aux":
            if self.aux_layer is None or self.aux_layer < 0:
                raise ValueError("aux_layer must be a non-negative integer when feature_source='aux'")
            export_feat_layers = [self.aux_layer]
        else:
            raise ValueError(f"Unsupported feature_source: {self.feature_source}")

        feats, aux_feats = da3_net.backbone(
            x,
            cam_token=None,
            export_feat_layers=export_feat_layers,
            ref_view_strategy=self.ref_view_strategy,
        )

        if feature_source == "final":
            patch_tokens, global_token = self._extract_final_features(feats)
        else:
            patch_tokens, global_token = self._extract_aux_features(aux_feats)

        feature_map, spatial_shape = self._patch_tokens_to_feature_map(patch_tokens, x)
        if not torch.isfinite(feature_map).all():
            raise ValueError("feature_map contains non-finite values")
        if not torch.isfinite(global_token).all():
            raise ValueError("global_token contains non-finite values")
        return {
            "patch_tokens": patch_tokens,
            "feature_map": feature_map,
            "global_token": global_token,
            "spatial_shape": spatial_shape,
        }
