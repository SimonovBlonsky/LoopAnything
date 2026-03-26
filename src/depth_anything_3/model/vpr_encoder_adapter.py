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
        aux_layers=None,
        layer_combine="single",
        layer_weights=None,
        layer_scale=1.0,
        post_fusion_norm="none",
    ):
        super().__init__()
        self.da3_model = da3_model
        self.feat_layer = feat_layer
        self.ref_view_strategy = ref_view_strategy
        self.patch_size = patch_size
        self.feature_source = feature_source
        self.aux_layer = aux_layer
        self.aux_layers = aux_layers
        self.layer_combine = layer_combine
        self.layer_weights = layer_weights
        self.layer_scale = layer_scale
        self.post_fusion_norm = post_fusion_norm

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

    def _normalized_layer_combine(self):
        return str(self.layer_combine).lower()

    def _normalized_post_fusion_norm(self):
        return str(self.post_fusion_norm).lower()

    def _has_explicit_aux_fusion_args(self):
        return (
            self.aux_layers is not None
            or self._normalized_layer_combine() != "single"
            or self.layer_weights is not None
            or self.layer_scale != 1.0
            or self._normalized_post_fusion_norm() != "none"
        )

    def _normalize_aux_layers(self):
        raw_layers = self.aux_layers if self.aux_layers is not None else [self.aux_layer]
        if isinstance(raw_layers, int):
            raw_layers = [raw_layers]
        else:
            raw_layers = list(raw_layers)
        if len(raw_layers) == 0:
            raise ValueError("aux_layers must contain at least one layer index")

        aux_layers = []
        for layer in raw_layers:
            if not isinstance(layer, int):
                raise ValueError("aux_layers must contain integer indices")
            if layer < 0:
                raise ValueError("aux_layers must not contain negative indices")
            aux_layers.append(layer)
        return aux_layers

    def _normalize_aux_feature(self, aux_feat):
        if not torch.is_tensor(aux_feat) or aux_feat.ndim != 4 or aux_feat.shape[1] != 1:
            raise ValueError("Expected auxiliary patch tokens with shape [B, 1, N, C]")
        return aux_feat[:, 0]

    def _validate_aux_feature_source(self, feature_source):
        if feature_source == "final":
            if self._has_explicit_aux_fusion_args():
                raise ValueError("feature_source='final' does not accept explicit AUX fusion args")
            return []
        if feature_source == "aux":
            return self._normalize_aux_layers()
        raise ValueError(f"Unsupported feature_source: {self.feature_source}")

    def _extract_final_features(self, feats):
        selected_feat = feats[self.feat_layer]
        if not isinstance(selected_feat, (tuple, list)) or len(selected_feat) != 2:
            raise ValueError("DA3EncoderAdapter expected (patch_tokens, camera_tokens) from backbone")
        patch_tokens, camera_tokens = selected_feat
        if not torch.is_tensor(patch_tokens) or patch_tokens.ndim != 4 or patch_tokens.shape[1] != 1:
            raise ValueError("DA3EncoderAdapter expected final patch_tokens with shape [B, 1, N, C]")
        if not torch.is_tensor(camera_tokens) or camera_tokens.ndim != 3 or camera_tokens.shape[1] != 1:
            raise ValueError("DA3EncoderAdapter expected final camera_tokens with shape [B, 1, C]")
        if patch_tokens.shape[0] != camera_tokens.shape[0]:
            raise ValueError("DA3EncoderAdapter expected matching batch dimensions in final features")
        if patch_tokens.shape[-1] != camera_tokens.shape[-1]:
            raise ValueError("DA3EncoderAdapter expected matching final feature dimensions")
        patch_tokens = patch_tokens[:, 0]
        global_token = camera_tokens[:, 0]
        return patch_tokens, global_token

    def _apply_post_fusion_norm(self, patch_tokens, x):
        post_fusion_norm = self._normalized_post_fusion_norm()
        if post_fusion_norm == "none":
            return patch_tokens
        if post_fusion_norm == "token_l2":
            return torch.nn.functional.normalize(patch_tokens, p=2, dim=-1)
        if post_fusion_norm == "feature_layernorm":
            feature_map, _ = self._patch_tokens_to_feature_map(patch_tokens, x)
            feature_map = torch.nn.functional.layer_norm(
                feature_map.permute(0, 2, 3, 1),
                normalized_shape=(feature_map.shape[1],),
            ).permute(0, 3, 1, 2).contiguous()
            return feature_map.permute(0, 2, 3, 1).reshape(x.shape[0], -1, feature_map.shape[1])
        raise ValueError(f"Unsupported post_fusion_norm: {self.post_fusion_norm}")

    def _extract_aux_features(self, aux_feats, requested_aux_layers):
        if len(requested_aux_layers) == 0:
            raise ValueError("aux_layers must contain at least one layer index")
        if aux_feats is None or len(aux_feats) == 0:
            raise ValueError("No auxiliary features were returned by the backbone")
        if len(aux_feats) != len(requested_aux_layers):
            raise ValueError("Malformed auxiliary features returned by the backbone")

        patch_tokens_list = [self._normalize_aux_feature(aux_feat) for aux_feat in aux_feats]
        first_shape = patch_tokens_list[0].shape
        for patch_tokens in patch_tokens_list[1:]:
            if patch_tokens.shape != first_shape:
                raise ValueError("Malformed/incompatible AUX features returned by the backbone")
        combine = self._normalized_layer_combine()

        if combine == "single":
            if len(patch_tokens_list) != 1:
                raise ValueError("single layer_combine requires exactly one aux layer")
            patch_tokens = patch_tokens_list[0]
        elif combine == "avg":
            if len(patch_tokens_list) == 1:
                raise ValueError("avg layer_combine requires multiple aux layers")
            patch_tokens = torch.stack(patch_tokens_list, dim=0).mean(dim=0)
        elif combine == "sum":
            if len(patch_tokens_list) == 1:
                raise ValueError("sum layer_combine requires multiple aux layers")
            patch_tokens = torch.stack(patch_tokens_list, dim=0).sum(dim=0)
        elif combine == "weighted_avg":
            if len(patch_tokens_list) == 1:
                raise ValueError("weighted_avg requires multiple aux layers")
            if self.layer_weights is None:
                raise ValueError("weighted_avg requires layer_weights to match aux_layers")
            try:
                weights = list(self.layer_weights)
            except TypeError as exc:
                raise ValueError("layer_weights must be a sequence") from exc
            if len(weights) != len(patch_tokens_list):
                raise ValueError("weighted_avg requires layer_weights to match aux_layers")
            weights = torch.as_tensor(weights, dtype=patch_tokens_list[0].dtype, device=patch_tokens_list[0].device)
            if not torch.isfinite(weights).all():
                raise ValueError("weighted_avg requires finite layer_weights")
            total_weight = weights.sum()
            if total_weight == 0:
                raise ValueError("weighted_avg requires a non-zero total weight")
            patch_tokens = (
                torch.stack(patch_tokens_list, dim=0) * weights.view(-1, 1, 1, 1)
            ).sum(dim=0) / total_weight
        else:
            raise ValueError(f"Unsupported layer_combine: {self.layer_combine}")

        return patch_tokens

    def forward(self, x):
        x = self._normalize_input(x)
        da3_net = self._unwrap_da3_net()

        feature_source = str(self.feature_source).lower()
        export_feat_layers = self._validate_aux_feature_source(feature_source)

        feats, aux_feats = da3_net.backbone(
            x,
            cam_token=None,
            export_feat_layers=export_feat_layers,
            ref_view_strategy=self.ref_view_strategy,
        )

        if feature_source == "final":
            patch_tokens, global_token = self._extract_final_features(feats)
        else:
            patch_tokens = self._extract_aux_features(aux_feats, export_feat_layers)
            patch_tokens = self._apply_post_fusion_norm(patch_tokens, x)
            patch_tokens = patch_tokens * self.layer_scale
            global_token = patch_tokens.mean(dim=1)

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
