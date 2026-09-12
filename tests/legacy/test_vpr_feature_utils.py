import pytest
import torch


def test_extract_aux_patch_tokens():
    from depth_anything_3.model.vpr_feature_utils import extract_aux_patch_tokens
    # Simulate aux_feats from backbone: [B, N, P, C] where N=1 (single view)
    aux_feat = torch.randn(2, 1, 1296, 768)
    aux_feats = [aux_feat]
    patch_tokens = extract_aux_patch_tokens(aux_feats)
    assert patch_tokens.shape == (2, 1296, 768), f"Expected (2, 1296, 768), got {patch_tokens.shape}"


def test_patch_tokens_to_feature_dict():
    from depth_anything_3.model.vpr_feature_utils import patch_tokens_to_feature_dict
    patch_tokens = torch.randn(2, 1296, 768)
    patch_size = 14
    image_h, image_w = 504, 504
    feat_dict = patch_tokens_to_feature_dict(patch_tokens, image_h, image_w, patch_size)
    assert feat_dict["patch_tokens"].shape == (2, 1296, 768)
    assert feat_dict["feature_map"].shape == (2, 768, 36, 36)
    assert feat_dict["global_token"].shape == (2, 768)
    assert feat_dict["spatial_shape"] == (36, 36)


def test_extract_final_layer_features():
    from depth_anything_3.model.vpr_feature_utils import extract_final_layer_features
    # Simulate feats from backbone: list of (patch_tokens, camera_tokens)
    B, N, P, C = 2, 1, 1296, 1536
    feats = [
        (torch.randn(B, N, P, C), torch.randn(B, 1, C)),  # layer 0
        (torch.randn(B, N, P, C), torch.randn(B, 1, C)),  # layer 1
        (torch.randn(B, N, P, C), torch.randn(B, 1, C)),  # layer 2
        (torch.randn(B, N, P, C), torch.randn(B, 1, C)),  # layer 3
    ]
    patch_tokens, camera_token = extract_final_layer_features(feats)
    assert patch_tokens.shape == (B, P, C), f"Expected ({B}, {P}, {C}), got {patch_tokens.shape}"
    assert camera_token.shape == (B, C), f"Expected ({B}, {C}), got {camera_token.shape}"
