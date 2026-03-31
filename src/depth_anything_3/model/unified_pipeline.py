from __future__ import annotations

import torch
import torch.nn as nn
from addict import Dict

from depth_anything_3.model.cross_view_fusion import CrossViewFusion
from depth_anything_3.model.retrieval_strategy import BaseRetrievalStrategy
from depth_anything_3.model.vpr_feature_utils import (
    extract_aux_patch_tokens,
    extract_final_layer_features,
    patch_tokens_to_feature_dict,
)


class UnifiedPipeline(nn.Module):
    """Unified image retrieval + pose regression pipeline.

    Single backbone pass: intermediate layer → VPR side branch (adapter + SALAD),
    final layer → cross-view fusion with retrieved candidates → Head → pose.

    Three calling modes:
        forward():        full unified feed-forward
        retrieval_only(): extract global descriptors only
        pose_only():      given pre-selected candidates, run pose regression
    """

    PATCH_SIZE = 14

    def __init__(
        self,
        da3_backbone: nn.Module,
        feature_adapter: nn.Module,
        aggregator: nn.Module,
        retrieval_strategy: BaseRetrievalStrategy,
        cross_view_fusion: CrossViewFusion,
        da3_head: nn.Module,
        cam_dec: nn.Module,
        aux_layer: int = 5,
    ):
        super().__init__()
        self.da3_backbone = da3_backbone
        self.feature_adapter = feature_adapter
        self.aggregator = aggregator
        self.retrieval_strategy = retrieval_strategy
        self.cross_view_fusion = cross_view_fusion
        self.da3_head = da3_head
        self.cam_dec = cam_dec
        self.aux_layer = aux_layer

    def _run_backbone(self, x: torch.Tensor):
        """Run backbone once, exporting aux layer features.

        Args:
            x: [B, 1, 3, H, W] single-view input

        Returns:
            feats: list of (patch_tokens, camera_tokens) tuples
            aux_feats: list of aux feature tensors
            image_h, image_w: image dimensions
        """
        image_h, image_w = x.shape[-2], x.shape[-1]
        feats, aux_feats = self.da3_backbone(
            x, cam_token=None, export_feat_layers=[self.aux_layer], ref_view_strategy="saddle_balanced"
        )
        return feats, aux_feats, image_h, image_w

    def _run_vpr_branch(self, aux_feats, image_h, image_w):
        """VPR side branch: aux features → adapter → SALAD → descriptor.

        Args:
            aux_feats: aux features from backbone
            image_h, image_w: image dimensions for spatial reshape

        Returns:
            descriptor: [B, D] global descriptor
        """
        patch_tokens = extract_aux_patch_tokens(aux_feats)
        feat_dict = patch_tokens_to_feature_dict(patch_tokens, image_h, image_w, self.PATCH_SIZE)
        feat_dict = self.feature_adapter(feat_dict)
        descriptor = self.aggregator((feat_dict["feature_map"], feat_dict["global_token"]))
        return descriptor

    def _run_pose_branch(self, feats, candidate_patch_tokens, candidate_camera_tokens,
                         candidate_weights, image_h, image_w):
        """Pose branch: fusion + head + cam_dec.

        Args:
            feats: backbone feats (list of tuples)
            candidate_patch_tokens: [B, K, P, C] candidate final layer patch tokens
            candidate_camera_tokens: [B, K, C] candidate final layer camera tokens
            candidate_weights: [B, K] retrieval weights
            image_h, image_w: image dimensions

        Returns:
            output dict with pose_enc and head outputs
        """
        query_patch, query_cam = extract_final_layer_features(feats)

        enhanced_patch, enhanced_cam = self.cross_view_fusion(
            query_patch, query_cam,
            candidate_patch_tokens, candidate_camera_tokens,
            candidate_weights,
        )

        # Rebuild feats for DualDPT: replace final layer with fused features
        fused_feats = []
        for i, (patch, cam) in enumerate(feats):
            if i < len(feats) - 1:
                fused_feats.append((patch, cam))
            else:
                # Replace final layer: restore N=1 dim
                fused_patch = enhanced_patch.unsqueeze(1)  # [B, 1, P, C]
                fused_cam = enhanced_cam.unsqueeze(1)  # [B, 1, C]
                fused_feats.append((fused_patch, fused_cam))

        output = Dict()
        head_out = self.da3_head(fused_feats, image_h, image_w, patch_start_idx=0)
        output.update(head_out)

        pose_enc = self.cam_dec(enhanced_cam.unsqueeze(1))  # [B, 1, 9]
        output.pose_enc = pose_enc

        return output

    def forward(
        self,
        query: torch.Tensor,
        candidate_patch_tokens: torch.Tensor,
        candidate_camera_tokens: torch.Tensor,
        candidate_descriptors: torch.Tensor,
    ) -> Dict:
        """Full unified forward pass.

        Args:
            query: [B, 1, 3, H, W] query image
            candidate_patch_tokens: [B, K, P, C] pre-extracted candidate final features
            candidate_camera_tokens: [B, K, C] pre-extracted candidate camera tokens
            candidate_descriptors: [K, D] pre-extracted candidate global descriptors

        Returns:
            Dict with keys: pose_enc, query_descriptor, candidate_weights, and head outputs
        """
        feats, aux_feats, image_h, image_w = self._run_backbone(query)

        query_descriptor = self._run_vpr_branch(aux_feats, image_h, image_w)

        # Retrieval: get candidate weights
        candidate_weights = self.retrieval_strategy(query_descriptor[0], candidate_descriptors)
        candidate_weights = candidate_weights.unsqueeze(0).expand(query.shape[0], -1)  # [B, K]

        output = self._run_pose_branch(
            feats, candidate_patch_tokens, candidate_camera_tokens,
            candidate_weights, image_h, image_w,
        )
        output.query_descriptor = query_descriptor
        output.candidate_weights = candidate_weights
        return output

    def retrieval_only(self, images: torch.Tensor) -> torch.Tensor:
        """Extract global descriptors only (for ablation experiments).

        Args:
            images: [B, 1, 3, H, W] input images

        Returns:
            descriptors: [B, D] global descriptors
        """
        _, aux_feats, image_h, image_w = self._run_backbone(images)
        return self._run_vpr_branch(aux_feats, image_h, image_w)

    def pose_only(
        self,
        query_image: torch.Tensor,
        candidate_images: torch.Tensor,
    ) -> Dict:
        """Pose-only mode: given pre-selected candidates, run pose regression.

        Args:
            query_image: [B, 1, 3, H, W] query image
            candidate_images: [B, K, 3, H, W] pre-selected candidate images

        Returns:
            Dict with keys: pose_enc and head outputs
        """
        B, K = candidate_images.shape[:2]

        query_feats, _, image_h, image_w = self._run_backbone(query_image)

        cand_patch_list = []
        cand_cam_list = []
        for k in range(K):
            cand_input = candidate_images[:, k:k+1]  # [B, 1, 3, H, W]
            cand_feats, _, _, _ = self._run_backbone(cand_input)
            cand_patch, cand_cam = extract_final_layer_features(cand_feats)
            cand_patch_list.append(cand_patch)
            cand_cam_list.append(cand_cam)

        candidate_patch_tokens = torch.stack(cand_patch_list, dim=1)  # [B, K, P, C]
        candidate_camera_tokens = torch.stack(cand_cam_list, dim=1)  # [B, K, C]

        candidate_weights = torch.ones(B, K, device=query_image.device) / K

        return self._run_pose_branch(
            query_feats, candidate_patch_tokens, candidate_camera_tokens,
            candidate_weights, image_h, image_w,
        )

    @torch.no_grad()
    def extract_database_features(self, images: torch.Tensor):
        """Offline: extract descriptors + final features for database images.

        Args:
            images: [B, 1, 3, H, W] database images (process in batches)

        Returns:
            descriptors: [B, D] global descriptors
            patch_tokens: [B, P, C] final layer patch tokens
            camera_tokens: [B, C] final layer camera tokens
        """
        feats, aux_feats, image_h, image_w = self._run_backbone(images)
        descriptors = self._run_vpr_branch(aux_feats, image_h, image_w)
        patch_tokens, camera_tokens = extract_final_layer_features(feats)
        return descriptors, patch_tokens, camera_tokens
