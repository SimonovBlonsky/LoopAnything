from __future__ import annotations

import torch
import torch.nn as nn
from addict import Dict

from depth_anything_3.model.retrieval_strategy import BaseRetrievalStrategy
from depth_anything_3.model.vpr_feature_utils import (
    extract_aux_patch_tokens,
    patch_tokens_to_feature_dict,
)


class UnifiedPipeline(nn.Module):
    """Unified image retrieval + pose regression pipeline v1.1.

    Two-stage architecture:
        Stage 1 (retrieval): query single-view backbone → aux features → VPR descriptor
        Stage 2 (pose): [query, top-M candidates] multi-view backbone → alternate attention → cam_dec → pose

    Three calling modes:
        forward():        full pipeline (retrieval + pose)
        retrieval_only(): extract global descriptors only
        pose_only():      given pre-selected candidates, multi-view backbone → pose
    """

    PATCH_SIZE = 14

    def __init__(
        self,
        da3_backbone: nn.Module,
        feature_adapter: nn.Module,
        aggregator: nn.Module,
        retrieval_strategy: BaseRetrievalStrategy,
        da3_head: nn.Module,
        cam_dec: nn.Module,
        aux_layer: int = 5,
        pose_top_m: int = 3,
    ):
        super().__init__()
        self.da3_backbone = da3_backbone
        self.feature_adapter = feature_adapter
        self.aggregator = aggregator
        self.retrieval_strategy = retrieval_strategy
        self.da3_head = da3_head
        self.cam_dec = cam_dec
        self.aux_layer = aux_layer
        self.pose_top_m = pose_top_m

    # ----- Stage 1: Retrieval -----

    def _run_backbone_single(self, x: torch.Tensor):
        """Run backbone in single-view mode, exporting aux layer features.

        Args:
            x: [B, 1, 3, H, W] single-view input

        Returns:
            feats: list of (patch_tokens, camera_tokens) tuples
            aux_feats: list of aux feature tensors
            image_h, image_w: image dimensions
        """
        image_h, image_w = x.shape[-2], x.shape[-1]
        feats, aux_feats = self.da3_backbone(
            x, cam_token=None, export_feat_layers=[self.aux_layer],
            ref_view_strategy="saddle_balanced",
        )
        return feats, aux_feats, image_h, image_w

    def _run_vpr_branch(self, aux_feats, image_h, image_w):
        """VPR side branch: aux features -> adapter -> SALAD -> descriptor.

        Args:
            aux_feats: aux features from backbone
            image_h, image_w: image dimensions for spatial reshape

        Returns:
            descriptor: [B, D] global descriptor
        """
        patch_tokens = extract_aux_patch_tokens(aux_feats)
        feat_dict = patch_tokens_to_feature_dict(
            patch_tokens, image_h, image_w, self.PATCH_SIZE,
        )
        feat_dict = self.feature_adapter(feat_dict)
        descriptor = self.aggregator(
            (feat_dict["feature_map"], feat_dict["global_token"]),
        )
        return descriptor

    # ----- Stage 2: Multi-view Pose -----

    def _run_backbone_multiview(self, multi_view_input: torch.Tensor):
        """Run backbone in multi-view mode with alternate attention.

        Args:
            multi_view_input: [B, 1+M, 3, H, W] query (view 0) + M candidates

        Returns:
            feats: list of (patch_tokens [B, 1+M, P, C], camera_tokens [B, 1+M, C])
            image_h, image_w: image dimensions
        """
        image_h, image_w = multi_view_input.shape[-2], multi_view_input.shape[-1]
        feats, _ = self.da3_backbone(
            multi_view_input, cam_token=None, export_feat_layers=[],
            ref_view_strategy="saddle_balanced",
        )
        return feats, image_h, image_w

    def _run_pose_cam_dec(self, feats, image_h, image_w):
        """Pose via cam_dec path (differentiable, for training).

        Args:
            feats: backbone multi-view feats
            image_h, image_w: image dimensions

        Returns:
            Dict with pose_enc [B, 1+M, 9] and decoded extrinsics/intrinsics
        """
        from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri
        from depth_anything_3.utils.geometry import affine_inverse

        # cam_dec expects camera tokens [B, S, C]
        camera_tokens = feats[-1][1]  # [B, 1+M, C]
        pose_enc = self.cam_dec(camera_tokens)  # [B, 1+M, 9]

        c2w, ixt = pose_encoding_to_extri_intri(pose_enc, (image_h, image_w))

        output = Dict()
        output.pose_enc = pose_enc
        output.extrinsics = affine_inverse(c2w)
        output.intrinsics = ixt
        return output

    def _run_pose_ray(self, feats, image_h, image_w):
        """Pose via ray path (non-differentiable, for inference).

        Args:
            feats: backbone multi-view feats
            image_h, image_w: image dimensions

        Returns:
            Dict with extrinsics and intrinsics from ray map
        """
        from depth_anything_3.utils.geometry import affine_inverse
        from depth_anything_3.utils.ray_utils import get_extrinsic_from_camray

        head_out = self.da3_head(feats, image_h, image_w, patch_start_idx=0)

        output = Dict()
        if "ray" in head_out and "ray_conf" in head_out:
            pred_ext, pred_fl, pred_pp = get_extrinsic_from_camray(
                head_out.ray, head_out.ray_conf,
                head_out.ray.shape[-3], head_out.ray.shape[-2],
            )
            pred_ext = affine_inverse(pred_ext)  # w2c -> c2w
            pred_ext = pred_ext[:, :, :3, :]

            pred_ixt = torch.eye(3, 3)[None, None].repeat(
                pred_ext.shape[0], pred_ext.shape[1], 1, 1,
            ).clone().to(pred_ext.device)
            pred_ixt[:, :, 0, 0] = pred_fl[:, :, 0] / 2 * image_w
            pred_ixt[:, :, 1, 1] = pred_fl[:, :, 1] / 2 * image_h
            pred_ixt[:, :, 0, 2] = pred_pp[:, :, 0] * image_w * 0.5
            pred_ixt[:, :, 1, 2] = pred_pp[:, :, 1] * image_h * 0.5

            output.extrinsics = pred_ext
            output.intrinsics = pred_ixt

        if "depth" in head_out:
            output.depth = head_out.depth

        return output

    # ----- Public API -----

    def forward(
        self,
        query_image: torch.Tensor,
        candidate_images: torch.Tensor,
    ) -> Dict:
        """Full unified forward: retrieval + multi-view pose.

        Args:
            query_image: [B, 1, 3, H, W]
            candidate_images: [B, K, 3, H, W] all K candidates (top-M selected internally)

        Returns:
            Dict with: pose_enc, extrinsics, intrinsics, query_descriptor,
                selected_indices [B, M] (per-sample retrieval top-M)
        """
        B, K = candidate_images.shape[:2]

        # Stage 1: Retrieval
        _, aux_feats, image_h, image_w = self._run_backbone_single(query_image)
        query_descriptor = self._run_vpr_branch(aux_feats, image_h, image_w)

        # Get candidate descriptors (no grad, frozen backbone)
        with torch.no_grad():
            cand_descs = []
            for k in range(K):
                cand_input = candidate_images[:, k:k+1]
                _, cand_aux, _, _ = self._run_backbone_single(cand_input)
                cand_desc = self._run_vpr_branch(cand_aux, image_h, image_w)
                cand_descs.append(cand_desc)
            cand_descs = torch.stack(cand_descs, dim=1)  # [B, K, D]

        # Select top-M
        M = min(self.pose_top_m, K)
        sims = torch.nn.functional.cosine_similarity(
            query_descriptor.unsqueeze(1), cand_descs, dim=-1,
        )
        topm_indices = sims.topk(M, dim=1).indices  # [B, M]

        # Gather top-M candidate images
        gather_idx = topm_indices[:, :, None, None, None].expand(-1, -1, *candidate_images.shape[2:])
        selected_cands = torch.gather(candidate_images, dim=1, index=gather_idx)  # [B, M, 3, H, W]

        # Stage 2: Multi-view pose
        multi_view = torch.cat([query_image, selected_cands], dim=1)  # [B, 1+M, 3, H, W]
        feats, _, _ = self._run_backbone_multiview(multi_view)

        output = self._run_pose_cam_dec(feats, image_h, image_w)
        output.query_descriptor = query_descriptor
        output.selected_indices = topm_indices
        return output

    def retrieval_only(self, images: torch.Tensor) -> torch.Tensor:
        """Extract global descriptors only.

        Args:
            images: [B, 1, 3, H, W]

        Returns:
            descriptors: [B, D]
        """
        _, aux_feats, image_h, image_w = self._run_backbone_single(images)
        return self._run_vpr_branch(aux_feats, image_h, image_w)

    def pose_only(
        self,
        query_image: torch.Tensor,
        candidate_images: torch.Tensor,
        pose_path: str = "cam_dec",
    ) -> Dict:
        """Pose-only: given pre-selected candidates, run multi-view backbone.

        Args:
            query_image: [B, 1, 3, H, W]
            candidate_images: [B, M, 3, H, W] pre-selected candidates
            pose_path: "cam_dec", "ray", or "both"

        Returns:
            Dict with pose results
        """
        multi_view = torch.cat([query_image, candidate_images], dim=1)  # [B, 1+M, 3, H, W]
        feats, image_h, image_w = self._run_backbone_multiview(multi_view)

        output = Dict()

        if pose_path in ("cam_dec", "both"):
            cam_out = self._run_pose_cam_dec(feats, image_h, image_w)
            output.update(cam_out)

        if pose_path in ("ray", "both"):
            with torch.no_grad():
                ray_out = self._run_pose_ray(feats, image_h, image_w)
            if pose_path == "ray":
                output.update(ray_out)
            else:
                output.ray_extrinsics = ray_out.get("extrinsics")
                output.ray_intrinsics = ray_out.get("intrinsics")

        return output

    @torch.no_grad()
    def extract_database_features(self, images: torch.Tensor) -> torch.Tensor:
        """Offline: extract VPR descriptors for database images.

        Args:
            images: [B, 1, 3, H, W]

        Returns:
            descriptors: [B, D]
        """
        _, aux_feats, image_h, image_w = self._run_backbone_single(images)
        return self._run_vpr_branch(aux_feats, image_h, image_w)
