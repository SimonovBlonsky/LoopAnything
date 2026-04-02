from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
REPO_ROOT = PROJECT_ROOT.parents[2]
for path in (str(SRC_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from depth_anything_3.data.unified_visloc_dataset import UnifiedVislocDataset
from depth_anything_3.model.unified_pipeline_helper import build_unified_pipeline, load_config
from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri
from depth_anything_3.model.vpr_feature_utils import extract_final_layer_features


# Import metrics
RELOC3R_ROOT = REPO_ROOT / "reloc3r"
if str(RELOC3R_ROOT) not in sys.path:
    sys.path.insert(0, str(RELOC3R_ROOT))
from reloc3r.utils.metric import get_rot_err


def geodesic_rotation_loss(pred_R: torch.Tensor, gt_R: torch.Tensor) -> torch.Tensor:
    """Geodesic distance between rotation matrices.

    Args:
        pred_R: [B, 3, 3] predicted rotation
        gt_R: [B, 3, 3] ground truth rotation

    Returns:
        Scalar loss (mean geodesic distance in radians)
    """
    R_rel = torch.bmm(pred_R.transpose(1, 2), gt_R)
    trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
    # Clamp away from ±1 to avoid infinite acos gradient (critical for FP16)
    cos_angle = torch.clamp((trace - 1) / 2, -1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(cos_angle)
    return angle.mean()


def translation_loss(pred_t: torch.Tensor, gt_t: torch.Tensor) -> torch.Tensor:
    """L2 distance between predicted and GT translations.

    Args:
        pred_t: [B, 3]
        gt_t: [B, 3]

    Returns:
        Scalar loss (mean L2 distance)
    """
    return (pred_t - gt_t).norm(dim=1).mean()


class UnifiedPipelineLightningModule(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.model_config = config["model"]
        self.train_config = config["training"]

        self.pipeline = build_unified_pipeline(self.model_config)

        self.rotation_weight = self.train_config["loss"].get("rotation_weight", 1.0)
        self.translation_weight = self.train_config["loss"].get("translation_weight", 1.0)

        self.save_hyperparameters(config)

    def _extract_candidate_features(self, candidate_images):
        """Run candidates through backbone to get final features.

        Args:
            candidate_images: [B, K, 3, H, W]

        Returns:
            cand_patch_tokens: [B, K, P, C]
            cand_camera_tokens: [B, K, C]
            cand_descriptors: [B*K, D]
        """
        B, K = candidate_images.shape[:2]
        all_patches = []
        all_cams = []
        all_descs = []

        for k in range(K):
            cand_input = candidate_images[:, k:k+1]  # [B, 1, 3, H, W]
            desc, patch, cam = self.pipeline.extract_database_features(cand_input)
            all_patches.append(patch)
            all_cams.append(cam)
            all_descs.append(desc)

        cand_patches = torch.stack(all_patches, dim=1)  # [B, K, P, C]
        cand_cams = torch.stack(all_cams, dim=1)  # [B, K, C]
        cand_descs = torch.cat(all_descs, dim=0)  # [B*K, D]
        return cand_patches, cand_cams, cand_descs

    @staticmethod
    def _nan_check(tensor, name, batch_idx):
        """Check for NaN/Inf and print diagnostic if found. Returns True if bad."""
        if not torch.isfinite(tensor).all():
            nan_count = torch.isnan(tensor).sum().item()
            inf_count = torch.isinf(tensor).sum().item()
            finite_mask = torch.isfinite(tensor)
            if finite_mask.any():
                vmin = f"{tensor[finite_mask].min().item():.6g}"
                vmax = f"{tensor[finite_mask].max().item():.6g}"
            else:
                vmin, vmax = "N/A", "N/A"
            print(
                f"\n[NaN DEBUG] batch={batch_idx} | '{name}' has "
                f"{nan_count} NaN, {inf_count} Inf | "
                f"shape={list(tensor.shape)} min={vmin} max={vmax}",
                flush=True,
            )
            return True
        return False

    def training_step(self, batch, batch_idx):
        query_image = batch["query_image"].unsqueeze(1)  # [B, 1, 3, H, W]
        query_pose = batch["query_pose"]  # [B, 4, 4]
        candidate_images = batch["candidate_images"]  # [B, K, 3, H, W]

        # Extract candidate features (no grad for frozen backbone)
        with torch.no_grad():
            cand_patches, cand_cams, cand_descs = self._extract_candidate_features(candidate_images)

        self._nan_check(cand_patches, "cand_patches (backbone out)", batch_idx)
        self._nan_check(cand_cams, "cand_cams (backbone out)", batch_idx)
        self._nan_check(cand_descs, "cand_descs (backbone out)", batch_idx)

        # ---- Step-by-step forward with NaN checks at each stage ----
        K = candidate_images.shape[1]
        pipeline = self.pipeline

        # 1) Query backbone
        feats, aux_feats, image_h, image_w = pipeline._run_backbone(query_image)
        from depth_anything_3.model.vpr_feature_utils import extract_final_layer_features
        q_patch, q_cam = extract_final_layer_features(feats)
        self._nan_check(q_patch, "query_patch (backbone final)", batch_idx)
        self._nan_check(q_cam, "query_cam (backbone final)", batch_idx)

        # 2) VPR branch
        query_descriptor = pipeline._run_vpr_branch(aux_feats, image_h, image_w)
        self._nan_check(query_descriptor, "query_descriptor (VPR)", batch_idx)

        # 3) Retrieval weights
        candidate_weights = pipeline.retrieval_strategy(query_descriptor[0], cand_descs[:K])
        candidate_weights = candidate_weights.unsqueeze(0).expand(query_image.shape[0], -1)
        self._nan_check(candidate_weights, "candidate_weights (retrieval)", batch_idx)

        # 4) Cross-view fusion
        enhanced_patch, enhanced_cam = pipeline.cross_view_fusion(
            q_patch, q_cam, cand_patches, cand_cams, candidate_weights,
        )
        self._nan_check(enhanced_patch, "enhanced_patch (fusion out)", batch_idx)
        self._nan_check(enhanced_cam, "enhanced_cam (fusion out)", batch_idx)

        # 5) DA3 head
        fused_feats = []
        for i, (patch, cam) in enumerate(feats):
            if i < len(feats) - 1:
                fused_feats.append((patch, cam))
            else:
                fused_feats.append((enhanced_patch.unsqueeze(1), enhanced_cam.unsqueeze(1)))
        head_out = pipeline.da3_head(fused_feats, image_h, image_w, patch_start_idx=0)

        # 6) CamDec → pose_enc
        pose_enc = pipeline.cam_dec(enhanced_cam.unsqueeze(1))  # [B, 1, 9]
        self._nan_check(pose_enc, "pose_enc (cam_dec out)", batch_idx)

        # ---- Decode and compute loss ----
        image_size = (query_image.shape[-2], query_image.shape[-1])
        c2w, _ixt = pose_encoding_to_extri_intri(pose_enc, image_size)
        self._nan_check(c2w, "c2w (decoded pose)", batch_idx)

        pred_R = c2w[:, 0, :3, :3]  # [B, 3, 3]
        pred_t = c2w[:, 0, :3, 3]   # [B, 3]

        gt_R = query_pose[:, :3, :3]
        gt_t = query_pose[:, :3, 3]

        rot_loss = geodesic_rotation_loss(pred_R, gt_R)
        trans_loss = translation_loss(pred_t, gt_t)
        total_loss = self.rotation_weight * rot_loss + self.translation_weight * trans_loss

        if self._nan_check(rot_loss, "rot_loss", batch_idx) or \
           self._nan_check(trans_loss, "trans_loss", batch_idx):
            R_rel = torch.bmm(pred_R.transpose(1, 2), gt_R)
            trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
            cos_angle = (trace - 1) / 2
            print(f"[NaN DEBUG]   trace range: [{trace.min().item():.6g}, {trace.max().item():.6g}]", flush=True)
            print(f"[NaN DEBUG]   cos_angle (pre-clamp): [{cos_angle.min().item():.6g}, {cos_angle.max().item():.6g}]", flush=True)
            print(f"[NaN DEBUG]   pred_t range: [{pred_t.min().item():.6g}, {pred_t.max().item():.6g}]", flush=True)
            print(f"[NaN DEBUG]   candidate_weights: {candidate_weights[0].tolist()}", flush=True)

        self.log("train/rot_loss", rot_loss, prog_bar=True)
        self.log("train/trans_loss", trans_loss, prog_bar=True)
        self.log("train/total_loss", total_loss, prog_bar=True)

        # Log fusion gate values to track how fast fusion contribution grows
        fusion = pipeline.cross_view_fusion
        self.log("train/patch_gate", fusion.patch_gate.item(), prog_bar=False)
        self.log("train/cam_gate", fusion.cam_gate.item(), prog_bar=False)

        return total_loss

    def validation_step(self, batch, batch_idx):
        query_image = batch["query_image"].unsqueeze(1)
        query_pose = batch["query_pose"]
        candidate_images = batch["candidate_images"]

        with torch.no_grad():
            cand_patches, cand_cams, cand_descs = self._extract_candidate_features(candidate_images)

        K = candidate_images.shape[1]
        self.pipeline.retrieval_strategy.eval()
        output = self.pipeline(query_image, cand_patches, cand_cams, cand_descs[:K])

        # Decode pose_enc to c2w
        pose_enc = output.pose_enc  # [B, 1, 9]
        image_size = (query_image.shape[-2], query_image.shape[-1])
        c2w, _ixt = pose_encoding_to_extri_intri(pose_enc, image_size)

        pred_R = c2w[:, 0, :3, :3].cpu().numpy()
        pred_t = c2w[:, 0, :3, 3].cpu().numpy()
        gt_R = query_pose[:, :3, :3].cpu().numpy()
        gt_t = query_pose[:, :3, 3].cpu().numpy()

        for i in range(pred_R.shape[0]):
            rerr = get_rot_err(pred_R[i], gt_R[i])
            terr = np.linalg.norm(pred_t[i] - gt_t[i])
            self.log("val/rot_err_deg", rerr, on_step=False, on_epoch=True)
            self.log("val/trans_err_m", terr, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        opt_config = self.train_config["optimizer"]
        params = [p for p in self.pipeline.parameters() if p.requires_grad]
        if not params:
            raise ValueError("No trainable parameters found")

        if opt_config["name"] == "adamw":
            optimizer = torch.optim.AdamW(params, lr=opt_config["lr"], weight_decay=opt_config.get("weight_decay", 1e-4))
        elif opt_config["name"] == "adam":
            optimizer = torch.optim.Adam(params, lr=opt_config["lr"], weight_decay=opt_config.get("weight_decay", 1e-4))
        elif opt_config["name"] == "sgd":
            optimizer = torch.optim.SGD(params, lr=opt_config["lr"], weight_decay=opt_config.get("weight_decay", 1e-4), momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {opt_config['name']}")

        sched_config = self.train_config.get("scheduler", {})
        if sched_config.get("name") == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=sched_config.get("T_max", 50))
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

        return optimizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train unified visual localization pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", type=int, default=1)
    return parser.parse_args()


def _build_datasets(train_config):
    """Load scene entries and build UnifiedVislocDataset instances."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from eval_unified_visloc import load_scene_images_and_poses

    dataset_name = train_config["dataset"]
    sampling = train_config.get("candidate_sampling", {})
    image_size = tuple(train_config.get("image_size", [504, 504]))
    data_root = train_config.get("data_root", None)
    num_candidates = train_config.get("eval", {}).get("top_k", 10)

    def make_dataset(scene, split):
        entries = load_scene_images_and_poses(dataset_name, scene, split, data_root=data_root)
        return UnifiedVislocDataset(
            entries=entries,
            num_candidates=num_candidates,
            pos_threshold=sampling.get("pos_threshold", 1.0),
            neg_threshold=sampling.get("neg_threshold", 3.0),
            pos_ratio=sampling.get("pos_ratio", 0.7),
            distance_alpha=sampling.get("distance_alpha", 0.5),
            image_size=image_size,
        )

    return make_dataset


def build_scene_dataloaders(train_config):
    """Build train and (optionally) val dataloaders."""
    make_dataset = _build_datasets(train_config)
    scenes = train_config["scenes"]
    batch_size = train_config["batch_size"]
    num_workers = train_config.get("num_workers", 8)

    # Train: all scenes, train split
    train_datasets = [make_dataset(scene, "train") for scene in scenes]
    combined_train = torch.utils.data.ConcatDataset(train_datasets)
    train_loader = DataLoader(
        combined_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Val: pick scene(s) from config, test split
    eval_config = train_config.get("eval", {})
    val_scenes = eval_config.get("val_scenes", None)
    if val_scenes is None:
        # Default: use the first scene's test split
        val_scenes = [scenes[0]]
    elif isinstance(val_scenes, str):
        val_scenes = [val_scenes]

    val_datasets = [make_dataset(scene, "test") for scene in val_scenes]
    combined_val = torch.utils.data.ConcatDataset(val_datasets)
    val_loader = DataLoader(
        combined_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Train: {len(combined_train)} samples from {len(scenes)} scenes")
    print(f"Val:   {len(combined_val)} samples from {val_scenes}")

    return train_loader, val_loader


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    config = load_config(args.config)
    train_config = config["training"]

    model = UnifiedPipelineLightningModule(config)
    train_loader, val_loader = build_scene_dataloaders(train_config)

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        monitor="val/trans_err_m",
        filename="unified_{epoch:02d}_terr{val/trans_err_m:.4f}",
        auto_insert_metric_name=False,
        save_top_k=3,
        save_last=True,
        mode="min",
    )

    eval_config = train_config.get("eval", {})
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        max_epochs=train_config["max_epochs"],
        check_val_every_n_epoch=eval_config.get("eval_every_n_epoch", 1),
        callbacks=[checkpoint_cb],
        precision="16-mixed",
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        default_root_dir="./logs/unified_pipeline/",
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume_from,
    )


if __name__ == "__main__":
    main()
