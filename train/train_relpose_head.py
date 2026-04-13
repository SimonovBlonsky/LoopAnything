"""Train a pairwise relative pose head on DA3's frozen backbone.

The DA3 backbone is frozen; only the lightweight RelPoseHead (~18M params) is
trained. Loss is Reloc3r's angular translation + geodesic rotation, applied
symmetrically (both A->B and B->A directions).

Usage (local smoke test with dummy data):
    PYTHONPATH=src python train/train_relpose_head.py \
        --config configs/train_relpose_head.yaml --dataset-mode dummy --max-epochs 2

Usage (server, 4x RTX3090):
    torchrun --nproc_per_node=4 train/train_relpose_head.py \
        --config configs/train_relpose_head.yaml
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
REPO_ROOT = PROJECT_ROOT.parents[2]
RELOC3R_ROOT = REPO_ROOT / "reloc3r"
for p in (str(SRC_ROOT), str(REPO_ROOT), str(RELOC3R_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from depth_anything_3.model.rel_pose_head import RelPoseHead
from depth_anything_3.model.unified_pipeline_helper import build_unified_pipeline, load_config


# ---------------------------------------------------------------------------
# Loss (copied from reloc3r/loss.py:154-197 for self-containedness)
# ---------------------------------------------------------------------------
class RelPoseLoss(nn.Module):
    """Angular translation + geodesic rotation loss (same as Reloc3r)."""

    def forward(self, pose_pred: torch.Tensor, pose_gt: torch.Tensor):
        t = pose_pred[:, :3, 3]
        tgt = pose_gt[:, :3, 3]
        R = pose_pred[:, :3, :3]
        Rgt = pose_gt[:, :3, :3]
        trans_loss = self._transl_ang_loss(t, tgt)
        rot_loss = self._rot_ang_loss(R, Rgt)
        return trans_loss + rot_loss, trans_loss, rot_loss

    @staticmethod
    def _transl_ang_loss(t, tgt, eps=1e-6):
        t_normed = t / (t.norm(dim=1, keepdim=True) + eps)
        tgt_normed = tgt / (tgt.norm(dim=1, keepdim=True) + eps)
        cosine = (t_normed * tgt_normed).sum(dim=1)
        return torch.acos(cosine.clamp(-1.0 + eps, 1.0 - eps)).mean()

    @staticmethod
    def _rot_ang_loss(R, Rgt, eps=1e-6):
        residual = R.transpose(1, 2) @ Rgt
        trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
        cosine = (trace - 1) / 2
        return torch.acos(cosine.clamp(-1.0 + eps, 1.0 - eps)).mean()


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------
class RelPoseHeadModule(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.train_config = config["training"]

        # Build DA3 backbone (frozen, no grad).
        model_cfg = load_config(config["model"]["unified_config"])
        self.pipeline = build_unified_pipeline(model_cfg, device="cpu")
        for p in self.pipeline.parameters():
            p.requires_grad = False

        # Trainable RelPoseHead.
        token_dim = config["model"].get("token_dim", 2048)
        self.rel_pose_head = RelPoseHead(token_dim=token_dim)

        self.criterion = RelPoseLoss()
        self.save_hyperparameters(config)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor):
        """Extract camera tokens and predict relative pose.

        Args:
            img1: [B, 3, H, W] ImageNet-normalized.
            img2: [B, 3, H, W] ImageNet-normalized.

        Returns:
            pred_2to1: [B, 4, 4] relative pose from view2 to view1.
            pred_1to2: [B, 4, 4] relative pose from view1 to view2.
        """
        multi_view = torch.stack([img1, img2], dim=1)  # [B, 2, 3, H, W]

        with torch.no_grad():
            feats, _h, _w = self.pipeline._run_backbone_multiview(multi_view)
            cam_tokens = feats[-1][1]  # [B, 2, C]

        cam1, cam2 = cam_tokens[:, 0], cam_tokens[:, 1]

        pred_2to1 = self.rel_pose_head(cam2, cam1)
        pred_1to2 = self.rel_pose_head(cam1, cam2)
        return pred_2to1, pred_1to2

    def training_step(self, batch, batch_idx):
        pred_2to1, pred_1to2 = self(batch["img1"], batch["img2"])
        gt_2to1 = batch["rel_pose_2to1"]
        gt_1to2 = batch["rel_pose_1to2"]

        with torch.amp.autocast("cuda", enabled=False):
            loss_fwd, t_err_fwd, r_err_fwd = self.criterion(
                pred_2to1.float(), gt_2to1.float()
            )
            loss_bwd, t_err_bwd, r_err_bwd = self.criterion(
                pred_1to2.float(), gt_1to2.float()
            )

        loss = loss_fwd + loss_bwd

        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.log("train/t_err_deg", (t_err_fwd + t_err_bwd).item() / 2 * 180 / math.pi, sync_dist=True)
        self.log("train/r_err_deg", (r_err_fwd + r_err_bwd).item() / 2 * 180 / math.pi, sync_dist=True)
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred_2to1, pred_1to2 = self(batch["img1"], batch["img2"])
        gt_2to1 = batch["rel_pose_2to1"]
        gt_1to2 = batch["rel_pose_1to2"]

        with torch.amp.autocast("cuda", enabled=False):
            loss_fwd, t_fwd, r_fwd = self.criterion(pred_2to1.float(), gt_2to1.float())
            loss_bwd, t_bwd, r_bwd = self.criterion(pred_1to2.float(), gt_1to2.float())

        self.log("val/loss", loss_fwd + loss_bwd, prog_bar=True, sync_dist=True)
        self.log("val/t_err_deg", (t_fwd + t_bwd).item() / 2 * 180 / math.pi, sync_dist=True)
        self.log("val/r_err_deg", (r_fwd + r_bwd).item() / 2 * 180 / math.pi, sync_dist=True)

    def configure_optimizers(self):
        opt_cfg = self.train_config["optimizer"]
        params = list(self.rel_pose_head.parameters())
        optimizer = torch.optim.AdamW(
            params,
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg.get("weight_decay", 0.05),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.95])),
        )

        sched_cfg = self.train_config.get("scheduler", {})
        if sched_cfg.get("name") == "cosine":
            warmup_epochs = sched_cfg.get("warmup_epochs", 5)
            max_epochs = self.train_config["max_epochs"]
            min_lr = opt_cfg.get("min_lr", 1e-6)
            base_lr = opt_cfg["lr"]

            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return epoch / max(warmup_epochs, 1)
                progress = (epoch - warmup_epochs) / max(max_epochs - warmup_epochs, 1)
                cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_lr / base_lr + (1.0 - min_lr / base_lr) * cosine_factor

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

        return optimizer


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_dataloaders(config: dict, dataset_mode: str | None = None):
    train_cfg = config["training"]
    mode = dataset_mode or train_cfg.get("dataset_mode", "dummy")
    batch_size = train_cfg["batch_size"]
    num_workers = train_cfg.get("num_workers", 4)

    if mode == "dummy":
        from depth_anything_3.data.reloc3r_adapter import DummyStereoDataset
        train_ds = DummyStereoDataset(length=256)
        val_ds = DummyStereoDataset(length=32)
    elif mode == "reloc3r":
        from depth_anything_3.data.reloc3r_adapter import build_training_dataset, build_test_dataset
        train_ds = build_training_dataset(epoch=0, use_augmentation=True)
        val_ds = build_test_dataset()
    else:
        raise ValueError(f"Unknown dataset_mode: {mode}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train RelPoseHead on DA3 backbone")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--dataset-mode", type=str, default=None,
                        help="Override config dataset_mode (dummy / reloc3r)")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Override config max_epochs")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed)

    config = load_config(args.config)
    if args.max_epochs is not None:
        config["training"]["max_epochs"] = args.max_epochs

    train_loader, val_loader = build_dataloaders(config, dataset_mode=args.dataset_mode)
    module = RelPoseHeadModule(config)

    train_cfg = config["training"]
    output_dir = train_cfg.get("output_dir", "checkpoints/relpose_head")

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=output_dir,
            filename="last",
            save_last=True,
            every_n_epochs=1,
        ),
        pl.callbacks.ModelCheckpoint(
            dirpath=output_dir,
            filename="epoch-{epoch:03d}",
            every_n_epochs=train_cfg.get("save_every_n_epochs", 10),
            save_top_k=-1,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs=train_cfg["max_epochs"],
        accelerator="auto",
        devices=args.devices,
        strategy="ddp_find_unused_parameters_true" if args.devices > 1 else "auto",
        precision=train_cfg.get("precision", "16-mixed"),
        gradient_clip_val=train_cfg.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=train_cfg.get("gradient_accum", 1),
        check_val_every_n_epoch=train_cfg.get("val_check_interval", 5),
        callbacks=callbacks,
        default_root_dir=output_dir,
        log_every_n_steps=10,
    )

    trainer.fit(
        module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume_from,
    )


if __name__ == "__main__":
    main()
