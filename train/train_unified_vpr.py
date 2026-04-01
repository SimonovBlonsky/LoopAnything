"""Train VPR (adapter + SALAD) within the unified pipeline framework.

Uses GSVCities for training and Pittsburgh for validation, same as the
standalone train_da3_adapter_salad.py but operating on the unified pipeline's
retrieval_only() path.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SALAD_ROOT = PROJECT_ROOT / "da3_streaming" / "loop_utils" / "salad"
for path in (str(SRC_ROOT), str(PROJECT_ROOT), str(SALAD_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from depth_anything_3.model.unified_pipeline_helper import build_unified_pipeline, load_config


# Lazy imports for SALAD utilities
_salad_utils = None
_GSVCitiesDataModule = None

PROXY_ENV_KEYS = (
    "ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy",
    "HTTPS_PROXY", "https_proxy",
)


def _sanitize_proxy_env():
    """Remove unsupported socks:// proxy env vars for httpx compatibility."""
    for key in PROXY_ENV_KEYS:
        val = os.environ.get(key)
        if val and val.split("://", 1)[0].lower() == "socks":
            os.environ.pop(key, None)


def _get_salad_utils():
    global _salad_utils
    if _salad_utils is None:
        from da3_streaming.loop_utils.salad import utils as su
        _salad_utils = su
    return _salad_utils


def _get_gsvcities_datamodule():
    global _GSVCitiesDataModule
    if _GSVCitiesDataModule is None:
        from da3_streaming.loop_utils.salad.dataloaders.GSVCitiesDataloader import (
            GSVCitiesDataModule,
        )
        _GSVCitiesDataModule = GSVCitiesDataModule
    return _GSVCitiesDataModule


class UnifiedVPRLightningModule(pl.LightningModule):
    """Train VPR components (adapter + SALAD) through the unified pipeline."""

    def __init__(self, config: dict):
        super().__init__()
        _sanitize_proxy_env()

        self.config = config
        self.model_config = config["model"]
        self.train_config = config["training"]

        # Build unified pipeline with freeze config
        # (backbone=frozen, vpr=trainable, fusion+head=frozen)
        self.pipeline = build_unified_pipeline(self.model_config)

        # Loss & miner
        salad_utils = _get_salad_utils()
        self.loss_fn = salad_utils.get_loss("MultiSimilarityLoss")
        self.miner = salad_utils.get_miner("MultiSimilarityMiner", 0.1)

        # Optimizer config
        opt_config = self.train_config["optimizer"]
        self.adapter_lr = opt_config.get("adapter_lr", 1e-4)
        self.aggregator_lr = opt_config.get("aggregator_lr", 6e-5)
        self.weight_decay = opt_config.get("weight_decay", 9.5e-9)
        self.optimizer_name = opt_config.get("name", "adamw")

        # Scheduler config
        self.sched_config = self.train_config.get("scheduler", {})

        # Tracking
        self.batch_acc: list[float] = []
        self.val_outputs: list[list[torch.Tensor]] = []
        self.faiss_gpu = self.train_config.get("faiss_gpu", False)

        self._validate_trainable_params()
        self._print_diagnostics()
        self.save_hyperparameters(config)

    def _validate_trainable_params(self):
        trainable = [(n, p) for n, p in self.named_parameters() if p.requires_grad]
        if not trainable:
            raise ValueError("No trainable parameters found. Check freeze config.")

        invalid = [
            n for n, _ in trainable
            if not n.startswith("pipeline.feature_adapter.")
            and not n.startswith("pipeline.aggregator.")
        ]
        if invalid:
            raise ValueError(
                f"Only adapter + aggregator should be trainable, but found: {invalid[:5]}"
            )

    def _print_diagnostics(self):
        def count(m, trainable_only=False):
            ps = m.parameters() if not trainable_only else (p for p in m.parameters() if p.requires_grad)
            return sum(p.numel() for p in ps)

        print(f"[VPR Train] Frozen backbone params: {count(self.pipeline.da3_backbone):,}")
        print(f"[VPR Train] Trainable adapter params: {count(self.pipeline.feature_adapter, True):,}")
        print(f"[VPR Train] Trainable aggregator params: {count(self.pipeline.aggregator, True):,}")

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep backbone frozen in eval mode
        self.pipeline.da3_backbone.eval()
        self.pipeline.cross_view_fusion.eval()
        self.pipeline.da3_head.eval()
        self.pipeline.cam_dec.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass through VPR path only.

        Args:
            images: [B, C, H, W] — already normalized by GSVCities transforms

        Returns:
            descriptors: [B, D]
        """
        # retrieval_only expects [B, 1, 3, H, W]
        return self.pipeline.retrieval_only(images.unsqueeze(1))

    def _loss_function(self, descriptors: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        miner_outputs = self.miner(descriptors, labels)
        loss = self.loss_fn(descriptors, labels, miner_outputs)

        num_samples = descriptors.shape[0]
        mined_indices = miner_outputs[0].detach().cpu().tolist()
        batch_acc = 1.0 - (len(set(mined_indices)) / num_samples if num_samples else 0.0)
        self.batch_acc.append(float(batch_acc))
        self.log("b_acc", sum(self.batch_acc) / len(self.batch_acc),
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    def training_step(self, batch, batch_idx):
        places, labels = batch  # places: [B, N, C, H, W], labels: [B]
        B, N, C, H, W = places.shape
        images = places.reshape(B * N, C, H, W)
        labels = labels.reshape(-1)

        descriptors = self(images)

        if not torch.isfinite(descriptors).all():
            raise ValueError(f"Non-finite descriptors in training, shape {descriptors.shape}")

        loss = self._loss_function(descriptors, labels)
        self.log("loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    def on_train_epoch_end(self):
        self.batch_acc = []

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _ = batch  # places: [B, C, H, W] for validation
        if dataloader_idx is None:
            dataloader_idx = 0

        descriptors = self(places)

        if not torch.isfinite(descriptors).all():
            raise ValueError(f"Non-finite descriptors in validation dl {dataloader_idx}")

        self.val_outputs[dataloader_idx].append(descriptors.detach().cpu())
        return descriptors.detach().cpu()

    def on_validation_epoch_start(self):
        val_datasets = getattr(self.trainer.datamodule, "val_datasets", [])
        self.val_outputs = [[] for _ in range(len(val_datasets))]

    def on_validation_epoch_end(self):
        dm = self.trainer.datamodule
        val_step_outputs = self.val_outputs

        for i, (val_set_name, val_dataset) in enumerate(zip(dm.val_set_names, dm.val_datasets)):
            feats = torch.concat(val_step_outputs[i], dim=0)

            if "pitts" in val_set_name:
                num_references = val_dataset.dbStruct.numDb
                positives = val_dataset.getPositives()
            elif "msls" in val_set_name:
                num_references = val_dataset.num_references
                positives = val_dataset.pIdx
            else:
                raise NotImplementedError(f"Unsupported validation set: {val_set_name}")

            reference_descriptors = feats[:num_references]
            query_descriptors = feats[num_references:]
            recalls = _get_salad_utils().get_validation_recalls(
                r_list=reference_descriptors,
                q_list=query_descriptors,
                k_values=[1, 5, 10, 15, 20, 50, 100],
                gt=positives,
                print_results=True,
                dataset_name=val_set_name,
                faiss_gpu=self.faiss_gpu,
            )
            self.log(f"{val_set_name}/R1", recalls[1], prog_bar=False, logger=True)
            self.log(f"{val_set_name}/R5", recalls[5], prog_bar=False, logger=True)
            self.log(f"{val_set_name}/R10", recalls[10], prog_bar=False, logger=True)

        self.val_outputs = []

    def configure_optimizers(self):
        adapter_params = [p for p in self.pipeline.feature_adapter.parameters() if p.requires_grad]
        aggregator_params = [p for p in self.pipeline.aggregator.parameters() if p.requires_grad]

        if not aggregator_params:
            raise ValueError("No trainable aggregator parameters")

        param_groups = []
        if adapter_params:
            param_groups.append({"params": adapter_params, "lr": self.adapter_lr})
        param_groups.append({"params": aggregator_params, "lr": self.aggregator_lr})

        if self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)
        elif self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(param_groups, weight_decay=self.weight_decay)
        elif self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(param_groups, weight_decay=self.weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        sched_name = self.sched_config.get("name", "linear")
        if sched_name == "linear":
            scheduler = lr_scheduler.LinearLR(
                optimizer,
                start_factor=self.sched_config.get("start_factor", 1.0),
                end_factor=self.sched_config.get("end_factor", 0.2),
                total_iters=self.sched_config.get("total_iters", 4000),
            )
        elif sched_name == "cosine":
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.sched_config.get("T_max", 50),
            )
        else:
            raise ValueError(f"Unknown scheduler: {sched_name}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Train VPR within unified pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to train_unified_vpr.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--faiss-gpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    config = load_config(args.config)
    train_config = config["training"]

    if args.faiss_gpu:
        train_config["faiss_gpu"] = True

    # Build data module
    image_size = tuple(train_config.get("image_size", [224, 224]))
    val_set_names = train_config.get("validation", {}).get("val_set_names", ["pitts30k_val", "pitts30k_test"])

    dm_cls = _get_gsvcities_datamodule()
    datamodule = dm_cls(
        batch_size=train_config.get("batch_size", 60),
        img_per_place=train_config.get("img_per_place", 4),
        min_img_per_place=train_config.get("min_img_per_place", 4),
        shuffle_all=False,
        random_sample_from_each_place=True,
        image_size=image_size,
        num_workers=train_config.get("num_workers", 10),
        show_data_stats=True,
        val_set_names=val_set_names,
    )

    model = UnifiedVPRLightningModule(config)

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        monitor="pitts30k_val/R1",
        filename="unified_vpr_({epoch:02d})_R1[{pitts30k_val/R1:.4f}]_R5[{pitts30k_val/R5:.4f}]",
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=3,
        save_last=True,
        mode="max",
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        default_root_dir="./logs/unified_vpr/",
        num_sanity_val_steps=0,
        precision="16-mixed",
        max_epochs=train_config.get("max_epochs", 4),
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_cb],
        reload_dataloaders_every_n_epochs=1,
        log_every_n_steps=20,
    )

    print(f"Starting VPR training with val sets: {val_set_names}")
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
