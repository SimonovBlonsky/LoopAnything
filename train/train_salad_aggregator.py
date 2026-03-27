from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SALAD_ROOT = PROJECT_ROOT / "da3_streaming" / "loop_utils" / "salad"
for path in (PROJECT_ROOT, SRC_ROOT, SALAD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from da3_streaming.loop_utils.salad import utils as salad_utils
from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule
from depth_anything_3.api import DepthAnything3
from depth_anything_3.model.VPRaggregators import SALAD
from depth_anything_3.model.vpr_helper import load_aggregator_weights_from_salad_ckpt
from depth_anything_3.model.vpr_model import VPRModel


class DA3Layer5CamTokenEncoder(nn.Module):
    PATCH_SIZE = 14
    AUX_LAYER = 5
    AUX_DIM = 768

    def __init__(self) -> None:
        super().__init__()
        self.da3 = DepthAnything3.from_pretrained("depth-anything/DA3-BASE")
        for parameter in self.da3.parameters():
            parameter.requires_grad = False
        self.da3.eval()

    def train(self, mode: bool = True) -> DA3Layer5CamTokenEncoder:
        super().train(mode)
        self.da3.eval()
        return self

    def forward(self, images: torch.Tensor) -> dict[str, Any]:
        if images.ndim != 4:
            raise ValueError(f"Expected images with shape [B, 3, H, W], got {tuple(images.shape)}")
        if images.shape[1] != 3:
            raise ValueError(f"Expected 3-channel images, got {images.shape[1]}")

        batch_size, _, height, width = images.shape
        if height % self.PATCH_SIZE != 0 or width % self.PATCH_SIZE != 0:
            raise ValueError(
                f"Input spatial size must be divisible by {self.PATCH_SIZE}, got {(height, width)}"
            )

        da3_device = next(self.da3.parameters()).device
        if da3_device != images.device:
            raise RuntimeError(
                f"DA3 device mismatch: encoder is on {da3_device}, inputs are on {images.device}. "
                "Move the parent module onto the input device before calling forward."
            )

        hp = height // self.PATCH_SIZE
        wp = width // self.PATCH_SIZE

        transformer = self.da3.model.backbone.pretrained
        views = images.unsqueeze(1)
        with torch.inference_mode():
            _, aux_outputs = transformer._get_intermediate_layers_not_chunked(
                views,
                n=[],
                export_feat_layers=[self.AUX_LAYER],
            )

        assert len(aux_outputs) == 1, (
            f"Expected one aux output for layer {self.AUX_LAYER}, got {len(aux_outputs)}"
        )
        tokens = transformer.norm(aux_outputs[0])[:, 0]
        assert tokens.ndim == 3, (
            f"Expected normalized tokens with shape [B, T, C], got {tuple(tokens.shape)}"
        )
        assert tokens.shape[0] == batch_size, f"Expected batch size {batch_size}, got {tokens.shape[0]}"
        assert tokens.shape[-1] == self.AUX_DIM, (
            f"Expected aux token dim {self.AUX_DIM}, got {tokens.shape[-1]}"
        )

        global_token = tokens[:, 0]
        patch_tokens = tokens[:, 1:]
        assert patch_tokens.shape[1] == hp * wp, (
            f"Expected {hp * wp} patch tokens from spatial shape {(hp, wp)}, got {patch_tokens.shape[1]}"
        )

        feature_map = patch_tokens.transpose(1, 2).reshape(batch_size, self.AUX_DIM, hp, wp)
        assert feature_map.shape == (batch_size, self.AUX_DIM, hp, wp), (
            f"Unexpected feature map shape {tuple(feature_map.shape)}"
        )
        assert global_token.shape == (batch_size, self.AUX_DIM), (
            f"Unexpected global token shape {tuple(global_token.shape)}"
        )

        if not torch.isfinite(tokens).all():
            raise ValueError("Non-finite normalized DA3 aux tokens")
        if not torch.isfinite(global_token).all():
            raise ValueError("Non-finite DA3 global token")
        if not torch.isfinite(patch_tokens).all():
            raise ValueError("Non-finite DA3 patch tokens")
        if not torch.isfinite(feature_map).all():
            raise ValueError("Non-finite DA3 feature map")

        return {
            "patch_tokens": patch_tokens,
            "feature_map": feature_map,
            "global_token": global_token,
            "spatial_shape": (hp, wp),
        }


class DA3SALADLightningModule(pl.LightningModule):
    AGGREGATOR_CONFIG = {
        "num_channels": 768,
        "num_clusters": 16,
        "cluster_dim": 32,
        "token_dim": 32,
    }
    AGGREGATOR_WEIGHTS = (
        PROJECT_ROOT / "da3_streaming" / "loop_utils" / "salad" / "weights" / "dino_salad_512_32.ckpt"
    )

    def __init__(
        self,
        lr: float = 6e-5,
        optimizer_name: str = "adamw",
        weight_decay: float = 9.5e-9,
        momentum: float = 0.9,
        lr_sched: str = "linear",
        lr_sched_args: dict[str, Any] | None = None,
        faiss_gpu: bool = False,
    ) -> None:
        super().__init__()
        if lr_sched_args is None:
            lr_sched_args = {
                "start_factor": 1.0,
                "end_factor": 0.2,
                "total_iters": 4000,
            }

        self.lr = lr
        self.optimizer_name = optimizer_name
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lr_sched = lr_sched
        self.lr_sched_args = lr_sched_args
        self.faiss_gpu = faiss_gpu
        self.encoder_arch = DA3Layer5CamTokenEncoder.__name__
        self.agg_arch = "SALAD"
        self.agg_config = dict(self.AGGREGATOR_CONFIG)

        encoder = DA3Layer5CamTokenEncoder()
        aggregator = SALAD(**self.AGGREGATOR_CONFIG)
        self._load_aggregator_weights(aggregator)

        self.vpr_model = VPRModel(encoder=encoder, aggregator=aggregator, agg_arch="SALAD")
        self.loss_fn = salad_utils.get_loss("MultiSimilarityLoss")
        self.miner = salad_utils.get_miner("MultiSimilarityMiner", 0.1)
        self.batch_acc: list[float] = []
        self.val_outputs: list[list[torch.Tensor]] = []

        self.save_hyperparameters(
            {
                "lr": lr,
                "optimizer_name": optimizer_name,
                "weight_decay": weight_decay,
                "momentum": momentum,
                "lr_sched": lr_sched,
                "lr_sched_args": lr_sched_args,
                "faiss_gpu": faiss_gpu,
                "encoder_arch": self.encoder_arch,
                "agg_arch": self.agg_arch,
                "agg_config": self.agg_config,
                "aggregator_config": self.agg_config,
            }
        )

    def _load_aggregator_weights(self, aggregator: SALAD) -> None:
        load_aggregator_weights_from_salad_ckpt(
            aggregator,
            self.AGGREGATOR_WEIGHTS,
            strict=True,
        )

    def _loss_function(self, descriptors: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.miner is not None:
            miner_outputs = self.miner(descriptors, labels)
            loss = self.loss_fn(descriptors, labels, miner_outputs)
            num_samples = descriptors.shape[0]
            mined_indices = miner_outputs[0].detach().cpu().tolist()
            batch_acc = 1.0 - (len(set(mined_indices)) / num_samples if num_samples else 0.0)
        else:
            loss = self.loss_fn(descriptors, labels)
            batch_acc = 0.0
            if isinstance(loss, tuple):
                loss, batch_acc = loss

        self.batch_acc.append(float(batch_acc))
        self.log(
            "b_acc",
            sum(self.batch_acc) / len(self.batch_acc),
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )
        return loss

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.vpr_model(x, **kwargs)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        places, labels = batch
        if places.ndim != 5:
            raise ValueError(
                f"Expected training batch places with shape [B, N, C, H, W], got {tuple(places.shape)}"
            )

        batch_size, num_views, channels, height, width = places.shape
        images = places.reshape(batch_size * num_views, channels, height, width)
        labels = labels.reshape(-1)

        descriptors = self(images)
        if torch.isnan(descriptors).any():
            raise ValueError("NaNs in descriptors")

        loss = self._loss_function(descriptors, labels)
        self.log("loss", loss, logger=True, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def on_train_epoch_end(self) -> None:
        self.batch_acc = []

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int | None = None,
    ) -> torch.Tensor:
        del batch_idx
        places, _ = batch
        if dataloader_idx is None:
            dataloader_idx = 0

        descriptors = self(places)
        self.val_outputs[dataloader_idx].append(descriptors.detach().cpu())
        return descriptors.detach().cpu()

    def on_validation_epoch_start(self) -> None:
        val_datasets = getattr(self.trainer.datamodule, "val_datasets", [])
        self.val_outputs = [[] for _ in range(len(val_datasets))]

    def on_validation_epoch_end(self) -> None:
        dm = self.trainer.datamodule
        val_step_outputs = self.val_outputs
        if len(dm.val_datasets) == 1:
            val_step_outputs = [val_step_outputs[0]]

        for i, (val_set_name, val_dataset) in enumerate(zip(dm.val_set_names, dm.val_datasets)):
            feats = torch.concat(val_step_outputs[i], dim=0)

            if "pitts" in val_set_name:
                num_references = val_dataset.dbStruct.numDb
                positives = val_dataset.getPositives()
            elif "msls" in val_set_name:
                num_references = val_dataset.num_references
                positives = val_dataset.pIdx
            else:
                raise NotImplementedError(
                    f"Validation splitting is only implemented for Pitts/MSLS datasets, got {val_set_name}"
                )

            reference_descriptors = feats[:num_references]
            query_descriptors = feats[num_references:]
            recalls = salad_utils.get_validation_recalls(
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

        print("\n\n")
        self.val_outputs = []

    def configure_optimizers(self) -> Any:
        trainable_parameters = [
            parameter for parameter in self.vpr_model.aggregator.parameters() if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise ValueError("No trainable aggregator parameters found for optimization")

        optimizer_name = self.optimizer_name.lower()
        if optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                trainable_parameters,
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
            )
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                trainable_parameters,
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                trainable_parameters,
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

        scheduler_name = self.lr_sched.lower()
        if scheduler_name == "multistep":
            scheduler = lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.lr_sched_args["milestones"],
                gamma=self.lr_sched_args["gamma"],
            )
        elif scheduler_name == "cosine":
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.lr_sched_args["T_max"])
        elif scheduler_name == "linear":
            scheduler = lr_scheduler.LinearLR(
                optimizer,
                start_factor=self.lr_sched_args["start_factor"],
                end_factor=self.lr_sched_args["end_factor"],
                total_iters=self.lr_sched_args["total_iters"],
            )
        else:
            raise ValueError(f"Unsupported lr scheduler: {self.lr_sched}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        optimizer_closure: Any,
    ) -> None:
        del epoch, batch_idx
        optimizer.step(closure=optimizer_closure)


def build_datamodule() -> GSVCitiesDataModule:
    return GSVCitiesDataModule(
        batch_size=60,
        img_per_place=4,
        min_img_per_place=4,
        shuffle_all=False,
        random_sample_from_each_place=True,
        image_size=(224, 224),
        num_workers=10,
        show_data_stats=True,
        val_set_names=["pitts30k_val", "pitts30k_test"],
    )


def build_trainer(model: pl.LightningModule) -> pl.Trainer:
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        monitor="pitts30k_val/R1",
        filename=(
            f"{model.encoder_arch}" + "_({epoch:02d})_R1[{pitts30k_val/R1:.4f}]"
            "_R5[{pitts30k_val/R5:.4f}]"
        ),
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=3,
        save_last=True,
        mode="max",
    )

    return pl.Trainer(
        accelerator="gpu",
        devices=1,
        default_root_dir="./logs/",
        num_nodes=1,
        num_sanity_val_steps=0,
        precision="16-mixed",
        max_epochs=4,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_cb],
        reload_dataloaders_every_n_epochs=1,
        log_every_n_steps=20,
    )


def main() -> None:
    raise SystemExit(
        "Task 1 scaffold only: train/train_salad_aggregator.py is import-safe, but execution is not implemented yet."
    )


if __name__ == "__main__":
    main()
