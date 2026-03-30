from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SALAD_ROOT = PROJECT_ROOT / "da3_streaming" / "loop_utils" / "salad"
for path in (PROJECT_ROOT, SRC_ROOT):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from depth_anything_3.model.vpr_helper import build_vpr_model


DEFAULT_AGGREGATOR_CKPT_PATH = SALAD_ROOT / "weights" / "dino_salad_512_32.ckpt"
DEFAULT_AGGREGATOR_CONFIG = {
    "num_channels": 768,
    "num_clusters": 16,
    "cluster_dim": 32,
    "token_dim": 32,
}
DEFAULT_VAL_SET_NAMES = ["pitts30k_val", "pitts30k_test"]
PROTOCOL_SEEDS = (0, 1, 2)
PINNED_ADAPTER_LR = 1e-4
PINNED_AGGREGATOR_LR = 6e-5
PINNED_PATCH_ONLY_BOTTLENECK = 640
PINNED_BATCH_SIZE = 60
PINNED_IMG_PER_PLACE = 4
PINNED_MIN_IMG_PER_PLACE = 4
PINNED_IMAGE_SIZE = (224, 224)
PINNED_NUM_WORKERS = 10
PINNED_PRECISION = "16-mixed"
PINNED_MAX_EPOCHS = 4
PINNED_CHECK_VAL_EVERY_N_EPOCH = 1
PINNED_NUM_SANITY_VAL_STEPS = 0
PINNED_SAVE_TOP_K = 3
PINNED_SAVE_LAST = True
METRICS_TO_WATCH = (
    "pitts30k_val/R1",
    "pitts30k_val/R5",
    "pitts30k_val/R10",
    "pitts30k_test/R1",
    "pitts30k_test/R5",
    "pitts30k_test/R10",
)
PROXY_ENV_KEYS = (
    "ALL_PROXY",
    "all_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
)
_PROXY_ENV_SANITIZED = False
GSVCitiesDataModule = None
salad_utils = None


def startup_log(message: str) -> None:
    print(f"[startup] {message}", flush=True)


def sanitize_unsupported_proxy_env_vars() -> None:
    global _PROXY_ENV_SANITIZED
    if _PROXY_ENV_SANITIZED:
        return

    removed_proxy_vars: list[str] = []
    for env_key in PROXY_ENV_KEYS:
        env_value = os.environ.get(env_key)
        if not env_value:
            continue
        scheme = env_value.split("://", 1)[0].lower()
        if scheme == "socks":
            removed_proxy_vars.append(env_key)
            os.environ.pop(env_key, None)

    if removed_proxy_vars:
        formatted = ", ".join(removed_proxy_vars)
        startup_log(
            "Sanitized unsupported proxy env vars with socks:// scheme for httpx compatibility: "
            f"{formatted}"
        )
        startup_log("Left all remaining proxy environment variables unchanged.")

    _PROXY_ENV_SANITIZED = True


def _normalize_aggregator_ckpt_path(path: str | None) -> str | None:
    if path is None:
        return None
    normalized = str(path).strip()
    if not normalized or normalized.lower() == "none":
        return None
    return normalized


def _validate_local_aggregator_ckpt_path(path: str | None) -> str | None:
    if path is None:
        return None
    checkpoint_path = Path(path)
    if checkpoint_path.is_file():
        return str(checkpoint_path)
    raise FileNotFoundError(
        f"Missing local SALAD warm-start checkpoint at {checkpoint_path}. "
        "Provide a local --aggregator-ckpt-path or pass --aggregator-ckpt-path none to disable warm start."
    )


def _count_parameters(module: nn.Module, *, trainable_only: bool = False) -> int:
    parameters = module.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


def _get_gsvcities_datamodule():
    global GSVCitiesDataModule
    if GSVCitiesDataModule is None:
        try:
            from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule as _GSVCitiesDataModule
        except Exception as exc:
            raise RuntimeError(
                f"Failed to import GSVCitiesDataModule from the active worktree SALAD path at {SALAD_ROOT}. "
                "Ensure the worktree-local SALAD checkout is available."
            ) from exc

        GSVCitiesDataModule = _GSVCitiesDataModule
    return GSVCitiesDataModule


def _get_salad_utils():
    global salad_utils
    if salad_utils is None:
        try:
            from da3_streaming.loop_utils.salad import utils as _salad_utils
        except Exception as exc:
            raise RuntimeError(
                f"Failed to import SALAD utilities from the active worktree path at {SALAD_ROOT}. "
                "Ensure the worktree-local SALAD checkout is available."
            ) from exc
        salad_utils = _salad_utils
    return salad_utils


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a dual-branch DA3 feature adapter with a SALAD aggregator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seed",
        type=int,
        choices=list(PROTOCOL_SEEDS),
        default=0,
        help="Pinned three-seed evaluation protocol for adapter training.",
    )
    parser.add_argument(
        "--da3-model-name-or-path",
        type=str,
        default="depth-anything/DA3-BASE",
        help="Depth Anything 3 checkpoint identifier passed to build_vpr_model().",
    )
    parser.add_argument(
        "--feature-source",
        type=str,
        default="aux",
        choices=["final", "aux"],
        help="Encoder feature source used by the DA3 encoder adapter.",
    )
    parser.add_argument(
        "--aux-layer",
        type=int,
        default=5,
        help="Auxiliary DA3 layer exported for adapter training.",
    )
    parser.add_argument(
        "--aux-global-token-mode",
        type=str,
        default="cls_token",
        choices=["mean_patch", "cls_token"],
        help="How to derive the global token from the aux feature path.",
    )
    parser.add_argument(
        "--feature-adapter-arch",
        type=str,
        default="dual_branch",
        choices=["identity", "patch_only", "dual_branch"],
        help="Feature adapter architecture to train on top of frozen DA3 features.",
    )
    parser.add_argument(
        "--adapter-local-bottleneck",
        type=int,
        default=256,
        help="Local-branch bottleneck dimension for the dual-branch adapter.",
    )
    parser.add_argument(
        "--adapter-global-hidden-dim",
        type=int,
        default=256,
        help="Global-branch hidden dimension for the dual-branch adapter.",
    )
    parser.add_argument(
        "--agg-arch",
        type=str,
        default="salad",
        choices=["salad"],
        help="VPR aggregator architecture.",
    )
    parser.add_argument(
        "--aggregator-ckpt-path",
        type=str,
        default=str(DEFAULT_AGGREGATOR_CKPT_PATH),
        help="SALAD warm-start checkpoint. Use 'none' or an empty string to disable warm start.",
    )
    parser.add_argument(
        "--optimizer-name",
        type=str,
        default="adamw",
        choices=["sgd", "adamw", "adam"],
        help="Optimizer for the trainable adapter and aggregator parameter groups.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=9.5e-9,
        help="Weight decay aligned with the SALAD baseline trainer.",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum used when --optimizer-name sgd is selected.",
    )
    parser.add_argument(
        "--lr-sched",
        type=str,
        default="linear",
        choices=["linear"],
        help="Learning rate scheduler aligned with the SALAD baseline trainer.",
    )
    parser.add_argument(
        "--lr-sched-start-factor",
        type=float,
        default=1.0,
        help="LinearLR start factor.",
    )
    parser.add_argument(
        "--lr-sched-end-factor",
        type=float,
        default=0.2,
        help="LinearLR end factor.",
    )
    parser.add_argument(
        "--lr-sched-total-iters",
        type=int,
        default=4000,
        help="LinearLR total iters.",
    )
    parser.add_argument(
        "--faiss-gpu",
        action="store_true",
        help="Use GPU FAISS when computing validation recalls.",
    )
    return parser.parse_args(argv)


def _build_feature_adapter_config(args) -> dict[str, Any] | None:
    if args.feature_adapter_arch == "dual_branch":
        return {
            "local_bottleneck": args.adapter_local_bottleneck,
            "global_hidden_dim": args.adapter_global_hidden_dim,
        }
    if args.feature_adapter_arch == "patch_only":
        return {
            "bottleneck": PINNED_PATCH_ONLY_BOTTLENECK,
        }
    if args.feature_adapter_arch == "identity":
        return None
    raise ValueError(f"Unsupported feature adapter architecture: {args.feature_adapter_arch}")


class DA3AdapterSALADLightningModule(pl.LightningModule):
    def __init__(self, args=None):
        super().__init__()
        if args is None:
            args = parse_args([])

        sanitize_unsupported_proxy_env_vars()

        self.args = args
        self.adapter_lr = PINNED_ADAPTER_LR
        self.lr = PINNED_AGGREGATOR_LR
        self.optimizer_name = args.optimizer_name
        self.weight_decay = args.weight_decay
        self.momentum = args.momentum
        self.lr_sched = args.lr_sched
        self.lr_sched_args = {
            "start_factor": args.lr_sched_start_factor,
            "end_factor": args.lr_sched_end_factor,
            "total_iters": args.lr_sched_total_iters,
        }
        self.faiss_gpu = args.faiss_gpu
        self.da3_model_name_or_path = args.da3_model_name_or_path
        self.feature_source = args.feature_source
        self.aux_layer = args.aux_layer
        self.aux_global_token_mode = args.aux_global_token_mode
        self.feature_adapter_arch = args.feature_adapter_arch
        self.adapter_local_bottleneck = args.adapter_local_bottleneck
        self.adapter_global_hidden_dim = args.adapter_global_hidden_dim
        self.agg_arch = args.agg_arch.lower()
        self.agg_config = dict(DEFAULT_AGGREGATOR_CONFIG)
        self.aggregator_ckpt_path = _validate_local_aggregator_ckpt_path(
            _normalize_aggregator_ckpt_path(args.aggregator_ckpt_path)
        )

        self.vpr_model = build_vpr_model(
            da3_model_name_or_path=self.da3_model_name_or_path,
            feature_source=self.feature_source,
            aux_layer=self.aux_layer,
            aux_global_token_mode=self.aux_global_token_mode,
            feature_adapter_arch=self.feature_adapter_arch,
            feature_adapter_config=_build_feature_adapter_config(args),
            agg_arch="salad",
            agg_config=self.agg_config,
            aggregator_ckpt_path=self.aggregator_ckpt_path,
        )
        self.encoder_arch = self.vpr_model.encoder.__class__.__name__

        self._freeze_encoder()
        current_salad_utils = _get_salad_utils()
        self.loss_fn = current_salad_utils.get_loss("MultiSimilarityLoss")
        self.miner = current_salad_utils.get_miner("MultiSimilarityMiner", 0.1)
        self.batch_acc: list[float] = []
        self.val_outputs: list[list[torch.Tensor]] = []
        self._validate_startup_state()
        self._print_startup_diagnostics()

        self.save_hyperparameters(
            {
                "seed": args.seed,
                "da3_model_name_or_path": self.da3_model_name_or_path,
                "feature_source": self.feature_source,
                "aux_layer": self.aux_layer,
                "aux_global_token_mode": self.aux_global_token_mode,
                "feature_adapter_arch": self.feature_adapter_arch,
                "adapter_local_bottleneck": self.adapter_local_bottleneck,
                "adapter_global_hidden_dim": self.adapter_global_hidden_dim,
                "agg_arch": self.agg_arch,
                "agg_config": self.agg_config,
                "aggregator_ckpt_path": self.aggregator_ckpt_path,
                "adapter_lr": self.adapter_lr,
                "lr": self.lr,
                "optimizer_name": self.optimizer_name,
                "weight_decay": self.weight_decay,
                "momentum": self.momentum,
                "lr_sched": self.lr_sched,
                "lr_sched_args": self.lr_sched_args,
                "faiss_gpu": self.faiss_gpu,
            }
        )

    def _freeze_encoder(self) -> None:
        self.vpr_model.encoder.requires_grad_(False)
        self.vpr_model.encoder.eval()

    def _named_trainable_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [
            (name, parameter)
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]

    def _validate_startup_state(self) -> None:
        trainable_parameters = self._named_trainable_parameters()
        if not trainable_parameters:
            raise ValueError(
                "Empty trainable-parameter list: expected feature adapter and SALAD aggregator parameters."
            )

        invalid_trainable = [
            name
            for name, _ in trainable_parameters
            if not name.startswith("vpr_model.feature_adapter.")
            and not name.startswith("vpr_model.aggregator.")
        ]
        if invalid_trainable:
            preview = ", ".join(invalid_trainable[:5])
            raise ValueError(
                "Only the feature adapter and aggregator should be trainable; "
                f"found encoder or other trainable parameters: {preview}"
            )

    def _print_startup_diagnostics(self) -> None:
        startup_log(f"DA3 source: {self.da3_model_name_or_path}")
        startup_log(f"Feature source: {self.feature_source}")
        startup_log(f"Aux layer: {self.aux_layer}")
        startup_log(f"Aux token mode: {self.aux_global_token_mode}")
        startup_log(f"Feature adapter arch: {self.feature_adapter_arch}")
        startup_log(f"Adapter local bottleneck: {self.adapter_local_bottleneck}")
        startup_log(f"Adapter global hidden dim: {self.adapter_global_hidden_dim}")
        startup_log(
            f"Frozen encoder param count: {_count_parameters(self.vpr_model.encoder, trainable_only=False):,}"
        )
        startup_log(
            f"Trainable adapter param count: {_count_parameters(self.vpr_model.feature_adapter, trainable_only=True):,}"
        )
        startup_log(
            f"Trainable aggregator param count: {_count_parameters(self.vpr_model.aggregator, trainable_only=True):,}"
        )
        startup_log(
            "SALAD warm start: "
            f"{self.aggregator_ckpt_path if self.aggregator_ckpt_path is not None else 'disabled'}"
        )

    @staticmethod
    def _ensure_finite_descriptors(descriptors: torch.Tensor, stage: str) -> None:
        if not torch.isfinite(descriptors).all():
            raise ValueError(
                f"Non-finite descriptors detected during {stage} with shape {tuple(descriptors.shape)}"
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

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
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
        self._ensure_finite_descriptors(descriptors, "training")

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
        self._ensure_finite_descriptors(descriptors, f"validation dataloader {dataloader_idx}")
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

        print("\n\n")
        self.val_outputs = []

    def configure_optimizers(self) -> Any:
        adapter_parameters = [
            parameter
            for parameter in self.vpr_model.feature_adapter.parameters()
            if parameter.requires_grad
        ]
        aggregator_parameters = [
            parameter
            for parameter in self.vpr_model.aggregator.parameters()
            if parameter.requires_grad
        ]
        if not aggregator_parameters:
            raise ValueError("No trainable aggregator parameters found for optimization")

        optimizer_params = []
        if adapter_parameters:
            optimizer_params.append({"params": adapter_parameters, "lr": self.adapter_lr})
        optimizer_params.append({"params": aggregator_parameters, "lr": self.lr})

        optimizer_name = self.optimizer_name.lower()
        if optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                optimizer_params,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
            )
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                optimizer_params,
                weight_decay=self.weight_decay,
            )
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                optimizer_params,
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


def build_datamodule(args):
    datamodule_cls = _get_gsvcities_datamodule()
    return datamodule_cls(
        batch_size=PINNED_BATCH_SIZE,
        img_per_place=PINNED_IMG_PER_PLACE,
        min_img_per_place=PINNED_MIN_IMG_PER_PLACE,
        shuffle_all=False,
        random_sample_from_each_place=True,
        image_size=PINNED_IMAGE_SIZE,
        num_workers=PINNED_NUM_WORKERS,
        show_data_stats=True,
        val_set_names=list(DEFAULT_VAL_SET_NAMES),
    )


def build_trainer(args, model: DA3AdapterSALADLightningModule) -> pl.Trainer:
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        monitor="pitts30k_val/R1",
        filename=(
            f"{model.encoder_arch}" + "_({epoch:02d})_R1[{pitts30k_val/R1:.4f}]"
            "_R5[{pitts30k_val/R5:.4f}]"
        ),
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=PINNED_SAVE_TOP_K,
        save_last=PINNED_SAVE_LAST,
        mode="max",
    )

    return pl.Trainer(
        accelerator="gpu",
        devices=1,
        default_root_dir="./logs/",
        num_nodes=1,
        num_sanity_val_steps=PINNED_NUM_SANITY_VAL_STEPS,
        precision=PINNED_PRECISION,
        max_epochs=PINNED_MAX_EPOCHS,
        check_val_every_n_epoch=PINNED_CHECK_VAL_EVERY_N_EPOCH,
        callbacks=[checkpoint_cb],
        reload_dataloaders_every_n_epochs=1,
        log_every_n_steps=20,
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    pl.seed_everything(args.seed, workers=True)

    startup_log(f"Validation metrics to watch: {', '.join(METRICS_TO_WATCH)}")

    datamodule = build_datamodule(args)
    startup_log(
        "Datamodule ready: "
        f"{datamodule.__class__.__name__} with validation sets {datamodule.val_set_names}"
    )

    model = DA3AdapterSALADLightningModule(args=args)
    trainer = build_trainer(args, model)
    startup_log(
        "Trainer ready: "
        f"accelerator={trainer.accelerator.__class__.__name__}, "
        f"devices={trainer.num_devices}, precision={trainer.precision}"
    )
    startup_log("Starting trainer.fit(...)")
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
