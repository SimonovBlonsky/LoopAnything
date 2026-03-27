from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
SALAD_ROOT = PROJECT_ROOT / "da3_streaming" / "loop_utils" / "salad"
for path in (PROJECT_ROOT, SRC_ROOT, SALAD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from depth_anything_3.api import DepthAnything3
from depth_anything_3.model.VPRaggregators import SALAD
from depth_anything_3.model.vpr_model import VPRModel


VAL_DATASETS = ["SPED", "pitts30k_test", "pitts250k_test"]
AGGREGATOR_STATE_PREFIXES = (
    "vpr_model.aggregator.",
    "aggregator.",
    "model.aggregator.",
    "module.aggregator.",
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
PittsburghDataset = None
SPEDDataset = None


class DA3BaseLayer5CamTokenEncoder(nn.Module):
    MODEL_NAME_OR_PATH = "depth-anything/DA3-BASE"
    PATCH_SIZE = 14
    AUX_LAYER = 5
    AUX_DIM = 768

    def __init__(self) -> None:
        super().__init__()
        sanitize_unsupported_proxy_env_vars()
        da3_model = DepthAnything3.from_pretrained(self.MODEL_NAME_OR_PATH)
        if da3_model is None:
            raise RuntimeError(f"DA3 model load returned None for '{self.MODEL_NAME_OR_PATH}'")
        self.da3 = da3_model
        for parameter in self.da3.parameters():
            parameter.requires_grad = False
        self.da3.eval()

    def train(self, mode: bool = True) -> DA3BaseLayer5CamTokenEncoder:
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
        with torch.no_grad():
            _, aux_outputs = transformer._get_intermediate_layers_not_chunked(
                views,
                n=[],
                export_feat_layers=[self.AUX_LAYER],
            )

        if len(aux_outputs) != 1:
            raise RuntimeError(
                f"Expected exactly one aux output for layer {self.AUX_LAYER}, got {len(aux_outputs)}"
            )

        tokens = transformer.norm(aux_outputs[0])[:, 0]
        if tokens.ndim != 3:
            raise RuntimeError(f"Expected normalized tokens [B, T, C], got {tuple(tokens.shape)}")
        if tokens.shape[0] != batch_size:
            raise RuntimeError(f"Expected batch size {batch_size}, got {tokens.shape[0]}")
        if tokens.shape[-1] != self.AUX_DIM:
            raise RuntimeError(f"Expected token dim {self.AUX_DIM}, got {tokens.shape[-1]}")

        global_token = tokens[:, 0]
        patch_tokens = tokens[:, 1:]
        if patch_tokens.shape[1] != hp * wp:
            raise RuntimeError(
                f"Expected {hp * wp} patch tokens from spatial shape {(hp, wp)}, "
                f"got {patch_tokens.shape[1]}"
            )

        feature_map = patch_tokens.transpose(1, 2).reshape(batch_size, self.AUX_DIM, hp, wp)

        if not torch.isfinite(global_token).all():
            raise ValueError("Non-finite cam token in DA3 layer-5 encoder")
        if not torch.isfinite(patch_tokens).all():
            raise ValueError("Non-finite patch tokens in DA3 layer-5 encoder")
        if not torch.isfinite(feature_map).all():
            raise ValueError("Non-finite feature map in DA3 layer-5 encoder")

        return {
            "patch_tokens": patch_tokens,
            "feature_map": feature_map,
            "global_token": global_token,
            "spatial_shape": (hp, wp),
        }


def sanitize_unsupported_proxy_env_vars() -> None:
    global _PROXY_ENV_SANITIZED
    if _PROXY_ENV_SANITIZED:
        return

    for env_key in PROXY_ENV_KEYS:
        env_value = os.environ.get(env_key)
        if not env_value:
            continue
        scheme = env_value.split("://", 1)[0].lower()
        if scheme == "socks":
            os.environ.pop(env_key, None)

    _PROXY_ENV_SANITIZED = True


def unwrap_checkpoint_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Expected checkpoint to contain a mapping of parameters")
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], Mapping):
        return checkpoint["state_dict"]
    if "model" in checkpoint and isinstance(checkpoint["model"], Mapping):
        return checkpoint["model"]
    return checkpoint


def extract_prefixed_state_dict(
    state_dict: Mapping[str, Any], prefixes: tuple[str, ...] = AGGREGATOR_STATE_PREFIXES
) -> dict[str, Any]:
    extracted: dict[str, Any] = {}
    for key, value in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                extracted[key[len(prefix) :]] = value
                break
    return extracted


def load_aggregator_state_dict_from_checkpoint(ckpt_path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(Path(ckpt_path), map_location="cpu")
    state_dict = unwrap_checkpoint_state_dict(checkpoint)
    aggregator_state_dict = extract_prefixed_state_dict(state_dict)
    if aggregator_state_dict:
        return aggregator_state_dict
    if not state_dict:
        raise ValueError(f"No parameters found in checkpoint: {ckpt_path}")
    return dict(state_dict)


def input_transform(image_size=None):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if image_size:
        return T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )
    return T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])


def _get_sped_dataset_cls():
    global SPEDDataset
    if SPEDDataset is None:
        from da3_streaming.loop_utils.salad.dataloaders.val.SPEDDataset import SPEDDataset as _SPEDDataset

        SPEDDataset = _SPEDDataset
    return SPEDDataset


def _get_pittsburgh_dataset_cls():
    global PittsburghDataset
    if PittsburghDataset is None:
        from da3_streaming.loop_utils.salad.dataloaders.val.PittsburghDataset import (
            PittsburghDataset as _PittsburghDataset,
        )

        PittsburghDataset = _PittsburghDataset
    return PittsburghDataset


def get_val_dataset(dataset_name, image_size=None):
    dataset_name = dataset_name.lower()
    transform = input_transform(image_size=image_size)

    if "pitts" in dataset_name:
        dataset_cls = _get_pittsburgh_dataset_cls()
        ds = dataset_cls(which_ds=dataset_name, input_transform=transform)
    elif "sped" in dataset_name:
        dataset_cls = _get_sped_dataset_cls()
        ds = dataset_cls(input_transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return ds, ds.num_references, ds.num_queries, ds.ground_truth


def get_validation_recalls_fn():
    from da3_streaming.loop_utils.salad.utils.validation import get_validation_recalls

    return get_validation_recalls


def get_descriptors(model, dataloader, device):
    descriptors = []
    device_type = torch.device(device).type
    autocast_enabled = device_type == "cuda"

    with torch.no_grad():
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=autocast_enabled):
            for imgs, _ in tqdm(dataloader, desc="Calculating descriptors"):
                output = model(imgs.to(device, non_blocking=True)).cpu()
                descriptors.append(output)

    return torch.cat(descriptors, dim=0)


def load_model(args):
    if args.device.startswith("cuda") and torch.cuda.is_available() is False:
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    encoder = DA3BaseLayer5CamTokenEncoder()
    aggregator = SALAD(
        num_channels=args.agg_num_channels,
        num_clusters=args.agg_num_clusters,
        cluster_dim=args.agg_cluster_dim,
        token_dim=args.agg_token_dim,
        dropout=args.agg_dropout,
    )
    aggregator_state_dict = load_aggregator_state_dict_from_checkpoint(args.agg_ckpt_path)
    aggregator.load_state_dict(aggregator_state_dict, strict=args.strict)

    model = VPRModel(encoder=encoder, aggregator=aggregator, agg_arch="SALAD")
    model = model.to(args.device)
    model.eval()
    print(f"Loaded DA3 layer-5 cam-token VPR model on {args.device}")
    print(f"  DA3 source: {DA3BaseLayer5CamTokenEncoder.MODEL_NAME_OR_PATH}")
    print(f"  Encoder contract: aux layer {DA3BaseLayer5CamTokenEncoder.AUX_LAYER}, cam_token")
    print(f"  Aggregator checkpoint: {args.agg_ckpt_path}")
    return model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate DA3 layer-5 cam-token SALAD pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--val_datasets",
        nargs="+",
        default=["SPED"],
        choices=VAL_DATASETS,
        help="Validation datasets to use",
    )
    parser.add_argument("--image_size", nargs="*", default=None, help="Image size (int, tuple or None)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--faiss_gpu", action="store_true", help="Use FAISS GPU index for recall evaluation")
    parser.add_argument("--agg-ckpt-path", type=str, required=True)
    parser.add_argument("--agg-num-channels", type=int, default=768)
    parser.add_argument("--agg-num-clusters", type=int, default=16)
    parser.add_argument("--agg-cluster-dim", type=int, default=32)
    parser.add_argument("--agg-token-dim", type=int, default=32)
    parser.add_argument("--agg-dropout", type=float, default=0.0)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args(argv)

    if args.image_size:
        if len(args.image_size) == 1:
            args.image_size = (int(args.image_size[0]), int(args.image_size[0]))
        elif len(args.image_size) == 2:
            args.image_size = tuple(map(int, args.image_size))
        else:
            raise ValueError("Invalid image size, must be int, tuple or None")

    return args


def main(argv=None):
    torch.backends.cudnn.benchmark = True

    args = parse_args(argv)
    model = load_model(args)
    get_validation_recalls = get_validation_recalls_fn()

    for val_name in args.val_datasets:
        val_dataset, num_references, num_queries, ground_truth = get_val_dataset(val_name, args.image_size)
        val_loader = DataLoader(
            val_dataset,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=args.device.startswith("cuda"),
        )

        print(f"Evaluating on {val_name}")
        descriptors = get_descriptors(model, val_loader, args.device)

        print(f"Descriptor dimension {descriptors.shape[1]}")
        r_list = descriptors[:num_references]
        q_list = descriptors[num_references:]

        print("total_size", descriptors.shape[0], num_queries + num_references)

        get_validation_recalls(
            r_list=r_list,
            q_list=q_list,
            k_values=[1, 5, 10, 15, 20, 25],
            gt=ground_truth,
            print_results=True,
            dataset_name=val_name,
            faiss_gpu=args.faiss_gpu,
            testing=False,
        )

        del descriptors
        print("========> DONE.\n")


if __name__ == "__main__":
    main()
