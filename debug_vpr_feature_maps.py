from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
SALAD_ROOT = REPO_ROOT / "da3_streaming/loop_utils/salad"
for _p in (SRC_ROOT, SALAD_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
import torchvision.transforms as T

from depth_anything_3.model.vpr_helper import build_vpr_model
from models.aggregators.salad import SALAD as SaladAggregator, get_matching_probs
from models.backbones.dinov2 import DINOv2


DEFAULT_SPED_ROOT = REPO_ROOT / "da3_streaming/loop_utils/salad/data/SPEDTEST"
DEFAULT_SPED_GT_ROOT = REPO_ROOT / "da3_streaming/loop_utils/salad/datasets/SPED"
DEFAULT_CKPT = REPO_ROOT / "da3_streaming/loop_utils/salad/weights/dino_salad_512_32.ckpt"
DEFAULT_OUT = REPO_ROOT / "workspace/vpr_feature_debug"
IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


def input_transform(image_size=None):
    if image_size:
        return T.Compose(
            [
                T.Resize(image_size, interpolation=InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=IMAGE_MEAN.tolist(), std=IMAGE_STD.tolist()),
            ]
        )
    return T.Compose([T.ToTensor(), T.Normalize(mean=IMAGE_MEAN.tolist(), std=IMAGE_STD.tolist())])


def unwrap_checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        return checkpoint["state_dict"]
    return checkpoint


def extract_prefix_state_dict(state_dict, prefix):
    return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}


def select_sped_pairs(q_images, db_images, ground_truth, pair_indices: Iterable[int]):
    pairs = []
    for pair_index in pair_indices:
        ref_index = int(np.asarray(ground_truth[pair_index]).reshape(-1)[0])
        pairs.append(
            {
                "pair_index": int(pair_index),
                "query_index": int(pair_index),
                "reference_index": ref_index,
                "query_relpath": str(q_images[pair_index]),
                "reference_relpath": str(db_images[ref_index]),
            }
        )
    return pairs


def run_salad_aggregator_debug(aggregator, feature_map, token):
    cluster_feature_map = aggregator.cluster_features(feature_map)
    score_map = aggregator.score(feature_map)
    token_features = aggregator.token_features(token)

    cluster_flat = cluster_feature_map.flatten(2)
    score_flat = score_map.flatten(2)
    matching_log_probs = get_matching_probs(score_flat, aggregator.dust_bin, 3)
    matching_probs = torch.exp(matching_log_probs)[:, :-1, :]

    p = matching_probs.unsqueeze(1).repeat(1, aggregator.cluster_dim, 1, 1)
    f = cluster_flat.unsqueeze(2).repeat(1, 1, aggregator.num_clusters, 1)
    aggregated_local_features = F.normalize((f * p).sum(dim=-1), p=2, dim=1)

    descriptor = torch.cat(
        [
            F.normalize(token_features, p=2, dim=-1),
            aggregated_local_features.flatten(1),
        ],
        dim=-1,
    )
    descriptor = F.normalize(descriptor, p=2, dim=-1)

    return {
        "cluster_feature_map": cluster_feature_map.detach().cpu(),
        "score_map": score_map.detach().cpu(),
        "token_features": token_features.detach().cpu(),
        "matching_probs": matching_probs.detach().cpu(),
        "aggregated_local_features": aggregated_local_features.detach().cpu(),
        "descriptor": descriptor.detach().cpu(),
    }


def resolve_token(raw_token, token_mode):
    if token_mode == "zero":
        return torch.zeros_like(raw_token)
    if token_mode == "raw":
        return raw_token
    raise ValueError(f"Unsupported token mode: {token_mode}")


def tensor_summary(tensor):
    t = tensor.detach().float().cpu()
    return {
        "shape": list(t.shape),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "mean": float(t.mean().item()),
        "std": float(t.std().item()),
    }


def _ensure_chw_tensor(tensor: torch.Tensor) -> torch.Tensor:
    t = tensor.detach().float().cpu()
    if t.ndim == 4:
        if t.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for visualization, got shape {tuple(t.shape)}")
        t = t[0]
    if t.ndim != 3:
        raise ValueError(f"Expected 3D CHW tensor or 4D BCHW tensor, got shape {tuple(t.shape)}")
    return t


def _normalize_unit_interval(array: np.ndarray) -> np.ndarray:
    arr = array.astype(np.float32, copy=False)
    arr = arr - float(arr.min())
    scale = float(arr.max())
    if scale <= 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return arr / scale


def _colorize_scalar_map(array_2d: np.ndarray) -> np.ndarray:
    x = _normalize_unit_interval(array_2d)
    r = np.clip(1.8 * x - 0.3, 0.0, 1.0)
    g = np.clip(1.5 - 3.0 * np.abs(x - 0.5), 0.0, 1.0)
    b = np.clip(1.3 - 1.8 * x, 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def feature_map_to_mean_abs_image(feature_map: torch.Tensor) -> Image.Image:
    chw = _ensure_chw_tensor(feature_map)
    heat = chw.abs().mean(dim=0).numpy()
    return Image.fromarray(_colorize_scalar_map(heat), mode="RGB")


def feature_map_to_pca_rgb_image(feature_map: torch.Tensor) -> Image.Image:
    chw = _ensure_chw_tensor(feature_map)
    channels, height, width = chw.shape
    pixels = chw.permute(1, 2, 0).reshape(-1, channels)
    pixels = pixels - pixels.mean(dim=0, keepdim=True)

    num_components = min(3, pixels.shape[0], pixels.shape[1])
    if num_components == 0:
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        return Image.fromarray(rgb, mode="RGB")

    try:
        _, _, v = torch.pca_lowrank(pixels, q=num_components, center=False)
        projected = pixels @ v[:, :num_components]
    except RuntimeError:
        projected = pixels[:, :num_components]

    if projected.shape[1] < 3:
        projected = F.pad(projected, (0, 3 - projected.shape[1]))

    rgb = projected.reshape(height, width, 3).numpy()
    rgb = np.stack([_normalize_unit_interval(rgb[..., i]) for i in range(3)], axis=-1)
    return Image.fromarray((rgb * 255.0).astype(np.uint8), mode="RGB")


def tensor_to_rgb_image(image_tensor: torch.Tensor) -> Image.Image:
    chw = _ensure_chw_tensor(image_tensor)
    image = chw * IMAGE_STD[:, None, None] + IMAGE_MEAN[:, None, None]
    image = image.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return Image.fromarray((image * 255.0).astype(np.uint8), mode="RGB")


def is_visualizable_feature_map(tensor: torch.Tensor) -> bool:
    t = tensor.detach().cpu()
    if t.ndim == 4:
        return t.shape[0] == 1 and t.shape[2] > 1 and t.shape[3] > 1
    if t.ndim == 3:
        return t.shape[1] > 1 and t.shape[2] > 1
    return False


def save_visualizations(out_dir: Path, data: dict[str, torch.Tensor]):
    for name, tensor in data.items():
        if not is_visualizable_feature_map(tensor):
            continue
        feature_map_to_mean_abs_image(tensor).save(out_dir / f"{name}_mean_abs.png")
        feature_map_to_pca_rgb_image(tensor).save(out_dir / f"{name}_pca_rgb.png")


def save_tensor_dict(out_dir: Path, data: dict[str, torch.Tensor]):
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, tensor in data.items():
        torch.save(tensor, out_dir / f"{name}.pt")
        summary[name] = tensor_summary(tensor)
    save_visualizations(out_dir, data)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def patch_tokens_to_feature_map(patch_tokens: torch.Tensor, spatial_shape: tuple[int, int]) -> torch.Tensor:
    tokens = patch_tokens
    if tokens.ndim == 4:
        if tokens.shape[1] != 1:
            raise ValueError(f"Expected a single view in aux features, got shape {tuple(tokens.shape)}")
        tokens = tokens[:, 0]
    if tokens.ndim != 3:
        raise ValueError(f"Expected [B, N, C] or [B, 1, N, C], got shape {tuple(tokens.shape)}")
    hp, wp = spatial_shape
    if tokens.shape[1] != hp * wp:
        raise ValueError(
            f"Cannot reshape {tokens.shape[1]} tokens into spatial shape {(hp, wp)}"
        )
    return tokens.reshape(tokens.shape[0], hp, wp, tokens.shape[-1]).permute(0, 3, 1, 2).contiguous()


def extract_dino_intermediate_feature_maps(backbone, image_tensor, device, export_feat_layers):
    export_feat_layers = [] if export_feat_layers is None else list(export_feat_layers)
    if not export_feat_layers:
        return {}
    if min(export_feat_layers) < 0:
        raise ValueError("export_feat_layers must be non-negative")

    x = image_tensor.unsqueeze(0).to(device)
    batch_size, _, height, width = x.shape
    patch_h, patch_w = height // 14, width // 14
    outputs = {}

    with torch.no_grad():
        tokens = backbone.model.prepare_tokens_with_masks(x)
        max_layer = max(export_feat_layers)
        if max_layer >= len(backbone.model.blocks):
            raise ValueError(
                f"Requested layer {max_layer}, but backbone only has {len(backbone.model.blocks)} blocks"
            )

        for layer_idx, block in enumerate(backbone.model.blocks):
            tokens = block(tokens)
            if layer_idx not in export_feat_layers:
                continue

            layer_tokens = backbone.model.norm(tokens) if backbone.norm_layer else tokens
            patch_tokens = layer_tokens[:, 1:]
            feature_map = patch_tokens.reshape(
                batch_size, patch_h, patch_w, backbone.num_channels
            ).permute(0, 3, 1, 2)
            outputs[f"intermediate_layer_{layer_idx:02d}_feature_map"] = feature_map.detach().cpu()

    return outputs


def load_image(image_path: Path, transform):
    image = Image.open(image_path).convert("RGB")
    return transform(image)


def load_sped_triplets(sped_root: Path, sped_gt_root: Path, pair_indices):
    db_images = np.load(sped_gt_root / "SPED_dbImages.npy", allow_pickle=True)
    q_images = np.load(sped_gt_root / "SPED_qImages.npy", allow_pickle=True)
    ground_truth = np.load(sped_gt_root / "SPED_gt.npy", allow_pickle=True)
    pairs = select_sped_pairs(q_images, db_images, ground_truth, pair_indices)
    for pair in pairs:
        pair["query_path"] = str(sped_root / pair["query_relpath"])
        pair["reference_path"] = str(sped_root / pair["reference_relpath"])
    return pairs


def extract_da3_debug(model, image_tensor, device, token_mode, export_feat_layers=None):
    export_feat_layers = [] if export_feat_layers is None else list(export_feat_layers)
    with torch.no_grad():
        model_input = image_tensor.unsqueeze(0).to(device)
        features = model.extract_features(model_input)
        raw_token = features["global_token"]
        used_token = resolve_token(raw_token, token_mode)
        agg_debug = run_salad_aggregator_debug(model.aggregator, features["feature_map"], used_token)

        debug = {
            "backbone_feature_map": features["feature_map"].detach().cpu(),
            "backbone_patch_tokens": features["patch_tokens"].detach().cpu(),
            "raw_token": raw_token.detach().cpu(),
            "used_token": used_token.detach().cpu(),
            **agg_debug,
        }

        if export_feat_layers:
            da3_net = model.encoder._unwrap_da3_net()
            backbone_input = model.encoder._normalize_input(model_input)
            _, aux_feats = da3_net.backbone(
                backbone_input,
                cam_token=None,
                export_feat_layers=export_feat_layers,
                ref_view_strategy=model.encoder.ref_view_strategy,
            )
            for layer_idx, aux_feat in zip(export_feat_layers, aux_feats):
                feature_map = patch_tokens_to_feature_map(aux_feat, features["spatial_shape"])
                debug[f"intermediate_layer_{layer_idx:02d}_feature_map"] = feature_map.detach().cpu()

    return debug


def load_pure_salad_models(device, ckpt_path, agg_num_channels, agg_num_clusters, agg_cluster_dim, agg_token_dim, agg_dropout):
    backbone = DINOv2(
        model_name="dinov2_vitb14",
        num_trainable_blocks=4,
        norm_layer=True,
        return_token=True,
    )
    aggregator = SaladAggregator(
        num_channels=agg_num_channels,
        num_clusters=agg_num_clusters,
        cluster_dim=agg_cluster_dim,
        token_dim=agg_token_dim,
        dropout=agg_dropout,
    )

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = unwrap_checkpoint_state_dict(checkpoint)
    backbone.load_state_dict(extract_prefix_state_dict(state_dict, "backbone."), strict=True)
    aggregator.load_state_dict(extract_prefix_state_dict(state_dict, "aggregator."), strict=True)

    backbone = backbone.to(device).eval()
    aggregator = aggregator.to(device).eval()
    return backbone, aggregator


def extract_pure_salad_debug(
    backbone, aggregator, image_tensor, device, token_mode, export_feat_layers=None
):
    with torch.no_grad():
        feature_map, raw_token = backbone(image_tensor.unsqueeze(0).to(device))
        used_token = resolve_token(raw_token, token_mode)
        agg_debug = run_salad_aggregator_debug(aggregator, feature_map, used_token)

    debug = {
        "backbone_feature_map": feature_map.detach().cpu(),
        "raw_token": raw_token.detach().cpu(),
        "used_token": used_token.detach().cpu(),
        **agg_debug,
    }
    debug.update(extract_dino_intermediate_feature_maps(backbone, image_tensor, device, export_feat_layers))
    return debug


def load_da3_model(args):
    build_kwargs = {
        "feat_layer": args.feat_layer,
        "ref_view_strategy": args.ref_view_strategy,
        "patch_size": args.patch_size,
        "agg_arch": "SALAD",
        "agg_config": {
            "num_channels": args.agg_num_channels,
            "num_clusters": args.agg_num_clusters,
            "cluster_dim": args.agg_cluster_dim,
            "token_dim": args.agg_token_dim,
            "dropout": args.agg_dropout,
        },
        "aggregator_ckpt_path": str(args.da3_agg_ckpt_path),
        "strict": True,
    }
    if args.da3_model_name_or_path is not None:
        build_kwargs["da3_model_name_or_path"] = args.da3_model_name_or_path
    else:
        build_kwargs["da3_config_path"] = str(args.da3_config_path)
        build_kwargs["da3_weight_path"] = str(args.da3_weight_path)
    model = build_vpr_model(**build_kwargs).to(args.device).eval()
    return model


def pair_dir_name(pair):
    q = Path(pair["query_relpath"]).stem
    r = Path(pair["reference_relpath"]).stem
    return f"pair_{pair['pair_index']:03d}_q{q}_r{r}"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Debug DA3+SALAD vs pure DINO SALAD feature maps on SPED pairs")
    parser.add_argument("--sped-root", type=Path, default=DEFAULT_SPED_ROOT)
    parser.add_argument("--sped-gt-root", type=Path, default=DEFAULT_SPED_GT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pair-indices", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--image-size", nargs=2, type=int, default=(322, 322))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--token-mode", choices=["zero", "raw"], default="zero")

    parser.add_argument("--da3-export-feat-layers", nargs="*", type=int, default=list(range(12)))
    parser.add_argument("--salad-export-feat-layers", nargs="*", type=int, default=list(range(12)))
    parser.add_argument("--da3-model-name-or-path", type=str, default="depth-anything/DA3-SMALL-1.1")
    parser.add_argument("--da3-config-path", type=Path, default=None)
    parser.add_argument("--da3-weight-path", type=Path, default=None)
    parser.add_argument("--da3-agg-ckpt-path", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--feat-layer", type=int, default=-1)
    parser.add_argument("--ref-view-strategy", type=str, default="saddle_balanced")
    parser.add_argument("--patch-size", type=int, default=14)

    parser.add_argument("--salad-ckpt-path", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--agg-num-channels", type=int, default=768)
    parser.add_argument("--agg-num-clusters", type=int, default=16)
    parser.add_argument("--agg-cluster-dim", type=int, default=32)
    parser.add_argument("--agg-token-dim", type=int, default=32)
    parser.add_argument("--agg-dropout", type=float, default=0.0)

    args = parser.parse_args(argv)
    if args.da3_config_path is not None and args.da3_weight_path is None:
        parser.error("--da3-weight-path is required when --da3-config-path is used")
    if args.da3_config_path is None and args.da3_weight_path is not None:
        parser.error("--da3-config-path is required when --da3-weight-path is used")
    return args


def main(argv=None):
    args = parse_args(argv)
    transform = input_transform(tuple(args.image_size))
    pairs = load_sped_triplets(args.sped_root, args.sped_gt_root, args.pair_indices)

    da3_model = load_da3_model(args)
    pure_backbone, pure_aggregator = load_pure_salad_models(
        args.device,
        str(args.salad_ckpt_path),
        args.agg_num_channels,
        args.agg_num_clusters,
        args.agg_cluster_dim,
        args.agg_token_dim,
        args.agg_dropout,
    )

    manifest = {
        "pairs": pairs,
        "token_mode": args.token_mode,
        "image_size": list(args.image_size),
        "da3_model_name_or_path": args.da3_model_name_or_path,
        "da3_export_feat_layers": list(args.da3_export_feat_layers),
        "salad_export_feat_layers": list(args.salad_export_feat_layers),
        "da3_agg_ckpt_path": str(args.da3_agg_ckpt_path),
        "salad_ckpt_path": str(args.salad_ckpt_path),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    for pair in pairs:
        pair_out = args.output_dir / pair_dir_name(pair)
        pair_out.mkdir(parents=True, exist_ok=True)
        (pair_out / "metadata.json").write_text(json.dumps(pair, indent=2))

        for role, path_key in (("query", "query_path"), ("reference", "reference_path")):
            image_path = Path(pair[path_key])
            image_tensor = load_image(image_path, transform)
            torch.save(image_tensor, pair_out / f"{role}_input_tensor.pt")
            tensor_to_rgb_image(image_tensor).save(pair_out / f"{role}_input_preview.png")

            da3_debug = extract_da3_debug(
                da3_model, image_tensor, args.device, args.token_mode, args.da3_export_feat_layers
            )
            save_tensor_dict(pair_out / "da3_salad" / role, da3_debug)

            pure_debug = extract_pure_salad_debug(
                pure_backbone,
                pure_aggregator,
                image_tensor,
                args.device,
                args.token_mode,
                args.salad_export_feat_layers,
            )
            save_tensor_dict(pair_out / "dino_salad" / role, pure_debug)

            print(f"Saved debug tensors and visualizations for {role}: {image_path}")


if __name__ == "__main__":
    main()
