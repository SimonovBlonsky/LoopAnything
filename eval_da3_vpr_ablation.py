from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from depth_anything_3.model.vpr_helper import build_vpr_model


VAL_DATASETS = ["SPED"]
DEFAULT_AGG_CKPT = (
    Path(__file__).resolve().parent / "da3_streaming/loop_utils/salad/weights/dino_salad_512_32.ckpt"
)


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


def get_val_dataset(dataset_name, image_size=None):
    dataset_name = dataset_name.lower()
    transform = input_transform(image_size=image_size)

    if dataset_name == "sped":
        from da3_streaming.loop_utils.salad.dataloaders.val.SPEDDataset import SPEDDataset

        ds = SPEDDataset(input_transform=transform)
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
    agg_config = {
        "num_channels": args.agg_num_channels,
        "num_clusters": args.agg_num_clusters,
        "cluster_dim": args.agg_cluster_dim,
        "token_dim": args.agg_token_dim,
        "dropout": args.agg_dropout,
    }

    build_kwargs = {
        "feat_layer": args.feat_layer,
        "ref_view_strategy": args.ref_view_strategy,
        "patch_size": args.patch_size,
        "agg_arch": args.agg_arch,
        "agg_config": agg_config,
        "aggregator_ckpt_path": args.agg_ckpt_path,
        "strict": args.strict,
    }

    if args.da3_model_name_or_path is not None:
        build_kwargs["da3_model_name_or_path"] = args.da3_model_name_or_path
    else:
        build_kwargs["da3_config_path"] = args.da3_config_path
        build_kwargs["da3_weight_path"] = args.da3_weight_path

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    model = build_vpr_model(**build_kwargs)
    if hasattr(model, "encoder"):
        model.encoder.feature_source = args.feature_source
        model.encoder.aux_layer = args.aux_layer
        model.encoder.aux_layers = args.aux_layers
        model.encoder.layer_combine = args.layer_combine
        model.encoder.layer_weights = args.layer_weights
        model.encoder.layer_scale = args.layer_scale
        model.encoder.post_fusion_norm = args.post_fusion_norm
    model = model.to(args.device)
    print(f"Loaded DA3 VPR model on {args.device}")
    print(f"  DA3 source: {args.da3_model_name_or_path or (args.da3_config_path, args.da3_weight_path)}")
    print(f"  Aggregator checkpoint: {args.agg_ckpt_path}")
    print(f"  Feature source: {args.feature_source}")
    if args.feature_source == "aux":
        if args.aux_layers is None:
            print(f"  Aux layer: {args.aux_layer}")
        else:
            print(f"  Aux layers: {args.aux_layers}")
        print(f"  Layer combine: {args.layer_combine}")
        print(f"  Layer scale: {args.layer_scale}")
        print(f"  Post-fusion norm: {args.post_fusion_norm}")
    return model


def _argv_has_flag(argv, flag_name):
    for idx, token in enumerate(argv):
        if token == flag_name:
            return True
        if token.startswith(f"{flag_name}="):
            return True
    return False


def parse_args(argv=None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Evaluate DA3 encoder + VPR aggregator on retrieval benchmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--val_datasets",
        nargs="+",
        default=VAL_DATASETS,
        choices=VAL_DATASETS,
        help="Validation datasets to use",
    )
    parser.add_argument("--image_size", nargs="*", default=None, help="Image size (int, tuple or None)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--faiss_gpu", action="store_true", help="Use FAISS GPU index for recall evaluation")

    parser.add_argument("--agg-arch", dest="agg_arch", type=str, default="SALAD")
    parser.add_argument("--agg-ckpt-path", dest="agg_ckpt_path", type=str, default=str(DEFAULT_AGG_CKPT))
    parser.add_argument("--agg-num-channels", type=int, default=768)
    parser.add_argument("--agg-num-clusters", type=int, default=16)
    parser.add_argument("--agg-cluster-dim", type=int, default=32)
    parser.add_argument("--agg-token-dim", type=int, default=32)
    parser.add_argument("--agg-dropout", type=float, default=0.0)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--feat-layer", type=int, default=-1)
    parser.add_argument("--feature-source", choices=["final", "aux"], default="final")
    parser.add_argument("--aux-layer", type=int, default=3)
    parser.add_argument("--aux-layers", nargs="+", type=int, default=None)
    parser.add_argument("--layer-combine", type=str, default="single")
    parser.add_argument("--layer-weights", nargs="+", type=float, default=None)
    parser.add_argument("--layer-scale", type=float, default=1.0)
    parser.add_argument("--post-fusion-norm", type=str, default="none")
    parser.add_argument("--ref-view-strategy", type=str, default="saddle_balanced")
    parser.add_argument("--patch-size", type=int, default=14)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--da3-model-name-or-path", type=str, default=None)
    group.add_argument("--da3-config-path", type=str, default=None)
    parser.add_argument("--da3-weight-path", type=str, default=None)

    args = parser.parse_args(raw_argv)

    if args.image_size:
        if len(args.image_size) == 1:
            args.image_size = (int(args.image_size[0]), int(args.image_size[0]))
        elif len(args.image_size) == 2:
            args.image_size = tuple(map(int, args.image_size))
        else:
            raise ValueError("Invalid image size, must be int, tuple or None")

    if args.da3_config_path is not None and args.da3_weight_path is None:
        parser.error("--da3-weight-path is required when --da3-config-path is used")
    if args.da3_config_path is None and args.da3_weight_path is not None:
        parser.error("--da3-config-path is required when --da3-weight-path is used")

    args.layer_combine = args.layer_combine.lower()
    if args.layer_combine not in {"single", "avg", "sum", "weighted_avg"}:
        parser.error(f"Unsupported --layer-combine value: {args.layer_combine}")

    args.post_fusion_norm = args.post_fusion_norm.lower()
    if args.post_fusion_norm not in {"none", "token_l2", "feature_layernorm"}:
        parser.error(f"Unsupported --post-fusion-norm value: {args.post_fusion_norm}")

    layer_combine_explicit = _argv_has_flag(raw_argv, "--layer-combine")
    layer_scale_explicit = _argv_has_flag(raw_argv, "--layer-scale")
    post_fusion_norm_explicit = _argv_has_flag(raw_argv, "--post-fusion-norm")
    explicit_aux_fusion_args = (
        args.aux_layers is not None
        or args.layer_weights is not None
        or layer_combine_explicit
        or args.layer_combine != "single"
        or layer_scale_explicit
        or args.layer_scale != 1.0
        or post_fusion_norm_explicit
        or args.post_fusion_norm != "none"
    )
    if args.feature_source == "final" and explicit_aux_fusion_args:
        parser.error("--feature-source final does not accept explicit AUX fusion args")

    if args.feature_source == "aux" and args.aux_layer < 0:
        parser.error("--aux-layer must be non-negative when --feature-source aux is used")
    if args.feature_source == "aux":
        if args.aux_layers is None:
            if args.layer_combine != "single":
                parser.error("--feature-source aux without --aux-layers only supports --layer-combine single")
            if args.layer_weights is not None:
                parser.error("--layer-weights requires --aux-layers")
        elif args.layer_combine == "single" and len(args.aux_layers) != 1:
            parser.error("single layer_combine requires exactly one aux layer")
        elif args.layer_combine == "weighted_avg" and args.layer_weights is None:
            parser.error("weighted_avg requires layer_weights to match aux_layers")
        elif args.layer_combine == "weighted_avg" and len(args.layer_weights) != len(args.aux_layers):
            parser.error("weighted_avg requires layer_weights to match aux_layers")

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
        print("========> DONE!\n")


if __name__ == "__main__":
    main()
