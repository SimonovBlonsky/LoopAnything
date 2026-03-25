from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
SALAD_ROOT = REPO_ROOT / "da3_streaming/loop_utils/salad"
for path in (SRC_ROOT, SALAD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from vpr_model import VPRModel
from utils.validation import get_validation_recalls
from dataloaders.val.MapillaryTestDataset import MSLSTest
from dataloaders.val.SPEDDataset import SPEDDataset

VAL_DATASETS = ["MSLS", "MSLS_Test", "pitts30k_test", "pitts250k_test", "Nordland", "SPED"]


class IntermediateLayerDINOv2Backbone(torch.nn.Module):
    def __init__(self, backbone, layer_index: int):
        super().__init__()
        self.backbone = backbone
        self.model = backbone.model
        self.num_channels = backbone.num_channels
        self.norm_layer = backbone.norm_layer
        self.return_token = backbone.return_token
        self.layer_index = int(layer_index)
        self.patch_size = self._resolve_patch_size()
        self.num_blocks = len(self.model.blocks)
        if self.layer_index < 0 or self.layer_index >= self.num_blocks:
            raise ValueError(
                f"Invalid backbone layer {self.layer_index}; valid range is [0, {self.num_blocks - 1}]"
            )

    def _resolve_patch_size(self) -> int:
        patch_size = getattr(self.model.patch_embed, "patch_size", 14)
        if isinstance(patch_size, tuple):
            if patch_size[0] != patch_size[1]:
                raise ValueError("Only square patch sizes are supported")
            return int(patch_size[0])
        return int(patch_size)

    def forward(self, x):
        b, _, h, w = x.shape
        tokens = self.model.prepare_tokens_with_masks(x)
        for i, blk in enumerate(self.model.blocks):
            tokens = blk(tokens)
            if i == self.layer_index:
                break

        if self.norm_layer:
            tokens = self.model.norm(tokens)

        cls_token = tokens[:, 0]
        patch_tokens = tokens[:, 1:]
        hp, wp = h // self.patch_size, w // self.patch_size
        if patch_tokens.shape[1] != hp * wp:
            raise ValueError("Intermediate patch tokens cannot be reshaped into a spatial feature map")
        feature_map = patch_tokens.reshape((b, hp, wp, self.num_channels)).permute(0, 3, 1, 2)

        if self.return_token:
            return feature_map, cls_token
        return feature_map


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

    if "sped" in dataset_name:
        ds = SPEDDataset(input_transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return ds, ds.num_references, ds.num_queries, ds.ground_truth


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
    model = VPRModel(
        backbone_arch="dinov2_vitb14",
        backbone_config={
            "num_trainable_blocks": 4,
            "return_token": True,
            "norm_layer": True,
        },
        agg_arch=args.agg_arch,
        agg_config={
            "num_channels": args.agg_num_channels,
            "num_clusters": args.agg_num_clusters,
            "cluster_dim": args.agg_cluster_dim,
            "token_dim": args.agg_token_dim,
        },
    )

    state_dict = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)

    if args.backbone_layer >= 0:
        model.backbone = IntermediateLayerDINOv2Backbone(model.backbone, args.backbone_layer)

    model = model.eval().to(args.device)
    print(f"Loaded model from {args.ckpt_path} successfully")
    if args.backbone_layer >= 0:
        print(f"  Using intermediate DINO layer: {args.backbone_layer}")
    else:
        print("  Using final DINO backbone output")
    return model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Eval SALAD VPR model with optional DINO intermediate-layer features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the checkpoint")
    parser.add_argument(
        "--val_datasets",
        nargs="+",
        default=["SPED"],
        help="Validation datasets to use",
        choices=VAL_DATASETS,
    )
    parser.add_argument("--image_size", nargs="*", default=None, help="Image size (int, tuple or None)")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of dataloader workers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--faiss_gpu", action="store_true", help="Use FAISS GPU index for recall evaluation")
    parser.add_argument("--agg-arch", type=str, default="SALAD")
    parser.add_argument("--agg-num-channels", type=int, default=768)
    parser.add_argument("--agg-num-clusters", type=int, default=16)
    parser.add_argument("--agg-cluster-dim", type=int, default=32)
    parser.add_argument("--agg-token-dim", type=int, default=32)
    parser.add_argument(
        "--backbone-layer",
        type=int,
        default=-1,
        help="0-based DINO block index to use as the backbone output; -1 uses the original final output",
    )

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

        testing = isinstance(val_dataset, MSLSTest)
        preds = get_validation_recalls(
            r_list=r_list,
            q_list=q_list,
            k_values=[1, 5, 10, 15, 20, 25],
            gt=ground_truth,
            print_results=True,
            dataset_name=val_name,
            faiss_gpu=args.faiss_gpu,
            testing=testing,
        )

        if testing:
            val_dataset.save_predictions(preds, args.ckpt_path + "." + model.agg_arch + ".preds.txt")

        del descriptors
        print("========> DONE!\n")


if __name__ == "__main__":
    main()
