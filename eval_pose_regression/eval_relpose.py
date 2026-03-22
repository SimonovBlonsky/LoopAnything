from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as tvf
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True

REPO_ROOT = Path(__file__).resolve().parents[2]
LOOPANYTHING_ROOT = REPO_ROOT / "LoopAnything"
LOOPANYTHING_SRC = LOOPANYTHING_ROOT / "src"
for path in (REPO_ROOT, LOOPANYTHING_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import datasets.megadepth_valid as megadepth_valid_module  # noqa: E402
import datasets.scannet1500 as scannet1500_module  # noqa: E402
from depth_anything_3.api import DepthAnything3  # noqa: E402
from depth_anything_3.utils.geometry import as_homogeneous  # noqa: E402
from utils.metric import error_auc, get_rot_err, get_transl_ang_err  # noqa: E402


TO_TENSOR = tvf.ToTensor()


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quick relative pose evaluation for Depth Anything 3")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["scannet1500", "megadepth1500"],
        help="Evaluation dataset name",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="depth-anything/DA3-LARGE-1.1",
        help="Hugging Face repo id or local DA3 pretrained directory",
    )
    parser.add_argument(
        "--scannet-root",
        type=str,
        default=str(REPO_ROOT / "reloc3r" / "data" / "scannet1500"),
        help="Path to reloc3r/data/scannet1500",
    )
    parser.add_argument(
        "--megadepth-root",
        type=str,
        default=str(REPO_ROOT / "reloc3r" / "data" / "megadepth1500"),
        help="Path to reloc3r/data/megadepth1500",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Number of pairs per forward pass")
    parser.add_argument(
        "--dataset-width",
        type=int,
        default=512,
        help="Shared reloc3r-style dataset crop width",
    )
    parser.add_argument(
        "--dataset-height",
        type=int,
        default=384,
        help="Shared reloc3r-style dataset crop height",
    )
    parser.add_argument("--seed", type=int, default=777, help="Dataset RNG seed")
    parser.add_argument(
        "--process-res",
        type=int,
        default=504,
        help="DA3 preprocessing resolution after the shared dataset crop",
    )
    parser.add_argument(
        "--process-res-method",
        type=str,
        default="upper_bound_resize",
        choices=[
            "upper_bound_resize",
            "lower_bound_resize",
            "upper_bound_crop",
            "lower_bound_crop",
        ],
        help="DA3 preprocessing method",
    )
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=8,
        help="Workers used by DA3 InputProcessor",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device",
    )
    parser.add_argument(
        "--use-ray-pose",
        action="store_true",
        help="Use DA3 ray-based pose estimation instead of camera decoder",
    )
    parser.add_argument(
        "--ref-view-strategy",
        type=str,
        default="saddle_balanced",
        help="Reference-view selection strategy passed to DA3",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=-1,
        help="Debug option: only evaluate the first N pairs (-1 means all)",
    )
    parser.add_argument(
        "--save-errors",
        type=str,
        default="",
        help="Optional path to save per-pair rotation/translation errors as an npz file",
    )
    return parser


def batched_indices(total: int, batch_size: int):
    for start in range(0, total, batch_size):
        yield range(start, min(total, start + batch_size))


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_dataset(args: argparse.Namespace):
    common_kwargs = dict(
        resolution=(args.dataset_width, args.dataset_height),
        seed=args.seed,
        transform=TO_TENSOR,
    )
    if args.dataset == "scannet1500":
        scannet1500_module.DATA_ROOT = args.scannet_root
        return scannet1500_module.ScanNet1500(**common_kwargs)
    if args.dataset == "megadepth1500":
        megadepth_valid_module.DATA_ROOT = args.megadepth_root
        return megadepth_valid_module.MegaDepth_valid(**common_kwargs)
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def pair_id_from_views(views: list[dict]) -> str:
    left = f"{views[0]['label']}/{views[0]['instance']}"
    right = f"{views[1]['label']}/{views[1]['instance']}"
    return f"{left} || {right}"


def gt_pose2to1_from_views(views: list[dict]) -> np.ndarray:
    return np.linalg.inv(views[0]["camera_pose"]) @ views[1]["camera_pose"]


def view_tensor_to_numpy_image(view: dict) -> np.ndarray:
    img = view["img"]
    if not torch.is_tensor(img):
        raise TypeError(f"Expected a tensor image from shared dataset, got {type(img)!r}")
    img = img.detach().cpu().clamp(0.0, 1.0)
    img = img.permute(1, 2, 0).numpy()
    return np.clip(np.round(img * 255.0), 0.0, 255.0).astype(np.uint8)


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, float]:
    device = torch.device(args.device)
    dataset = build_dataset(args)
    total_pairs = len(dataset) if args.max_pairs < 0 else min(len(dataset), args.max_pairs)

    print(f"Loading DA3 model from: {args.model_path}")
    model = DepthAnything3.from_pretrained(args.model_path).to(device)
    model.eval()

    rerrs: list[float] = []
    terrs: list[float] = []
    pair_ids: list[str] = []
    processed_pairs = 0

    preprocess_time = 0.0
    model_time = 0.0
    metric_time = 0.0

    total_start = time.perf_counter()
    pbar = tqdm(total=total_pairs, desc=f"Evaluating {args.dataset}", unit="pair")
    try:
        for batch_ids in batched_indices(total_pairs, args.batch_size):
            pair_views_batch = [dataset[i] for i in batch_ids]
            shared_crops = [view_tensor_to_numpy_image(view) for views in pair_views_batch for view in views]

            t0 = time.perf_counter()
            imgs_cpu, _, _ = model.input_processor(
                shared_crops,
                process_res=args.process_res,
                process_res_method=args.process_res_method,
                num_workers=args.preprocess_workers,
                print_progress=False,
                sequential=args.preprocess_workers <= 1,
                desc=None,
            )
            imgs = imgs_cpu.view(len(pair_views_batch), 2, *imgs_cpu.shape[1:]).to(device, non_blocking=True).float()
            preprocess_time += time.perf_counter() - t0

            synchronize_if_cuda(device)
            t1 = time.perf_counter()
            raw_output = model.forward(
                imgs,
                extrinsics=None,
                intrinsics=None,
                export_feat_layers=[],
                infer_gs=False,
                use_ray_pose=args.use_ray_pose,
                ref_view_strategy=args.ref_view_strategy,
            )
            synchronize_if_cuda(device)
            model_time += time.perf_counter() - t1

            pred_ext = raw_output.get("extrinsics", None)
            if pred_ext is None:
                raise RuntimeError(
                    "DA3 forward pass did not return extrinsics. "
                    "Check the model type or pose-estimation settings."
                )

            t2 = time.perf_counter()
            pred_ext = as_homogeneous(pred_ext)
            pred_rel_2to1 = pred_ext[:, 0] @ torch.linalg.inv(pred_ext[:, 1])
            pred_rel_2to1 = pred_rel_2to1.detach().cpu().numpy()

            for sid, views in enumerate(pair_views_batch):
                gt_pose2to1 = gt_pose2to1_from_views(views)
                pr_pose2to1 = pred_rel_2to1[sid]

                rerr = get_rot_err(pr_pose2to1[:3, :3], gt_pose2to1[:3, :3])

                transl = pr_pose2to1[:3, 3]
                gt_transl = gt_pose2to1[:3, 3]
                terr = get_transl_ang_err(transl, gt_transl)

                rerrs.append(rerr)
                terrs.append(terr)
                pair_ids.append(pair_id_from_views(views))
            metric_time += time.perf_counter() - t2

            processed_pairs += len(pair_views_batch)
            elapsed = time.perf_counter() - total_start
            pbar.update(len(pair_views_batch))
            pbar.set_postfix(pair_per_s=f"{processed_pairs / max(elapsed, 1e-6):.2f}")
    finally:
        pbar.close()

    total_time = time.perf_counter() - total_start
    rerrs_np = np.asarray(rerrs, dtype=np.float32)
    terrs_np = np.asarray(terrs, dtype=np.float32)
    aucs = error_auc(rerrs_np, terrs_np, thresholds=[5, 10, 20])

    speed = {
        "pairs": int(len(rerrs_np)),
        "batch_size": int(args.batch_size),
        "preprocess_time_s": preprocess_time,
        "model_time_s": model_time,
        "metric_time_s": metric_time,
        "total_time_s": total_time,
        "ms_per_pair_model_only": 1000.0 * model_time / max(len(rerrs_np), 1),
        "ms_per_pair_end_to_end": 1000.0 * total_time / max(len(rerrs_np), 1),
        "pairs_per_sec_model_only": len(rerrs_np) / max(model_time, 1e-6),
        "pairs_per_sec_end_to_end": len(rerrs_np) / max(total_time, 1e-6),
    }

    print(f"In total {len(rerrs_np)} pairs")
    print(json.dumps(aucs, indent=2))
    print(json.dumps(speed, indent=2))

    if args.save_errors:
        save_path = Path(args.save_errors)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_path, pair_id=np.asarray(pair_ids), rerr=rerrs_np, terr=terrs_np)
        print(f"Saved per-pair errors to {save_path}")

    return aucs


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    evaluate(args)
