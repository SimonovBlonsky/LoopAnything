from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True

REPO_ROOT = Path(__file__).resolve().parents[2]
LOOPANYTHING_ROOT = REPO_ROOT / "LoopAnything"
LOOPANYTHING_SRC = LOOPANYTHING_ROOT / "src"
if str(LOOPANYTHING_SRC) not in sys.path:
    sys.path.insert(0, str(LOOPANYTHING_SRC))

from depth_anything_3.api import DepthAnything3  # noqa: E402
from depth_anything_3.utils.geometry import as_homogeneous  # noqa: E402

from datasets import MegaDepth1500Pairs, ScanNet1500Pairs  # noqa: E402
from metrics import error_auc, get_rot_err, get_transl_ang_err  # noqa: E402


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
        "--process-res",
        type=int,
        default=504,
        help="DA3 preprocessing resolution (also used as reloc3r-style square target for MegaDepth)",
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
    if args.dataset == "scannet1500":
        return ScanNet1500Pairs(args.scannet_root)
    if args.dataset == "megadepth1500":
        return MegaDepth1500Pairs(args.megadepth_root)
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def build_input_items(dataset, samples, process_res: int):
    if isinstance(dataset, MegaDepth1500Pairs):
        resolution = (process_res, process_res)
        inputs = []
        for sample in samples:
            inputs.extend(dataset.prepare_pair_images_for_da3(sample, resolution))
        return inputs

    image_paths: list[str] = []
    for sample in samples:
        image_paths.extend([sample.image1, sample.image2])
    return image_paths


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
            samples = [dataset[i] for i in batch_ids]
            input_items = build_input_items(dataset, samples, args.process_res)

            t0 = time.perf_counter()
            imgs_cpu, _, _ = model.input_processor(
                input_items,
                process_res=args.process_res,
                process_res_method=args.process_res_method,
                num_workers=args.preprocess_workers,
                print_progress=False,
                sequential=args.preprocess_workers <= 1,
                desc=None,
            )
            imgs = imgs_cpu.view(len(samples), 2, *imgs_cpu.shape[1:]).to(device, non_blocking=True).float()
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

            for sid, sample in enumerate(samples):
                gt_pose2to1 = sample.gt_pose2to1
                pr_pose2to1 = pred_rel_2to1[sid]

                rerr = get_rot_err(pr_pose2to1[:3, :3], gt_pose2to1[:3, :3])

                transl = pr_pose2to1[:3, 3]
                gt_transl = gt_pose2to1[:3, 3]
                transl_dir = transl / (np.linalg.norm(transl) + 1e-8)
                gt_transl_dir = gt_transl / (np.linalg.norm(gt_transl) + 1e-8)
                terr = get_transl_ang_err(transl_dir, gt_transl_dir)

                rerrs.append(rerr)
                terrs.append(terr)
                pair_ids.append(sample.pair_id)
            metric_time += time.perf_counter() - t2

            processed_pairs += len(samples)
            elapsed = time.perf_counter() - total_start
            pbar.update(len(samples))
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
