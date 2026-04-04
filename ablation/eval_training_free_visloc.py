from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import torch


SUPPORTED_BACKENDS = ("dino_salad", "da3_salad")
SUPPORTED_POSE_PATHS = ("cam_dec", "ray", "both")


def validate_retrieval_backend(backend: str) -> str:
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported retrieval backend: {backend}. "
            f"Expected one of {SUPPORTED_BACKENDS}."
        )
    return backend


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training-free visual localization baseline scaffold")
    parser.add_argument("--dataset", type=str, required=True, choices=["7scenes", "cambridge"])
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--backend", type=str, default="dino_salad")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-m", type=int, default=3)
    parser.add_argument("--pose-path", type=str, default="cam_dec", choices=list(SUPPORTED_POSE_PATHS))
    args = parser.parse_args(argv)
    args.backend = validate_retrieval_backend(args.backend)
    if args.top_m > args.top_k:
        raise ValueError("top_m must be <= top_k")
    return args


def select_topk_topm(
    sims: torch.Tensor,
    top_k: int,
    top_m: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sims.ndim != 1:
        raise ValueError("sims must be a 1D tensor")
    if top_m > top_k:
        raise ValueError("top_m must be <= top_k")
    if top_k <= 0 or top_m <= 0:
        raise ValueError("top_k and top_m must be positive")

    k = min(top_k, int(sims.numel()))
    topk_indices = sims.topk(k).indices
    m = min(top_m, k)
    topm_indices = topk_indices[:m]
    return topk_indices, topm_indices


def _get_field(output: Any, field: str) -> Any:
    if isinstance(output, dict):
        return output.get(field)
    return getattr(output, field, None)


def _resolve_cam_dec(output: Any) -> dict[str, Any]:
    extrinsics = _get_field(output, "extrinsics")
    intrinsics = _get_field(output, "intrinsics")
    if extrinsics is None or intrinsics is None:
        raise ValueError("cam_dec output requires `extrinsics` and `intrinsics`.")
    return {
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "pose_enc": _get_field(output, "pose_enc"),
    }


def _resolve_ray(output: Any) -> dict[str, Any]:
    extrinsics = _get_field(output, "ray_extrinsics")
    intrinsics = _get_field(output, "ray_intrinsics")
    if extrinsics is None or intrinsics is None:
        extrinsics = _get_field(output, "extrinsics")
        intrinsics = _get_field(output, "intrinsics")
    if extrinsics is None or intrinsics is None:
        raise ValueError("ray output requires ray-specific or generic extrinsics/intrinsics fields.")
    return {
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
    }


def resolve_pose_output(output: Any, pose_path: str) -> dict[str, dict[str, Any]]:
    if pose_path not in SUPPORTED_POSE_PATHS:
        raise ValueError(f"Unsupported pose_path: {pose_path}")
    if pose_path == "cam_dec":
        return {"cam_dec": _resolve_cam_dec(output)}
    if pose_path == "ray":
        return {"ray": _resolve_ray(output)}
    return {"cam_dec": _resolve_cam_dec(output), "ray": _resolve_ray(output)}


def _as_pose_array(poses: np.ndarray | list[np.ndarray], name: str) -> np.ndarray:
    arr = np.asarray(poses, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1:] != (4, 4):
        raise ValueError(f"{name} must have shape [N, 4, 4], got {arr.shape}")
    return arr


def _estimate_sim3(src_pts: np.ndarray, dst_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Estimate Sim(3) from src to dst: dst = s * R * src + t."""
    if src_pts.shape != dst_pts.shape:
        raise ValueError("src_pts and dst_pts must share shape [N, 3]")
    if src_pts.ndim != 2 or src_pts.shape[1] != 3:
        raise ValueError("src_pts and dst_pts must have shape [N, 3]")
    if src_pts.shape[0] < 2:
        raise ValueError("At least 2 points are required for Sim(3) estimation.")

    n = src_pts.shape[0]
    src_mean = src_pts.mean(axis=0)
    dst_mean = dst_pts.mean(axis=0)
    src_centered = src_pts - src_mean
    dst_centered = dst_pts - dst_mean

    cov = (dst_centered.T @ src_centered) / n
    u, svals, vt = np.linalg.svd(cov)
    d = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        d[2, 2] = -1.0
    rot = u @ d @ vt

    src_var = np.sum(src_centered**2) / n
    if src_var <= 1e-12:
        raise ValueError("Degenerate source geometry for Sim(3) estimation.")
    scale = float(np.sum(svals * np.diag(d)) / src_var)
    trans = dst_mean - scale * (rot @ src_mean)
    return rot, trans, scale


def _apply_sim3_to_pose(pose: np.ndarray, rot: np.ndarray, trans: np.ndarray, scale: float) -> np.ndarray:
    aligned = np.eye(4, dtype=np.float64)
    aligned[:3, :3] = rot @ pose[:3, :3]
    aligned[:3, 3] = scale * (rot @ pose[:3, 3]) + trans
    return aligned


def align_query_pose_multi_ref(pred_group: np.ndarray, ref_gt: np.ndarray) -> np.ndarray:
    pred_group = _as_pose_array(pred_group, "pred_group")
    ref_gt = _as_pose_array(ref_gt, "ref_gt")
    if pred_group.shape[0] < 2:
        raise ValueError("pred_group must contain query plus at least one reference pose.")
    if pred_group.shape[0] - 1 != ref_gt.shape[0]:
        raise ValueError("ref_gt count must match number of reference poses in pred_group.")

    pred_refs_centers = pred_group[1:, :3, 3]
    gt_refs_centers = ref_gt[:, :3, 3]
    rot, trans, scale = _estimate_sim3(pred_refs_centers, gt_refs_centers)
    return _apply_sim3_to_pose(pred_group[0], rot, trans, scale)


def align_query_pose_top1_anchor(pred_group: np.ndarray, ref_gt: np.ndarray) -> np.ndarray:
    pred_group = _as_pose_array(pred_group, "pred_group")
    ref_gt = _as_pose_array(ref_gt, "ref_gt")
    if pred_group.shape[0] < 2:
        raise ValueError("pred_group must contain query plus at least one reference pose.")
    if ref_gt.shape[0] < 1:
        raise ValueError("ref_gt must contain at least one reference pose.")

    ref_transform = ref_gt[0] @ np.linalg.inv(pred_group[1])
    return ref_transform @ pred_group[0]


def main() -> None:
    _ = parse_args()
    raise NotImplementedError(
        "This script currently exposes protocol helpers only. "
        "Full training-free scene evaluation is implemented in Task 3."
    )


if __name__ == "__main__":
    main()
