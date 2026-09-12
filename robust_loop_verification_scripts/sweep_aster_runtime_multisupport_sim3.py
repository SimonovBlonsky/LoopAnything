#!/usr/bin/env python3
"""Sweep multi-support Sim3 scale alignment for AsterSLAM runtime DA3 loops.

This script is intentionally geometry-only: it does not run or modify the
safety gate. It reuses the accepted runtime loop pairs, selects candidate-near
supports, reruns DA3 N-view groups, and measures q-c factor accuracy after
median support-scale alignment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from robust_loop_verifier.da3_runner import (
    RealDa3Runner,
    RealDa3RunnerConfig,
    _field,
    convert_da3_batched_extrinsics_to_c2w,
)
from robust_loop_verifier.geometry import invert_transform, rotation_matrix_from_quat_xyzw

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from retest_aster_runtime_loops_offline import extract_images_from_bag


DEFAULT_RUN_DIR = (
    "/home/chenguyuan/code/NeurIPS26/AsterSLAM_ws/src/AsterSLAM/aster_slam/workspace/"
    "ntu_viral_loopanything_runtime/eee_01_loopanything_pose_debug_20260622_084829"
)
DEFAULT_GT_PATH = "/data/datasets/NTU-VIRAL/processed_gt_tum/eee_01.txt"
DEFAULT_BAG = "/data/datasets/NTU-VIRAL/data/eee_01/eee_01.bag"
DEFAULT_PAIRS = ((52, 2), (53, 2), (54, 2), (119, 88))
DEFAULT_T_CAMERA_LIDAR = (
    "0.0218308,0.99976,-0.00201407,0.122993,"
    "-0.0131205,0.00230088,0.999911,0.0398643,"
    "0.999676,-0.0218025,0.0131676,-0.0577101,"
    "0.0,0.0,0.0,1.0"
)
SUPPORT_SELECTION_MODES = ("temporal", "spatial", "temporal_stride", "da3_consistency_topk")
CONSISTENCY_POOL_MODES = ("temporal", "spatial", "temporal_stride")
DEFAULT_TEMPORAL_STRIDE_PATTERN = "1,2,3,5,8,13,21,34,55,89"


@dataclass(frozen=True)
class SupportCandidate:
    idx: int
    baseline_m: float


@dataclass(frozen=True)
class Da3ViewGroup:
    image_paths: tuple[str, ...]
    keyframe_indices: tuple[int, ...]
    view_roles: tuple[str, ...]


@dataclass(frozen=True)
class Da3ViewGroupResult:
    predicted_c2w: np.ndarray
    keyframe_indices: tuple[int, ...]
    view_roles: tuple[str, ...]


@dataclass(frozen=True)
class MedianScaleAlignment:
    valid: bool
    reason: str
    scale: float
    scale_mad: float
    support_rmse_m: float
    support_max_direction_error_deg: float
    aligned_query_c2w: np.ndarray
    aligned_candidate_c2w: np.ndarray
    per_support: list[dict[str, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep support=1..N median Sim3 scale alignment on AsterSLAM runtime loops."
    )
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--bag", default=DEFAULT_BAG)
    parser.add_argument("--image-topic", default="/left/image_raw")
    parser.add_argument("--gt-path", default=DEFAULT_GT_PATH)
    parser.add_argument("--pairs", nargs="*", default=[], help="Pairs as q:c.")
    parser.add_argument("--max-supports", type=int, default=10)
    parser.add_argument("--support-pool-size", type=int, default=10)
    parser.add_argument("--support-mode", choices=SUPPORT_SELECTION_MODES, default="temporal")
    parser.add_argument(
        "--support-selection-strategy",
        choices=SUPPORT_SELECTION_MODES,
        default=None,
        help="Alias for --support-mode. If set, overrides --support-mode.",
    )
    parser.add_argument(
        "--consistency-pool-mode",
        choices=CONSISTENCY_POOL_MODES,
        default="temporal_stride",
        help="Runtime-feasible support pool used before DA3 consistency ranking.",
    )
    parser.add_argument("--temporal-stride-pattern", default=DEFAULT_TEMPORAL_STRIDE_PATTERN)
    parser.add_argument("--min-support-baseline-m", type=float, default=0.5)
    parser.add_argument("--exclude-recent-from-query", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-name", default="depth-anything/DA3-LARGE-1.1")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--process-res-method", default="upper_bound_resize")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--triplet-batch-size", type=int, default=4)
    parser.add_argument("--image-sync-tolerance-sec", type=float, default=0.15)
    parser.add_argument("--t-camera-lidar", default=DEFAULT_T_CAMERA_LIDAR)
    parser.add_argument("--save-visualizations", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def parse_pairs(values: Iterable[str]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for value in values:
        if ":" not in value:
            raise ValueError(f"Pair must use q:c format, got {value!r}")
        query, candidate = value.split(":", 1)
        pairs.append((int(query), int(candidate)))
    return pairs


def parse_int_pattern(value: str) -> tuple[int, ...]:
    pattern = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not pattern or any(item <= 0 for item in pattern):
        raise ValueError(f"temporal stride pattern must contain positive integers, got {value!r}")
    return tuple(dict.fromkeys(pattern))


def load_tum_poses(path: Path) -> tuple[list[float], dict[int, np.ndarray]]:
    timestamps: list[float] = []
    poses: dict[int, np.ndarray] = {}
    for row, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 8:
            continue
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = np.asarray([float(v) for v in fields[1:4]], dtype=np.float64)
        pose[:3, :3] = rotation_matrix_from_quat_xyzw(
            np.asarray([float(v) for v in fields[4:8]], dtype=np.float64)
        )
        timestamps.append(float(fields[0]))
        poses[row] = pose
    return timestamps, poses


def select_candidate_nearest_supports(
    *,
    candidate_idx: int,
    query_idx: int,
    odom_by_idx: dict[int, np.ndarray],
    max_supports: int,
    exclude_recent_from_query: int,
    min_baseline_m: float,
) -> list[SupportCandidate]:
    if candidate_idx not in odom_by_idx:
        raise ValueError(f"candidate_idx {candidate_idx} not found in odometry poses")
    candidate_position = odom_by_idx[candidate_idx][:3, 3]
    blocked = {candidate_idx, query_idx}
    if exclude_recent_from_query > 0:
        blocked.update(
            idx
            for idx in odom_by_idx
            if abs(int(idx) - int(query_idx)) <= exclude_recent_from_query
        )

    supports: list[SupportCandidate] = []
    for idx, pose in odom_by_idx.items():
        if idx in blocked:
            continue
        baseline = float(np.linalg.norm(pose[:3, 3] - candidate_position))
        if not np.isfinite(baseline) or baseline < min_baseline_m:
            continue
        supports.append(SupportCandidate(idx=int(idx), baseline_m=baseline))
    supports.sort(key=lambda support: (support.baseline_m, support.idx))
    return supports[:max_supports]


def select_candidate_temporal_supports(
    *,
    candidate_idx: int,
    query_idx: int,
    odom_by_idx: dict[int, np.ndarray],
    max_supports: int,
    exclude_recent_from_query: int,
    min_baseline_m: float,
) -> list[SupportCandidate]:
    if candidate_idx not in odom_by_idx:
        raise ValueError(f"candidate_idx {candidate_idx} not found in odometry poses")
    candidate_position = odom_by_idx[candidate_idx][:3, 3]
    blocked = {candidate_idx, query_idx}
    if exclude_recent_from_query > 0:
        blocked.update(
            idx
            for idx in odom_by_idx
            if abs(int(idx) - int(query_idx)) <= exclude_recent_from_query
        )

    supports: list[SupportCandidate] = []
    for idx, pose in odom_by_idx.items():
        if idx in blocked:
            continue
        baseline = float(np.linalg.norm(pose[:3, 3] - candidate_position))
        if not np.isfinite(baseline) or baseline < min_baseline_m:
            continue
        supports.append(SupportCandidate(idx=int(idx), baseline_m=baseline))
    supports.sort(key=lambda support: (abs(support.idx - candidate_idx), support.idx))
    return supports[:max_supports]


def select_candidate_temporal_stride_supports(
    *,
    candidate_idx: int,
    query_idx: int,
    odom_by_idx: dict[int, np.ndarray],
    max_supports: int,
    exclude_recent_from_query: int,
    min_baseline_m: float,
    stride_pattern: tuple[int, ...] = tuple(int(v) for v in DEFAULT_TEMPORAL_STRIDE_PATTERN.split(",")),
) -> list[SupportCandidate]:
    if candidate_idx not in odom_by_idx:
        raise ValueError(f"candidate_idx {candidate_idx} not found in odometry poses")
    candidate_position = odom_by_idx[candidate_idx][:3, 3]
    blocked = {candidate_idx, query_idx}
    if exclude_recent_from_query > 0:
        blocked.update(
            idx
            for idx in odom_by_idx
            if abs(int(idx) - int(query_idx)) <= exclude_recent_from_query
        )

    supports: list[SupportCandidate] = []
    seen: set[int] = set()
    for stride in stride_pattern:
        for idx in (candidate_idx - stride, candidate_idx + stride):
            if idx in seen or idx in blocked or idx not in odom_by_idx:
                continue
            baseline = float(np.linalg.norm(odom_by_idx[idx][:3, 3] - candidate_position))
            if not np.isfinite(baseline) or baseline < min_baseline_m:
                continue
            supports.append(SupportCandidate(idx=int(idx), baseline_m=baseline))
            seen.add(int(idx))
            if len(supports) >= max_supports:
                return supports
    return supports[:max_supports]


def select_supports(
    *,
    mode: str,
    candidate_idx: int,
    query_idx: int,
    odom_by_idx: dict[int, np.ndarray],
    max_supports: int,
    exclude_recent_from_query: int,
    min_baseline_m: float,
    temporal_stride_pattern: tuple[int, ...] = tuple(int(v) for v in DEFAULT_TEMPORAL_STRIDE_PATTERN.split(",")),
) -> list[SupportCandidate]:
    if mode == "temporal":
        return select_candidate_temporal_supports(
            candidate_idx=candidate_idx,
            query_idx=query_idx,
            odom_by_idx=odom_by_idx,
            max_supports=max_supports,
            exclude_recent_from_query=exclude_recent_from_query,
            min_baseline_m=min_baseline_m,
        )
    if mode == "spatial":
        return select_candidate_nearest_supports(
            candidate_idx=candidate_idx,
            query_idx=query_idx,
            odom_by_idx=odom_by_idx,
            max_supports=max_supports,
            exclude_recent_from_query=exclude_recent_from_query,
            min_baseline_m=min_baseline_m,
        )
    if mode == "temporal_stride":
        return select_candidate_temporal_stride_supports(
            candidate_idx=candidate_idx,
            query_idx=query_idx,
            odom_by_idx=odom_by_idx,
            max_supports=max_supports,
            exclude_recent_from_query=exclude_recent_from_query,
            min_baseline_m=min_baseline_m,
            stride_pattern=temporal_stride_pattern,
        )
    raise ValueError(f"Unsupported support mode {mode!r}")


def build_da3_view_group(
    *,
    query_idx: int,
    candidate_idx: int,
    supports: list[SupportCandidate],
    image_by_idx: dict[int, Path],
) -> Da3ViewGroup:
    keyframe_indices = (int(query_idx), int(candidate_idx), *[support.idx for support in supports])
    return Da3ViewGroup(
        image_paths=tuple(str(image_by_idx[idx]) for idx in keyframe_indices),
        keyframe_indices=keyframe_indices,
        view_roles=("query", "candidate", *[f"support_{idx + 1}" for idx in range(len(supports))]),
    )


def run_da3_view_group(runner: RealDa3Runner, group: Da3ViewGroup) -> Da3ViewGroupResult:
    from PIL import Image
    import torch

    images = []
    for image_path in group.image_paths:
        with Image.open(image_path) as image:
            images.append(image.convert("RGB").copy())

    model = runner._load_model()
    imgs_cpu, _, _ = model.input_processor(
        images,
        extrinsics=None,
        intrinsics=None,
        process_res=runner.config.process_res,
        process_res_method=runner.config.process_res_method,
        num_workers=1,
        print_progress=False,
        sequential=True,
        desc=None,
    )
    if imgs_cpu.ndim == 5 and imgs_cpu.shape[0] == 1:
        imgs_cpu = imgs_cpu[0]
    if imgs_cpu.ndim != 4 or imgs_cpu.shape[0] != len(group.image_paths):
        raise ValueError("DA3 preprocessed view group must have shape (N,C,H,W)")
    image_batch = imgs_cpu[None].to(runner.config.device, non_blocking=True).float()
    output = model.forward(
        image_batch,
        extrinsics=None,
        intrinsics=None,
        export_feat_layers=[],
        infer_gs=False,
        use_ray_pose=False,
        ref_view_strategy=runner.config.ref_view_strategy,
    )
    predicted_c2w_batch = convert_da3_batched_extrinsics_to_c2w(
        _field(output, "extrinsics"),
        extrinsics_are_c2w=runner.config.extrinsics_are_c2w,
    )
    if predicted_c2w_batch.shape != (1, len(group.image_paths), 4, 4):
        raise ValueError(
            "DA3 batched c2w poses must have shape (1,N,4,4), got "
            f"{predicted_c2w_batch.shape}"
        )
    predicted_c2w = predicted_c2w_batch[0]
    if not np.all(np.isfinite(predicted_c2w)):
        raise ValueError("DA3 c2w poses must contain only finite values")
    return Da3ViewGroupResult(
        predicted_c2w=predicted_c2w,
        keyframe_indices=group.keyframe_indices,
        view_roles=group.view_roles,
    )


def run_da3_view_groups(runner: RealDa3Runner, groups: list[Da3ViewGroup]) -> list[Da3ViewGroupResult]:
    return [run_da3_view_group(runner, group) for group in groups]


def align_with_median_support_scale(
    *,
    da3_query_c2w: np.ndarray,
    da3_candidate_c2w: np.ndarray,
    da3_support_c2w: list[np.ndarray],
    odom_candidate_c2w: np.ndarray,
    odom_support_c2w: list[np.ndarray],
    baseline_epsilon: float = 1e-9,
) -> MedianScaleAlignment:
    if len(da3_support_c2w) != len(odom_support_c2w) or not da3_support_c2w:
        return _invalid_alignment("support_count_mismatch")

    r_align = odom_candidate_c2w[:3, :3] @ da3_candidate_c2w[:3, :3].T
    scales = []
    per_support = []
    for support_idx, (da3_support, odom_support) in enumerate(
        zip(da3_support_c2w, odom_support_c2w)
    ):
        da3_delta = da3_support[:3, 3] - da3_candidate_c2w[:3, 3]
        odom_delta = odom_support[:3, 3] - odom_candidate_c2w[:3, 3]
        da3_norm = float(np.linalg.norm(da3_delta))
        odom_norm = float(np.linalg.norm(odom_delta))
        if da3_norm <= baseline_epsilon or odom_norm <= baseline_epsilon:
            continue
        scale = odom_norm / da3_norm
        if not np.isfinite(scale) or scale <= 0.0:
            continue
        direction_error = direction_error_deg(r_align @ da3_delta, odom_delta)
        scales.append(scale)
        per_support.append(
            {
                "support_order": float(support_idx),
                "scale": float(scale),
                "da3_baseline_m": da3_norm,
                "odom_baseline_m": odom_norm,
                "direction_error_deg": float(direction_error),
            }
        )

    if not scales:
        return _invalid_alignment("no_valid_support_scale")

    scale = float(np.median(np.asarray(scales, dtype=np.float64)))
    scale_mad = float(np.median(np.abs(np.asarray(scales, dtype=np.float64) - scale)))
    aligned_query = apply_candidate_anchored_sim3(
        da3_query_c2w, da3_candidate_c2w, odom_candidate_c2w, r_align, scale
    )
    aligned_candidate = apply_candidate_anchored_sim3(
        da3_candidate_c2w, da3_candidate_c2w, odom_candidate_c2w, r_align, scale
    )

    support_errors = []
    direction_errors = []
    for da3_support, odom_support in zip(da3_support_c2w, odom_support_c2w):
        aligned_support = apply_candidate_anchored_sim3(
            da3_support, da3_candidate_c2w, odom_candidate_c2w, r_align, scale
        )
        support_errors.append(
            float(np.linalg.norm(aligned_support[:3, 3] - odom_support[:3, 3]))
        )
        direction_errors.append(
            direction_error_deg(
                aligned_support[:3, 3] - aligned_candidate[:3, 3],
                odom_support[:3, 3] - odom_candidate_c2w[:3, 3],
            )
        )
    rmse = float(np.sqrt(np.mean(np.square(support_errors))))
    max_direction = float(np.nanmax(np.asarray(direction_errors, dtype=np.float64)))
    return MedianScaleAlignment(
        valid=True,
        reason="",
        scale=scale,
        scale_mad=scale_mad,
        support_rmse_m=rmse,
        support_max_direction_error_deg=max_direction,
        aligned_query_c2w=aligned_query,
        aligned_candidate_c2w=aligned_candidate,
        per_support=per_support,
    )


def align_single_support_scale(
    *,
    da3_query_c2w: np.ndarray,
    da3_candidate_c2w: np.ndarray,
    da3_support_c2w: np.ndarray,
    odom_candidate_c2w: np.ndarray,
    odom_support_c2w: np.ndarray,
    baseline_epsilon: float = 1e-9,
) -> MedianScaleAlignment:
    return align_with_median_support_scale(
        da3_query_c2w=da3_query_c2w,
        da3_candidate_c2w=da3_candidate_c2w,
        da3_support_c2w=[da3_support_c2w],
        odom_candidate_c2w=odom_candidate_c2w,
        odom_support_c2w=[odom_support_c2w],
        baseline_epsilon=baseline_epsilon,
    )


def select_supports_by_da3_consistency(
    *,
    supports: list[SupportCandidate],
    predicted_c2w: np.ndarray,
    candidate_idx: int,
    odom_camera_by_idx: dict[int, np.ndarray],
    max_supports: int,
) -> tuple[list[SupportCandidate], dict[str, Any]]:
    if max_supports <= 0:
        raise ValueError("max_supports must be positive")
    if predicted_c2w.shape[0] < len(supports) + 2:
        raise ValueError(
            "predicted_c2w must contain query, candidate, and all support poses "
            f"({predicted_c2w.shape[0]} < {len(supports) + 2})"
        )
    if candidate_idx not in odom_camera_by_idx:
        raise ValueError(f"candidate_idx {candidate_idx} not found in odometry camera poses")

    alignment = align_with_median_support_scale(
        da3_query_c2w=predicted_c2w[0],
        da3_candidate_c2w=predicted_c2w[1],
        da3_support_c2w=[predicted_c2w[2 + idx] for idx in range(len(supports))],
        odom_candidate_c2w=odom_camera_by_idx[candidate_idx],
        odom_support_c2w=[odom_camera_by_idx[support.idx] for support in supports],
    )
    if not alignment.valid:
        return supports[:max_supports], {
            "selection_strategy": "da3_consistency_topk",
            "selection_reason": alignment.reason,
            "ranked_support_indices": [support.idx for support in supports],
            "ranked_support_scores": [],
        }

    scale = alignment.scale
    scored: list[tuple[float, int, SupportCandidate, dict[str, float]]] = []
    for order, (support, support_metric) in enumerate(zip(supports, alignment.per_support)):
        support_pose = predicted_c2w[2 + order]
        aligned_support = apply_candidate_anchored_sim3(
            support_pose,
            predicted_c2w[1],
            odom_camera_by_idx[candidate_idx],
            odom_camera_by_idx[candidate_idx][:3, :3] @ predicted_c2w[1][:3, :3].T,
            scale,
        )
        residual = float(
            np.linalg.norm(aligned_support[:3, 3] - odom_camera_by_idx[support.idx][:3, 3])
        )
        scale_error = abs(float(support_metric["scale"]) - scale) / max(abs(scale), 1e-9)
        direction_error = float(support_metric["direction_error_deg"])
        if not np.isfinite(direction_error):
            direction_error = 180.0
        score = residual + scale_error + direction_error / 45.0
        scored.append(
            (
                float(score),
                order,
                support,
                {
                    "score": float(score),
                    "residual_m": residual,
                    "scale_error_ratio": float(scale_error),
                    "direction_error_deg": direction_error,
                    "support_scale": float(support_metric["scale"]),
                },
            )
        )

    scored.sort(key=lambda item: (item[0], item[1]))
    selected = [item[2] for item in scored[:max_supports]]
    metrics = {
        "selection_strategy": "da3_consistency_topk",
        "selection_reason": "",
        "ranking_pool_scale": alignment.scale,
        "ranking_pool_scale_mad": alignment.scale_mad,
        "ranking_pool_support_rmse_m": alignment.support_rmse_m,
        "ranked_support_indices": [item[2].idx for item in scored],
        "ranked_support_scores": [item[3] for item in scored],
    }
    return selected, metrics


def aggregate_backend_factors_by_translation_median(factors: list[np.ndarray]) -> np.ndarray:
    if not factors:
        raise ValueError("factors must be non-empty")
    factor_array = [np.asarray(factor, dtype=np.float64) for factor in factors]
    translations = np.stack([factor[:3, 3] for factor in factor_array], axis=0)
    norms = np.linalg.norm(translations, axis=1)
    median_norm = float(np.median(norms))
    median_translation = np.median(translations, axis=0)
    if float(np.linalg.norm(median_translation)) <= 1e-12:
        median_index = int(np.argsort(np.abs(norms - median_norm))[0])
        median_translation = translations[median_index]

    rotation_index = int(np.argsort(np.abs(norms - median_norm))[0])
    aggregate = np.array(factor_array[rotation_index], dtype=np.float64, copy=True)
    aggregate[:3, 3] = median_translation
    return aggregate


def _invalid_alignment(reason: str) -> MedianScaleAlignment:
    return MedianScaleAlignment(
        valid=False,
        reason=reason,
        scale=float("nan"),
        scale_mad=float("nan"),
        support_rmse_m=float("nan"),
        support_max_direction_error_deg=float("nan"),
        aligned_query_c2w=np.eye(4, dtype=np.float64),
        aligned_candidate_c2w=np.eye(4, dtype=np.float64),
        per_support=[],
    )


def apply_candidate_anchored_sim3(
    pose: np.ndarray,
    da3_candidate_c2w: np.ndarray,
    odom_candidate_c2w: np.ndarray,
    r_align: np.ndarray,
    scale: float,
) -> np.ndarray:
    aligned = np.eye(4, dtype=np.float64)
    aligned[:3, :3] = r_align @ pose[:3, :3]
    aligned[:3, 3] = odom_candidate_c2w[:3, 3] + scale * (
        r_align @ (pose[:3, 3] - da3_candidate_c2w[:3, 3])
    )
    return aligned


def camera_pose_from_backend_pose(world_lidar: np.ndarray, t_camera_lidar: np.ndarray) -> np.ndarray:
    return world_lidar @ invert_transform(t_camera_lidar)


def camera_relative_to_backend_relative(
    camera_relative: np.ndarray, t_camera_lidar: np.ndarray
) -> np.ndarray:
    return invert_transform(t_camera_lidar) @ camera_relative @ t_camera_lidar


def pose_between(a_c2w: np.ndarray, b_c2w: np.ndarray) -> np.ndarray:
    return invert_transform(a_c2w) @ b_c2w


def direction_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm <= 0.0 or b_norm <= 0.0:
        return float("nan")
    cosine = float(np.clip(np.dot(a / a_norm, b / b_norm), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def run_triplets_in_chunks(runner, triplets: list[Any], batch_size: int) -> list[Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    results = []
    for start in range(0, len(triplets), batch_size):
        results.extend(runner.run_triplets(triplets[start : start + batch_size]))
    return results


def interp_positions(timestamps: list[float], poses: dict[int, np.ndarray], query_times: list[float]) -> np.ndarray:
    pose_times = np.asarray(timestamps, dtype=np.float64)
    positions = np.stack([poses[idx][:3, 3] for idx in sorted(poses)], axis=0)
    output = np.empty((len(query_times), 3), dtype=np.float64)
    query = np.asarray(query_times, dtype=np.float64)
    for axis in range(3):
        output[:, axis] = np.interp(query, pose_times, positions[:, axis])
    return output


def umeyama_positions(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, float]:
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    covariance = (dst_centered.T @ src_centered) / src.shape[0]
    u_matrix, singular_values, vt_matrix = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(u_matrix @ vt_matrix) < 0.0:
        correction[-1, -1] = -1.0
    rotation = u_matrix @ correction @ vt_matrix
    variance = float(np.mean(np.sum(src_centered * src_centered, axis=1)))
    scale = float(np.sum(singular_values * np.diag(correction)) / variance)
    translation = dst_mean - scale * (rotation @ src_mean)
    aligned = (scale * (rotation @ src.T)).T + translation
    rmse = float(np.sqrt(np.mean(np.sum((aligned - dst) ** 2, axis=1))))
    return scale, rotation, translation, rmse


def write_support_visualization(
    *,
    output_dir: Path,
    query_idx: int,
    candidate_idx: int,
    support_count: int,
    strategy: str,
    image_by_idx: dict[int, Path],
    selected_supports: list[SupportCandidate],
    pool_supports: list[SupportCandidate],
    extra_metrics: dict[str, Any],
) -> Path:
    from PIL import Image, ImageDraw

    output_dir.mkdir(parents=True, exist_ok=True)
    view_indices = [query_idx, candidate_idx, *[support.idx for support in selected_supports]]
    roles = ["query", "candidate", *[f"support_{idx + 1}" for idx in range(len(selected_supports))]]
    tile_w, tile_h = 220, 170
    header_h = 34
    gap = 8
    tiles = []
    for role, idx in zip(roles, view_indices):
        with Image.open(image_by_idx[idx]) as image:
            image = image.convert("RGB")
            image.thumbnail((tile_w, tile_h - header_h), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (tile_w, tile_h), color=(245, 245, 245))
            x = (tile_w - image.width) // 2
            y = header_h + (tile_h - header_h - image.height) // 2
            tile.paste(image, (x, y))
        draw = ImageDraw.Draw(tile)
        baseline = ""
        for support in selected_supports:
            if support.idx == idx:
                baseline = f" b={support.baseline_m:.2f}m"
                break
        draw.text((6, 6), f"{role}: kf {idx}{baseline}", fill=(0, 0, 0))
        tiles.append(tile)

    cols = min(4, len(tiles))
    rows = int(math.ceil(len(tiles) / cols))
    summary_h = 46
    canvas = Image.new(
        "RGB",
        (cols * tile_w + (cols - 1) * gap, rows * tile_h + (rows - 1) * gap + summary_h),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)
    selected_ids = " ".join(str(support.idx) for support in selected_supports)
    pool_ids = " ".join(str(support.idx) for support in pool_supports)
    rel_error = extra_metrics.get("relative_error_pct", "nan")
    dir_error = extra_metrics.get("direction_error_deg", "nan")
    scale_mad = extra_metrics.get("sim3_scale_mad", extra_metrics.get("scale_mad", "nan"))
    draw.text(
        (6, 5),
        f"{strategy} q{query_idx} c{candidate_idx} K={support_count} "
        f"rel={_fmt_float(rel_error)}% dir={_fmt_float(dir_error)} scale_mad={_fmt_float(scale_mad)}",
        fill=(0, 0, 0),
    )
    draw.text((6, 24), f"selected: {selected_ids} | pool: {pool_ids}", fill=(0, 0, 0))
    for tile_idx, tile in enumerate(tiles):
        row, col = divmod(tile_idx, cols)
        canvas.paste(tile, (col * (tile_w + gap), summary_h + row * (tile_h + gap)))

    png_path = output_dir / (
        f"{strategy}_q{query_idx:06d}_c{candidate_idx:06d}_k{support_count:02d}.png"
    )
    canvas.save(png_path)
    manifest = {
        "strategy": strategy,
        "query_idx": query_idx,
        "candidate_idx": candidate_idx,
        "support_count": support_count,
        "selected_supports": [
            {"idx": support.idx, "baseline_m": support.baseline_m}
            for support in selected_supports
        ],
        "pool_supports": [
            {"idx": support.idx, "baseline_m": support.baseline_m}
            for support in pool_supports
        ],
        "extra_metrics": extra_metrics,
    }
    png_path.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=True),
        encoding="utf-8",
    )
    return png_path


def _fmt_float(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "nan"
    return f"{number:.1f}"


def main() -> int:
    args = parse_args()
    support_mode = args.support_selection_strategy or args.support_mode
    temporal_stride_pattern = parse_int_pattern(args.temporal_stride_pattern)
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else run_dir / "multisupport_sim3_sweep"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = parse_pairs(args.pairs) if args.pairs else list(DEFAULT_PAIRS)
    keyframe_timestamps, odom_lidar_by_idx = load_tum_poses(run_dir / "trajectory_keyframes.txt")
    gt_timestamps, gt_by_idx = load_tum_poses(Path(args.gt_path))
    t_camera_lidar = np.asarray([float(v) for v in args.t_camera_lidar.split(",")]).reshape(4, 4)
    odom_camera_by_idx = {
        idx: camera_pose_from_backend_pose(pose, t_camera_lidar)
        for idx, pose in odom_lidar_by_idx.items()
    }

    support_by_pair: dict[tuple[int, int], list[SupportCandidate]] = {}
    pool_by_pair: dict[tuple[int, int], list[SupportCandidate]] = {}
    needed_indices = set()
    for query_idx, candidate_idx in pairs:
        pool_mode = args.consistency_pool_mode if support_mode == "da3_consistency_topk" else support_mode
        supports = select_supports(
            mode=pool_mode,
            candidate_idx=candidate_idx,
            query_idx=query_idx,
            odom_by_idx=odom_lidar_by_idx,
            max_supports=max(args.max_supports, args.support_pool_size),
            exclude_recent_from_query=args.exclude_recent_from_query,
            min_baseline_m=args.min_support_baseline_m,
            temporal_stride_pattern=temporal_stride_pattern,
        )[: args.support_pool_size]
        if len(supports) < args.max_supports:
            raise RuntimeError(
                f"Pair q{query_idx} c{candidate_idx} only has {len(supports)} supports; "
                f"need {args.max_supports}"
            )
        pool_by_pair[(query_idx, candidate_idx)] = supports
        support_by_pair[(query_idx, candidate_idx)] = supports
        needed_indices.add(query_idx)
        needed_indices.add(candidate_idx)
        needed_indices.update(support.idx for support in supports)

    image_records = extract_images_from_bag(
        bag_path=Path(args.bag),
        image_topic=args.image_topic,
        keyframe_timestamps={idx: keyframe_timestamps[idx] for idx in needed_indices},
        output_dir=output_dir / "images",
        tolerance_sec=args.image_sync_tolerance_sec,
    )

    image_by_idx = {idx: image_record.path for idx, image_record in image_records.items()}
    runner = RealDa3Runner(
        RealDa3RunnerConfig(
            model_name=args.model_name,
            device=args.device,
            process_res=args.process_res,
            process_res_method=args.process_res_method,
            triplet_batch_size=args.triplet_batch_size,
            local_files_only=args.local_files_only,
            extrinsics_are_c2w=False,
        )
    )

    selection_meta_by_group: dict[tuple[int, int, int], dict[str, Any]] = {}
    if support_mode == "da3_consistency_topk":
        pool_groups = []
        pool_meta = []
        for query_idx, candidate_idx in pairs:
            pool = pool_by_pair[(query_idx, candidate_idx)]
            pool_groups.append(
                build_da3_view_group(
                    query_idx=query_idx,
                    candidate_idx=candidate_idx,
                    supports=pool,
                    image_by_idx=image_by_idx,
                )
            )
            pool_meta.append((query_idx, candidate_idx))
        pool_results = run_da3_view_groups(runner, pool_groups)
        pool_predicted_by_pair = {
            meta: result.predicted_c2w for meta, result in zip(pool_meta, pool_results)
        }
        for query_idx, candidate_idx in pairs:
            pool = pool_by_pair[(query_idx, candidate_idx)]
            selected_all, metrics = select_supports_by_da3_consistency(
                supports=pool,
                predicted_c2w=pool_predicted_by_pair[(query_idx, candidate_idx)],
                candidate_idx=candidate_idx,
                odom_camera_by_idx=odom_camera_by_idx,
                max_supports=args.max_supports,
            )
            support_by_pair[(query_idx, candidate_idx)] = selected_all
            for support_count in range(1, args.max_supports + 1):
                selection_meta_by_group[(query_idx, candidate_idx, support_count)] = metrics

    groups = []
    group_meta = []
    for query_idx, candidate_idx in pairs:
        supports = support_by_pair[(query_idx, candidate_idx)][: args.max_supports]
        for support_count in range(1, args.max_supports + 1):
            selected = supports[:support_count]
            groups.append(
                build_da3_view_group(
                    query_idx=query_idx,
                    candidate_idx=candidate_idx,
                    supports=selected,
                    image_by_idx=image_by_idx,
                )
            )
            group_meta.append((query_idx, candidate_idx, support_count))

    da3_results = run_da3_view_groups(runner, groups)
    da3_by_group = {
        meta: result.predicted_c2w for meta, result in zip(group_meta, da3_results)
    }

    keyframe_gt_positions = interp_positions(
        gt_timestamps,
        gt_by_idx,
        [keyframe_timestamps[idx] for idx in sorted(odom_lidar_by_idx)],
    )
    odom_positions = np.stack(
        [odom_lidar_by_idx[idx][:3, 3] for idx in sorted(odom_lidar_by_idx)], axis=0
    )
    gt_to_odom_scale, gt_to_odom_r, gt_to_odom_t, gt_to_odom_rmse = umeyama_positions(
        keyframe_gt_positions, odom_positions
    )

    records: list[dict[str, Any]] = []
    for query_idx, candidate_idx in pairs:
        supports = support_by_pair[(query_idx, candidate_idx)][: args.max_supports]
        pool_supports = pool_by_pair[(query_idx, candidate_idx)]
        gt_pair_positions = interp_positions(
            gt_timestamps,
            gt_by_idx,
            [keyframe_timestamps[query_idx], keyframe_timestamps[candidate_idx]],
        )
        gt_pair_odom = (gt_to_odom_scale * (gt_to_odom_r @ gt_pair_positions.T)).T + gt_to_odom_t
        gt_delta_world = gt_pair_odom[1] - gt_pair_odom[0]
        gt_distance = float(np.linalg.norm(gt_delta_world))
        gt_direction_backend = odom_lidar_by_idx[query_idx][:3, :3].T @ gt_delta_world

        for support_count in range(1, args.max_supports + 1):
            selected = supports[:support_count]
            selection_meta = selection_meta_by_group.get(
                (query_idx, candidate_idx, support_count), {}
            )
            predicted = da3_by_group[(query_idx, candidate_idx, support_count)]
            raw_camera_relative = pose_between(predicted[0], predicted[1])
            raw_backend_relative = camera_relative_to_backend_relative(
                raw_camera_relative,
                t_camera_lidar,
            )
            alignment = align_with_median_support_scale(
                da3_query_c2w=predicted[0],
                da3_candidate_c2w=predicted[1],
                da3_support_c2w=[predicted[2 + idx] for idx in range(support_count)],
                odom_candidate_c2w=odom_camera_by_idx[candidate_idx],
                odom_support_c2w=[odom_camera_by_idx[support.idx] for support in selected],
            )
            if alignment.valid:
                aligned_camera_relative = pose_between(
                    alignment.aligned_query_c2w,
                    alignment.aligned_candidate_c2w,
                )
                backend_relative = camera_relative_to_backend_relative(
                    aligned_camera_relative,
                    t_camera_lidar,
                )
                backend_norm = float(np.linalg.norm(backend_relative[:3, 3]))
                direction_error = direction_error_deg(
                    backend_relative[:3, 3],
                    gt_direction_backend,
                )
                signed_error = backend_norm - gt_distance
                relative_error_pct = 100.0 * signed_error / gt_distance
                scale = alignment.scale
                scale_mad = alignment.scale_mad
                support_rmse = alignment.support_rmse_m
                support_max_direction = alignment.support_max_direction_error_deg
                per_support = [
                    {
                        "support_idx": float(selected[idx].idx),
                        "support_baseline_m": float(selected[idx].baseline_m),
                        **alignment.per_support[idx],
                    }
                    for idx in range(len(alignment.per_support))
                ]
                valid = True
                reason = ""
            else:
                backend_norm = float("nan")
                direction_error = float("nan")
                signed_error = float("nan")
                relative_error_pct = float("nan")
                scale = float("nan")
                scale_mad = float("nan")
                support_rmse = float("nan")
                support_max_direction = float("nan")
                per_support = []
                valid = False
                reason = alignment.reason

            record = {
                "query_idx": query_idx,
                "candidate_idx": candidate_idx,
                "support_count": support_count,
                "support_strategy": support_mode,
                "support_pool_mode": (
                    args.consistency_pool_mode
                    if support_mode == "da3_consistency_topk"
                    else support_mode
                ),
                "support_indices": [support.idx for support in selected],
                "support_baselines_m": [support.baseline_m for support in selected],
                "pool_support_indices": [support.idx for support in pool_supports],
                "pool_support_baselines_m": [support.baseline_m for support in pool_supports],
                "valid": valid,
                "reason": reason,
                "sim3_scale": scale,
                "sim3_scale_mad": scale_mad,
                "support_rmse_m": support_rmse,
                "support_max_direction_error_deg": support_max_direction,
                "raw_backend_norm_m": float(np.linalg.norm(raw_backend_relative[:3, 3])),
                "gt_distance_m": gt_distance,
                "backend_norm_m": backend_norm,
                "signed_error_m": signed_error,
                "relative_error_pct": relative_error_pct,
                "direction_error_deg": direction_error,
                "per_support": per_support,
                "selection_meta": selection_meta,
            }
            if args.save_visualizations:
                viz_path = write_support_visualization(
                    output_dir=output_dir / "support_visualizations",
                    query_idx=query_idx,
                    candidate_idx=candidate_idx,
                    support_count=support_count,
                    strategy=support_mode,
                    image_by_idx=image_by_idx,
                    selected_supports=selected,
                    pool_supports=pool_supports,
                    extra_metrics=record,
                )
                record["support_visualization"] = str(viz_path)
            records.append(record)

    summary = {
        "run_dir": str(run_dir),
        "bag": str(Path(args.bag)),
        "image_topic": args.image_topic,
        "gt_path": str(Path(args.gt_path)),
        "output_dir": str(output_dir),
        "config": {
            "max_supports": args.max_supports,
            "support_pool_size": args.support_pool_size,
            "support_mode": support_mode,
            "support_pool_mode": args.consistency_pool_mode,
            "temporal_stride_pattern": temporal_stride_pattern,
            "save_visualizations": args.save_visualizations,
            "min_support_baseline_m": args.min_support_baseline_m,
            "exclude_recent_from_query": args.exclude_recent_from_query,
            "process_res": args.process_res,
            "process_res_method": args.process_res_method,
            "triplet_batch_size": args.triplet_batch_size,
            "gt_to_odom_scale": gt_to_odom_scale,
            "gt_to_odom_rmse_m": gt_to_odom_rmse,
        },
        "records": records,
    }
    (output_dir / "multisupport_sim3_sweep_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True),
        encoding="utf-8",
    )
    write_csv(output_dir / "multisupport_sim3_sweep.csv", records)
    print_table(records)
    print(f"wrote {output_dir / 'multisupport_sim3_sweep_summary.json'}")
    print(f"wrote {output_dir / 'multisupport_sim3_sweep.csv'}")
    return 0


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "query_idx",
        "candidate_idx",
        "support_count",
        "support_strategy",
        "support_pool_mode",
        "valid",
        "sim3_scale",
        "sim3_scale_mad",
        "support_rmse_m",
        "support_max_direction_error_deg",
        "raw_backend_norm_m",
        "gt_distance_m",
        "backend_norm_m",
        "signed_error_m",
        "relative_error_pct",
        "direction_error_deg",
        "support_indices",
        "support_baselines_m",
        "pool_support_indices",
        "support_visualization",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {field: record.get(field) for field in fields}
            row["support_indices"] = " ".join(str(v) for v in record["support_indices"])
            row["support_baselines_m"] = " ".join(
                f"{float(v):.6f}" for v in record["support_baselines_m"]
            )
            row["pool_support_indices"] = " ".join(
                str(v) for v in record.get("pool_support_indices", [])
            )
            writer.writerow(row)


def print_table(records: list[dict[str, Any]]) -> None:
    print(
        "pair support strategy scale scale_mad gt_norm est_norm rel_err_pct dir_err_deg "
        "support_rmse support_indices"
    )
    for record in records:
        pair = f"{record['query_idx']}-{record['candidate_idx']}"
        print(
            f"{pair:>7} {record['support_count']:>2d} "
            f"{record.get('support_strategy', ''):>18s} "
            f"{record['sim3_scale']:>8.3f} {record['sim3_scale_mad']:>8.3f} "
            f"{record['gt_distance_m']:>8.3f} {record['backend_norm_m']:>8.3f} "
            f"{record['relative_error_pct']:>9.1f} {record['direction_error_deg']:>9.1f} "
            f"{record['support_rmse_m']:>9.3f} "
            f"{' '.join(str(v) for v in record['support_indices'])}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
