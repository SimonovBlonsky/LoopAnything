#!/usr/bin/env python3
"""Offline retest for AsterSLAM runtime DA3 LoopAnything accepted loops."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from robust_loop_verifier.da3_runner import RealDa3Runner, RealDa3RunnerConfig, build_da3_triplet
from robust_loop_verifier.geometry import (
    invert_transform,
    pose_between,
    rotation_matrix_from_quat_xyzw,
    se3_log,
)
from robust_loop_verifier.pgo import trajectory_deformation_rmse


DEFAULT_PAIRS = ((52, 2), (53, 2), (54, 2), (119, 88))
NSEC_PER_SEC = 1_000_000_000


@dataclass(frozen=True)
class ImageRecord:
    keyframe_idx: int
    target_timestamp: float
    image_timestamp: float
    sync_delta_sec: float
    path: Path


@dataclass(frozen=True)
class LoopFactor:
    from_idx: int
    to_idx: int
    pose_between: np.ndarray
    noise_sigmas: tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class SafetyPgoRetestResult:
    converged: bool
    failure_reason: str
    original_poses: list[np.ndarray]
    optimized_poses: list[np.ndarray]
    error_before: float
    error_after: float
    prior_chi2_before: float
    prior_chi2_after: float
    loop_chi2_after: float
    odom_chi2_before: float
    odom_chi2_after: float
    odom_strain_chi2_after: float
    factor_count: int
    history_loop_factor_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retest selected AsterSLAM runtime accepted DA3 loops with offline "
            "LoopAnything DA3 forward, Sim3 alignment, PGO, and visualization."
        )
    )
    parser.add_argument("--run-dir", required=True, help="AsterSLAM runtime run directory.")
    parser.add_argument("--bag", default="/data/datasets/NTU-VIRAL/data/eee_01/eee_01.bag")
    parser.add_argument("--image-topic", default="/left/image_raw")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--pairs", nargs="*", default=[], help="Pairs as q:c. Defaults to runtime accepted pairs.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-name", default="depth-anything/DA3-LARGE-1.1")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--process-res-method", default="upper_bound_resize")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-sync-tolerance-sec", type=float, default=0.15)
    parser.add_argument("--max-nodes", type=int, default=80)
    parser.add_argument("--min-nodes", type=int, default=4)
    parser.add_argument("--include-accepted-loops", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-deformation-rmse", type=float, default=0.5)
    parser.add_argument("--max-pgo-error-per-factor", type=float, default=1.0)
    parser.add_argument("--max-odom-strain-chi2", type=float, default=50.0)
    parser.add_argument("--max-loop-chi2", type=float, default=25.0)
    parser.add_argument("--loop-rotation-sigma", type=float, default=0.3)
    parser.add_argument("--loop-translation-sigma", type=float, default=2.0)
    parser.add_argument(
        "--t-camera-lidar",
        default=(
            "0.0218308,0.99976,-0.00201407,0.122993,"
            "-0.0131205,0.00230088,0.999911,0.0398643,"
            "0.999676,-0.0218025,0.0131676,-0.0577101,"
            "0.0,0.0,0.0,1.0"
        ),
    )
    return parser.parse_args()


def load_tum_poses(path: Path) -> tuple[list[int], list[float], dict[int, np.ndarray]]:
    indices: list[int] = []
    timestamps: list[float] = []
    poses: dict[int, np.ndarray] = {}
    for row, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 8:
            continue
        timestamp = float(fields[0])
        translation = np.asarray([float(v) for v in fields[1:4]], dtype=np.float64)
        quat = np.asarray([float(v) for v in fields[4:8]], dtype=np.float64)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = rotation_matrix_from_quat_xyzw(quat)
        pose[:3, 3] = translation
        indices.append(row)
        timestamps.append(timestamp)
        poses[row] = pose
    return indices, timestamps, poses


def parse_pairs(values: Iterable[str]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for value in values:
        if ":" not in value:
            raise ValueError(f"Pair must use q:c format, got {value!r}")
        query, candidate = value.split(":", 1)
        pairs.append((int(query), int(candidate)))
    return pairs


def parse_matrix_label(value: str) -> np.ndarray:
    vals = [float(v) for v in value.split(",")]
    if len(vals) != 16:
        raise ValueError(f"matrix label must contain 16 values, got {len(vals)}")
    return np.asarray(vals, dtype=np.float64).reshape(4, 4)


def matrix_to_list(matrix: np.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(matrix, dtype=np.float64).reshape(-1)]


def load_runtime_rows(loop_log: Path) -> list[dict[str, Any]]:
    rows = []
    for line in loop_log.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def accepted_runtime_pairs(rows: list[dict[str, Any]]) -> list[tuple[int, int]]:
    return [
        (int(row["query_idx"]), int(row["candidate_idx"]))
        for row in rows
        if row.get("status") == "accepted"
    ]


def selected_runtime_rows(
    rows: list[dict[str, Any]], pairs: list[tuple[int, int]]
) -> list[dict[str, Any]]:
    by_pair = {
        (int(row["query_idx"]), int(row["candidate_idx"])): row
        for row in rows
        if row.get("status") == "accepted"
    }
    selected = []
    for pair in pairs:
        if pair not in by_pair:
            raise ValueError(f"Accepted runtime pair {pair} was not found in loop_log")
        selected.append(by_pair[pair])
    return selected


def stamp_to_nsec(stamp: float) -> int:
    return int(round(float(stamp) * NSEC_PER_SEC))


def extract_images_from_bag(
    *,
    bag_path: Path,
    image_topic: str,
    keyframe_timestamps: dict[int, float],
    output_dir: Path,
    tolerance_sec: float,
) -> dict[int, ImageRecord]:
    try:
        import cv2
        import rosbag
        from cv_bridge import CvBridge
    except (ModuleNotFoundError, OSError):
        return extract_images_from_bag_with_system_python(
            bag_path=bag_path,
            image_topic=image_topic,
            keyframe_timestamps=keyframe_timestamps,
            output_dir=output_dir,
            tolerance_sec=tolerance_sec,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    bridge = CvBridge()
    pending = {
        idx: {
            "timestamp": ts,
            "best_delta": tolerance_sec,
            "best_stamp": None,
            "best_image": None,
        }
        for idx, ts in keyframe_timestamps.items()
    }
    if not pending:
        return {}

    min_ts = min(keyframe_timestamps.values()) - tolerance_sec
    max_ts = max(keyframe_timestamps.values()) + tolerance_sec
    with rosbag.Bag(str(bag_path), "r") as bag:
        for _, msg, _ in bag.read_messages(topics=[image_topic]):
            stamp = msg.header.stamp.to_sec()
            if stamp < min_ts:
                continue
            if stamp > max_ts:
                break
            for item in pending.values():
                delta = abs(stamp - item["timestamp"])
                if delta <= item["best_delta"]:
                    item["best_delta"] = delta
                    item["best_stamp"] = stamp
                    item["best_image"] = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    records: dict[int, ImageRecord] = {}
    for idx, item in pending.items():
        if item["best_image"] is None or item["best_stamp"] is None:
            raise RuntimeError(
                f"No {image_topic} image within {tolerance_sec:.3f}s for keyframe {idx} "
                f"at {item['timestamp']:.9f}"
            )
        path = output_dir / f"{idx:06d}.png"
        if not cv2.imwrite(str(path), item["best_image"]):
            raise RuntimeError(f"Failed to write {path}")
        records[idx] = ImageRecord(
            keyframe_idx=idx,
            target_timestamp=float(item["timestamp"]),
            image_timestamp=float(item["best_stamp"]),
            sync_delta_sec=float(item["best_delta"]),
            path=path,
        )
    return records


def extract_images_from_bag_with_system_python(
    *,
    bag_path: Path,
    image_topic: str,
    keyframe_timestamps: dict[int, float],
    output_dir: Path,
    tolerance_sec: float,
) -> dict[int, ImageRecord]:
    output_dir.mkdir(parents=True, exist_ok=True)
    request_path = output_dir / "_rosbag_extract_request.json"
    manifest_path = output_dir / "_rosbag_extract_manifest.json"
    helper_path = output_dir / "_rosbag_extract_helper.py"
    request = {
        "bag": str(bag_path),
        "image_topic": image_topic,
        "keyframe_timestamps": {str(k): float(v) for k, v in keyframe_timestamps.items()},
        "output_dir": str(output_dir),
        "tolerance_sec": float(tolerance_sec),
    }
    request_path.write_text(json.dumps(request, sort_keys=True), encoding="utf-8")
    helper_path.write_text(
        r'''
import json
import sys
from pathlib import Path

import cv2
import rosbag
from cv_bridge import CvBridge


def main():
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    manifest_path = Path(sys.argv[2])
    bag_path = request["bag"]
    image_topic = request["image_topic"]
    tolerance = float(request["tolerance_sec"])
    output_dir = Path(request["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = {
        int(idx): {
            "timestamp": float(timestamp),
            "best_delta": tolerance,
            "best_stamp": None,
            "best_image": None,
        }
        for idx, timestamp in request["keyframe_timestamps"].items()
    }
    if not pending:
        manifest_path.write_text("{}", encoding="utf-8")
        return

    min_ts = min(item["timestamp"] for item in pending.values()) - tolerance
    max_ts = max(item["timestamp"] for item in pending.values()) + tolerance
    bridge = CvBridge()
    with rosbag.Bag(bag_path, "r") as bag:
        for _, msg, _ in bag.read_messages(topics=[image_topic]):
            stamp = msg.header.stamp.to_sec()
            if stamp < min_ts:
                continue
            if stamp > max_ts:
                break
            for item in pending.values():
                delta = abs(stamp - item["timestamp"])
                if delta <= item["best_delta"]:
                    item["best_delta"] = delta
                    item["best_stamp"] = stamp
                    item["best_image"] = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    records = {}
    for idx, item in pending.items():
        if item["best_image"] is None or item["best_stamp"] is None:
            raise RuntimeError(
                f"No {image_topic} image within {tolerance:.3f}s for keyframe {idx} "
                f"at {item['timestamp']:.9f}"
            )
        path = output_dir / f"{idx:06d}.png"
        if not cv2.imwrite(str(path), item["best_image"]):
            raise RuntimeError(f"Failed to write {path}")
        records[str(idx)] = {
            "keyframe_idx": idx,
            "target_timestamp": item["timestamp"],
            "image_timestamp": item["best_stamp"],
            "sync_delta_sec": item["best_delta"],
            "path": str(path),
        }
    manifest_path.write_text(json.dumps(records, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
'''.lstrip(),
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = ":".join(
        [
            "/opt/ros/noetic/lib/python3/dist-packages",
            "/usr/lib/python3/dist-packages",
            env.get("PYTHONPATH", ""),
        ]
    )
    subprocess.run(
        ["/usr/bin/python3", str(helper_path), str(request_path), str(manifest_path)],
        check=True,
        env=env,
    )
    raw_records = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        int(idx): ImageRecord(
            keyframe_idx=int(record["keyframe_idx"]),
            target_timestamp=float(record["target_timestamp"]),
            image_timestamp=float(record["image_timestamp"]),
            sync_delta_sec=float(record["sync_delta_sec"]),
            path=Path(record["path"]),
        )
        for idx, record in raw_records.items()
    }


def camera_pose_from_backend_pose(world_lidar: np.ndarray, t_camera_lidar: np.ndarray) -> np.ndarray:
    return world_lidar @ invert_transform(t_camera_lidar)


def query_camera_to_candidate_camera(query_c2w: np.ndarray, candidate_c2w: np.ndarray) -> np.ndarray:
    return invert_transform(query_c2w) @ candidate_c2w


def camera_relative_to_backend_relative(
    camera_relative: np.ndarray, t_camera_lidar: np.ndarray
) -> np.ndarray:
    return invert_transform(t_camera_lidar) @ camera_relative @ t_camera_lidar


def project_rotation_to_so3(rotation: np.ndarray) -> np.ndarray:
    u_matrix, _, vt_matrix = np.linalg.svd(np.asarray(rotation, dtype=np.float64))
    projected = u_matrix @ vt_matrix
    if np.linalg.det(projected) < 0.0:
        u_matrix[:, -1] *= -1.0
        projected = u_matrix @ vt_matrix
    return projected


def project_transform_to_se3(transform: np.ndarray) -> np.ndarray:
    projected = np.asarray(transform, dtype=np.float64).copy()
    projected[:3, :3] = project_rotation_to_so3(projected[:3, :3])
    projected[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return projected


def align_pose_prior(
    predicted_c2w: np.ndarray,
    prior_candidate_support_c2w: list[np.ndarray],
    *,
    min_scale: float = 0.05,
    max_scale: float = 20.0,
    max_alignment_rmse: float = 2.0,
    max_direction_error_deg: float = 45.0,
) -> tuple[bool, str, dict[str, float], np.ndarray | None, np.ndarray | None]:
    pred_query, pred_candidate, pred_support = predicted_c2w[:3]
    prior_candidate, prior_support = prior_candidate_support_c2w[:2]
    pred_delta = pred_support[:3, 3] - pred_candidate[:3, 3]
    prior_delta = prior_support[:3, 3] - prior_candidate[:3, 3]
    pred_norm = float(np.linalg.norm(pred_delta))
    prior_norm = float(np.linalg.norm(prior_delta))
    metrics: dict[str, float] = {"sim3_support_count": 1.0}
    if pred_norm <= 1e-12 or prior_norm <= 1e-12:
        return False, "sim3_alignment_failed", metrics, None, None
    scale = prior_norm / pred_norm
    metrics["sim3_scale"] = float(scale)
    if not np.isfinite(scale) or scale < min_scale or scale > max_scale:
        return False, "sim3_scale_out_of_range", metrics, None, None

    r_align = prior_candidate[:3, :3] @ pred_candidate[:3, :3].T
    t_align = prior_candidate[:3, 3] - scale * (r_align @ pred_candidate[:3, 3])

    def apply_sim3(pose: np.ndarray) -> np.ndarray:
        aligned = np.eye(4, dtype=np.float64)
        aligned[:3, :3] = r_align @ pose[:3, :3]
        aligned[:3, 3] = scale * (r_align @ pose[:3, 3]) + t_align
        return aligned

    aligned_query = apply_sim3(pred_query)
    aligned_candidate = apply_sim3(pred_candidate)
    aligned_support = apply_sim3(pred_support)
    residual = aligned_support[:3, 3] - prior_support[:3, 3]
    rmse = float(np.linalg.norm(residual))
    aligned_delta = aligned_support[:3, 3] - aligned_candidate[:3, 3]
    direction_error = direction_error_deg(aligned_delta, prior_delta)
    metrics["sim3_alignment_rmse"] = rmse
    metrics["sim3_direction_error_deg"] = direction_error
    if not np.isfinite(direction_error) or direction_error > max_direction_error_deg:
        return False, "sim3_direction_error_too_high", metrics, None, None
    if not np.isfinite(rmse) or rmse > max_alignment_rmse:
        return False, "sim3_alignment_residual_too_high", metrics, None, None
    return True, "", metrics, aligned_query, aligned_candidate


def direction_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return float("nan")
    dot = float(np.clip(np.dot(a / na, b / nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def select_local_nodes(
    candidate_idx: int,
    query_idx: int,
    *,
    cache_order: list[int],
    max_nodes: int,
) -> list[int]:
    first_idx = min(candidate_idx, query_idx)
    last_idx = max(candidate_idx, query_idx)
    nodes = [idx for idx in cache_order if first_idx <= idx <= last_idx]
    if len(nodes) > max_nodes:
        keep = max_nodes
        last = len(nodes) - 1
        sampled = []
        for i in range(keep):
            source = last if i == keep - 1 else (i * last + (keep - 1) // 2) // (keep - 1)
            sampled.append(nodes[source])
        nodes = sampled
    return nodes


def _to_gtsam_pose3(gtsam_module, transform: np.ndarray):
    transform = project_transform_to_se3(transform)
    return gtsam_module.Pose3(gtsam_module.Rot3(transform[:3, :3]), transform[:3, 3])


def _between_chi2(
    left: np.ndarray,
    right: np.ndarray,
    measured: np.ndarray,
    sigmas: tuple[float, float, float, float, float, float],
) -> float:
    predicted = pose_between(project_transform_to_se3(left), project_transform_to_se3(right))
    residual_transform = invert_transform(project_transform_to_se3(measured)) @ predicted
    residual = se3_log(project_transform_to_se3(residual_transform))
    sigma_array = np.asarray(sigmas, dtype=np.float64)
    return float(np.dot(residual / sigma_array, residual / sigma_array))


def _prior_chi2(
    pose: np.ndarray,
    measured: np.ndarray,
    sigmas: tuple[float, float, float, float, float, float],
) -> float:
    residual_transform = invert_transform(project_transform_to_se3(measured)) @ project_transform_to_se3(pose)
    residual = se3_log(project_transform_to_se3(residual_transform))
    sigma_array = np.asarray(sigmas, dtype=np.float64)
    return float(np.dot(residual / sigma_array, residual / sigma_array))


def run_runtime_style_safety_pgo(
    *,
    nodes: list[int],
    odom_by_idx: dict[int, np.ndarray],
    query_idx: int,
    candidate_idx: int,
    loop_factor: np.ndarray,
    accepted_history: list[LoopFactor],
    include_accepted_loops: bool,
    prior_sigmas: tuple[float, float, float, float, float, float],
    odom_sigmas: tuple[float, float, float, float, float, float],
    loop_sigmas: tuple[float, float, float, float, float, float],
) -> SafetyPgoRetestResult:
    original_poses = [project_transform_to_se3(odom_by_idx[idx]) for idx in nodes]
    try:
        import gtsam

        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()
        key_by_idx = {idx: int(idx) for idx in nodes}
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.asarray(prior_sigmas, dtype=np.float64))
        odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.asarray(odom_sigmas, dtype=np.float64))
        loop_noise = gtsam.noiseModel.Diagonal.Sigmas(np.asarray(loop_sigmas, dtype=np.float64))

        for idx, pose in zip(nodes, original_poses):
            initial.insert(key_by_idx[idx], _to_gtsam_pose3(gtsam, pose))

        graph.add(
            gtsam.PriorFactorPose3(
                key_by_idx[nodes[0]],
                _to_gtsam_pose3(gtsam, original_poses[0]),
                prior_noise,
            )
        )
        odom_measurements = []
        for left_idx, right_idx in zip(nodes[:-1], nodes[1:]):
            measurement = pose_between(odom_by_idx[left_idx], odom_by_idx[right_idx])
            measurement = project_transform_to_se3(measurement)
            odom_measurements.append(measurement)
            graph.add(
                gtsam.BetweenFactorPose3(
                    key_by_idx[left_idx],
                    key_by_idx[right_idx],
                    _to_gtsam_pose3(gtsam, measurement),
                    odom_noise,
                )
            )

        history_count = 0
        if include_accepted_loops:
            for factor in accepted_history:
                if factor.from_idx not in key_by_idx or factor.to_idx not in key_by_idx:
                    continue
                graph.add(
                    gtsam.BetweenFactorPose3(
                        key_by_idx[factor.from_idx],
                        key_by_idx[factor.to_idx],
                        _to_gtsam_pose3(gtsam, factor.pose_between),
                        gtsam.noiseModel.Diagonal.Sigmas(
                            np.asarray(factor.noise_sigmas, dtype=np.float64)
                        ),
                    )
                )
                history_count += 1

        graph.add(
            gtsam.BetweenFactorPose3(
                key_by_idx[query_idx],
                key_by_idx[candidate_idx],
                _to_gtsam_pose3(gtsam, loop_factor),
                loop_noise,
            )
        )
        factor_count = int(graph.size())
        error_before = float(graph.error(initial))
        params = gtsam.LevenbergMarquardtParams()
        if hasattr(params, "setVerbosityLM"):
            params.setVerbosityLM("SILENT")
        optimized_values = gtsam.LevenbergMarquardtOptimizer(graph, initial, params).optimize()
        error_after = float(graph.error(optimized_values))
        optimized = [
            np.asarray(optimized_values.atPose3(key_by_idx[idx]).matrix(), dtype=np.float64)
            for idx in nodes
        ]
        optimized = [project_transform_to_se3(pose) for pose in optimized]

        prior_chi2_before = _prior_chi2(original_poses[0], original_poses[0], prior_sigmas)
        prior_chi2_after = _prior_chi2(optimized[0], original_poses[0], prior_sigmas)
        odom_chi2_before = 0.0
        odom_chi2_after = 0.0
        for position, measurement in enumerate(odom_measurements):
            odom_chi2_before += _between_chi2(
                original_poses[position],
                original_poses[position + 1],
                measurement,
                odom_sigmas,
            )
            odom_chi2_after += _between_chi2(
                optimized[position],
                optimized[position + 1],
                measurement,
                odom_sigmas,
            )
        from_position = nodes.index(query_idx)
        to_position = nodes.index(candidate_idx)
        loop_chi2_after = _between_chi2(
            optimized[from_position],
            optimized[to_position],
            loop_factor,
            loop_sigmas,
        )
        odom_strain = max(0.0, odom_chi2_after - odom_chi2_before)
        converged = bool(
            len(optimized) == len(nodes)
            and np.isfinite(error_before)
            and np.isfinite(error_after)
            and error_after <= error_before + 1e-9
        )
        return SafetyPgoRetestResult(
            converged=converged,
            failure_reason="" if converged else "safety_gate_optimizer_not_converged",
            original_poses=original_poses,
            optimized_poses=optimized,
            error_before=error_before,
            error_after=error_after,
            prior_chi2_before=float(prior_chi2_before),
            prior_chi2_after=float(prior_chi2_after),
            loop_chi2_after=float(loop_chi2_after),
            odom_chi2_before=float(odom_chi2_before),
            odom_chi2_after=float(odom_chi2_after),
            odom_strain_chi2_after=float(odom_strain),
            factor_count=factor_count,
            history_loop_factor_count=history_count,
        )
    except Exception as exc:
        return SafetyPgoRetestResult(
            converged=False,
            failure_reason=f"{type(exc).__name__}: {exc}",
            original_poses=original_poses,
            optimized_poses=original_poses,
            error_before=float("inf"),
            error_after=float("inf"),
            prior_chi2_before=float("inf"),
            prior_chi2_after=float("inf"),
            loop_chi2_after=float("inf"),
            odom_chi2_before=float("inf"),
            odom_chi2_after=float("inf"),
            odom_strain_chi2_after=float("inf"),
            factor_count=0,
            history_loop_factor_count=0,
        )


def safety_gate_retest(
    *,
    query_idx: int,
    candidate_idx: int,
    pose_between_matrix: np.ndarray,
    odom_by_idx: dict[int, np.ndarray],
    cache_order: list[int],
    accepted_history: list[LoopFactor],
    args: argparse.Namespace,
) -> tuple[bool, str, dict[str, float]]:
    nodes = select_local_nodes(
        candidate_idx,
        query_idx,
        cache_order=cache_order,
        max_nodes=args.max_nodes,
    )
    metrics: dict[str, float] = {
        "safety_node_count": float(len(nodes)),
        "safety_history_loop_factor_count": 0.0,
    }
    if len(nodes) < args.min_nodes:
        return False, "safety_gate_too_few_nodes", metrics
    if nodes[0] != min(candidate_idx, query_idx) or nodes[-1] != max(candidate_idx, query_idx):
        return False, "safety_gate_missing_endpoint", metrics

    prior_sigmas = (1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-2)
    odom_sigmas = (0.05, 0.05, 0.05, 0.10, 0.10, 0.10)
    loop_sigmas = (
        args.loop_rotation_sigma,
        args.loop_rotation_sigma,
        args.loop_rotation_sigma,
        args.loop_translation_sigma,
        args.loop_translation_sigma,
        args.loop_translation_sigma,
    )
    pgo = run_runtime_style_safety_pgo(
        nodes=nodes,
        odom_by_idx=odom_by_idx,
        query_idx=query_idx,
        candidate_idx=candidate_idx,
        loop_factor=pose_between_matrix,
        accepted_history=accepted_history,
        include_accepted_loops=args.include_accepted_loops,
        prior_sigmas=prior_sigmas,
        odom_sigmas=odom_sigmas,
        loop_sigmas=loop_sigmas,
    )
    metrics.update(
        {
            "safety_factor_count": float(pgo.factor_count),
            "safety_history_loop_factor_count": float(pgo.history_loop_factor_count),
            "safety_pgo_converged": 1.0 if pgo.converged else 0.0,
            "safety_pgo_error_before": float(pgo.error_before),
            "safety_pgo_error_after": float(pgo.error_after),
            "safety_prior_chi2_before": float(pgo.prior_chi2_before),
            "safety_prior_chi2_after": float(pgo.prior_chi2_after),
            "safety_loop_chi2_after": float(pgo.loop_chi2_after),
            "safety_odom_chi2_before": float(pgo.odom_chi2_before),
            "safety_odom_chi2_after": float(pgo.odom_chi2_after),
            "safety_odom_strain_chi2_after": float(pgo.odom_strain_chi2_after),
        }
    )
    if not pgo.converged:
        return False, pgo.failure_reason or "safety_gate_pgo_failed", metrics
    deformation = trajectory_deformation_rmse(
        [odom_by_idx[idx] for idx in nodes],
        pgo.optimized_poses,
    )
    factor_count = max(1, pgo.factor_count)
    error_per_factor = pgo.error_after / factor_count
    metrics["safety_trajectory_deformation_rmse"] = float(deformation)
    metrics["safety_pgo_error_per_factor_after"] = float(error_per_factor)
    if not np.isfinite(deformation) or not np.isfinite(error_per_factor):
        return False, "safety_gate_pgo_failed", metrics
    if deformation > args.max_deformation_rmse:
        return False, "safety_gate_deformation_too_high", metrics
    if error_per_factor > args.max_pgo_error_per_factor:
        return False, "safety_gate_residual_too_high", metrics
    if metrics["safety_odom_strain_chi2_after"] > args.max_odom_strain_chi2:
        return False, "safety_gate_odom_strain_too_high", metrics
    if metrics["safety_loop_chi2_after"] > args.max_loop_chi2:
        return False, "safety_gate_loop_chi2_too_high", metrics
    return True, "safety_gate_passed", metrics


def norm(matrix: np.ndarray) -> float:
    return float(np.linalg.norm(matrix[:3, 3]))


def matrix_delta(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    delta = invert_transform(a) @ b
    cos_angle = float(np.clip((np.trace(delta[:3, :3]) - 1.0) * 0.5, -1.0, 1.0))
    return {
        "translation_norm": float(np.linalg.norm(delta[:3, 3])),
        "rotation_deg": float(np.degrees(np.arccos(cos_angle))),
    }


def make_visualization(
    *,
    query: ImageRecord,
    candidate: ImageRecord,
    support: ImageRecord,
    record: dict[str, Any],
    output_path: Path,
) -> None:
    images = []
    for title, image_record in (("query", query), ("candidate", candidate), ("support", support)):
        image = Image.open(image_record.path).convert("RGB")
        image.thumbnail((360, 240))
        canvas = Image.new("RGB", (360, 280), "white")
        canvas.paste(image, ((360 - image.width) // 2, 30))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), f"{title} kf={image_record.keyframe_idx}", fill=(0, 0, 0))
        draw.text((8, 252), f"dt={image_record.sync_delta_sec:.3f}s", fill=(0, 0, 0))
        images.append(canvas)
    text_h = 150
    out = Image.new("RGB", (1080, 280 + text_h), "white")
    for idx, image in enumerate(images):
        out.paste(image, (idx * 360, 0))
    draw = ImageDraw.Draw(out)
    lines = [
        f"pair q{record['query_idx']} c{record['candidate_idx']} support {record['support_idx']}",
        (
            f"runtime: raw {record['runtime']['raw_norm']:.3f}, aligned "
            f"{record['runtime']['aligned_norm']:.3f}, backend {record['runtime']['backend_norm']:.3f}"
        ),
        (
            f"offline: raw {record['offline']['raw_norm']:.3f}, aligned "
            f"{record['offline']['aligned_norm']:.3f}, backend {record['offline']['backend_norm']:.3f}"
        ),
        (
            f"sim3 runtime/offline scale {record['runtime']['sim3_scale']:.3f}/"
            f"{record['offline']['sim3_scale']:.3f}, rmse "
            f"{record['runtime']['sim3_alignment_rmse']:.3f}/"
            f"{record['offline']['sim3_alignment_rmse']:.3f}"
        ),
        (
            f"safety runtime/offline {record['runtime']['safety_status']} / "
            f"{record['offline']['safety_status']} ({record['offline']['safety_reason']})"
        ),
    ]
    y = 292
    for line in lines:
        draw.text((10, y), line, fill=(0, 0, 0))
        y += 24
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else run_dir / "offline_retest"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_runtime_rows(run_dir / "loop_log.jsonl")
    pairs = parse_pairs(args.pairs) if args.pairs else accepted_runtime_pairs(rows)
    if not pairs:
        pairs = list(DEFAULT_PAIRS)
    runtime_rows = selected_runtime_rows(rows, pairs)

    cache_order, keyframe_timestamps, odom_by_idx = load_tum_poses(run_dir / "trajectory_keyframes.txt")
    t_camera_lidar = np.asarray([float(v) for v in args.t_camera_lidar.split(",")]).reshape(4, 4)

    needed_indices = set()
    for row in runtime_rows:
        needed_indices.add(int(row["query_idx"]))
        needed_indices.add(int(row["candidate_idx"]))
        needed_indices.add(int(row["metrics"]["selected_support_idx"]))
    image_records = extract_images_from_bag(
        bag_path=Path(args.bag),
        image_topic=args.image_topic,
        keyframe_timestamps={idx: keyframe_timestamps[idx] for idx in needed_indices},
        output_dir=output_dir / "images",
        tolerance_sec=args.image_sync_tolerance_sec,
    )

    da3_runner = RealDa3Runner(
        RealDa3RunnerConfig(
            model_name=args.model_name,
            device=args.device,
            process_res=args.process_res,
            process_res_method=args.process_res_method,
            local_files_only=args.local_files_only,
            triplet_batch_size=1,
            extrinsics_are_c2w=False,
        )
    )

    records = []
    accepted_history: list[LoopFactor] = []
    loop_noise = (
        args.loop_rotation_sigma,
        args.loop_rotation_sigma,
        args.loop_rotation_sigma,
        args.loop_translation_sigma,
        args.loop_translation_sigma,
        args.loop_translation_sigma,
    )

    for row in runtime_rows:
        query_idx = int(row["query_idx"])
        candidate_idx = int(row["candidate_idx"])
        support_idx = int(row["metrics"]["selected_support_idx"])
        triplet = build_da3_triplet(
            image_records[query_idx].path,
            image_records[candidate_idx].path,
            image_records[support_idx].path,
            query_idx=query_idx,
            candidate_idx=candidate_idx,
            support_idx=support_idx,
        )
        da3_result = da3_runner.run_triplet(triplet)
        predicted_c2w = da3_result.predicted_c2w
        raw = query_camera_to_candidate_camera(predicted_c2w[0], predicted_c2w[1])

        prior_candidate_support = [
            camera_pose_from_backend_pose(odom_by_idx[candidate_idx], t_camera_lidar),
            camera_pose_from_backend_pose(odom_by_idx[support_idx], t_camera_lidar),
        ]
        alignment_valid, alignment_reason, alignment_metrics, aligned_query, aligned_candidate = (
            align_pose_prior(predicted_c2w, prior_candidate_support)
        )
        if alignment_valid and aligned_query is not None and aligned_candidate is not None:
            aligned = query_camera_to_candidate_camera(aligned_query, aligned_candidate)
            backend = camera_relative_to_backend_relative(aligned, t_camera_lidar)
            safety_ok, safety_reason, safety_metrics = safety_gate_retest(
                query_idx=query_idx,
                candidate_idx=candidate_idx,
        pose_between_matrix=project_transform_to_se3(backend),
                odom_by_idx=odom_by_idx,
                cache_order=cache_order,
                accepted_history=accepted_history,
                args=args,
            )
        else:
            aligned = np.eye(4, dtype=np.float64)
            backend = np.eye(4, dtype=np.float64)
            safety_ok = False
            safety_reason = alignment_reason
            safety_metrics = {}

        if safety_ok:
            accepted_history.append(
                LoopFactor(query_idx, candidate_idx, backend.copy(), loop_noise)
            )

        runtime_models = row["model_fields"]
        runtime_metrics = row["metrics"]
        runtime_raw = parse_matrix_label(runtime_models["raw_da3_relative_camera_rowmajor16"])
        runtime_aligned = parse_matrix_label(runtime_models["aligned_camera_relative_rowmajor16"])
        runtime_backend = parse_matrix_label(runtime_models["backend_pose_between_rowmajor16"])

        record = {
            "query_idx": query_idx,
            "candidate_idx": candidate_idx,
            "support_idx": support_idx,
            "image_sync": {
                str(idx): {
                    "target_timestamp": image_records[idx].target_timestamp,
                    "image_timestamp": image_records[idx].image_timestamp,
                    "sync_delta_sec": image_records[idx].sync_delta_sec,
                    "path": str(image_records[idx].path),
                }
                for idx in (query_idx, candidate_idx, support_idx)
            },
            "runtime": {
                "score": float(row["score"]),
                "raw_norm": norm(runtime_raw),
                "aligned_norm": norm(runtime_aligned),
                "backend_norm": norm(runtime_backend),
                "sim3_scale": float(runtime_metrics.get("sim3_scale", float("nan"))),
                "sim3_alignment_rmse": float(
                    runtime_metrics.get("sim3_alignment_rmse", float("nan"))
                ),
                "sim3_direction_error_deg": float(
                    runtime_metrics.get("sim3_direction_error_deg", float("nan"))
                ),
                "safety_status": runtime_models.get("safety_gate_status", ""),
                "safety_trajectory_deformation_rmse": float(
                    runtime_metrics.get("safety_trajectory_deformation_rmse", float("nan"))
                ),
                "safety_pgo_error_per_factor_after": float(
                    runtime_metrics.get("safety_pgo_error_per_factor_after", float("nan"))
                ),
                "safety_loop_chi2_after": float(
                    runtime_metrics.get("safety_loop_chi2_after", float("nan"))
                ),
                "raw_matrix": matrix_to_list(runtime_raw),
                "aligned_matrix": matrix_to_list(runtime_aligned),
                "backend_matrix": matrix_to_list(runtime_backend),
            },
            "offline": {
                "alignment_valid": bool(alignment_valid),
                "alignment_reason": alignment_reason,
                "raw_norm": norm(raw),
                "aligned_norm": norm(aligned) if alignment_valid else None,
                "backend_norm": norm(backend) if alignment_valid else None,
                "sim3_scale": alignment_metrics.get("sim3_scale"),
                "sim3_alignment_rmse": alignment_metrics.get("sim3_alignment_rmse"),
                "sim3_direction_error_deg": alignment_metrics.get("sim3_direction_error_deg"),
                "safety_status": "accepted" if safety_ok else "rejected",
                "safety_reason": safety_reason,
                "safety_metrics": safety_metrics,
                "raw_matrix": matrix_to_list(raw),
                "aligned_matrix": matrix_to_list(aligned) if alignment_valid else None,
                "backend_matrix": matrix_to_list(backend) if alignment_valid else None,
                "raw_delta_vs_runtime": matrix_delta(runtime_raw, raw),
                "aligned_delta_vs_runtime": (
                    matrix_delta(runtime_aligned, aligned) if alignment_valid else None
                ),
                "backend_delta_vs_runtime": (
                    matrix_delta(runtime_backend, backend) if alignment_valid else None
                ),
            },
        }
        records.append(record)
        make_visualization(
            query=image_records[query_idx],
            candidate=image_records[candidate_idx],
            support=image_records[support_idx],
            record=record,
            output_path=output_dir / "visualizations" / f"q{query_idx:06d}_c{candidate_idx:06d}.png",
        )

    summary = {
        "run_dir": str(run_dir),
        "bag": str(Path(args.bag)),
        "image_topic": args.image_topic,
        "output_dir": str(output_dir),
        "config": {
            "model_name": args.model_name,
            "device": args.device,
            "process_res": args.process_res,
            "process_res_method": args.process_res_method,
            "local_files_only": args.local_files_only,
            "image_sync_tolerance_sec": args.image_sync_tolerance_sec,
            "max_nodes": args.max_nodes,
            "min_nodes": args.min_nodes,
            "include_accepted_loops": args.include_accepted_loops,
            "loop_noise_sigmas": loop_noise,
        },
        "records": records,
    }
    (output_dir / "offline_retest_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True),
        encoding="utf-8",
    )
    with (output_dir / "offline_retest_records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=True) + "\n")
    print(f"Wrote offline retest to {output_dir}")
    for record in records:
        print(
            "q{query_idx:03d}-c{candidate_idx:03d}: "
            "runtime backend={runtime_backend:.3f}, offline backend={offline_backend}, "
            "safety={safety}".format(
                query_idx=record["query_idx"],
                candidate_idx=record["candidate_idx"],
                runtime_backend=record["runtime"]["backend_norm"],
                offline_backend=(
                    "None"
                    if record["offline"]["backend_norm"] is None
                    else f"{record['offline']['backend_norm']:.3f}"
                ),
                safety=(
                    f"{record['offline']['safety_status']}:"
                    f"{record['offline']['safety_reason']}"
                ),
            )
        )


if __name__ == "__main__":
    main()
