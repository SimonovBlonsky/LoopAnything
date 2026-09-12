import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from robust_loop_verifier.geometry import invert_transform
from robust_loop_verifier.io import (
    associate_tum_by_timestamp,
    read_json,
    read_jsonl,
    read_tum_trajectory,
    write_json,
    write_jsonl,
)
from robust_loop_verifier.schema import RobustLoopVerifierConfig


def preprocess_fusionportable_sequence(
    raw_dir: Path,
    gt_trajectory_file: Optional[Path],
    sequence_name: str,
    config: RobustLoopVerifierConfig,
    max_gt_delta_sec: float = 0.05,
) -> Path:
    raw_dir = Path(raw_dir)

    odom_trajectory_file = raw_dir / "trajectory_keyframes.txt"
    keyframe_rows = list(read_jsonl(raw_dir / "keyframes_with_images.jsonl"))
    odom_records = read_tum_trajectory(odom_trajectory_file)
    t_camera_lidar = _read_camera_lidar_extrinsic(raw_dir / "sequence_meta.json")
    t_lidar_camera = invert_transform(t_camera_lidar)

    _validate_keyframe_stream_order(keyframe_rows)
    _validate_odom_keyframe_alignment(keyframe_rows, odom_records)
    dataset_name = _safe_path_segment(config.dataset_name, "dataset_name")
    platform = _safe_path_segment(config.platform, "platform")
    sequence_name = _safe_path_segment(sequence_name, "sequence_name")
    if _uses_aster_slam_label_trajectory(platform):
        gt_trajectory_file = odom_trajectory_file
        gt_records = odom_records
        gt_label_source = "aster_slam_trajectory_keyframes"
    else:
        if gt_trajectory_file is None:
            raise ValueError("gt_trajectory_file is required for non-handheld/legged platforms")
        gt_trajectory_file = Path(gt_trajectory_file)
        gt_records = read_tum_trajectory(gt_trajectory_file)
        gt_label_source = "external_trajectory_timestamp_association"

    out_dir, image_out_dir = _prepare_output_dir(
        config.output_root,
        dataset_name,
        platform,
        sequence_name,
    )

    keyframes = []
    skipped_gt_association_indices: list[int] = []
    for row, odom_record in zip(keyframe_rows, odom_records):
        idx = int(row["keyframe_idx"])
        timestamp = float(row["timestamp"])
        if gt_label_source == "aster_slam_trajectory_keyframes":
            gt_record = odom_record
        else:
            try:
                gt_record = associate_tum_by_timestamp(gt_records, timestamp, max_gt_delta_sec)
            except ValueError as error:
                if str(error) != "Nearest TUM timestamp exceeds max_delta_sec":
                    raise
                skipped_gt_association_indices.append(idx)
                continue
        image_path = _link_image(raw_dir, image_out_dir, idx, row)
        odom_camera_pose = odom_record.pose @ t_lidar_camera
        gt_camera_pose = gt_record.pose @ t_lidar_camera

        keyframes.append(
            {
                "idx": idx,
                "timestamp": timestamp,
                "image_path": image_path,
                "odom_pose": _flatten_pose(odom_camera_pose),
                "gt_pose": _flatten_pose(gt_camera_pose),
                "source_raw_dir": str(raw_dir),
                "source_gt_trajectory_file": str(gt_trajectory_file),
                "gt_label_source": gt_label_source,
            }
        )

    if not keyframes:
        raise ValueError("No keyframes remain after GT timestamp association")

    positives = _build_online_causal_positives(
        keyframes,
        positive_radius_m=config.positive_radius_m,
        positive_max_rotation_deg=config.positive_max_rotation_deg,
        recent_exclusion_keyframes=config.recent_exclusion_keyframes,
    )

    write_jsonl(out_dir / "keyframes.jsonl", keyframes)
    write_jsonl(out_dir / "positives.jsonl", positives)
    write_json(
        out_dir / "manifest.json",
        {
            "dataset_name": config.dataset_name,
            "platform": config.platform,
            "sequence_name": sequence_name,
            "keyframe_count": len(keyframes),
            "raw_dir": str(raw_dir),
            "gt_trajectory_file": str(gt_trajectory_file),
            "gt_label_source": gt_label_source,
            "pose_frame": "camera",
            "source_trajectory_pose_frame": "lidar",
            "T_camera_lidar": t_camera_lidar.reshape(-1).tolist(),
            "positive_radius_m": config.positive_radius_m,
            "positive_max_rotation_deg": config.positive_max_rotation_deg,
            "recent_exclusion_keyframes": config.recent_exclusion_keyframes,
            "keyframe_count_before_gt_filter": len(keyframe_rows),
            "skipped_gt_association_count": len(skipped_gt_association_indices),
            "skipped_gt_association_indices": skipped_gt_association_indices,
            "max_gt_delta_sec": max_gt_delta_sec,
        },
    )

    return out_dir


def _read_camera_lidar_extrinsic(sequence_meta_path: Path) -> np.ndarray:
    meta = read_json(sequence_meta_path)
    if "T_camera_lidar" not in meta:
        raise ValueError("sequence_meta.json must contain T_camera_lidar")

    transform = np.asarray(meta["T_camera_lidar"], dtype=np.float64)
    if transform.size != 16:
        raise ValueError("T_camera_lidar must contain 16 values")
    transform = transform.reshape(4, 4)
    if not np.all(np.isfinite(transform)):
        raise ValueError("T_camera_lidar must contain only finite values")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7, rtol=0.0):
        raise ValueError("T_camera_lidar bottom row must be [0, 0, 0, 1]")

    rotation = transform[:3, :3]
    u, _, vt = np.linalg.svd(rotation)
    projected_rotation = u @ vt
    if np.linalg.det(projected_rotation) < 0.0:
        u[:, -1] *= -1.0
        projected_rotation = u @ vt
    if np.max(np.abs(projected_rotation - rotation)) > 1e-3:
        raise ValueError("T_camera_lidar rotation is not sufficiently close to SO(3)")

    projected_transform = transform.copy()
    projected_transform[:3, :3] = projected_rotation
    try:
        invert_transform(projected_transform)
    except ValueError as error:
        raise ValueError(f"invalid T_camera_lidar: {error}") from error
    return projected_transform


def _uses_aster_slam_label_trajectory(platform: str) -> bool:
    return platform.lower() in {
        "handheld",
        "legged",
        "ntu-viral",
        "ntu_viral",
        "ugv",
        "offroad",
    }


def _build_online_causal_positives(
    keyframes: List[Mapping[str, Any]],
    positive_radius_m: float,
    positive_max_rotation_deg: float,
    recent_exclusion_keyframes: int,
) -> List[Dict[str, Any]]:
    positives = []
    gt_poses = [
        np.asarray(keyframe["gt_pose"], dtype=np.float64).reshape(4, 4) for keyframe in keyframes
    ]

    for query_pose, query_keyframe in zip(gt_poses, keyframes):
        query_idx = int(query_keyframe["idx"])
        positive_indices = []
        for candidate_pose, candidate_keyframe in zip(gt_poses, keyframes):
            candidate_idx = int(candidate_keyframe["idx"])
            if candidate_idx >= query_idx - recent_exclusion_keyframes:
                continue
            distance = float(np.linalg.norm(query_pose[:3, 3] - candidate_pose[:3, 3]))
            rotation_deg = _rotation_angle_deg(query_pose[:3, :3], candidate_pose[:3, :3])
            if distance <= positive_radius_m and rotation_deg <= positive_max_rotation_deg:
                positive_indices.append(candidate_idx)
        positives.append(
            {
                "query_idx": query_idx,
                "positive_indices": positive_indices,
            }
        )

    return positives


def _rotation_angle_deg(query_rotation: np.ndarray, candidate_rotation: np.ndarray) -> float:
    relative_rotation = query_rotation.T @ candidate_rotation
    cosine = (float(np.trace(relative_rotation)) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _safe_path_segment(value: Any, field_name: str) -> str:
    segment = str(value)
    if not segment or not segment.strip():
        raise ValueError("{} must be a non-empty relative path segment".format(field_name))
    if Path(segment).is_absolute() or "\\" in segment or "/" in segment:
        raise ValueError("{} must be a safe relative path segment".format(field_name))
    if segment == "." or any(part in ("", ".", "..") for part in Path(segment).parts):
        raise ValueError("{} must be a safe relative path segment".format(field_name))
    return segment


def _prepare_output_dir(
    output_root: Path,
    dataset_name: str,
    platform: str,
    sequence_name: str,
) -> tuple:
    root = Path(output_root)
    resolved_root = root.resolve(strict=False)
    _ensure_real_directory(root, resolved_root, allow_equal=True)

    current = root
    for segment in (dataset_name, platform, sequence_name, "images"):
        current = current / segment
        _ensure_real_directory(current, resolved_root)

    out_dir = root / dataset_name / platform / sequence_name
    return out_dir, out_dir / "images"


def _ensure_real_directory(path: Path, resolved_root: Path, allow_equal: bool = False) -> None:
    if path.is_symlink():
        raise ValueError("output directory must not be a symlink")
    if path.exists():
        if not path.is_dir():
            raise ValueError("output path exists and is not a directory")
    else:
        if allow_equal:
            path.mkdir(parents=True)
        else:
            path.mkdir()

    if path.is_symlink():
        raise ValueError("output directory must not be a symlink")
    resolved_path = path.resolve(strict=True)
    if resolved_path == resolved_root:
        if allow_equal:
            return
        raise ValueError("output directory must be under output_root")
    if not _is_relative_to(resolved_path, resolved_root):
        raise ValueError("output directory must resolve under output_root")


def _validate_keyframe_stream_order(keyframe_rows: List[Mapping[str, Any]]) -> None:
    previous_idx = None
    for row in keyframe_rows:
        idx = int(row["keyframe_idx"])
        if previous_idx is not None and idx <= previous_idx:
            raise ValueError("keyframe_idx must be strictly increasing")
        previous_idx = idx


def _validate_odom_keyframe_alignment(
    keyframe_rows: List[Mapping[str, Any]],
    odom_records: List[Any],
) -> None:
    if len(odom_records) != len(keyframe_rows):
        raise ValueError("Odom trajectory length must match keyframe rows length")


def _resolve_image_source(raw_dir: Path, image_path: Any) -> Optional[Path]:
    if not image_path:
        return None

    image_path_str = str(image_path)
    relative_path = Path(image_path_str)
    if relative_path.is_absolute() or "\\" in image_path_str:
        raise ValueError("image_path must be a safe relative path under raw_dir")
    if any(part in ("", "..") for part in relative_path.parts):
        raise ValueError("image_path must be a safe relative path under raw_dir")

    raw_root = raw_dir.resolve(strict=True)
    source = (raw_dir / relative_path).resolve(strict=False)
    if not _is_relative_to(source, raw_root):
        raise ValueError("image_path must resolve under raw_dir")
    if not source.exists():
        return None
    if not source.is_file():
        raise ValueError("image_path must resolve to a regular file")
    return source


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _link_image(
    raw_dir: Path,
    image_out_dir: Path,
    keyframe_idx: int,
    row: Mapping[str, Any],
) -> Optional[str]:
    src = _resolve_image_source(raw_dir, row.get("image_path"))
    if src is None:
        return None

    dst = image_out_dir / "{:06d}{}".format(keyframe_idx, src.suffix)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink():
            try:
                if dst.resolve(strict=True) == src.resolve(strict=True):
                    return str(dst.relative_to(image_out_dir.parent))
            except FileNotFoundError:
                pass
            dst.unlink()
        elif dst.is_file():
            dst.unlink()
        else:
            raise ValueError("output image path exists and is not a regular file")

    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)

    return str(dst.relative_to(image_out_dir.parent))


def _flatten_pose(pose: np.ndarray) -> List[float]:
    return [float(value) for value in np.asarray(pose, dtype=np.float64).reshape(-1)]
