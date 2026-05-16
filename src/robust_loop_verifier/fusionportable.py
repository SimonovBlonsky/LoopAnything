import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from robust_loop_verifier.io import (
    associate_tum_by_timestamp,
    read_jsonl,
    read_tum_trajectory,
    write_json,
    write_jsonl,
)
from robust_loop_verifier.schema import RobustLoopVerifierConfig


def preprocess_fusionportable_sequence(
    raw_dir: Path,
    gt_trajectory_file: Path,
    sequence_name: str,
    config: RobustLoopVerifierConfig,
    max_gt_delta_sec: float = 0.05,
) -> Path:
    raw_dir = Path(raw_dir)
    gt_trajectory_file = Path(gt_trajectory_file)

    odom_trajectory_file = raw_dir / "trajectory_keyframes.txt"
    keyframe_rows = list(read_jsonl(raw_dir / "keyframes_with_images.jsonl"))
    odom_records = read_tum_trajectory(odom_trajectory_file)
    gt_records = read_tum_trajectory(gt_trajectory_file)

    _validate_keyframe_stream_order(keyframe_rows)
    _validate_odom_keyframe_alignment(keyframe_rows, odom_records)
    dataset_name = _safe_path_segment(config.dataset_name, "dataset_name")
    platform = _safe_path_segment(config.platform, "platform")
    sequence_name = _safe_path_segment(sequence_name, "sequence_name")

    out_dir, image_out_dir = _prepare_output_dir(
        config.output_root,
        dataset_name,
        platform,
        sequence_name,
    )

    keyframes = []
    for row, odom_record in zip(keyframe_rows, odom_records):
        idx = int(row["keyframe_idx"])
        timestamp = float(row["timestamp"])
        gt_record = associate_tum_by_timestamp(gt_records, timestamp, max_gt_delta_sec)
        image_path = _link_image(raw_dir, image_out_dir, idx, row)

        keyframes.append(
            {
                "idx": idx,
                "timestamp": timestamp,
                "image_path": image_path,
                "odom_pose": _flatten_pose(odom_record.pose),
                "gt_pose": _flatten_pose(gt_record.pose),
                "source_raw_dir": str(raw_dir),
                "source_gt_trajectory_file": str(gt_trajectory_file),
            }
        )

    positives = _build_online_causal_positives(
        keyframes,
        positive_radius_m=config.positive_radius_m,
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
            "positive_radius_m": config.positive_radius_m,
            "recent_exclusion_keyframes": config.recent_exclusion_keyframes,
        },
    )

    return out_dir


def _build_online_causal_positives(
    keyframes: List[Mapping[str, Any]],
    positive_radius_m: float,
    recent_exclusion_keyframes: int,
) -> List[Dict[str, Any]]:
    positives = []
    gt_positions = [
        np.asarray(keyframe["gt_pose"], dtype=np.float64).reshape(4, 4)[:3, 3]
        for keyframe in keyframes
    ]

    for query_position, query_keyframe in zip(gt_positions, keyframes):
        query_idx = int(query_keyframe["idx"])
        positive_indices = []
        for candidate_position, candidate_keyframe in zip(gt_positions, keyframes):
            candidate_idx = int(candidate_keyframe["idx"])
            if candidate_idx >= query_idx - recent_exclusion_keyframes:
                continue
            distance = float(np.linalg.norm(query_position - candidate_position))
            if distance <= positive_radius_m:
                positive_indices.append(candidate_idx)
        positives.append(
            {
                "query_idx": query_idx,
                "positive_indices": positive_indices,
            }
        )

    return positives


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
