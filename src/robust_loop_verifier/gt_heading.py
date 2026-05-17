from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

_IDENTITY_QUAT_XYZW = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def convert_tum_file_to_translation_heading(
    input_file: Path,
    output_file: Path,
    *,
    min_segment_translation_m: float = 1e-3,
    replace_only_identity: bool = True,
) -> dict:
    records = _read_tum_rows(Path(input_file))
    if len(records) < 2:
        raise ValueError("At least two TUM records are required to infer heading")
    if min_segment_translation_m <= 0.0:
        raise ValueError("min_segment_translation_m must be positive")

    timestamps = np.array([record[0] for record in records], dtype=np.float64)
    positions = np.array([record[1:4] for record in records], dtype=np.float64)
    original_quats = np.array([record[4:8] for record in records], dtype=np.float64)
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("TUM timestamps must be strictly increasing")

    heading_yaws = _interpolate_heading_yaws(
        timestamps,
        positions,
        min_segment_translation_m=min_segment_translation_m,
    )
    heading_quats = np.array([_quat_xyzw_from_yaw(yaw) for yaw in heading_yaws])

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    replaced_count = 0
    with output_file.open("w", encoding="utf-8") as handle:
        for record, original_quat, heading_quat in zip(records, original_quats, heading_quats):
            should_replace = (not replace_only_identity) or _is_identity_quat(original_quat)
            quat = heading_quat if should_replace else _normalize_quat(original_quat)
            if should_replace:
                replaced_count += 1
            handle.write(_format_tum_row(record[0], record[1:4], quat))
            handle.write("\n")

    return {
        "input_file": str(input_file),
        "output_file": str(output_file),
        "record_count": len(records),
        "replaced_identity_count": replaced_count,
        "min_segment_translation_m": min_segment_translation_m,
    }


def process_fusionportable_platform_gt(
    input_root: Path,
    output_root: Path,
    *,
    platforms: Iterable[str] = ("handheld", "legged"),
    min_segment_translation_m: float = 1e-3,
    replace_only_identity: bool = True,
) -> list[dict]:
    input_root = Path(input_root)
    output_root = Path(output_root)

    summaries = []
    for platform in platforms:
        platform_dir = input_root / platform
        if not platform_dir.is_dir():
            continue
        for sequence_dir in sorted(path for path in platform_dir.iterdir() if path.is_dir()):
            input_file = sequence_dir / f"{sequence_dir.name}.txt"
            if not input_file.is_file():
                continue
            output_file = output_root / platform / sequence_dir.name / input_file.name
            summary = convert_tum_file_to_translation_heading(
                input_file,
                output_file,
                min_segment_translation_m=min_segment_translation_m,
                replace_only_identity=replace_only_identity,
            )
            summary["platform"] = platform
            summary["sequence_name"] = sequence_dir.name
            summaries.append(summary)

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return summaries


def _interpolate_heading_yaws(
    timestamps: np.ndarray,
    positions: np.ndarray,
    *,
    min_segment_translation_m: float,
) -> np.ndarray:
    segment_vectors = np.diff(positions[:, :2], axis=0)
    segment_distances = np.linalg.norm(segment_vectors, axis=1)
    valid = segment_distances >= min_segment_translation_m
    if not np.any(valid):
        raise ValueError("Trajectory has no segment long enough to infer heading")

    segment_mid_timestamps = 0.5 * (timestamps[:-1] + timestamps[1:])
    segment_yaws = np.arctan2(segment_vectors[:, 1], segment_vectors[:, 0])
    valid_mid_timestamps = segment_mid_timestamps[valid]
    valid_yaws = np.unwrap(segment_yaws[valid])

    return np.interp(timestamps, valid_mid_timestamps, valid_yaws)


def _read_tum_rows(path: Path) -> list[tuple[float, float, float, float, float, float, float, float]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) != 8:
                raise ValueError(f"TUM row {line_number} must have 8 fields")
            values = tuple(float(field) for field in fields)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"TUM row {line_number} contains a non-finite value")
            rows.append(values)
    return rows


def _format_tum_row(timestamp: float, translation, quat_xyzw: np.ndarray) -> str:
    tx, ty, tz = translation
    qx, qy, qz, qw = quat_xyzw
    return (
        f"{timestamp:.9f} {tx:.9f} {ty:.9f} {tz:.9f} "
        f"{qx:.12f} {qy:.12f} {qz:.12f} {qw:.12f}"
    )


def _quat_xyzw_from_yaw(yaw: float) -> np.ndarray:
    half_yaw = 0.5 * float(yaw)
    return np.array([0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)], dtype=np.float64)


def _normalize_quat(quat_xyzw: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quat_xyzw))
    if norm <= 0.0:
        raise ValueError("Quaternion norm must be positive")
    return np.asarray(quat_xyzw, dtype=np.float64) / norm


def _is_identity_quat(quat_xyzw: np.ndarray) -> bool:
    quat = _normalize_quat(quat_xyzw)
    return np.allclose(quat, _IDENTITY_QUAT_XYZW, atol=1e-9) or np.allclose(
        quat,
        -_IDENTITY_QUAT_XYZW,
        atol=1e-9,
    )
