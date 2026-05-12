from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from loop_policy.schema import (
    KeyframeRecord,
    PoseRecord,
    dataclass_to_json_dict,
)


@dataclass(frozen=True)
class AsterRawSequence:
    raw_dir: Path
    meta: Dict[str, Any]
    sequence_name: str
    platform: str
    loop_closure_enabled: bool
    image_topic: Optional[str]
    t_camera_lidar: np.ndarray
    keyframes: List[KeyframeRecord]
    trajectory: List[PoseRecord]


def _iter_jsonl_records(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if stripped:
                yield line_no, json.loads(stripped)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [record for _, record in _iter_jsonl_records(path)]


def write_jsonl(path: Path, records: Iterable[Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dataclass_to_json_dict(record), sort_keys=True) + "\n")


def read_tum_trajectory(path: Path) -> List[PoseRecord]:
    path = Path(path)
    poses: List[PoseRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            fields = stripped.split()
            if len(fields) != 8:
                raise ValueError(
                    f"{path}:{line_no}: expected 8 TUM fields, got {len(fields)} "
                    f"in line {stripped!r}"
                )

            try:
                timestamp, tx, ty, tz, qx, qy, qz, qw = (float(field) for field in fields)
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_no}: invalid TUM float fields {fields!r} in line {stripped!r}"
                ) from exc
            poses.append(
                PoseRecord(
                    timestamp=timestamp,
                    position=(tx, ty, tz),
                    quaternion_xyzw=(qx, qy, qz, qw),
                )
            )
    return poses


def _read_keyframes(path: Path) -> List[KeyframeRecord]:
    path = Path(path)
    keyframes: List[KeyframeRecord] = []
    for line_no, row in _iter_jsonl_records(path):
        keyframe_idx = row.get("keyframe_idx", row.get("idx"))
        if keyframe_idx is None:
            raise ValueError(f"{path}:{line_no}: missing keyframe_idx or idx in {row!r}")

        trajectory_idx = row.get("trajectory_idx")
        try:
            parsed_keyframe_idx = int(keyframe_idx)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{path}:{line_no}: invalid keyframe_idx {keyframe_idx!r} in {row!r}"
            ) from exc

        try:
            parsed_trajectory_idx = int(trajectory_idx) if trajectory_idx is not None else None
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{path}:{line_no}: invalid trajectory_idx {trajectory_idx!r} in {row!r}"
            ) from exc

        keyframes.append(
            KeyframeRecord(
                keyframe_idx=parsed_keyframe_idx,
                timestamp=float(row["timestamp"]),
                image_path=row.get("image_path"),
                trajectory_idx=parsed_trajectory_idx,
                raw=row,
            )
        )
    return keyframes


def load_aster_raw_sequence(raw_dir: Path) -> AsterRawSequence:
    raw_dir = Path(raw_dir)
    meta_path = raw_dir / "sequence_meta.json"
    keyframes_path = raw_dir / "keyframes_with_images.jsonl"
    trajectory_path = raw_dir / "trajectory.txt"

    for required_path in (meta_path, keyframes_path, trajectory_path):
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    t_camera_lidar = np.asarray(meta["T_camera_lidar"], dtype=np.float64)
    if t_camera_lidar.shape == (16,):
        t_camera_lidar = t_camera_lidar.reshape(4, 4)
    if t_camera_lidar.shape != (4, 4):
        raise ValueError("T_camera_lidar must be a 4x4 matrix")

    return AsterRawSequence(
        raw_dir=raw_dir,
        meta=meta,
        sequence_name=meta.get("sequence_name") or raw_dir.parent.name,
        platform=meta.get("platform") or raw_dir.parent.parent.name,
        loop_closure_enabled=bool(meta.get("loop_closure_enabled", True)),
        image_topic=meta.get("image_topic"),
        t_camera_lidar=t_camera_lidar,
        keyframes=_read_keyframes(keyframes_path),
        trajectory=read_tum_trajectory(trajectory_path),
    )
