from __future__ import annotations

import json
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np
import yaml

from .geometry import make_transform, rotation_matrix_from_quat_xyzw


@dataclass(frozen=True)
class TumPoseRecord:
    timestamp: float
    pose: np.ndarray


def read_jsonl(path: Path) -> Iterator[Mapping[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("JSONL rows must be objects")
            yield row


def read_json(path: Path) -> Mapping[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")
    return data


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("JSONL rows must be mappings")
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    if not isinstance(data, Mapping):
        raise ValueError("JSON root must be a mapping")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_yaml(path: Path) -> Mapping[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping")
    return data


def write_yaml(path: Path, data: Mapping[str, Any]) -> None:
    if not isinstance(data, Mapping):
        raise ValueError("YAML root must be a mapping")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=True)


def read_tum_trajectory(path: Path) -> list[TumPoseRecord]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) != 8:
                raise ValueError(f"TUM row {line_number} must have 8 fields")
            try:
                timestamp, tx, ty, tz, qx, qy, qz, qw = (float(field) for field in fields)
            except ValueError as exc:
                raise ValueError(
                    f"TUM row {line_number} contains a non-numeric field: {stripped!r}"
                ) from exc
            if not np.all(np.isfinite([timestamp, tx, ty, tz, qx, qy, qz, qw])):
                raise ValueError(
                    f"TUM row {line_number} contains a non-finite numeric field: {stripped!r}"
                )
            rotation = rotation_matrix_from_quat_xyzw([qx, qy, qz, qw])
            pose = make_transform(rotation, [tx, ty, tz])
            records.append(TumPoseRecord(timestamp=timestamp, pose=pose))

    return sorted(records, key=lambda record: record.timestamp)


def associate_tum_by_timestamp(
    records: list[TumPoseRecord], timestamp: float, max_delta_sec: float
) -> TumPoseRecord:
    if not records:
        raise ValueError("Cannot associate against an empty TUM trajectory")
    if not np.isfinite(timestamp):
        raise ValueError("timestamp must be finite")
    if not np.isfinite(max_delta_sec):
        raise ValueError("max_delta_sec must be finite")
    if max_delta_sec < 0.0:
        raise ValueError("max_delta_sec must be non-negative")

    sorted_records = sorted(records, key=lambda record: record.timestamp)
    timestamps = [record.timestamp for record in sorted_records]
    insertion_index = bisect_left(timestamps, timestamp)
    candidate_indices = []
    if insertion_index < len(sorted_records):
        candidate_indices.append(insertion_index)
    if insertion_index > 0:
        candidate_indices.append(insertion_index - 1)

    nearest = min(candidate_indices, key=lambda index: abs(timestamps[index] - timestamp))
    delta = abs(timestamps[nearest] - timestamp)
    if delta > max_delta_sec:
        raise ValueError("Nearest TUM timestamp exceeds max_delta_sec")

    return sorted_records[nearest]
