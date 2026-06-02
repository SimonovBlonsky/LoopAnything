#!/usr/bin/env python3
"""Convert NTU-VIRAL ground_truth.csv files to TUM trajectories.

The public NTU-VIRAL CSV trajectories in this workspace often carry identity
quaternions. For visual loop labels, a pure translation-positive definition is
too weak, so this converter can replace near-constant rotations with planar
heading rotations estimated from the GT trajectory.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    input_file = Path(args.input).resolve()
    output_file = Path(args.output).resolve()
    timestamps, positions, quaternions = _read_csv(input_file)
    source = "csv_quaternion"
    if args.rotation_source == "heading" or (
        args.rotation_source == "auto" and _rotation_extent_deg(quaternions) <= args.identity_deg
    ):
        quaternions = _heading_quaternions_from_positions(positions)
        source = "translation_heading"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    _write_tum(output_file, timestamps, positions, quaternions)
    print(f"output={output_file}")
    print(f"rotation_source={source}")
    print(f"frame_count={len(timestamps)}")
    return 0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="NTU-VIRAL ground_truth.csv")
    parser.add_argument("--output", required=True, help="Output TUM trajectory path")
    parser.add_argument(
        "--rotation-source",
        choices=("auto", "csv", "heading"),
        default="auto",
        help="Use CSV quaternion, trajectory heading, or auto fallback when CSV rotation is identity.",
    )
    parser.add_argument(
        "--identity-deg",
        type=float,
        default=1.0,
        help="Auto mode treats rotations with this max extent as identity.",
    )
    return parser.parse_args(argv)


def _read_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps: list[float] = []
    positions: list[list[float]] = []
    quaternions: list[list[float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            timestamps.append(_timestamp_to_seconds(row["%time"]))
            positions.append(
                [
                    float(row["field.pose.position.x"]),
                    float(row["field.pose.position.y"]),
                    float(row["field.pose.position.z"]),
                ]
            )
            quaternions.append(
                _normalize_quaternion(
                    [
                        float(row["field.pose.orientation.x"]),
                        float(row["field.pose.orientation.y"]),
                        float(row["field.pose.orientation.z"]),
                        float(row["field.pose.orientation.w"]),
                    ]
                )
            )
    if not timestamps:
        raise ValueError(f"No rows found in {path}")
    order = np.argsort(np.asarray(timestamps, dtype=np.float64))
    return (
        np.asarray(timestamps, dtype=np.float64)[order],
        np.asarray(positions, dtype=np.float64)[order],
        np.asarray(quaternions, dtype=np.float64)[order],
    )


def _timestamp_to_seconds(value: str) -> float:
    raw = float(value)
    return raw * 1e-9 if raw > 1e12 else raw


def _normalize_quaternion(values: Iterable[float]) -> list[float]:
    q = np.asarray(list(values), dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm <= 0.0:
        return [0.0, 0.0, 0.0, 1.0]
    q /= norm
    return q.tolist()


def _rotation_extent_deg(quaternions: np.ndarray) -> float:
    if len(quaternions) < 2:
        return 0.0
    first = quaternions[0]
    dots = np.abs(quaternions @ first)
    dots = np.clip(dots, -1.0, 1.0)
    return float(np.max(2.0 * np.degrees(np.arccos(dots))))


def _heading_quaternions_from_positions(positions: np.ndarray) -> np.ndarray:
    headings: list[float] = []
    last_heading = 0.0
    for idx in range(len(positions)):
        if len(positions) == 1:
            delta = np.zeros(3, dtype=np.float64)
        elif idx == 0:
            delta = positions[1] - positions[0]
        elif idx == len(positions) - 1:
            delta = positions[-1] - positions[-2]
        else:
            delta = positions[idx + 1] - positions[idx - 1]
        if float(np.linalg.norm(delta[:2])) > 1e-6:
            last_heading = math.atan2(float(delta[1]), float(delta[0]))
        headings.append(last_heading)
    return np.asarray([_yaw_to_quaternion(yaw) for yaw in headings], dtype=np.float64)


def _yaw_to_quaternion(yaw: float) -> list[float]:
    half = 0.5 * yaw
    return [0.0, 0.0, math.sin(half), math.cos(half)]


def _write_tum(
    path: Path,
    timestamps: np.ndarray,
    positions: np.ndarray,
    quaternions: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for timestamp, position, quaternion in zip(timestamps, positions, quaternions):
            handle.write(
                "{:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f}\n".format(
                    float(timestamp),
                    float(position[0]),
                    float(position[1]),
                    float(position[2]),
                    float(quaternion[0]),
                    float(quaternion[1]),
                    float(quaternion[2]),
                    float(quaternion[3]),
                )
            )


if __name__ == "__main__":
    raise SystemExit(main())
