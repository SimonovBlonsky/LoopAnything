"""Batch helpers for offline robust loop verifier experiments."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_RMSE_RE = re.compile(
    r"^\s*rmse\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class AteSequence:
    platform: str
    sequence_name: str
    ate_rmse_m: float
    sequence_root: Path
    ate_summary_file: Path

    def to_json(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "sequence_name": self.sequence_name,
            "ate_rmse_m": self.ate_rmse_m,
            "sequence_root": str(self.sequence_root),
            "ate_summary_file": str(self.ate_summary_file),
        }


@dataclass(frozen=True)
class BatchRunRoot:
    platform: str
    sequence_name: str
    run_root: Path
    ate_rmse_m: float | None = None


def parse_evo_ape_rmse(text: str) -> float:
    match = _RMSE_RE.search(text)
    if match is None:
        raise ValueError("evo APE summary does not contain an rmse row")
    value = float(match.group(1))
    if not math.isfinite(value):
        raise ValueError("evo APE rmse must be finite")
    return value


def discover_aster_slam_ate_sequences(
    loop_dataset_root: Path,
    *,
    platforms: Sequence[str],
    ate_rmse_threshold_m: float,
) -> list[AteSequence]:
    if ate_rmse_threshold_m <= 0.0 or not math.isfinite(ate_rmse_threshold_m):
        raise ValueError("ate_rmse_threshold_m must be a finite positive value")

    root = Path(loop_dataset_root)
    selected: list[AteSequence] = []
    for platform in platforms:
        platform_root = root / platform
        if not platform_root.is_dir():
            continue
        for sequence_root in sorted(path for path in platform_root.iterdir() if path.is_dir()):
            ate_summary_file = sequence_root / "raw" / "evo_ape_summary.txt"
            if not ate_summary_file.is_file():
                continue
            rmse = parse_evo_ape_rmse(ate_summary_file.read_text(encoding="utf-8"))
            if rmse < ate_rmse_threshold_m:
                selected.append(
                    AteSequence(
                        platform=platform,
                        sequence_name=sequence_root.name,
                        ate_rmse_m=rmse,
                        sequence_root=sequence_root,
                        ate_summary_file=ate_summary_file,
                    )
                )

    return sorted(selected, key=lambda item: (item.platform, item.sequence_name))


def read_batch_summary_run_roots(path: Path) -> list[BatchRunRoot]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("batch summary JSON root must be an object")
    sequences = data.get("sequences")
    if not isinstance(sequences, list):
        raise ValueError("batch summary must contain a sequences list")

    records: list[BatchRunRoot] = []
    for row in sequences:
        if not isinstance(row, Mapping):
            raise ValueError("batch summary sequence entries must be objects")
        records.append(
            BatchRunRoot(
                platform=str(row["platform"]),
                sequence_name=str(row["sequence_name"]),
                run_root=Path(str(row["run_root"])),
                ate_rmse_m=_optional_float(row.get("ate_rmse_m")),
            )
        )
    return records


def average_score_sweep_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)

    averages: list[dict[str, Any]] = []
    for method, method_rows in grouped.items():
        ap_values = [_finite_float(row["AP"], "AP") for row in method_rows]
        mr_values = [_finite_float(row["MR@100P"], "MR@100P") for row in method_rows]
        averages.append(
            {
                "method": method,
                "sequence_count": len(method_rows),
                "average_AP": sum(ap_values) / len(ap_values),
                "average_MR@100P": sum(mr_values) / len(mr_values),
            }
        )

    return sorted(averages, key=lambda row: row["method"])


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown_table(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    columns: Sequence[str],
) -> None:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = [_format_markdown_value(row.get(column)) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return _finite_float(value, "value")


def _finite_float(value: Any, field_name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field_name} must be finite")
    return numeric


def _format_markdown_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return "" if value is None else str(value)
