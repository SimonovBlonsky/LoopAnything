#!/usr/bin/env python3
"""Compare old appearance labels with automatic and reviewed geometry labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

LOOPANYTHING_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = LOOPANYTHING_ROOT / "src"
for import_path in (LOOPANYTHING_ROOT, SRC_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from robust_loop_verifier.rover_annotation import (  # noqa: E402
    sha256_file,
    verify_annotation_seal,
)

PAIR_FILENAME = "benchmark_pairs.jsonl"
BENCHMARK_MANIFEST_FILENAME = "manifest.json"
ANNOTATION_FILENAME = "annotations.jsonl"
ANNOTATION_SEAL_FILENAME = "annotation_seal.json"
PREDICTION_FILENAME = "geometry_predictions.jsonl"
PREDICTION_MANIFEST_FILENAME = "geometry_prediction_manifest.json"
FORMAT_VERSION = 1
LEGACY_THRESHOLD_FIELDS = {
    "max_translation_error_m",
    "max_rotation_error_deg",
    "max_translation_direction_error_deg",
    "min_direction_baseline_m",
}
SCALE_ADAPTIVE_THRESHOLD_FIELDS = LEGACY_THRESHOLD_FIELDS | {
    "min_translation_error_m",
    "translation_error_scale_ratio",
}
COMMON_DA3_CONFIG_FIELDS = {
    "backend",
    "device",
    "verifier_configs",
    "verifier_config_sha256",
    "salad_score_file",
    "salad_score_file_sha256",
    "salad_score_manifest",
    "salad_score_manifest_sha256",
    "da3_model_name",
    "da3_model_dir",
    "da3_model_path",
    "da3_model_path_sha256",
    "da3_checkpoint",
    "da3_checkpoint_sha256",
    "da3_snapshot",
    "da3_snapshot_sha256",
    "da3_runtime_by_dataset",
}
REAL_DA3_CONFIG_FIELDS = {
    "da3_cache_dir",
    "da3_cache_source",
    "da3_process_res",
    "da3_ref_view_strategy",
    "da3_triplet_batch_size",
}
REAL_DATASET_RUNTIME_FIELDS = {
    "da3_model_name",
    "da3_model_dir",
    "da3_model_path",
    "da3_model_path_sha256",
    "da3_checkpoint",
    "da3_checkpoint_sha256",
    "da3_snapshot",
    "da3_snapshot_sha256",
    "da3_cache_dir",
    "da3_cache_source",
}
MACRO_DEFINITION = {
    "count_fields": "sum across sequences",
    "rate_precision_recall_pose_fields": ("unweighted mean over sequences with finite values"),
    "pose_fields": "mean of per-sequence percentiles, not pooled percentiles",
}
ERROR_FIELDS = (
    "translation_error_m",
    "rotation_error_deg",
    "translation_direction_error_deg",
)
COUNT_FIELDS = (
    "pair_count",
    "old_positive_count",
    "automatic_positive_count",
    "reviewed_positive_count",
    "old_negative_to_reviewed_positive",
    "old_positive_to_reviewed_negative",
    "support_failed_count",
    "da3_failed_count",
    "sim3_failed_count",
    "other_factor_failure_count",
)
AVERAGE_FIELDS = (
    "old_positive_rate",
    "automatic_positive_rate",
    "reviewed_positive_rate",
    "automatic_precision",
    "automatic_recall",
    *tuple(
        f"reviewed_{label}_{field}_{statistic}"
        for label in ("positive", "negative")
        for field in ERROR_FIELDS
        for statistic in ("median", "p90")
    ),
)
METRIC_FIELDS = (
    *COUNT_FIELDS[:4],
    *AVERAGE_FIELDS[:3],
    *COUNT_FIELDS[4:6],
    *AVERAGE_FIELDS[3:5],
    *COUNT_FIELDS[6:],
    *AVERAGE_FIELDS[5:],
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_root", type=Path)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument(
        "--validate-predictions-only",
        action="store_true",
        help="validate the frozen pair copy and complete prediction manifest without labels",
    )
    parser.add_argument(
        "--allow-partial-predictions",
        action="store_true",
        help="allow a complete pair-limit target prefix during validate-only smoke checks",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.allow_partial_predictions and not args.validate_predictions_only:
        parser.error("--allow-partial-predictions requires --validate-predictions-only")
    if args.validate_predictions_only:
        validate_prediction_bundle(
            args.benchmark_root,
            args.geometry_root,
            require_full_pair_coverage=not args.allow_partial_predictions,
        )
        return 0
    report = summarize(args.benchmark_root, args.geometry_root)
    write_report(args.geometry_root, report)
    return 0


def summarize(benchmark_root: Path, geometry_root: Path) -> dict[str, object]:
    benchmark_root = Path(benchmark_root)
    geometry_root = Path(geometry_root)
    pairs, predictions, prediction_manifest = validate_prediction_bundle(
        benchmark_root,
        geometry_root,
        require_full_pair_coverage=True,
    )

    old_annotations = _verify_and_read_annotations(benchmark_root)
    new_annotations = _verify_and_read_annotations(geometry_root)
    pair_ids = [str(pair["pair_id"]) for pair in pairs]
    _require_ordered_pair_ids(old_annotations, pair_ids, "old annotations")
    _require_ordered_pair_ids(new_annotations, pair_ids, "new annotations")

    joined = [
        {
            "pair": pair,
            "old_label": old_annotation["label"],
            "reviewed_label": new_annotation["label"],
            "prediction": prediction,
        }
        for pair, old_annotation, new_annotation, prediction in zip(
            pairs,
            old_annotations,
            new_annotations,
            predictions,
        )
    ]
    grouped: OrderedDict[tuple[str, str, str], list[Mapping[str, object]]] = OrderedDict()
    for item in joined:
        pair = item["pair"]
        key = (
            _required_string(pair, "dataset", "benchmark pair"),
            _required_string(pair, "platform", "benchmark pair"),
            _required_string(pair, "sequence", "benchmark pair"),
        )
        grouped.setdefault(key, []).append(item)

    sequences = []
    for (dataset, platform, sequence), items in grouped.items():
        sequences.append(
            {
                "dataset": dataset,
                "platform": platform,
                "sequence": sequence,
                **_summarize_items(items),
            }
        )

    return {
        "format_version": 1,
        "provenance": {
            "benchmark_root": str(benchmark_root.resolve()),
            "geometry_root": str(geometry_root.resolve()),
            "thresholds": prediction_manifest["thresholds"],
            "source_pair_manifest_sha256": sha256_file(benchmark_root / PAIR_FILENAME),
            "geometry_prediction_manifest_sha256": sha256_file(
                geometry_root / PREDICTION_MANIFEST_FILENAME
            ),
            "old_annotation_seal_sha256": sha256_file(benchmark_root / ANNOTATION_SEAL_FILENAME),
            "new_annotation_seal_sha256": sha256_file(geometry_root / ANNOTATION_SEAL_FILENAME),
        },
        "macro_definition": dict(MACRO_DEFINITION),
        "sequences": sequences,
        "overall": _summarize_items(joined),
        "macro": _macro_summary(sequences),
    }


def validate_prediction_bundle(
    benchmark_root: Path,
    geometry_root: Path,
    *,
    require_full_pair_coverage: bool,
) -> tuple[list[dict], list[dict], dict]:
    benchmark_root = Path(benchmark_root)
    geometry_root = Path(geometry_root)
    for filename in (PAIR_FILENAME, BENCHMARK_MANIFEST_FILENAME):
        source = (benchmark_root / filename).read_bytes()
        copied = (geometry_root / filename).read_bytes()
        if copied != source:
            raise ValueError(f"frozen benchmark {filename} must be exact byte-for-byte copy")

    pairs = _read_jsonl_strict(geometry_root / PAIR_FILENAME)
    pair_ids = _validated_pair_ids(pairs, "benchmark pairs")
    pair_hash = sha256_file(geometry_root / PAIR_FILENAME)
    benchmark_manifest = _read_json_object(geometry_root / BENCHMARK_MANIFEST_FILENAME)
    if benchmark_manifest.get("pair_manifest_sha256") != pair_hash:
        raise ValueError(
            "benchmark manifest pair_manifest_sha256 does not match benchmark_pairs.jsonl"
        )

    prediction_manifest_path = geometry_root / PREDICTION_MANIFEST_FILENAME
    prediction_manifest = _read_json_object(prediction_manifest_path)
    if (
        prediction_manifest.get("format_version") != FORMAT_VERSION
        or type(prediction_manifest.get("format_version")) is not int
    ):
        raise ValueError("geometry prediction manifest format_version must be exact int 1")
    if prediction_manifest.get("prediction_file") != PREDICTION_FILENAME:
        raise ValueError(
            f"geometry prediction manifest prediction_file must be {PREDICTION_FILENAME}"
        )
    if type(prediction_manifest.get("complete")) is not bool:
        raise ValueError("geometry prediction manifest complete must be an exact bool")
    if prediction_manifest["complete"] is not True:
        raise ValueError("geometry prediction manifest complete must be true")
    if prediction_manifest.get("source_pair_manifest_sha256") != pair_hash:
        raise ValueError("geometry prediction manifest source_pair_manifest_sha256 mismatch")
    source_manifest_hash = sha256_file(geometry_root / BENCHMARK_MANIFEST_FILENAME)
    if prediction_manifest.get("source_manifest_sha256") != source_manifest_hash:
        raise ValueError("geometry prediction manifest source_manifest_sha256 mismatch")
    _require_exact_count(
        prediction_manifest,
        "source_pair_count",
        len(pair_ids),
    )

    predictions = _read_jsonl_strict(
        geometry_root / PREDICTION_FILENAME,
        allow_nonfinite_constants=True,
    )
    prediction_ids = _validated_pair_ids(predictions, "geometry predictions")
    target_count = prediction_manifest.get("target_pair_count")
    if type(target_count) is not int or target_count < 0:
        raise ValueError(
            "geometry prediction manifest target_pair_count must be a non-negative exact int"
        )
    if target_count != len(predictions):
        raise ValueError("geometry prediction manifest target_pair_count mismatch")
    pair_limit = _validate_pair_limit(prediction_manifest.get("pair_limit"))
    expected_target_count = len(pair_ids) if pair_limit is None else min(pair_limit, len(pair_ids))
    if target_count != expected_target_count:
        raise ValueError(
            "geometry prediction manifest pair_limit does not match target_pair_count"
        )
    _require_exact_count(
        prediction_manifest,
        "prediction_record_count",
        len(predictions),
    )
    expected_target_ids = pair_ids[:target_count]
    if prediction_ids != expected_target_ids:
        raise ValueError(
            "geometry prediction pair order and coverage must match target benchmark pairs"
        )
    if require_full_pair_coverage and prediction_ids != pair_ids:
        raise ValueError(
            "geometry predictions must have full pair order and coverage for UI/summary; "
            "partial target prefixes are only valid for pair-limit smoke validation"
        )
    expected_target_hash = _pair_ids_sha256(expected_target_ids)
    if prediction_manifest.get("target_pair_ids_sha256") != expected_target_hash:
        raise ValueError("geometry prediction manifest target pair IDs hash mismatch")
    prediction_hash = sha256_file(geometry_root / PREDICTION_FILENAME)
    if prediction_manifest.get("prediction_file_sha256") != prediction_hash:
        raise ValueError("geometry prediction file hash mismatch")

    thresholds = _validate_thresholds(
        prediction_manifest.get("thresholds"),
        "thresholds",
    )
    request_contract = _validate_request_contract(
        prediction_manifest.get("request_contract"),
        pair_limit=pair_limit,
        thresholds=thresholds,
    )
    da3_config = prediction_manifest.get("da3_config")
    da3_config_hash = prediction_manifest.get("da3_config_sha256")
    _validate_sha256(da3_config_hash, "da3_config_sha256")
    if da3_config_hash != _canonical_json_sha256(da3_config):
        raise ValueError("da3_config_sha256 does not match da3_config")
    _validate_da3_config(da3_config, request_contract)
    source_commit = prediction_manifest.get("source_commit")
    if not isinstance(source_commit, str) or not source_commit:
        raise ValueError("geometry prediction manifest source_commit must be non-empty")
    source_commit_error = prediction_manifest.get("source_commit_error")
    if source_commit_error is not None and (
        not isinstance(source_commit_error, str) or not source_commit_error
    ):
        raise ValueError("geometry prediction manifest source_commit_error must be non-empty")
    command = prediction_manifest.get("command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(item, str) or not item for item in command)
    ):
        raise ValueError("geometry prediction manifest command must contain non-empty strings")

    for index, prediction in enumerate(predictions, start=1):
        _reject_nonfinite_prediction_non_error_fields(prediction, index)
        label = prediction.get("automatic_label")
        if type(label) is not int or label not in (0, 1):
            raise ValueError(
                f"geometry prediction line {index} automatic_label must be exact int 0 or 1"
            )
        _required_string(prediction, "factor_status", f"geometry prediction line {index}")
        for field in ERROR_FIELDS:
            if field not in prediction:
                raise ValueError(f"geometry prediction line {index} is missing {field}")
            value = prediction[field]
            if value is not None and type(value) not in (int, float):
                raise ValueError(
                    f"geometry prediction line {index} {field} must be numeric or None"
                )
    return pairs, predictions, prediction_manifest


def _reject_nonfinite_prediction_non_error_fields(
    prediction: Mapping[str, object],
    line_number: int,
) -> None:
    for field, value in prediction.items():
        if field in ERROR_FIELDS:
            continue
        _reject_nonfinite_json_value(
            value,
            f"geometry prediction line {line_number} {field}",
        )


def _reject_nonfinite_json_value(value: object, context: str) -> None:
    if type(value) is float and not math.isfinite(value):
        raise ValueError(f"{context} contains non-finite value")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_nonfinite_json_value(child, f"{context}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_nonfinite_json_value(child, f"{context}[{index}]")


def write_report(geometry_root: Path, report: Mapping[str, object]) -> None:
    payloads = {
        "label_comparison.csv": _render_csv(report),
        "label_comparison.md": _render_markdown(report),
        "label_comparison.json": json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    }
    _write_payload_set_atomically(Path(geometry_root), payloads)


def _verify_and_read_annotations(root: Path) -> list[dict]:
    verify_annotation_seal(
        root / BENCHMARK_MANIFEST_FILENAME,
        root / PAIR_FILENAME,
        root / ANNOTATION_FILENAME,
        root / ANNOTATION_SEAL_FILENAME,
    )
    rows = _read_jsonl_strict(root / ANNOTATION_FILENAME)
    for index, row in enumerate(rows, start=1):
        label = row.get("label")
        if type(label) is not int or label not in (0, 1):
            raise ValueError(f"annotation line {index} label must be exact int 0 or 1")
    return rows


def _summarize_items(items: Sequence[Mapping[str, object]]) -> dict[str, object]:
    pair_count = len(items)
    old_positive_count = sum(item["old_label"] == 1 for item in items)
    automatic_positive_count = sum(item["prediction"]["automatic_label"] == 1 for item in items)
    reviewed_positive_count = sum(item["reviewed_label"] == 1 for item in items)
    true_positive = sum(
        item["prediction"]["automatic_label"] == 1 and item["reviewed_label"] == 1
        for item in items
    )
    false_positive = automatic_positive_count - true_positive
    false_negative = reviewed_positive_count - true_positive
    metrics: dict[str, object] = {
        "pair_count": pair_count,
        "old_positive_count": old_positive_count,
        "old_positive_rate": _safe_ratio(old_positive_count, pair_count),
        "automatic_positive_count": automatic_positive_count,
        "automatic_positive_rate": _safe_ratio(automatic_positive_count, pair_count),
        "reviewed_positive_count": reviewed_positive_count,
        "reviewed_positive_rate": _safe_ratio(reviewed_positive_count, pair_count),
        "old_negative_to_reviewed_positive": sum(
            item["old_label"] == 0 and item["reviewed_label"] == 1 for item in items
        ),
        "old_positive_to_reviewed_negative": sum(
            item["old_label"] == 1 and item["reviewed_label"] == 0 for item in items
        ),
        "automatic_precision": _safe_ratio(
            true_positive,
            true_positive + false_positive,
        ),
        "automatic_recall": _safe_ratio(
            true_positive,
            true_positive + false_negative,
        ),
    }
    statuses = [str(item["prediction"]["factor_status"]) for item in items]
    metrics.update(
        {
            "support_failed_count": statuses.count("support_failed"),
            "da3_failed_count": statuses.count("da3_failed"),
            "sim3_failed_count": statuses.count("sim3_failed"),
            "other_factor_failure_count": sum(
                status not in {"ok", "support_failed", "da3_failed", "sim3_failed"}
                for status in statuses
            ),
        }
    )
    for label_name, label_value in (("positive", 1), ("negative", 0)):
        for field in ERROR_FIELDS:
            values = [
                float(item["prediction"][field])
                for item in items
                if item["reviewed_label"] == label_value
                and _is_finite_number(item["prediction"][field])
            ]
            median, p90 = _percentiles(values)
            metrics[f"reviewed_{label_name}_{field}_median"] = median
            metrics[f"reviewed_{label_name}_{field}_p90"] = p90
    return metrics


def _macro_summary(sequences: Sequence[Mapping[str, object]]) -> dict[str, object]:
    macro: dict[str, object] = {"sequence_count": len(sequences)}
    for field in COUNT_FIELDS:
        macro[field] = sum(int(row[field]) for row in sequences)
    for field in AVERAGE_FIELDS:
        values = [float(row[field]) for row in sequences if _is_finite_number(row.get(field))]
        macro[field] = sum(values) / len(values) if values else None
    return macro


def _percentiles(values: Sequence[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    median, p90 = np.percentile(
        np.asarray(values, dtype=np.float64),
        [50.0, 90.0],
        method="linear",
    )
    return float(median), float(p90)


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _is_finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _render_csv(report: Mapping[str, object]) -> str:
    fieldnames = [
        "scope",
        "dataset",
        "platform",
        "sequence",
        "sequence_count",
        *METRIC_FIELDS,
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in _report_rows(report):
        writer.writerow({field: _format_tabular(row.get(field)) for field in fieldnames})
    return buffer.getvalue()


def _render_markdown(report: Mapping[str, object]) -> str:
    provenance = report["provenance"]
    macro_definition = report["macro_definition"]
    lines = [
        "# DA3 Geometry Label Comparison",
        "",
        f"- Source pair SHA256: `{provenance['source_pair_manifest_sha256']}`",
        (
            "- Prediction manifest SHA256: "
            f"`{provenance['geometry_prediction_manifest_sha256']}`"
        ),
        f"- Old annotation seal SHA256: `{provenance['old_annotation_seal_sha256']}`",
        f"- New annotation seal SHA256: `{provenance['new_annotation_seal_sha256']}`",
        f"- Thresholds: `{json.dumps(provenance['thresholds'], sort_keys=True)}`",
        (
            "- Macro counts: "
            f"{macro_definition['count_fields']}; rates, precision, recall, and "
            f"pose summaries: {macro_definition['rate_precision_recall_pose_fields']}; "
            f"pose macro: {macro_definition['pose_fields']}."
        ),
        "",
        "## Label Metrics",
        "",
        (
            "| scope | pairs | old + | auto + | reviewed + | old - to reviewed + | "
            "old + to reviewed - | precision | recall | support fail | DA3 fail | "
            "Sim3 fail | other fail |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in _report_rows(report):
        label = _scope_label(row)
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    _format_tabular(row.get("pair_count")),
                    _count_rate(row, "old_positive_count", "old_positive_rate"),
                    _count_rate(
                        row,
                        "automatic_positive_count",
                        "automatic_positive_rate",
                    ),
                    _count_rate(
                        row,
                        "reviewed_positive_count",
                        "reviewed_positive_rate",
                    ),
                    _format_tabular(row.get("old_negative_to_reviewed_positive")),
                    _format_tabular(row.get("old_positive_to_reviewed_negative")),
                    _format_tabular(row.get("automatic_precision")),
                    _format_tabular(row.get("automatic_recall")),
                    _format_tabular(row.get("support_failed_count")),
                    _format_tabular(row.get("da3_failed_count")),
                    _format_tabular(row.get("sim3_failed_count")),
                    _format_tabular(row.get("other_factor_failure_count")),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Pose Error Statistics",
            "",
            "| scope | reviewed label | metric | median | p90 |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for row in _report_rows(report):
        for label in ("positive", "negative"):
            for field in ERROR_FIELDS:
                prefix = f"reviewed_{label}_{field}"
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            _scope_label(row),
                            label,
                            field,
                            _format_tabular(row.get(f"{prefix}_median")),
                            _format_tabular(row.get(f"{prefix}_p90")),
                        ]
                    )
                    + " |"
                )
    lines.append("")
    return "\n".join(lines)


def _report_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows = [{"scope": "sequence", **dict(row)} for row in report["sequences"]]
    rows.append({"scope": "overall", **dict(report["overall"])})
    rows.append({"scope": "macro", **dict(report["macro"])})
    return rows


def _scope_label(row: Mapping[str, object]) -> str:
    if row["scope"] == "sequence":
        return str(row["sequence"])
    return str(row["scope"])


def _count_rate(
    row: Mapping[str, object],
    count_field: str,
    rate_field: str,
) -> str:
    return f"{_format_tabular(row.get(count_field))} " f"({_format_tabular(row.get(rate_field))})"


def _format_tabular(value: object) -> str:
    if value is None or value == "":
        return "N/A"
    if type(value) is float:
        return f"{value:.6g}"
    return str(value)


def _read_jsonl_strict(
    path: Path,
    *,
    allow_nonfinite_constants: bool = False,
) -> list[dict]:
    data = Path(path).read_bytes()
    if data and not data.endswith(b"\n"):
        raise ValueError(f"{path.name} must end with a newline")
    rows = []
    parse_kwargs = {} if allow_nonfinite_constants else {"parse_constant": _reject_json_constant}
    for line_number, raw_line in enumerate(data.splitlines(), start=1):
        if not raw_line.strip():
            raise ValueError(f"{path.name} line {line_number} is blank")
        try:
            row = json.loads(raw_line, **parse_kwargs)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"{path.name} line {line_number} is invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path.name} line {line_number} must be an object")
        rows.append(row)
    return rows


def _read_json_object(path: Path) -> dict:
    try:
        payload = json.loads(
            Path(path).read_bytes(),
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{path.name} is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _validated_pair_ids(rows: Sequence[Mapping[str, object]], context: str) -> list[str]:
    pair_ids = []
    for index, row in enumerate(rows, start=1):
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError(f"{context} line {index} has invalid pair_id")
        pair_ids.append(pair_id)
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError(f"{context} contains duplicate pair_id")
    return pair_ids


def _require_ordered_pair_ids(
    rows: Sequence[Mapping[str, object]],
    expected: Sequence[str],
    context: str,
) -> None:
    actual = _validated_pair_ids(rows, context)
    if actual != list(expected):
        raise ValueError(f"{context} pair IDs must have exact order and coverage")


def _required_string(
    mapping: Mapping[str, object],
    field: str,
    context: str,
) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} {field} must be a non-empty string")
    return value


def _require_exact_count(
    mapping: Mapping[str, object],
    field: str,
    expected: int,
) -> None:
    value = mapping.get(field)
    if type(value) is not int or value != expected:
        raise ValueError(f"geometry prediction manifest {field} mismatch")


def _validate_pair_limit(value: object) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value <= 0:
        raise ValueError(
            "geometry prediction manifest pair_limit must be a positive exact int or None"
        )
    return value


def _validate_thresholds(value: object, context: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) not in {
        frozenset(LEGACY_THRESHOLD_FIELDS),
        frozenset(SCALE_ADAPTIVE_THRESHOLD_FIELDS),
    }:
        raise ValueError(
            f"geometry prediction manifest {context} must contain exact threshold fields"
        )
    thresholds: dict[str, float] = {}
    for field in sorted(value):
        threshold = value[field]
        if type(threshold) is not float or not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError(
                f"geometry prediction manifest {context}.{field} "
                "must be a finite non-negative exact float"
            )
        thresholds[field] = threshold
    return thresholds


def _validate_request_contract(
    value: object,
    *,
    pair_limit: int | None,
    thresholds: Mapping[str, float],
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "backend",
        "device",
        "pair_limit",
        "thresholds",
    }:
        raise ValueError("geometry prediction manifest request_contract fields mismatch")
    backend = value.get("backend")
    if backend not in {"real", "mock"}:
        raise ValueError(
            "geometry prediction manifest request_contract.backend must be real or mock"
        )
    device = value.get("device")
    if not isinstance(device, str) or not device:
        raise ValueError("geometry prediction manifest request_contract.device must be non-empty")
    if _validate_pair_limit(value.get("pair_limit")) != pair_limit:
        raise ValueError("geometry prediction manifest request_contract.pair_limit mismatch")
    request_thresholds = _validate_thresholds(
        value.get("thresholds"),
        "request_contract.thresholds",
    )
    if request_thresholds != dict(thresholds):
        raise ValueError("geometry prediction manifest request_contract.thresholds mismatch")
    return {
        "backend": backend,
        "device": device,
        "pair_limit": pair_limit,
        "thresholds": request_thresholds,
    }


def _validate_da3_config(value: object, request_contract: Mapping[str, object]) -> None:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("geometry prediction manifest da3_config must be non-empty")
    missing_fields = COMMON_DA3_CONFIG_FIELDS - set(value)
    if missing_fields:
        raise ValueError(
            "geometry prediction manifest da3_config missing required field: "
            f"{sorted(missing_fields)[0]}"
        )
    for field in ("backend", "device"):
        config_value = value.get(field)
        if (
            not isinstance(config_value, str)
            or not config_value
            or config_value != request_contract[field]
        ):
            raise ValueError(f"geometry prediction manifest da3_config {field} mismatch")
    backend = str(value["backend"])
    if backend == "real":
        missing_real_fields = REAL_DA3_CONFIG_FIELDS - set(value)
        if missing_real_fields:
            raise ValueError(
                "geometry prediction manifest real da3_config missing required field: "
                f"{sorted(missing_real_fields)[0]}"
            )

    verifier_configs = value.get("verifier_configs")
    if not isinstance(verifier_configs, Mapping) or not verifier_configs:
        raise ValueError(
            "geometry prediction manifest da3_config verifier_configs " "must be non-empty"
        )
    for dataset, config_path in verifier_configs.items():
        if (
            not isinstance(dataset, str)
            or not dataset
            or not isinstance(config_path, str)
            or not config_path
        ):
            raise ValueError(
                "geometry prediction manifest da3_config verifier_configs "
                "must map non-empty strings"
            )

    config_hashes = value.get("verifier_config_sha256")
    if not isinstance(config_hashes, Mapping) or set(config_hashes) != set(verifier_configs):
        raise ValueError("geometry prediction manifest da3_config verifier_config_sha256 mismatch")
    for dataset, config_hash in config_hashes.items():
        if config_hash is not None:
            _validate_sha256(
                config_hash,
                f"da3_config.verifier_config_sha256[{dataset!r}]",
            )

    runtime_by_dataset = value.get("da3_runtime_by_dataset")
    if not isinstance(runtime_by_dataset, Mapping) or set(runtime_by_dataset) != set(
        verifier_configs
    ):
        raise ValueError("geometry prediction manifest da3_config da3_runtime_by_dataset mismatch")
    for dataset, runtime in runtime_by_dataset.items():
        if not isinstance(runtime, Mapping):
            raise ValueError(
                f"geometry prediction manifest da3 runtime {dataset!r} must be a mapping"
            )
        process_res = runtime.get("process_res")
        if type(process_res) is not int or process_res <= 0:
            raise ValueError(
                "geometry prediction manifest da3 runtime process_res "
                "must be a positive integer"
            )
        ref_view_strategy = runtime.get("ref_view_strategy")
        if not isinstance(ref_view_strategy, str) or not ref_view_strategy:
            raise ValueError(
                "geometry prediction manifest da3 runtime ref_view_strategy " "must be non-empty"
            )
        triplet_batch_size = runtime.get("triplet_batch_size")
        if type(triplet_batch_size) is not int or triplet_batch_size <= 0:
            raise ValueError(
                "geometry prediction manifest da3 runtime triplet_batch_size "
                "must be a positive integer"
            )
        if backend == "real":
            missing_runtime_fields = REAL_DATASET_RUNTIME_FIELDS - set(runtime)
            if missing_runtime_fields:
                raise ValueError(
                    "geometry prediction manifest real da3 runtime missing field: "
                    f"{sorted(missing_runtime_fields)[0]}"
                )
        _validate_optional_da3_provenance(runtime, f"runtime[{dataset!r}]")

    if backend == "real":
        for field in ("da3_process_res", "da3_triplet_batch_size"):
            field_value = value.get(field)
            if type(field_value) is not int or field_value <= 0:
                raise ValueError(
                    f"geometry prediction manifest real da3_config {field} "
                    "must be a positive integer"
                )
        ref_view_strategy = value.get("da3_ref_view_strategy")
        if not isinstance(ref_view_strategy, str) or not ref_view_strategy:
            raise ValueError(
                "geometry prediction manifest real da3_config "
                "da3_ref_view_strategy must be non-empty"
            )
    _validate_optional_da3_provenance(value, "da3_config")


def _validate_optional_da3_provenance(
    value: Mapping[str, object],
    context: str,
) -> None:
    optional_string_fields = (
        "da3_model_name",
        "da3_model_dir",
        "da3_model_path",
        "da3_checkpoint",
        "da3_snapshot",
        "da3_cache_dir",
        "da3_cache_source",
        "salad_score_file",
        "salad_score_manifest",
    )
    optional_hash_fields = (
        "da3_model_path_sha256",
        "da3_checkpoint_sha256",
        "da3_snapshot_sha256",
        "salad_score_file_sha256",
        "salad_score_manifest_sha256",
    )
    for field in (*optional_string_fields, *optional_hash_fields):
        if field not in value or value[field] is None:
            continue
        if field in optional_hash_fields:
            _validate_sha256(value[field], f"{context}.{field}")
        elif not isinstance(value[field], str) or not value[field]:
            raise ValueError(f"geometry prediction manifest {context}.{field} has invalid type")


def _validate_sha256(value: object, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError(f"{field_name} must be a 64-character hex SHA256")


def _canonical_json_sha256(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("da3_config must be canonical JSON-safe data") from exc
    return hashlib.sha256(encoded).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _pair_ids_sha256(pair_ids: Sequence[str]) -> str:
    content = "".join(f"{pair_id}\n" for pair_id in pair_ids).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _write_payload_set_atomically(
    output_dir: Path,
    payloads: Mapping[str, str],
) -> None:
    output_dir = Path(output_dir)
    output_parent = output_dir.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(
        tempfile.mkdtemp(
            dir=output_parent,
            prefix=f".{output_dir.name}.",
            suffix=".tmp",
        )
    )
    staged_dir = temp_root / "staged"
    backup_dir = temp_root / "backup"
    published: list[tuple[Path, Path | None]] = []
    try:
        staged_dir.mkdir()
        backup_dir.mkdir()
        for filename, content in payloads.items():
            _write_text_with_fsync(staged_dir / filename, content)

        output_dir.mkdir(parents=True, exist_ok=True)
        for filename in payloads:
            final_path = output_dir / filename
            if final_path.exists() and not final_path.is_file():
                raise IsADirectoryError(str(final_path))

        for filename in payloads:
            staged_path = staged_dir / filename
            final_path = output_dir / filename
            backup_path = backup_dir / filename
            if final_path.exists():
                published.append((final_path, backup_path))
                os.replace(final_path, backup_path)
            else:
                published.append((final_path, None))
            os.replace(staged_path, final_path)

        _fsync_directory(output_dir)
        _fsync_directory(output_parent)
    except BaseException:
        _restore_published_payloads(published)
        raise
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _write_text_with_fsync(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _restore_published_payloads(
    published: Sequence[tuple[Path, Path | None]],
) -> None:
    for final_path, backup_path in reversed(published):
        if backup_path is not None and backup_path.exists():
            if final_path.exists():
                final_path.unlink()
            os.replace(backup_path, final_path)
        elif backup_path is None and final_path.exists():
            final_path.unlink()


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            pass
    finally:
        os.close(descriptor)


if __name__ == "__main__":
    raise SystemExit(main())
