#!/usr/bin/env python3
"""Export visual failure cases from the ROVER-aligned manual benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for import_path in (REPO_ROOT, SRC_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from robust_loop_verifier.metrics import assign_failure_worst_scores

from robust_loop_verification_scripts.failure_case_analysis import (
    _copy_image_or_placeholder,
    _read_keyframes,
    _resolve_image_path,
    _slugify_method,
    _write_overview,
)


DEFAULT_BENCHMARK_ROOT = (
    REPO_ROOT / "workspace" / "rover_aligned_benchmark" / "benchmark_v1"
)
DEFAULT_METHODS = (
    "SALAD score only",
    "ROVER deformation only",
    "query_gate_graph:def=0.5,res=0.25,margin=0",
)
DEFAULT_REFERENCE_METHOD = "SALAD score only"
DEFAULT_TARGET_METHOD = "query_gate_graph:def=0.5,res=0.25,margin=0"


@dataclass(frozen=True)
class ExportSummaryRow:
    case_type: str
    dataset: str
    platform: str
    sequence: str
    method: str
    reference_method: str | None
    target_method: str | None
    candidate_count: int
    saved_count: int
    output_dir: Path


def run_rover_aligned_failure_case_export(
    benchmark_root: Path,
    *,
    output_root: Path,
    candidate_records: Path | None = None,
    methods: Sequence[str] = DEFAULT_METHODS,
    reference_method: str = DEFAULT_REFERENCE_METHOD,
    target_method: str = DEFAULT_TARGET_METHOD,
    max_false_positives_per_method: int = 24,
    max_regressions_per_sequence: int = 24,
    max_reference_rank: int | None = None,
) -> list[ExportSummaryRow]:
    benchmark_root = Path(benchmark_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    candidate_records = (
        Path(candidate_records)
        if candidate_records is not None
        else benchmark_root / "candidate_records.jsonl"
    )
    records = _records_with_labels(candidate_records, benchmark_root / "annotations.jsonl")
    sequence_contexts = _sequence_contexts(benchmark_root / "manifest.json")
    records_by_sequence = _records_by_sequence(records)
    summary_rows: list[ExportSummaryRow] = []

    for sequence_key, sequence_records in records_by_sequence.items():
        context = sequence_contexts[sequence_key]
        keyframes = _read_keyframes(context.cache_root / "keyframes.jsonl")
        cache_manifest = _read_json(context.cache_root / "manifest.json")
        for method in methods:
            method_scores = _method_scores(sequence_records, method)
            false_positives = _ranked_false_positives(sequence_records, method_scores)
            method_dir = (
                output_root
                / context.dataset
                / context.platform
                / context.sequence
                / "false_positives"
                / _slugify_method(method)
            )
            saved = 0
            for candidate in false_positives[:max_false_positives_per_method]:
                _export_case(
                    method_dir,
                    candidate.record,
                    case_type="false_positive",
                    case_name=(
                        f"rank{candidate.rank:03d}_"
                        f"q{candidate.query_idx:06d}_"
                        f"c{candidate.candidate_idx:06d}_"
                        f"{_support_part(candidate.support_idx)}"
                    ),
                    keyframes=keyframes,
                    cache_root=context.cache_root,
                    cache_manifest=cache_manifest,
                    payload={
                        "case_type": "false_positive",
                        "method": method,
                        "method_score": candidate.score,
                        "method_rank": candidate.rank,
                    },
                    overview_title=(
                        f"false positive  {method}  "
                        f"rank={candidate.rank}  score={candidate.score:.6f}"
                    ),
                )
                saved += 1
            summary_rows.append(
                ExportSummaryRow(
                    case_type="false_positive",
                    dataset=context.dataset,
                    platform=context.platform,
                    sequence=context.sequence,
                    method=method,
                    reference_method=None,
                    target_method=None,
                    candidate_count=len(false_positives),
                    saved_count=saved,
                    output_dir=method_dir,
                )
            )

        regressions = _ranked_target_regressions(
            sequence_records,
            reference_method=reference_method,
            target_method=target_method,
            max_reference_rank=max_reference_rank,
        )
        regression_dir = (
            output_root
            / context.dataset
            / context.platform
            / context.sequence
            / "regressions"
            / f"{_slugify_method(target_method)}_vs_{_slugify_method(reference_method)}"
        )
        saved = 0
        for regression in regressions[:max_regressions_per_sequence]:
            _export_case(
                regression_dir,
                regression.record,
                case_type="target_regression",
                case_name=(
                    f"refrank{regression.reference_rank:03d}_"
                    f"targetrank{regression.target_rank:03d}_"
                    f"q{regression.query_idx:06d}_"
                    f"c{regression.candidate_idx:06d}_"
                    f"{_support_part(regression.support_idx)}"
                ),
                keyframes=keyframes,
                cache_root=context.cache_root,
                cache_manifest=cache_manifest,
                payload={
                    "case_type": "target_regression",
                    "reference_method": reference_method,
                    "target_method": target_method,
                    "reference_score": regression.reference_score,
                    "target_score": regression.target_score,
                    "reference_rank": regression.reference_rank,
                    "target_rank": regression.target_rank,
                    "rank_drop": regression.rank_drop,
                },
                overview_title=(
                    "target regression  "
                    f"ref_rank={regression.reference_rank}  "
                    f"target_rank={regression.target_rank}  "
                    f"drop={regression.rank_drop}"
                ),
            )
            saved += 1
        summary_rows.append(
            ExportSummaryRow(
                case_type="target_regression",
                dataset=context.dataset,
                platform=context.platform,
                sequence=context.sequence,
                method=f"{target_method} vs {reference_method}",
                reference_method=reference_method,
                target_method=target_method,
                candidate_count=len(regressions),
                saved_count=saved,
                output_dir=regression_dir,
            )
        )

    _write_summary(output_root, summary_rows)
    return summary_rows


@dataclass(frozen=True)
class _SequenceContext:
    dataset: str
    platform: str
    sequence: str
    cache_root: Path


@dataclass(frozen=True)
class _RankedCase:
    record: Mapping[str, Any]
    score: float
    rank: int
    query_idx: int
    candidate_idx: int
    support_idx: int | None


@dataclass(frozen=True)
class _RegressionCase:
    record: Mapping[str, Any]
    reference_score: float
    target_score: float
    reference_rank: int
    target_rank: int
    rank_drop: int
    query_idx: int
    candidate_idx: int
    support_idx: int | None


def _records_with_labels(candidate_records: Path, annotations: Path) -> list[dict[str, Any]]:
    labels = {
        str(row["pair_id"]): _bool_label(row.get("label"))
        for row in _read_jsonl(annotations)
    }
    records = []
    for row in _read_jsonl(candidate_records):
        record = dict(row)
        pair_id = str(record["pair_id"])
        if pair_id in labels:
            record["label"] = labels[pair_id]
        elif type(record.get("label")) is not bool:
            raise ValueError(f"missing annotation label for pair_id={pair_id}")
        records.append(record)
    return records


def _sequence_contexts(manifest_path: Path) -> dict[str, _SequenceContext]:
    manifest = _read_json(manifest_path)
    rows = manifest.get("sequences")
    if not isinstance(rows, list):
        raise ValueError("benchmark manifest must contain a sequences list")
    contexts = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("benchmark manifest sequence rows must be mappings")
        dataset = str(row["dataset"])
        platform = str(row["platform"])
        sequence = str(row["sequence"])
        contexts[f"{dataset}/{platform}/{sequence}"] = _SequenceContext(
            dataset=dataset,
            platform=platform,
            sequence=sequence,
            cache_root=Path(str(row["cache"])),
        )
    return contexts


def _records_by_sequence(records: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = _sequence_key(record)
        grouped.setdefault(key, []).append(dict(record))
    return grouped


def _sequence_key(record: Mapping[str, Any]) -> str:
    sequence_key = record.get("sequence_key")
    if isinstance(sequence_key, str) and sequence_key:
        return sequence_key
    return f"{record['dataset']}/{record['platform']}/{record['sequence']}"


def _method_scores(records: Sequence[Mapping[str, Any]], method: str) -> list[float | None]:
    if method == "SALAD score only":
        return [
            _finite_float_or_none(record.get("score_salad", record.get("salad_score")))
            for record in records
        ]
    if method == "ROVER deformation only":
        return [
            _negative(_finite_float_or_none(record.get("trajectory_deformation_rmse")))
            for record in records
        ]
    if method == "PGO residual only":
        return [_negative(_log1p_or_none(record.get("pgo_error_after"))) for record in records]

    absolute_match = re.fullmatch(r"absolute_graph:def=([^,]+),res=([^,]+)", method)
    if absolute_match:
        return _raw_graph_scores(
            records,
            float(absolute_match.group(1)),
            float(absolute_match.group(2)),
        )

    query_gate_match = re.fullmatch(
        r"query_gate_graph:def=([^,]+),res=([^,]+),margin=([^,]+)",
        method,
    )
    if query_gate_match:
        return _query_gate_graph_scores(
            records,
            float(query_gate_match.group(1)),
            float(query_gate_match.group(2)),
            float(query_gate_match.group(3)),
        )
    raise ValueError(f"unsupported failure-case method: {method}")


def _ranked_false_positives(
    records: Sequence[Mapping[str, Any]],
    raw_scores: Sequence[float | None],
) -> list[_RankedCase]:
    assigned_scores = assign_failure_worst_scores(raw_scores)
    ranked_indices = _rank_indices(assigned_scores)
    rank_by_index = {index: rank for rank, index in enumerate(ranked_indices, start=1)}
    cases = []
    for index in ranked_indices:
        raw_score = raw_scores[index]
        if raw_score is None or _label(records[index]):
            continue
        cases.append(
            _RankedCase(
                record=records[index],
                score=float(assigned_scores[index]),
                rank=rank_by_index[index],
                query_idx=_int_value(records[index].get("query_idx")),
                candidate_idx=_int_value(records[index].get("candidate_idx")),
                support_idx=_optional_int_value(records[index].get("support_idx")),
            )
        )
    return cases


def _ranked_target_regressions(
    records: Sequence[Mapping[str, Any]],
    *,
    reference_method: str,
    target_method: str,
    max_reference_rank: int | None,
) -> list[_RegressionCase]:
    reference_raw = _method_scores(records, reference_method)
    target_raw = _method_scores(records, target_method)
    reference_scores = assign_failure_worst_scores(reference_raw)
    target_scores = assign_failure_worst_scores(target_raw)
    reference_rank = _rank_by_index(reference_scores)
    target_rank = _rank_by_index(target_scores)
    cases = []
    for index, record in enumerate(records):
        if not _label(record) or reference_raw[index] is None:
            continue
        if max_reference_rank is not None and reference_rank[index] > max_reference_rank:
            continue
        rank_drop = target_rank[index] - reference_rank[index]
        if rank_drop <= 0:
            continue
        cases.append(
            _RegressionCase(
                record=record,
                reference_score=float(reference_scores[index]),
                target_score=float(target_scores[index]),
                reference_rank=reference_rank[index],
                target_rank=target_rank[index],
                rank_drop=rank_drop,
                query_idx=_int_value(record.get("query_idx")),
                candidate_idx=_int_value(record.get("candidate_idx")),
                support_idx=_optional_int_value(record.get("support_idx")),
            )
        )
    return sorted(
        cases,
        key=lambda case: (-case.rank_drop, case.reference_rank, case.target_rank),
    )


def _raw_graph_scores(
    records: Sequence[Mapping[str, Any]],
    deformation_weight: float,
    residual_weight: float,
) -> list[float | None]:
    scores = []
    for record in records:
        deformation = _finite_float_or_none(record.get("trajectory_deformation_rmse"))
        residual = _log1p_or_none(record.get("pgo_error_after"))
        if deformation is None or residual is None:
            scores.append(None)
        else:
            scores.append(-(deformation_weight * deformation + residual_weight * residual))
    return scores


def _query_gate_graph_scores(
    records: Sequence[Mapping[str, Any]],
    deformation_weight: float,
    residual_weight: float,
    margin_weight: float,
) -> list[float | None]:
    base_scores = _raw_graph_scores(records, deformation_weight, residual_weight)
    scores: list[float | None] = [None for _ in records]
    query_to_indices: dict[int, list[int]] = {}
    for index, record in enumerate(records):
        query_to_indices.setdefault(_int_value(record.get("query_idx")), []).append(index)

    for indices in query_to_indices.values():
        valid_scores = [
            float(base_scores[index])
            for index in indices
            if base_scores[index] is not None
            and math.isfinite(float(base_scores[index]))
        ]
        if not valid_scores:
            continue
        valid_scores.sort(reverse=True)
        best_score = valid_scores[0]
        second_score = valid_scores[1] if len(valid_scores) > 1 else best_score
        query_confidence = best_score + margin_weight * max(0.0, best_score - second_score)
        for index in indices:
            if base_scores[index] is not None:
                scores[index] = float(base_scores[index]) + query_confidence
    return scores


def _export_case(
    output_dir: Path,
    record: Mapping[str, Any],
    *,
    case_type: str,
    case_name: str,
    keyframes: Mapping[int, Mapping[str, Any]],
    cache_root: Path,
    cache_manifest: Mapping[str, Any],
    payload: Mapping[str, Any],
    overview_title: str,
) -> None:
    case_dir = output_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    query_idx = _int_value(record.get("query_idx"))
    candidate_idx = _int_value(record.get("candidate_idx"))
    support_idx = _optional_int_value(record.get("support_idx"))
    image_specs = [("query", query_idx), ("candidate", candidate_idx)]
    if support_idx is not None:
        image_specs.append(("support", support_idx))

    exported_images = []
    for role, keyframe_idx in image_specs:
        src = _resolve_image_path(keyframe_idx, keyframes, cache_root, cache_manifest)
        dst = case_dir / f"{role}.png"
        _copy_image_or_placeholder(src, dst, f"{role} {keyframe_idx:06d}")
        exported_images.append((role, dst, keyframe_idx))
    _write_overview(case_dir / "overview.png", exported_images, title=overview_title)
    record_payload = dict(record)
    record_payload.update(dict(payload))
    record_payload["case_type"] = case_type
    record_payload["exported_images"] = {
        role: str(path.relative_to(case_dir)) for role, path, _ in exported_images
    }
    (case_dir / "record.json").write_text(
        json.dumps(_sanitize(record_payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_summary(output_root: Path, rows: Sequence[ExportSummaryRow]) -> None:
    summary_md = output_root / "summary.md"
    lines = [
        "| case_type | dataset | platform | sequence | method | reference | target | candidates | saved | output_dir |",
        "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                row.case_type,
                row.dataset,
                row.platform,
                row.sequence,
                row.method,
                row.reference_method or "",
                row.target_method or "",
                row.candidate_count,
                row.saved_count,
                row.output_dir,
            )
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case_type",
                "dataset",
                "platform",
                "sequence",
                "method",
                "reference_method",
                "target_method",
                "candidate_count",
                "saved_count",
                "output_dir",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.case_type,
                    row.dataset,
                    row.platform,
                    row.sequence,
                    row.method,
                    row.reference_method or "",
                    row.target_method or "",
                    row.candidate_count,
                    row.saved_count,
                    row.output_dir,
                ]
            )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    if not Path(path).is_file():
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def _rank_indices(scores: Sequence[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))


def _rank_by_index(scores: Sequence[float]) -> dict[int, int]:
    return {index: rank for rank, index in enumerate(_rank_indices(scores), start=1)}


def _label(record: Mapping[str, Any]) -> bool:
    label = record.get("label")
    if type(label) is not bool:
        raise ValueError(f"candidate record label must be bool: {label!r}")
    return bool(label)


def _bool_label(value: Any) -> bool:
    if type(value) is bool:
        return value
    if value in (0, 1):
        return bool(value)
    raise ValueError(f"annotation label must be binary: {value!r}")


def _finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _log1p_or_none(value: Any) -> float | None:
    value = _finite_float_or_none(value)
    if value is None or value < -1.0:
        return None
    return math.log1p(value)


def _negative(value: float | None) -> float | None:
    return None if value is None else -value


def _int_value(value: Any) -> int:
    if value is None:
        raise ValueError("missing integer value")
    return int(value)


def _optional_int_value(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _support_part(support_idx: int | None) -> str:
    return f"s{support_idx:06d}" if support_idx is not None else "snone"


def _sanitize(value: Any):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    return value


def _default_output_root() -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "workspace" / "failure_case_analysis" / f"rover_aligned_{run_id}"


def _parse_methods(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return DEFAULT_METHODS
    return tuple(values)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "benchmark_root",
        nargs="?",
        type=Path,
        default=DEFAULT_BENCHMARK_ROOT,
    )
    parser.add_argument(
        "--candidate-records",
        type=Path,
        default=None,
        help="Candidate records JSONL. Defaults to BENCHMARK_ROOT/candidate_records.jsonl.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output directory. Defaults to workspace/failure_case_analysis/rover_aligned_<time>.",
    )
    parser.add_argument(
        "--method",
        action="append",
        default=None,
        help="Method for false-positive export. Can be repeated.",
    )
    parser.add_argument("--reference-method", default=DEFAULT_REFERENCE_METHOD)
    parser.add_argument("--target-method", default=DEFAULT_TARGET_METHOD)
    parser.add_argument("--max-false-positives-per-method", type=int, default=24)
    parser.add_argument("--max-regressions-per-sequence", type=int, default=24)
    parser.add_argument(
        "--max-reference-rank",
        type=int,
        default=None,
        help="Optional maximum reference-method rank for regression cases.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    output_root = args.output_root or _default_output_root()
    rows = run_rover_aligned_failure_case_export(
        args.benchmark_root,
        output_root=output_root,
        candidate_records=args.candidate_records,
        methods=_parse_methods(args.method),
        reference_method=args.reference_method,
        target_method=args.target_method,
        max_false_positives_per_method=args.max_false_positives_per_method,
        max_regressions_per_sequence=args.max_regressions_per_sequence,
        max_reference_rank=args.max_reference_rank,
    )
    print(f"Wrote ROVER-aligned failure cases to {output_root}")
    for row in rows:
        print(
            "{} / {} / {} / {} / {}: candidates={} saved={}".format(
                row.dataset,
                row.platform,
                row.sequence,
                row.case_type,
                row.method,
                row.candidate_count,
                row.saved_count,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
