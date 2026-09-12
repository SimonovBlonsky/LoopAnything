from __future__ import annotations

import csv
import io
import json
import math
import os
import shutil
import tempfile
from numbers import Real
from pathlib import Path
from typing import Mapping, Sequence

from .io import read_json, read_jsonl
from .metrics import average_precision, max_recall_at_100_precision
from .rover_annotation import ANNOTATION_VERSION, verify_annotation_seal
from .rover_benchmark import sha256_file

SEQUENCE_COUNT = 10
ANNOTATION_FIELDS = frozenset(
    {"pair_id", "label", "annotated_at", "annotation_version"}
)


def validate_scores(
    pairs: Sequence[Mapping[str, object]],
    score_rows: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    pair_ids = _pair_ids(pairs)
    expected = set(pair_ids)
    raw_scores: dict[str, float | None] = {}

    for index, row in enumerate(score_rows):
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError(f"score row {index} pair_id must be a non-empty string")
        if pair_id in raw_scores:
            raise ValueError(f"duplicate score for pair_id={pair_id}")

        status = row.get("status")
        if status not in ("ok", "failed"):
            raise ValueError(f"invalid status for pair_id={pair_id}")
        score = row.get("score")
        if status == "ok":
            if isinstance(score, bool) or not isinstance(score, Real):
                raise ValueError(f"ok score must be finite numeric for pair_id={pair_id}")
            numeric_score = float(score)
            if not math.isfinite(numeric_score):
                raise ValueError(f"ok score must be finite numeric for pair_id={pair_id}")
            raw_scores[pair_id] = numeric_score
        else:
            if "score" not in row or score is not None:
                raise ValueError(f"failed score must be exactly None for pair_id={pair_id}")
            raw_scores[pair_id] = None

    missing = sorted(expected - raw_scores.keys())
    unknown = sorted(raw_scores.keys() - expected)
    if missing:
        raise ValueError(f"missing scores: {missing[:5]}")
    if unknown:
        raise ValueError(f"unknown scores: {unknown[:5]}")

    successful = [score for score in raw_scores.values() if score is not None]
    if not successful:
        return {pair_id: 0.0 for pair_id in pair_ids}
    if len(successful) == len(raw_scores):
        return {pair_id: float(raw_scores[pair_id]) for pair_id in pair_ids}

    failure_score = -math.pi / 2.0
    smallest_success = math.nextafter(failure_score, math.inf)
    transformed_successes: dict[float, float] = {}
    previous_transformed: float | None = None
    for score in sorted(set(successful)):
        # atan keeps ordinary scores bounded above the failure sentinel, but
        # binary64 rounding can collapse distinct extreme values to one float.
        transformed = max(math.atan(score), smallest_success)
        if previous_transformed is not None and transformed <= previous_transformed:
            transformed = math.nextafter(previous_transformed, math.inf)
        transformed_successes[score] = transformed
        previous_transformed = transformed

    normalized: dict[str, float] = {}
    for pair_id in pair_ids:
        score = raw_scores[pair_id]
        if score is None:
            normalized[pair_id] = failure_score
            continue
        normalized[pair_id] = transformed_successes[score]
    return normalized


def read_complete_annotations(
    pairs: Sequence[Mapping[str, object]],
    manifest_path: Path,
    annotations_path: Path,
    seal_path: Path,
) -> dict[str, int]:
    pairs_path = Path(manifest_path).parent / "benchmark_pairs.jsonl"
    verify_annotation_seal(manifest_path, pairs_path, annotations_path, seal_path)

    pair_ids = _pair_ids(pairs)
    expected = set(pair_ids)
    labels: dict[str, int] = {}
    actual_order: list[str] = []
    for index, row in enumerate(read_jsonl(annotations_path)):
        if set(row) != ANNOTATION_FIELDS:
            raise ValueError(f"annotation row {index} has invalid fields")
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError(f"annotation row {index} pair_id must be a non-empty string")
        if pair_id in labels:
            raise ValueError(f"duplicate annotation pair_id: {pair_id}")
        if pair_id not in expected:
            raise ValueError(f"unknown annotation pair_id: {pair_id}")
        label = row.get("label")
        if type(label) is not int or label not in (0, 1):
            raise ValueError(f"annotation row {index} label must be 0 or 1")
        if row.get("annotation_version") != ANNOTATION_VERSION:
            raise ValueError(f"annotation row {index} annotation_version mismatch")
        annotated_at = row.get("annotated_at")
        if not isinstance(annotated_at, str) or not annotated_at:
            raise ValueError(f"annotation row {index} annotated_at must be non-empty")
        labels[pair_id] = label
        actual_order.append(pair_id)

    if set(labels) != expected or len(labels) != len(pair_ids):
        raise ValueError("annotation coverage does not match benchmark pairs")
    if actual_order != pair_ids:
        raise ValueError("annotation pair_id order does not match benchmark pairs")
    return labels


def validate_sequence_groups(
    pairs: Sequence[Mapping[str, object]],
    sequence_order: Sequence[str],
) -> None:
    _validate_sequence_groups(pairs, sequence_order, required_count=SEQUENCE_COUNT)


def evaluate_score_file(
    method_name: str,
    pairs: Sequence[Mapping[str, object]],
    labels: Mapping[str, object],
    scores: Mapping[str, object],
    sequence_order: Sequence[str],
) -> dict[str, object]:
    if not isinstance(method_name, str) or not method_name:
        raise ValueError("method_name must be a non-empty string")
    _validate_sequence_groups(pairs, sequence_order, required_count=None)
    pair_ids = _pair_ids(pairs)
    _validate_value_coverage(pair_ids, labels, "annotations")
    _validate_value_coverage(pair_ids, scores, "scores")

    pairs_by_sequence: dict[str, list[str]] = {key: [] for key in sequence_order}
    for pair in pairs:
        pairs_by_sequence[_sequence_key(pair)].append(str(pair["pair_id"]))

    sequence_results: dict[str, dict[str, float | int]] = {}
    for sequence_key in sequence_order:
        sequence_pair_ids = pairs_by_sequence[sequence_key]
        sequence_labels: list[bool] = []
        sequence_scores: list[float] = []
        for pair_id in sequence_pair_ids:
            label = labels[pair_id]
            if type(label) is not int or label not in (0, 1):
                raise ValueError(f"annotation label must be 0 or 1 for pair_id={pair_id}")
            score = scores[pair_id]
            if isinstance(score, bool) or not isinstance(score, Real):
                raise ValueError(f"score must be finite numeric for pair_id={pair_id}")
            numeric_score = float(score)
            if not math.isfinite(numeric_score):
                raise ValueError(f"score must be finite numeric for pair_id={pair_id}")
            sequence_labels.append(bool(label))
            sequence_scores.append(numeric_score)

        sequence_results[sequence_key] = {
            "AP": average_precision(sequence_labels, sequence_scores),
            "MR@100P": max_recall_at_100_precision(sequence_labels, sequence_scores),
            "candidate_count": len(sequence_pair_ids),
            "positive_count": sum(sequence_labels),
        }

    sequence_count = len(sequence_order)
    return {
        "method": method_name,
        "sequences": sequence_results,
        "macro_average": {
            "AP": sum(float(row["AP"]) for row in sequence_results.values())
            / sequence_count,
            "MR@100P": sum(
                float(row["MR@100P"]) for row in sequence_results.values()
            )
            / sequence_count,
        },
    }


def verify_score_manifest(
    pair_manifest_path: Path,
    score_path: Path,
    score_manifest_path: Path,
    method_name: str | None = None,
) -> None:
    pair_manifest_path = Path(pair_manifest_path)
    score_path = Path(score_path)
    score_manifest_path = Path(score_manifest_path)
    manifest = read_json(score_manifest_path)

    for field in (
        "method",
        "pair_manifest_sha256",
        "score_file_sha256",
        "source_commit",
        "command",
    ):
        if field not in manifest:
            raise ValueError(f"score manifest missing {field}")

    method = manifest["method"]
    if not isinstance(method, str) or not method:
        raise ValueError("score manifest method must be a non-empty string")
    effective_method = manifest.get("evaluation_name", method)
    if not isinstance(effective_method, str) or not effective_method:
        raise ValueError("score manifest evaluation_name must be a non-empty string")
    if method_name is not None and effective_method != method_name:
        raise ValueError(
            f"score manifest method mismatch: expected {method_name}, got {effective_method}"
        )

    expected_pair_hash = manifest["pair_manifest_sha256"]
    if not isinstance(expected_pair_hash, str) or not expected_pair_hash:
        raise ValueError("score manifest pair_manifest_sha256 must be non-empty")
    if sha256_file(pair_manifest_path) != expected_pair_hash:
        raise ValueError("candidate-manifest hash does not match score manifest")

    expected_score_hash = manifest["score_file_sha256"]
    if not isinstance(expected_score_hash, str) or not expected_score_hash:
        raise ValueError("score manifest score_file_sha256 must be non-empty")
    if sha256_file(score_path) != expected_score_hash:
        raise ValueError("score-file hash does not match score manifest")

    source_commit = manifest["source_commit"]
    if not isinstance(source_commit, str) or not source_commit:
        raise ValueError("score manifest source_commit must be a non-empty string")
    command = manifest["command"]
    if isinstance(command, str):
        valid_command = bool(command)
    elif isinstance(command, list):
        valid_command = bool(command) and all(
            isinstance(part, str) and part for part in command
        )
    else:
        valid_command = False
    if not valid_command:
        raise ValueError("score manifest command must be non-empty")

    declared_score_file = manifest.get("score_file")
    if declared_score_file is not None:
        if not isinstance(declared_score_file, str) or not declared_score_file:
            raise ValueError("score manifest score_file must be a non-empty string")
        declared_path = Path(declared_score_file)
        if declared_path.is_absolute():
            resolved_declared = declared_path.resolve()
        else:
            resolved_declared = (pair_manifest_path.parent / declared_path).resolve()
        if resolved_declared != score_path.resolve():
            raise ValueError("score manifest score_file path mismatch")


def evaluate_methods(
    benchmark_root: Path,
    method_files: Mapping[str, Path] | Sequence[tuple[str, Path]],
) -> dict[str, object]:
    benchmark_root = Path(benchmark_root)
    pairs_path = benchmark_root / "benchmark_pairs.jsonl"
    manifest_path = benchmark_root / "manifest.json"
    annotations_path = benchmark_root / "annotations.jsonl"
    seal_path = benchmark_root / "annotation_seal.json"

    manifest = read_json(manifest_path)
    pairs = list(read_jsonl(pairs_path))
    sequence_order = _manifest_sequence_order(manifest)
    labels = read_complete_annotations(
        pairs,
        manifest_path,
        annotations_path,
        seal_path,
    )
    validate_sequence_groups(pairs, sequence_order)

    methods = _method_items(method_files)
    results_by_method: dict[str, object] = {}
    for method_name, configured_path in methods:
        score_path = (
            configured_path
            if configured_path.is_absolute()
            else benchmark_root / configured_path
        )
        score_manifest_path = score_path.with_suffix(".manifest.json")
        verify_score_manifest(
            pairs_path,
            score_path,
            score_manifest_path,
            method_name=method_name,
        )
        scores = validate_scores(pairs, list(read_jsonl(score_path)))
        results_by_method[method_name] = evaluate_score_file(
            method_name,
            pairs,
            labels,
            scores,
            sequence_order,
        )

    return {
        "sequence_order": list(sequence_order),
        "method_order": [name for name, _ in methods],
        "methods": results_by_method,
    }


def write_metrics_outputs(output_dir: Path, results: Mapping[str, object]) -> None:
    sequence_order, method_order, methods = _validated_results(results)
    markdown = render_table1_markdown(results, sequence_order, method_order)

    csv_buffer = io.StringIO(newline="")
    writer = csv.writer(csv_buffer, lineterminator="\n")
    writer.writerow(
        ["method", "sequence", "candidate_count", "positive_count", "AP", "MR@100P"]
    )
    for method_name in method_order:
        method_result = methods[method_name]
        sequences = method_result["sequences"]
        for sequence_key in sequence_order:
            row = sequences[sequence_key]
            writer.writerow(
                [
                    method_name,
                    sequence_key,
                    row["candidate_count"],
                    row["positive_count"],
                    _format_json_float(row["AP"]),
                    _format_json_float(row["MR@100P"]),
                ]
            )

    summary_text = json.dumps(results, indent=2, ensure_ascii=True) + "\n"
    payloads = {
        "metrics_per_sequence.csv": csv_buffer.getvalue(),
        "metrics_summary.json": summary_text,
        "table1.md": markdown,
    }
    _write_payload_set_atomically(Path(output_dir), payloads)


def _write_payload_set_atomically(output_dir: Path, payloads: Mapping[str, str]) -> None:
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
    with path.open("w", encoding="utf-8") as handle:
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
        try:
            os.close(descriptor)
        except OSError:
            pass


def render_table1_markdown(
    results: Mapping[str, object],
    sequence_order: Sequence[str],
    method_order: Sequence[str],
) -> str:
    validated_sequence_order, validated_method_order, methods = _validated_results(
        results,
        sequence_order=sequence_order,
        method_order=method_order,
    )
    first_method = methods[validated_method_order[0]]
    total_candidates = sum(
        int(first_method["sequences"][key]["candidate_count"])
        for key in validated_sequence_order
    )
    total_positives = sum(
        int(first_method["sequences"][key]["positive_count"])
        for key in validated_sequence_order
    )

    columns = ["Method", *validated_sequence_order, "Macro average"]
    lines = [
        "# Table 1",
        "",
        f"Candidates: {total_candidates}; positives: {total_positives}.",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]

    sequence_ranks = {
        key: {
            metric: _metric_ranks(
                validated_method_order,
                {
                    method: float(methods[method]["sequences"][key][metric])
                    for method in validated_method_order
                },
            )
            for metric in ("AP", "MR@100P")
        }
        for key in validated_sequence_order
    }
    macro_ranks = {
        metric: _metric_ranks(
            validated_method_order,
            {
                method: float(methods[method]["macro_average"][metric])
                for method in validated_method_order
            },
        )
        for metric in ("AP", "MR@100P")
    }

    for method_name in validated_method_order:
        cells = [method_name]
        for sequence_key in validated_sequence_order:
            row = methods[method_name]["sequences"][sequence_key]
            cells.append(
                _format_ranked(row["AP"], sequence_ranks[sequence_key]["AP"][method_name])
                + " / "
                + _format_ranked(
                    row["MR@100P"],
                    sequence_ranks[sequence_key]["MR@100P"][method_name],
                )
            )
        macro = methods[method_name]["macro_average"]
        cells.append(
            _format_ranked(macro["AP"], macro_ranks["AP"][method_name])
            + " / "
            + _format_ranked(macro["MR@100P"], macro_ranks["MR@100P"][method_name])
        )
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def _pair_ids(pairs: Sequence[Mapping[str, object]]) -> list[str]:
    pair_ids: list[str] = []
    seen: set[str] = set()
    for index, pair in enumerate(pairs):
        pair_id = pair.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError(f"benchmark pair {index} pair_id must be a non-empty string")
        if pair_id in seen:
            raise ValueError(f"duplicate pair_id in benchmark pairs: {pair_id}")
        seen.add(pair_id)
        pair_ids.append(pair_id)
    if not pair_ids:
        raise ValueError("benchmark pairs must be non-empty")
    return pair_ids


def _sequence_key(pair: Mapping[str, object]) -> str:
    components = []
    for field in ("dataset", "platform", "sequence"):
        value = pair.get(field)
        if not isinstance(value, str) or not value or "/" in value:
            raise ValueError(f"benchmark pair {field} must be a non-empty path component")
        components.append(value)
    return "/".join(components)


def _validate_sequence_groups(
    pairs: Sequence[Mapping[str, object]],
    sequence_order: Sequence[str],
    required_count: int | None,
) -> None:
    order = list(sequence_order)
    if required_count is not None and len(order) != required_count:
        raise ValueError(f"sequence_order must contain exactly {required_count} groups")
    if not order:
        raise ValueError("sequence_order must be non-empty")
    if any(not isinstance(key, str) or not key for key in order):
        raise ValueError("sequence_order entries must be non-empty strings")
    if len(set(order)) != len(order):
        raise ValueError("sequence_order contains duplicate groups")

    _pair_ids(pairs)
    pair_keys = [_sequence_key(pair) for pair in pairs]
    observed = set(pair_keys)
    expected = set(order)
    missing = [key for key in order if key not in observed]
    extra = sorted(observed - expected)
    if missing:
        raise ValueError(f"missing sequence groups: {missing}")
    if extra:
        raise ValueError(f"extra sequence groups: {extra}")

    group_order: list[str] = []
    for key in pair_keys:
        if not group_order or group_order[-1] != key:
            if key in group_order:
                raise ValueError(f"sequence group is not contiguous: {key}")
            group_order.append(key)
    if group_order != order:
        raise ValueError("benchmark sequence group order does not match sequence_order")


def _validate_value_coverage(
    pair_ids: Sequence[str],
    values: Mapping[str, object],
    value_name: str,
) -> None:
    expected = set(pair_ids)
    actual = set(values)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise ValueError(f"missing {value_name}: {missing[:5]}")
    if unknown:
        raise ValueError(f"unknown {value_name}: {unknown[:5]}")


def _manifest_sequence_order(manifest: Mapping[str, object]) -> list[str]:
    raw_order = manifest.get("sequence_order")
    if not isinstance(raw_order, list):
        raise ValueError("manifest sequence_order must be a list")
    order = list(raw_order)
    if len(order) != SEQUENCE_COUNT:
        raise ValueError(f"manifest sequence_order must contain exactly {SEQUENCE_COUNT} groups")
    if any(not isinstance(key, str) or not key for key in order):
        raise ValueError("manifest sequence_order entries must be non-empty strings")
    if len(set(order)) != len(order):
        raise ValueError("manifest sequence_order contains duplicates")

    sequence_rows = manifest.get("sequences")
    if not isinstance(sequence_rows, list) or len(sequence_rows) != SEQUENCE_COUNT:
        raise ValueError(f"manifest sequences must contain exactly {SEQUENCE_COUNT} rows")
    row_order: list[str] = []
    for row in sequence_rows:
        if not isinstance(row, Mapping):
            raise ValueError("manifest sequence rows must be mappings")
        row_order.append(_sequence_key(row))
    if row_order != order:
        raise ValueError("manifest sequence_order does not match sequences")
    return order


def _method_items(
    method_files: Mapping[str, Path] | Sequence[tuple[str, Path]],
) -> list[tuple[str, Path]]:
    raw_items = (
        list(method_files.items()) if isinstance(method_files, Mapping) else list(method_files)
    )
    if not raw_items:
        raise ValueError("at least one method score file is required")
    items: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for method_name, path in raw_items:
        if not isinstance(method_name, str) or not method_name:
            raise ValueError("method names must be non-empty strings")
        if method_name in seen:
            raise ValueError(f"duplicate method name: {method_name}")
        seen.add(method_name)
        items.append((method_name, Path(path)))
    return items


def _validated_results(
    results: Mapping[str, object],
    sequence_order: Sequence[str] | None = None,
    method_order: Sequence[str] | None = None,
) -> tuple[list[str], list[str], Mapping[str, Mapping[str, object]]]:
    raw_sequences = results.get("sequence_order") if sequence_order is None else sequence_order
    raw_methods = results.get("method_order") if method_order is None else method_order
    methods = results.get("methods")
    if not isinstance(raw_sequences, (list, tuple)) or not raw_sequences:
        raise ValueError("results sequence_order must be non-empty")
    if not isinstance(raw_methods, (list, tuple)) or not raw_methods:
        raise ValueError("results method_order must be non-empty")
    if not isinstance(methods, Mapping):
        raise ValueError("results methods must be a mapping")
    sequences = list(raw_sequences)
    method_names = list(raw_methods)
    if len(set(sequences)) != len(sequences):
        raise ValueError("results sequence_order contains duplicates")
    if len(set(method_names)) != len(method_names):
        raise ValueError("results method_order contains duplicates")
    if set(methods) != set(method_names):
        raise ValueError("results methods do not match method_order")

    expected_counts: dict[str, tuple[int, int]] = {}
    for method_name in method_names:
        result = methods[method_name]
        if not isinstance(result, Mapping):
            raise ValueError(f"result for method {method_name} must be a mapping")
        sequence_rows = result.get("sequences")
        macro = result.get("macro_average")
        if not isinstance(sequence_rows, Mapping) or set(sequence_rows) != set(sequences):
            raise ValueError(f"result sequences do not match sequence_order for {method_name}")
        if not isinstance(macro, Mapping):
            raise ValueError(f"macro_average must be a mapping for {method_name}")
        for metric in ("AP", "MR@100P"):
            _finite_metric(macro.get(metric), method_name, "macro", metric)
        for sequence_key in sequences:
            row = sequence_rows[sequence_key]
            if not isinstance(row, Mapping):
                raise ValueError(f"sequence result must be a mapping for {method_name}")
            for metric in ("AP", "MR@100P"):
                _finite_metric(row.get(metric), method_name, sequence_key, metric)
            candidate_count = row.get("candidate_count")
            positive_count = row.get("positive_count")
            if type(candidate_count) is not int or candidate_count <= 0:
                raise ValueError("candidate counts must be positive integers")
            if (
                type(positive_count) is not int
                or positive_count < 0
                or positive_count > candidate_count
            ):
                raise ValueError("positive counts must be valid integers")
            counts = (candidate_count, positive_count)
            if sequence_key in expected_counts and expected_counts[sequence_key] != counts:
                raise ValueError(f"counts differ across methods for {sequence_key}")
            expected_counts[sequence_key] = counts
    return sequences, method_names, methods  # type: ignore[return-value]


def _finite_metric(value: object, method: str, group: str, metric: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{method} {group} {metric} must be finite numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{method} {group} {metric} must be finite numeric")
    return result


def _metric_ranks(
    method_order: Sequence[str],
    values: Mapping[str, float],
) -> dict[str, int]:
    order_index = {method: index for index, method in enumerate(method_order)}
    ranked = sorted(method_order, key=lambda method: (-values[method], order_index[method]))
    return {method: rank for rank, method in enumerate(ranked)}


def _format_ranked(value: object, rank: int) -> str:
    text = f"{float(value):.4f}"
    if rank == 0:
        return f"**{text}**"
    if rank == 1:
        return f"<u>{text}</u>"
    return text


def _format_json_float(value: object) -> str:
    return format(float(value), ".17g")
