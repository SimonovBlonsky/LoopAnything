#!/usr/bin/env python3
"""Export false-loop visualizations for low-recall loop-verifier sequences."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw

from robust_loop_verifier.metrics import (
    assign_failure_worst_scores,
    average_precision,
    max_recall_at_100_precision,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METHODS = (
    "absolute_graph:def=0.5,res=0.25",
    "query_gate_graph:def=0.5,res=0.25,margin=0",
)


@dataclass(frozen=True)
class SequenceSpec:
    dataset: str
    platform: str
    sequence: str
    run_root: Path
    cache_root: Path


@dataclass(frozen=True)
class MethodResult:
    dataset: str
    platform: str
    sequence: str
    method: str
    record_count: int
    positive_count: int
    ap: float
    mr_at_100p: float
    first_false_rank: int | None
    true_before_first_false: int
    false_positive_count: int
    saved_false_positives: int
    output_dir: Path


DEFAULT_SEQUENCE_SPECS = {
    "ugv_parking02": SequenceSpec(
        dataset="FusionPortableV2",
        platform="ugv",
        sequence="ugv_parking02",
        run_root=REPO_ROOT / (
            "workspace/robust_loop_verifier_runs/FusionPortableV2/ugv/"
            "ugv_parking02/ugv_20260519_012615"
        ),
        cache_root=Path(
            "/data/datasets/FusionPortable/robust_loop_verifier_cache/"
            "FusionPortableV2/ugv/ugv_parking02"
        ),
    ),
    "Offroad01_beta": SequenceSpec(
        dataset="GEODE",
        platform="Offroad",
        sequence="Offroad01_beta",
        run_root=REPO_ROOT
        / "workspace/robust_loop_verifier_runs/GEODE/Offroad/Offroad01_beta/20260520_152209",
        cache_root=Path(
            "/data/datasets/GEODE/robust_loop_verifier_cache/GEODE/Offroad/Offroad01_beta"
        ),
    ),
    "eee_02": SequenceSpec(
        dataset="NTU-VIRAL",
        platform="NTU-VIRAL",
        sequence="eee_02",
        run_root=REPO_ROOT / (
            "workspace/robust_loop_verifier_runs/NTU-VIRAL/NTU-VIRAL/"
            "eee_02/20260521_aster_slam_labels"
        ),
        cache_root=Path(
            "/data/datasets/NTU-VIRAL/robust_loop_verifier_cache/NTU-VIRAL/NTU-VIRAL/eee_02"
        ),
    ),
}


def run_failure_case_analysis(
    sequence_specs: Sequence[SequenceSpec],
    *,
    output_root: Path,
    max_failures_per_method: int = 24,
    methods: Sequence[str] = DEFAULT_METHODS,
) -> list[MethodResult]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _initialize_summary_tables(output_root)

    results: list[MethodResult] = []
    for spec in sequence_specs:
        records = _read_jsonl(spec.run_root / "candidate_records.jsonl")
        keyframes = _read_keyframes(spec.cache_root / "keyframes.jsonl")
        manifest = _read_json(spec.cache_root / "manifest.json")
        sequence_results: list[MethodResult] = []

        for method in methods:
            raw_scores = _method_scores(records, method)
            result = _analyze_method(
                spec,
                records,
                raw_scores,
                method=method,
                output_root=output_root,
                keyframes=keyframes,
                manifest=manifest,
                max_failures=max_failures_per_method,
            )
            sequence_results.append(result)
            results.append(result)

        # Persist immediately after each sequence so a later long run cannot lose earlier results.
        _append_summary_rows(output_root, sequence_results)
        _write_sequence_table(output_root, spec, sequence_results)

    return results


def _analyze_method(
    spec: SequenceSpec,
    records: Sequence[Mapping[str, Any]],
    raw_scores: Sequence[float | None],
    *,
    method: str,
    output_root: Path,
    keyframes: Mapping[int, Mapping[str, Any]],
    manifest: Mapping[str, Any],
    max_failures: int,
) -> MethodResult:
    labels = [_label(record) for record in records]
    scores = assign_failure_worst_scores(raw_scores)
    ap = average_precision(labels, scores)
    mr = max_recall_at_100_precision(labels, scores)
    ranked_indices = sorted(range(len(records)), key=lambda index: (-scores[index], index))

    first_false_rank: int | None = None
    true_before_first_false = 0
    false_positive_indices: list[int] = []
    true_seen = 0
    for rank, index in enumerate(ranked_indices, start=1):
        if labels[index]:
            true_seen += 1
            continue
        if raw_scores[index] is None:
            continue
        false_positive_indices.append(index)
        if first_false_rank is None:
            first_false_rank = rank
            true_before_first_false = true_seen

    method_dir = (
        output_root / spec.dataset / spec.platform / spec.sequence / _slugify_method(method)
    )
    method_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    rank_lookup = {index: rank for rank, index in enumerate(ranked_indices, start=1)}
    for index in false_positive_indices[:max_failures]:
        _export_false_positive(
            method_dir,
            records[index],
            method=method,
            method_score=scores[index],
            score_rank=rank_lookup[index],
            keyframes=keyframes,
            cache_root=spec.cache_root,
            manifest=manifest,
        )
        saved += 1

    return MethodResult(
        dataset=spec.dataset,
        platform=spec.platform,
        sequence=spec.sequence,
        method=method,
        record_count=len(records),
        positive_count=sum(labels),
        ap=ap,
        mr_at_100p=mr,
        first_false_rank=first_false_rank,
        true_before_first_false=true_before_first_false,
        false_positive_count=len(false_positive_indices),
        saved_false_positives=saved,
        output_dir=method_dir,
    )


def _export_false_positive(
    method_dir: Path,
    record: Mapping[str, Any],
    *,
    method: str,
    method_score: float,
    score_rank: int,
    keyframes: Mapping[int, Mapping[str, Any]],
    cache_root: Path,
    manifest: Mapping[str, Any],
) -> None:
    query_idx = _int_value(record.get("query_idx"))
    candidate_idx = _int_value(record.get("candidate_idx"))
    support_idx = _optional_int_value(record.get("support_idx"))
    support_part = f"s{support_idx:06d}" if support_idx is not None else "snone"
    case_name = (
        f"rank{score_rank:03d}_q{query_idx:06d}_"
        f"c{candidate_idx:06d}_{support_part}"
    )
    case_dir = method_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    image_specs = [
        ("query", query_idx),
        ("candidate", candidate_idx),
    ]
    if support_idx is not None:
        image_specs.append(("support", support_idx))

    exported_images: list[tuple[str, Path, int]] = []
    for role, keyframe_idx in image_specs:
        src = _resolve_image_path(keyframe_idx, keyframes, cache_root, manifest)
        dst = case_dir / f"{role}.png"
        _copy_image_or_placeholder(src, dst, f"{role} {keyframe_idx:06d}")
        exported_images.append((role, dst, keyframe_idx))

    _write_overview(
        case_dir / "overview.png",
        exported_images,
        title=f"{method}  score_rank={score_rank}  score={method_score:.6f}",
    )
    record_payload = dict(record)
    record_payload.update(
        {
            "method": method,
            "method_score": method_score,
            "score_rank": score_rank,
            "exported_images": {
                role: str(path.relative_to(case_dir))
                for role, path, _ in exported_images
            },
        }
    )
    (case_dir / "record.json").write_text(
        json.dumps(record_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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

    raise ValueError(f"unsupported method for failure-case export: {method}")


def _raw_graph_scores(
    records: Sequence[Mapping[str, Any]],
    deformation_weight: float,
    residual_weight: float,
) -> list[float | None]:
    scores: list[float | None] = []
    for record in records:
        deformation = _finite_float_or_none(record.get("trajectory_deformation_rmse"))
        residual = _log1p_or_none(record.get("pgo_error_after"))
        if deformation is None or residual is None:
            scores.append(None)
            continue
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
            if base_scores[index] is not None and math.isfinite(float(base_scores[index]))
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


def _initialize_summary_tables(output_root: Path) -> None:
    summary_md = output_root / "summary.md"
    if not summary_md.exists():
        summary_md.write_text(
            (
                "| dataset | platform | sequence | method | records | positives | AP | "
                "MR@100P | first false rank | true before first false | "
                "false positives | saved | image dir |\n"
            )
            + (
                "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | "
                "---: | ---: | ---: | --- |\n"
            ),
            encoding="utf-8",
        )
    summary_csv = output_root / "summary.csv"
    if not summary_csv.exists():
        with summary_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "dataset",
                    "platform",
                    "sequence",
                    "method",
                    "record_count",
                    "positive_count",
                    "AP",
                    "MR@100P",
                    "first_false_rank",
                    "true_before_first_false",
                    "false_positive_count",
                    "saved_false_positives",
                    "output_dir",
                ]
            )


def _append_summary_rows(output_root: Path, rows: Sequence[MethodResult]) -> None:
    with (output_root / "summary.md").open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                (
                    "| {} | {} | {} | {} | {} | {} | {:.4f} | {:.4f} | "
                    "{} | {} | {} | {} | {} |\n"
                ).format(
                    row.dataset,
                    row.platform,
                    row.sequence,
                    row.method,
                    row.record_count,
                    row.positive_count,
                    row.ap,
                    row.mr_at_100p,
                    _optional_int_text(row.first_false_rank),
                    row.true_before_first_false,
                    row.false_positive_count,
                    row.saved_false_positives,
                    row.output_dir,
                )
            )
    with (output_root / "summary.csv").open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for row in rows:
            writer.writerow(
                [
                    row.dataset,
                    row.platform,
                    row.sequence,
                    row.method,
                    row.record_count,
                    row.positive_count,
                    f"{row.ap:.8f}",
                    f"{row.mr_at_100p:.8f}",
                    _optional_int_text(row.first_false_rank),
                    row.true_before_first_false,
                    row.false_positive_count,
                    row.saved_false_positives,
                    row.output_dir,
                ]
            )


def _write_sequence_table(
    output_root: Path,
    spec: SequenceSpec,
    rows: Sequence[MethodResult],
) -> None:
    sequence_dir = output_root / spec.dataset / spec.platform / spec.sequence
    sequence_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Failure Case Summary: {spec.dataset}/{spec.platform}/{spec.sequence}",
        "",
        (
            "| method | AP | MR@100P | first false rank | true before first false | "
            "saved false positives | image dir |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {} | {:.4f} | {:.4f} | {} | {} | {} | {} |".format(
                row.method,
                row.ap,
                row.mr_at_100p,
                _optional_int_text(row.first_false_rank),
                row.true_before_first_false,
                row.saved_false_positives,
                row.output_dir,
            )
        )
    (sequence_dir / "sequence_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _read_keyframes(path: Path) -> dict[int, Mapping[str, Any]]:
    return {_int_value(row["idx"]): row for row in _read_jsonl(path)}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_image_path(
    keyframe_idx: int,
    keyframes: Mapping[int, Mapping[str, Any]],
    cache_root: Path,
    manifest: Mapping[str, Any],
) -> Path | None:
    candidates: list[Path] = []
    keyframe = keyframes.get(keyframe_idx, {})
    image_path = keyframe.get("image_path")
    if isinstance(image_path, str) and image_path:
        path = Path(image_path)
        candidates.append(path if path.is_absolute() else cache_root / path)
    candidates.append(cache_root / "images" / f"{keyframe_idx:06d}.png")

    raw_dir = manifest.get("raw_dir") or keyframe.get("source_raw_dir")
    if isinstance(raw_dir, str) and raw_dir:
        candidates.append(Path(raw_dir) / "keyframe_images" / f"{keyframe_idx:06d}.png")

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _copy_image_or_placeholder(src: Path | None, dst: Path, label: str) -> None:
    if src is not None and src.is_file():
        shutil.copy2(src, dst)
        return
    image = Image.new("RGB", (320, 240), color=(40, 40, 40))
    draw = ImageDraw.Draw(image)
    draw.text((12, 12), f"missing\n{label}", fill=(255, 255, 255))
    image.save(dst)


def _write_overview(
    path: Path,
    exported_images: Sequence[tuple[str, Path, int]],
    *,
    title: str,
) -> None:
    card_width = 320
    card_height = 260
    title_height = 48
    canvas = Image.new(
        "RGB",
        (card_width * len(exported_images), title_height + card_height),
        color=(245, 245, 245),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 12), title, fill=(0, 0, 0))
    for column, (role, image_path, keyframe_idx) in enumerate(exported_images):
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((card_width, card_height - 28))
        x = column * card_width + (card_width - image.width) // 2
        y = title_height + 26
        canvas.paste(image, (x, y))
        draw.text(
            (column * card_width + 10, title_height + 6),
            f"{role}: {keyframe_idx:06d}",
            fill=(0, 0, 0),
        )
    canvas.save(path)


def _label(record: Mapping[str, Any]) -> bool:
    if type(record.get("label")) is not bool:
        raise ValueError(f"candidate record label must be bool: {record.get('label')!r}")
    return bool(record["label"])


def _finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _log1p_or_none(value: Any) -> float | None:
    number = _finite_float_or_none(value)
    if number is None or number < -1.0:
        return None
    return math.log1p(number)


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


def _optional_int_text(value: int | None) -> str:
    return "" if value is None else str(value)


def _slugify_method(method: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", method).strip("_").lower()
    return slug or "method"


def _parse_sequence_names(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_methods(
    value: str | None,
    method_values: Sequence[str] | None = None,
) -> tuple[str, ...]:
    if method_values:
        return tuple(method for method in method_values if method)
    if not value:
        return DEFAULT_METHODS
    separator = ";" if ";" in value else ","
    return tuple(item.strip() for item in value.split(separator) if item.strip())


def _default_output_root() -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "workspace" / "failure_case_analysis" / run_id


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export ranked false-positive loop visualizations from existing "
            "candidate_records.jsonl files."
        )
    )
    parser.add_argument(
        "--sequences",
        default="ugv_parking02,Offroad01_beta,eee_02",
        help="Comma-separated default sequence keys to analyze.",
    )
    parser.add_argument(
        "--method",
        action="append",
        default=None,
        help=(
            "Method to export. Can be passed multiple times. This is preferred because "
            "graph method names contain commas."
        ),
    )
    parser.add_argument(
        "--methods",
        default=None,
        help=(
            "Optional semicolon-separated methods string. Kept for ad-hoc use; "
            "prefer repeated --method when method names contain commas."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output directory. Defaults to workspace/failure_case_analysis/<timestamp>.",
    )
    parser.add_argument(
        "--max-failures-per-method",
        type=int,
        default=24,
        help="Maximum top-ranked false positives to export for each method and sequence.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    sequence_names = _parse_sequence_names(args.sequences)
    unknown = [name for name in sequence_names if name not in DEFAULT_SEQUENCE_SPECS]
    if unknown:
        raise SystemExit(
            "Unknown sequence key(s): {}. Available: {}".format(
                ", ".join(unknown),
                ", ".join(sorted(DEFAULT_SEQUENCE_SPECS)),
            )
        )
    output_root = args.output_root or _default_output_root()
    specs = [DEFAULT_SEQUENCE_SPECS[name] for name in sequence_names]
    results = run_failure_case_analysis(
        specs,
        output_root=output_root,
        max_failures_per_method=args.max_failures_per_method,
        methods=_parse_methods(args.methods, args.method),
    )
    print(f"Wrote failure-case analysis to {output_root}")
    for result in results:
        print(
            "{} / {} / {} / {}: AP={:.4f}, MR@100P={:.4f}, saved={}".format(
                result.dataset,
                result.platform,
                result.sequence,
                result.method,
                result.ap,
                result.mr_at_100p,
                result.saved_false_positives,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
