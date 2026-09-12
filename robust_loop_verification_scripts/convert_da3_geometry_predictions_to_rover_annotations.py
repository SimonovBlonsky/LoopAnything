#!/usr/bin/env python3
"""Convert DA3 geometry automatic labels into a sealed ROVER-aligned benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

LOOPANYTHING_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = LOOPANYTHING_ROOT / "src"
for import_path in (LOOPANYTHING_ROOT, SRC_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from robust_loop_verifier.io import read_json, read_jsonl, write_json, write_jsonl  # noqa: E402
from robust_loop_verifier.rover_annotation import (  # noqa: E402
    ANNOTATION_SEAL_VERSION,
    ANNOTATION_VERSION,
    sha256_file,
    verify_annotation_seal,
)

PAIR_FILENAME = "benchmark_pairs.jsonl"
MANIFEST_FILENAME = "manifest.json"
PREDICTION_FILENAME = "geometry_predictions.jsonl"
PREDICTION_MANIFEST_FILENAME = "geometry_prediction_manifest.json"
ANNOTATION_FILENAME = "annotations.jsonl"
ANNOTATION_SEAL_FILENAME = "annotation_seal.json"
CONVERSION_MANIFEST_FILENAME = "auto_label_conversion_manifest.json"
SCORES_DIRNAME = "scores"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_root", type=Path)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--copy-scores",
        action="store_true",
        help="Copy existing score files from BENCHMARK_ROOT/scores into OUTPUT_ROOT/scores.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing generated output files under OUTPUT_ROOT.",
    )
    parser.add_argument(
        "--review-only",
        action="store_true",
        help=(
            "Prepare an unsealed annotation-review root with frozen pairs, geometry "
            "predictions, and optional scores, but do not write annotations.jsonl."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        convert(
            benchmark_root=args.benchmark_root,
            geometry_root=args.geometry_root,
            output_root=args.output_root,
            copy_scores=bool(args.copy_scores),
            overwrite=bool(args.overwrite),
            review_only=bool(args.review_only),
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def convert(
    *,
    benchmark_root: Path,
    geometry_root: Path,
    output_root: Path,
    copy_scores: bool,
    overwrite: bool = False,
    review_only: bool = False,
) -> None:
    benchmark_root = Path(benchmark_root)
    geometry_root = Path(geometry_root)
    output_root = Path(output_root)
    _validate_roots(benchmark_root, geometry_root, output_root)

    pairs_path = benchmark_root / PAIR_FILENAME
    manifest_path = benchmark_root / MANIFEST_FILENAME
    prediction_path = geometry_root / PREDICTION_FILENAME
    if not pairs_path.is_file():
        raise FileNotFoundError(pairs_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)

    _validate_geometry_snapshot_matches_benchmark(benchmark_root, geometry_root)
    pairs = list(read_jsonl(pairs_path))
    pair_ids = _validate_pairs(pairs)
    predictions = list(read_jsonl(prediction_path))
    labels = _labels_from_predictions(predictions, pair_ids)
    manifest = read_json(manifest_path)
    expected_pair_hash = _required_string(manifest, "pair_manifest_sha256", "manifest")
    actual_pair_hash = sha256_file(pairs_path)
    if expected_pair_hash != actual_pair_hash:
        raise ValueError("manifest pair_manifest_sha256 does not match benchmark_pairs.jsonl")

    output_files = [
        output_root / PAIR_FILENAME,
        output_root / MANIFEST_FILENAME,
        output_root / PREDICTION_FILENAME,
        output_root / CONVERSION_MANIFEST_FILENAME,
    ]
    if not review_only:
        output_files.extend(
            [
                output_root / ANNOTATION_FILENAME,
                output_root / ANNOTATION_SEAL_FILENAME,
            ]
        )
    if (geometry_root / PREDICTION_MANIFEST_FILENAME).is_file():
        output_files.append(output_root / PREDICTION_MANIFEST_FILENAME)
    if not overwrite:
        existing = [path for path in output_files if path.exists()]
        if copy_scores and (output_root / SCORES_DIRNAME).exists():
            existing.append(output_root / SCORES_DIRNAME)
        if existing:
            raise FileExistsError(
                "output already exists; pass --overwrite to replace generated files: "
                + str(existing[0])
            )

    output_root.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in output_files:
            if path.exists():
                path.unlink()
        if copy_scores and (output_root / SCORES_DIRNAME).exists():
            shutil.rmtree(output_root / SCORES_DIRNAME)

    shutil.copy2(pairs_path, output_root / PAIR_FILENAME)
    shutil.copy2(manifest_path, output_root / MANIFEST_FILENAME)
    shutil.copy2(prediction_path, output_root / PREDICTION_FILENAME)
    if (geometry_root / PREDICTION_MANIFEST_FILENAME).is_file():
        shutil.copy2(
            geometry_root / PREDICTION_MANIFEST_FILENAME,
            output_root / PREDICTION_MANIFEST_FILENAME,
        )

    annotated_at = datetime.now(timezone.utc).isoformat()
    if not review_only:
        annotation_rows = [
            {
                "pair_id": pair_id,
                "label": labels[pair_id],
                "annotated_at": annotated_at,
                "annotation_version": ANNOTATION_VERSION,
            }
            for pair_id in pair_ids
        ]
        write_jsonl(output_root / ANNOTATION_FILENAME, annotation_rows)
        write_json(
            output_root / ANNOTATION_SEAL_FILENAME,
            {
                "version": ANNOTATION_SEAL_VERSION,
                "annotation_version": ANNOTATION_VERSION,
                "benchmark_version": _required_string(manifest, "benchmark_version", "manifest"),
                "pair_manifest_sha256": actual_pair_hash,
                "annotation_sha256": sha256_file(output_root / ANNOTATION_FILENAME),
                "count": len(annotation_rows),
                "completed_at": annotated_at,
            },
        )
        verify_annotation_seal(
            output_root / MANIFEST_FILENAME,
            output_root / PAIR_FILENAME,
            output_root / ANNOTATION_FILENAME,
            output_root / ANNOTATION_SEAL_FILENAME,
        )

    if copy_scores:
        _copy_scores(benchmark_root / SCORES_DIRNAME, output_root / SCORES_DIRNAME)

    positive_count = sum(labels.values())
    conversion_manifest = {
        "conversion_version": 1,
        "source_benchmark_root": str(benchmark_root.resolve()),
        "source_geometry_root": str(geometry_root.resolve()),
        "output_root": str(output_root.resolve()),
        "pair_count": len(pair_ids),
        "positive_count": positive_count,
        "negative_count": len(pair_ids) - positive_count,
        "pair_manifest_sha256": actual_pair_hash,
        "geometry_predictions_sha256": sha256_file(prediction_path),
        "created_at": annotated_at,
        "label_source": "geometry_predictions.automatic_label",
        "copy_scores": copy_scores,
        "review_only": review_only,
    }
    if not review_only:
        conversion_manifest["annotation_sha256"] = sha256_file(
            output_root / ANNOTATION_FILENAME
        )
    if (geometry_root / PREDICTION_MANIFEST_FILENAME).is_file():
        conversion_manifest["geometry_prediction_manifest_sha256"] = sha256_file(
            geometry_root / PREDICTION_MANIFEST_FILENAME
        )
    write_json(output_root / CONVERSION_MANIFEST_FILENAME, conversion_manifest)

    action = "Prepared DA3 automatic-label review benchmark" if review_only else (
        "Converted DA3 automatic labels to sealed benchmark"
    )
    print(f"{action}: {output_root}")
    print(f"  pairs: {len(pair_ids)}")
    print(f"  positives: {positive_count}")
    print(f"  negatives: {len(pair_ids) - positive_count}")


def _validate_roots(benchmark_root: Path, geometry_root: Path, output_root: Path) -> None:
    resolved_benchmark = benchmark_root.resolve()
    resolved_geometry = geometry_root.resolve()
    resolved_output = output_root.resolve()
    if not benchmark_root.is_dir():
        raise NotADirectoryError(benchmark_root)
    if not geometry_root.is_dir():
        raise NotADirectoryError(geometry_root)
    if resolved_output == resolved_benchmark:
        raise ValueError("output_root must differ from benchmark_root")
    if resolved_output == resolved_geometry:
        raise ValueError("output_root must differ from geometry_root")


def _validate_geometry_snapshot_matches_benchmark(
    benchmark_root: Path,
    geometry_root: Path,
) -> None:
    for filename in (PAIR_FILENAME, MANIFEST_FILENAME):
        geometry_path = geometry_root / filename
        benchmark_path = benchmark_root / filename
        if not geometry_path.is_file():
            raise FileNotFoundError(geometry_path)
        if sha256_file(geometry_path) != sha256_file(benchmark_path):
            raise ValueError(f"geometry {filename} does not match benchmark {filename}")


def _validate_pairs(pairs: Sequence[Mapping[str, object]]) -> list[str]:
    pair_ids: list[str] = []
    seen: set[str] = set()
    for index, pair in enumerate(pairs):
        pair_id = pair.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError(f"benchmark pair {index} pair_id must be a non-empty string")
        if pair_id in seen:
            raise ValueError(f"duplicate benchmark pair_id: {pair_id}")
        seen.add(pair_id)
        pair_ids.append(pair_id)
    if not pair_ids:
        raise ValueError("benchmark pairs must be non-empty")
    return pair_ids


def _labels_from_predictions(
    predictions: Sequence[Mapping[str, object]],
    pair_ids: Sequence[str],
) -> dict[str, int]:
    if len(predictions) != len(pair_ids):
        raise ValueError(
            f"geometry prediction count {len(predictions)} does not match pair count {len(pair_ids)}"
        )
    labels: dict[str, int] = {}
    for index, (prediction, expected_pair_id) in enumerate(zip(predictions, pair_ids)):
        pair_id = prediction.get("pair_id")
        if pair_id != expected_pair_id:
            raise ValueError(
                "geometry_predictions.jsonl pair_id order must match benchmark_pairs.jsonl "
                f"at row {index}: expected {expected_pair_id}, got {pair_id!r}"
            )
        label = prediction.get("automatic_label")
        if type(label) is not int or label not in (0, 1):
            raise ValueError(
                f"geometry prediction row {index} automatic_label must be exact int 0 or 1"
            )
        labels[str(pair_id)] = int(label)
    return labels


def _copy_scores(source: Path, target: Path) -> None:
    if not source.is_dir():
        raise NotADirectoryError(source)
    target.mkdir(parents=True, exist_ok=True)
    for source_path in sorted(source.iterdir()):
        if not source_path.is_file():
            continue
        shutil.copy2(source_path, target / source_path.name)


def _required_string(row: Mapping[str, object], field: str, context: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} {field} must be a non-empty string")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
