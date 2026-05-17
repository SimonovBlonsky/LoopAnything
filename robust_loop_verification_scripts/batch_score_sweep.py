#!/usr/bin/env python3
"""Batch residual-aware score sweep for robust loop verifier run roots."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from robust_loop_verifier.artifacts import write_json
from robust_loop_verifier.batch import (
    BatchRunRoot,
    average_score_sweep_rows,
    discover_aster_slam_ate_sequences,
    read_batch_summary_run_roots,
    write_csv,
    write_markdown_table,
)
from robust_loop_verifier.score_sweep import compute_score_sweep, read_candidate_records


def main() -> int:
    args = _parse_args()
    records = _resolve_run_roots(args)
    if not records:
        raise SystemExit("No run roots found for batch score sweep")

    output_root = Path(args.output_root or _default_output_root(args, records)).resolve()
    per_sequence_rows: list[dict[str, Any]] = []
    per_sequence_results: list[dict[str, Any]] = []

    for record in records:
        candidate_records = record.run_root / "candidate_records.jsonl"
        if not candidate_records.is_file():
            message = f"missing candidate_records.jsonl: {candidate_records}"
            if args.skip_missing:
                print(f"[skip] {message}")
                continue
            raise SystemExit(message)

        result = compute_score_sweep(
            read_candidate_records(candidate_records),
            graph_weights=_parse_float_csv(args.graph_weights),
            fusion_weights=_parse_float_csv(args.fusion_weights),
        )
        per_sequence_results.append(
            {
                "platform": record.platform,
                "sequence_name": record.sequence_name,
                "run_root": str(record.run_root),
                "ate_rmse_m": record.ate_rmse_m,
                "sweep": result,
            }
        )
        for row in result["scores"]:
            per_sequence_rows.append(
                {
                    "platform": record.platform,
                    "sequence": record.sequence_name,
                    "method": row["name"],
                    "AP": row["AP"],
                    "MR@100P": row["MR@100P"],
                }
            )

    average_rows = average_score_sweep_rows(per_sequence_rows)
    payload = {
        "sequence_count": len(
            {(row["platform"], row["sequence"]) for row in per_sequence_rows}
        ),
        "per_sequence": per_sequence_results,
        "average": average_rows,
        "best_by_average_ap": _best_average_row(average_rows, "average_AP"),
        "best_by_average_mr": _best_average_row(average_rows, "average_MR@100P"),
    }
    write_json(output_root / "batch_score_sweep.json", payload)
    write_csv(
        output_root / "batch_score_sweep_per_sequence.csv",
        per_sequence_rows,
        ["platform", "sequence", "method", "AP", "MR@100P"],
    )
    write_csv(
        output_root / "batch_score_sweep_average.csv",
        average_rows,
        ["method", "sequence_count", "average_AP", "average_MR@100P"],
    )
    write_markdown_table(
        output_root / "batch_score_sweep_average.md",
        sorted(average_rows, key=lambda row: (-row["average_AP"], -row["average_MR@100P"])),
        columns=["method", "sequence_count", "average_AP", "average_MR@100P"],
    )

    print(f"batch_score_sweep_json={output_root / 'batch_score_sweep.json'}")
    print(f"batch_score_sweep_average_csv={output_root / 'batch_score_sweep_average.csv'}")
    print(
        "batch_score_sweep_per_sequence_csv="
        f"{output_root / 'batch_score_sweep_per_sequence.csv'}"
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-summary", default=None)
    parser.add_argument(
        "--loop-dataset-root",
        default="/data/datasets/FusionPortable/fusionportable_loop_dataset",
    )
    parser.add_argument(
        "--runs-root",
        default=str(
            Path(__file__).resolve().parents[1]
            / "workspace"
            / "robust_loop_verifier_runs"
        ),
    )
    parser.add_argument("--dataset-name", default="FusionPortableV2")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--platforms", default="handheld,legged")
    parser.add_argument("--ate-rmse-threshold-m", type=float, default=0.1)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--graph-weights", default="0,0.25,0.5,1,2,4")
    parser.add_argument("--fusion-weights", default="0,0.25,0.5,1,2,4")
    parser.add_argument("--skip-missing", action="store_true")
    return parser.parse_args()


def _resolve_run_roots(args: argparse.Namespace) -> list[BatchRunRoot]:
    if args.batch_summary:
        return read_batch_summary_run_roots(Path(args.batch_summary))
    if not args.run_id:
        raise SystemExit("Set --batch-summary or --run-id")

    runs_root = Path(args.runs_root) / args.dataset_name
    selected = discover_aster_slam_ate_sequences(
        Path(args.loop_dataset_root),
        platforms=_parse_string_csv(args.platforms),
        ate_rmse_threshold_m=args.ate_rmse_threshold_m,
    )
    return [
        BatchRunRoot(
            platform=item.platform,
            sequence_name=item.sequence_name,
            run_root=runs_root / item.platform / item.sequence_name / args.run_id,
            ate_rmse_m=item.ate_rmse_m,
        )
        for item in selected
    ]


def _default_output_root(args: argparse.Namespace, records: list[BatchRunRoot]) -> Path:
    if args.batch_summary:
        return Path(args.batch_summary).resolve().parent / "batch_score_sweep"
    run_id = args.run_id or records[0].run_root.name
    return (
        Path(args.runs_root).resolve()
        / args.dataset_name
        / f"batch_handheld_legged_{run_id}"
        / "score_sweep"
    )


def _parse_string_csv(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("CSV value must not be empty")
    return items


def _parse_float_csv(value: str) -> tuple[float, ...]:
    items = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("weight list must not be empty")
    if any(not math.isfinite(item) or item < 0.0 for item in items):
        raise ValueError("weights must be finite non-negative values")
    return items


def _best_average_row(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    if not rows:
        return {}
    return dict(
        max(rows, key=lambda row: (row[metric], row["average_MR@100P"], row["method"]))
    )


if __name__ == "__main__":
    raise SystemExit(main())
