#!/usr/bin/env python3
"""Batch preprocess and run FusionPortable handheld/legged verifier experiments."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from robust_loop_verifier.batch import (
    AteSequence,
    discover_aster_slam_ate_sequences,
    write_csv,
    write_markdown_table,
)
from robust_loop_verifier.io import read_json, write_json


def main() -> int:
    args = _parse_args()
    repo_root = Path(args.repo_root).resolve()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    platforms = _parse_csv(args.platforms)
    output_base = Path(args.output_base).resolve()
    default_batch_root = output_base / args.dataset_name / f"batch_handheld_legged_{run_id}"
    batch_root = Path(args.batch_output_root or default_batch_root)
    selected = discover_aster_slam_ate_sequences(
        Path(args.loop_dataset_root),
        platforms=platforms,
        ate_rmse_threshold_m=args.ate_rmse_threshold_m,
    )

    print("Selected sequences with AsterSLAM ATE RMSE below threshold:")
    for item in selected:
        print(f"  {item.platform}/{item.sequence_name}: rmse={item.ate_rmse_m:.6f} m")
    if not selected:
        print("No eligible sequences found.")
        return 1

    summary_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for item in selected:
        row = _base_summary_row(item, args, run_id, output_base)
        summary_rows.append(row)
        commands = _commands_for_sequence(item, args, repo_root, run_id, output_base)
        if args.dry_run:
            row["status"] = "dry_run"
            print()
            print(f"[dry-run] {item.platform}/{item.sequence_name}")
            for command, _ in commands:
                print("  " + shlex.join(command))
            continue

        try:
            for command, env in commands:
                print()
                print(f"[run] {item.platform}/{item.sequence_name}: {shlex.join(command)}")
                subprocess.run(command, cwd=repo_root, env=env, check=True)
            metrics = read_json(Path(row["run_root"]) / "metrics.json")
            row["status"] = "ok"
            row["metrics"] = metrics
            metric_rows.extend(_flatten_metrics(item, metrics))
        except (OSError, subprocess.CalledProcessError, ValueError) as error:
            row["status"] = "error"
            row["error"] = str(error)
            print(f"[error] {item.platform}/{item.sequence_name}: {error}", file=sys.stderr)
            if not args.keep_going:
                _write_outputs(batch_root, args, run_id, summary_rows, metric_rows)
                return 1

    _write_outputs(batch_root, args, run_id, summary_rows, metric_rows)
    print()
    print(f"batch_summary_json={batch_root / 'batch_summary.json'}")
    print(f"batch_summary_csv={batch_root / 'batch_summary.csv'}")
    print(f"batch_metrics_csv={batch_root / 'batch_metrics.csv'}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument(
        "--loop-dataset-root",
        default="/data/datasets/FusionPortable/fusionportable_loop_dataset",
    )
    parser.add_argument(
        "--cache-root",
        default="/data/datasets/FusionPortable/robust_loop_verifier_cache",
    )
    parser.add_argument(
        "--output-base",
        default=str(
            Path(__file__).resolve().parents[1]
            / "workspace"
            / "robust_loop_verifier_runs"
        ),
    )
    parser.add_argument("--batch-output-root", default=None)
    parser.add_argument("--dataset-name", default="FusionPortableV2")
    parser.add_argument("--platforms", default="handheld,legged")
    parser.add_argument("--ate-rmse-threshold-m", type=float, default=0.1)
    parser.add_argument("--backend", default=os.environ.get("BACKEND", "real"))
    parser.add_argument("--query-limit", type=int, default=_optional_int_env("QUERY_LIMIT"))
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID"))
    parser.add_argument("--config", default=os.environ.get("CONFIG"))
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", sys.executable))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


def _commands_for_sequence(
    item: AteSequence,
    args: argparse.Namespace,
    repo_root: Path,
    run_id: str,
    output_base: Path,
) -> list[tuple[list[str], dict[str, str]]]:
    script_dir = repo_root / "robust_loop_verification_scripts"
    generate_script = script_dir / "generate_fusionportable_dataset_cache.sh"
    run_script = script_dir / "run_robust_loop_verifier_pipeline.sh"

    base_env = os.environ.copy()
    base_env["PYTHON_BIN"] = args.python_bin
    if args.config:
        base_env["CONFIG"] = args.config

    preprocess_env = base_env.copy()
    preprocess_env.update(
        {
            "PLATFORM": item.platform,
            "LOOP_DATASET_ROOT": str(Path(args.loop_dataset_root)),
            "OUTPUT_ROOT": str(Path(args.cache_root)),
        }
    )

    run_env = base_env.copy()
    run_env.update(
        {
            "PLATFORM": item.platform,
            "CACHE_ROOT": str(Path(args.cache_root)),
            "DATASET_NAME": args.dataset_name,
            "OUTPUT_BASE": str(output_base),
            "RUN_ID": run_id,
            "BACKEND": args.backend,
        }
    )
    if args.query_limit is not None:
        run_env["QUERY_LIMIT"] = str(args.query_limit)

    return [
        ([str(generate_script), item.sequence_name], preprocess_env),
        ([str(run_script), item.sequence_name], run_env),
    ]


def _base_summary_row(
    item: AteSequence,
    args: argparse.Namespace,
    run_id: str,
    output_base: Path,
) -> dict[str, Any]:
    cache_dir = Path(args.cache_root) / args.dataset_name / item.platform / item.sequence_name
    run_root = output_base / args.dataset_name / item.platform / item.sequence_name / run_id
    return {
        "platform": item.platform,
        "sequence_name": item.sequence_name,
        "ate_rmse_m": item.ate_rmse_m,
        "cache_dir": str(cache_dir),
        "run_root": str(run_root),
        "status": "pending",
        "error": "",
    }


def _flatten_metrics(item: AteSequence, metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, values in metrics.items():
        if not isinstance(values, Mapping):
            continue
        rows.append(
            {
                "platform": item.platform,
                "sequence": item.sequence_name,
                "method": method,
                "AP": float(values["AP"]),
                "MR@100P": float(values["MR@100P"]),
            }
        )
    return rows


def _write_outputs(
    batch_root: Path,
    args: argparse.Namespace,
    run_id: str,
    summary_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
) -> None:
    payload = {
        "dataset_name": args.dataset_name,
        "run_id": run_id,
        "ate_rmse_threshold_m": args.ate_rmse_threshold_m,
        "backend": args.backend,
        "query_limit": args.query_limit,
        "sequences": summary_rows,
    }
    write_json(batch_root / "batch_summary.json", payload)
    write_csv(
        batch_root / "batch_summary.csv",
        summary_rows,
        [
            "platform",
            "sequence_name",
            "ate_rmse_m",
            "status",
            "cache_dir",
            "run_root",
            "error",
        ],
    )
    write_markdown_table(
        batch_root / "batch_summary.md",
        summary_rows,
        columns=["platform", "sequence_name", "ate_rmse_m", "status", "run_root", "error"],
    )
    write_csv(
        batch_root / "batch_metrics.csv",
        metric_rows,
        ["platform", "sequence", "method", "AP", "MR@100P"],
    )
    write_markdown_table(
        batch_root / "batch_metrics.md",
        metric_rows,
        columns=["platform", "sequence", "method", "AP", "MR@100P"],
    )


def _parse_csv(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("CSV value must not be empty")
    return items


def _optional_int_env(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return int(value)


if __name__ == "__main__":
    raise SystemExit(main())
