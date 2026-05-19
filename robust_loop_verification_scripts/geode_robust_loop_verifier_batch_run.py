#!/usr/bin/env python3
"""Batch preprocess and run GEODE robust loop verifier experiments."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from robust_loop_verifier.batch import (
    average_score_sweep_rows,
    write_csv,
    write_markdown_table,
)
from robust_loop_verifier.io import read_json, read_yaml, write_json, write_yaml


# Edit this list when more GEODE raw exports are ready for the main table.
GEODE_SEQUENCE_NAMES: tuple[str, ...] = ("Offroad05_beta",)


@dataclass(frozen=True)
class SequencePlan:
    platform: str
    sequence_name: str
    raw_dir: Path
    gt_trajectory_file: Path
    gt_label_source: str
    cache_dir: Path
    run_root: Path


def main() -> int:
    args = _parse_args()
    repo_root = Path(args.repo_root).resolve()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(args.output_base).resolve()
    batch_root = Path(
        args.batch_output_root or output_base / args.dataset_name / f"geode_batch_{run_id}"
    ).resolve()

    plans = _select_sequences(args, run_id, output_base)
    if not plans:
        print("No sequences selected.", file=sys.stderr)
        return 1

    summary_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for plan in plans:
        row = _summary_row(plan, args, run_id)
        row["preprocess_status"] = _initial_preprocess_status(plan.cache_dir, args)
        row["query_limit"] = _query_limit_for_cache(plan.cache_dir, args.query_limit)
        summary_rows.append(row)

        try:
            if args.dry_run:
                row["status"] = "dry_run"
                _print_dry_run(plan, row)
                continue

            if args.overwrite_dataset or not _cache_ready(plan.cache_dir):
                _run_preprocess(plan, args, repo_root, batch_root)
                row["preprocess_status"] = "completed"
            else:
                row["preprocess_status"] = "skipped_existing"

            row["query_limit"] = _query_limit_for_cache(plan.cache_dir, args.query_limit)
            if row["query_limit"] is None:
                raise ValueError(f"cache manifest lacks keyframe_count: {plan.cache_dir}")

            config_path = _write_run_config(plan, args, batch_root)
            _run_pipeline(plan, args, config_path, int(row["query_limit"]), repo_root)
            metrics = read_json(plan.run_root / "metrics.json")
            row["status"] = "ok"
            row["metrics"] = metrics
            metric_rows.extend(_flatten_metrics(plan, metrics))
        except (OSError, subprocess.CalledProcessError, ValueError) as error:
            row["status"] = "error"
            row["error"] = str(error)
            print(f"[error] {plan.platform}/{plan.sequence_name}: {error}", file=sys.stderr)
            if not args.keep_going:
                _write_outputs(batch_root, args, run_id, summary_rows, metric_rows)
                return 1

    _write_outputs(batch_root, args, run_id, summary_rows, metric_rows)
    print(f"batch_summary_json={batch_root / 'batch_summary.json'}")
    print(f"batch_metrics_csv={batch_root / 'batch_metrics.csv'}")
    print(f"batch_metrics_average_csv={batch_root / 'batch_metrics_average.csv'}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--repo-root", default=str(repo_root))
    parser.add_argument(
        "--loop-dataset-root",
        default="/data/datasets/GEODE/geode_loop_dataset",
    )
    parser.add_argument("--gt-data-root", default="/data/datasets/GEODE/data/offroad")
    parser.add_argument(
        "--cache-root",
        default="/data/datasets/GEODE/robust_loop_verifier_cache",
    )
    parser.add_argument(
        "--output-base",
        default=str(repo_root / "workspace" / "robust_loop_verifier_runs"),
    )
    parser.add_argument("--batch-output-root", default=None)
    parser.add_argument("--dataset-name", default="GEODE")
    parser.add_argument("--platform", default="Offroad")
    parser.add_argument("--sequences", default=",".join(GEODE_SEQUENCE_NAMES))
    parser.add_argument("--backend", default=os.environ.get("BACKEND", "real"))
    parser.add_argument("--query-limit", type=int, default=_optional_int_env("QUERY_LIMIT"))
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID"))
    parser.add_argument(
        "--config",
        default=os.environ.get(
            "CONFIG",
            str(repo_root / "configs" / "robust_loop_verifier" / "geode_offroad.yaml"),
        ),
    )
    parser.set_defaults(support_ensemble=True)
    parser.add_argument(
        "--support-ensemble",
        dest="support_ensemble",
        action="store_true",
        help="Use the GEODE support-ensemble config. This is the default.",
    )
    parser.add_argument(
        "--no-support-ensemble",
        dest="support_ensemble",
        action="store_false",
        help="Disable support ensemble and use the base GEODE config.",
    )
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", sys.executable))
    parser.add_argument("--max-gt-delta-sec", type=float, default=0.1)
    parser.add_argument("--overwrite_dataset", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


def _select_sequences(
    args: argparse.Namespace,
    run_id: str,
    output_base: Path,
) -> list[SequencePlan]:
    selected: list[SequencePlan] = []
    for sequence_name in _parse_csv(args.sequences):
        selected.append(
            _make_plan(
                platform=args.platform,
                sequence_name=sequence_name,
                raw_dir=Path(args.loop_dataset_root) / args.platform / sequence_name / "raw",
                gt_trajectory_file=Path(args.gt_data_root) / _gt_filename_for_sequence(sequence_name),
                gt_label_source="external_gt_trajectory",
                cache_root=Path(args.cache_root),
                output_base=output_base,
                dataset_name=args.dataset_name,
                run_id=run_id,
            )
        )
    return sorted(selected, key=lambda item: (item.platform, item.sequence_name))


def _make_plan(
    *,
    platform: str,
    sequence_name: str,
    raw_dir: Path,
    gt_trajectory_file: Path,
    gt_label_source: str,
    cache_root: Path,
    output_base: Path,
    dataset_name: str,
    run_id: str,
) -> SequencePlan:
    return SequencePlan(
        platform=platform,
        sequence_name=sequence_name,
        raw_dir=raw_dir,
        gt_trajectory_file=gt_trajectory_file,
        gt_label_source=gt_label_source,
        cache_dir=cache_root / dataset_name / platform / sequence_name,
        run_root=output_base / dataset_name / platform / sequence_name / run_id,
    )


def _run_preprocess(
    plan: SequencePlan,
    args: argparse.Namespace,
    repo_root: Path,
    batch_root: Path,
) -> None:
    _validate_sequence_inputs(plan)
    config_path = _write_preprocess_config(plan, args, batch_root)
    command = [
        args.python_bin,
        "-m",
        "robust_loop_verifier.cli",
        "preprocess-fusionportable",
        "--config",
        str(config_path),
        "--raw-dir",
        str(plan.raw_dir),
        "--gt-trajectory-file",
        str(plan.gt_trajectory_file),
        "--sequence-name",
        plan.sequence_name,
        "--max-gt-delta-sec",
        str(args.max_gt_delta_sec),
    ]
    _run_command(command, repo_root)


def _run_pipeline(
    plan: SequencePlan,
    args: argparse.Namespace,
    config_path: Path,
    query_limit: int,
    repo_root: Path,
) -> None:
    command = [
        args.python_bin,
        "-m",
        "robust_loop_verifier.cli",
        "run-cache",
        "--config",
        str(config_path),
        "--sequence-cache",
        str(plan.cache_dir),
        "--output-root",
        str(plan.run_root),
        "--query-limit",
        str(query_limit),
        "--backend",
        args.backend,
    ]
    _run_command(command, repo_root)


def _write_preprocess_config(
    plan: SequencePlan,
    args: argparse.Namespace,
    batch_root: Path,
) -> Path:
    path = _config_output_path(plan, batch_root, "preprocess")
    data = _base_config(args)
    data["dataset_name"] = args.dataset_name
    data["platform"] = plan.platform
    data["input_root"] = str(Path(args.loop_dataset_root))
    data["gt_root"] = str(Path(args.gt_data_root))
    data["output_root"] = str(Path(args.cache_root))
    write_yaml(path, data)
    return path


def _write_run_config(
    plan: SequencePlan,
    args: argparse.Namespace,
    batch_root: Path,
) -> Path:
    path = _config_output_path(plan, batch_root, "run")
    data = _base_config(args)
    data["dataset_name"] = args.dataset_name
    data["platform"] = plan.platform
    data["input_root"] = str(Path(args.loop_dataset_root))
    data["gt_root"] = str(Path(args.gt_data_root))
    data["output_root"] = str(batch_root)
    write_yaml(path, data)
    return path


def _config_output_path(plan: SequencePlan, batch_root: Path, kind: str) -> Path:
    config_dir = batch_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / f"{plan.platform}_{plan.sequence_name}_{kind}.yaml"


def _base_config(args: argparse.Namespace) -> dict[str, Any]:
    return dict(read_yaml(_selected_config_path(args)))


def _selected_config_path(args: argparse.Namespace) -> Path:
    if args.support_ensemble:
        return (
            Path(args.repo_root)
            / "configs"
            / "robust_loop_verifier"
            / "geode_offroad_support_ensemble.yaml"
        )
    return Path(args.config)


def _run_command(command: list[str], repo_root: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    print("[run] " + shlex.join(command))
    subprocess.run(command, cwd=repo_root, env=env, check=True)


def _validate_sequence_inputs(plan: SequencePlan) -> None:
    if not plan.raw_dir.is_dir():
        raise ValueError(f"raw directory does not exist: {plan.raw_dir}")
    if not plan.gt_trajectory_file.is_file():
        raise ValueError(f"GT trajectory file does not exist: {plan.gt_trajectory_file}")


def _print_dry_run(plan: SequencePlan, row: Mapping[str, Any]) -> None:
    print(f"[dry-run] {plan.platform}/{plan.sequence_name}")
    print(f"  raw_dir: {plan.raw_dir}")
    print(f"  gt_trajectory_file: {plan.gt_trajectory_file}")
    print(f"  cache_dir: {plan.cache_dir}")
    print(f"  run_root: {plan.run_root}")
    print(f"  preprocess_status: {row['preprocess_status']}")
    print(f"  query_limit: {row['query_limit']}")


def _summary_row(
    plan: SequencePlan,
    args: argparse.Namespace,
    run_id: str,
) -> dict[str, Any]:
    return {
        "platform": plan.platform,
        "sequence_name": plan.sequence_name,
        "gt_label_source": plan.gt_label_source,
        "raw_dir": str(plan.raw_dir),
        "gt_trajectory_file": str(plan.gt_trajectory_file),
        "cache_dir": str(plan.cache_dir),
        "run_root": str(plan.run_root),
        "run_id": run_id,
        "query_limit": None,
        "preprocess_status": "pending",
        "status": "pending",
        "error": "",
    }


def _initial_preprocess_status(plan_cache: Path, args: argparse.Namespace) -> str:
    if args.overwrite_dataset:
        return "pending"
    if _cache_ready(plan_cache):
        return "skipped_existing"
    return "pending"


def _cache_ready(cache_dir: Path) -> bool:
    required = ("manifest.json", "keyframes.jsonl", "positives.jsonl")
    return all((cache_dir / name).is_file() for name in required)


def _query_limit_for_cache(cache_dir: Path, override: int | None) -> int | None:
    if override is not None:
        return override
    manifest = cache_dir / "manifest.json"
    if not manifest.is_file():
        return None
    data = read_json(manifest)
    value = data.get("keyframe_count")
    return None if value is None else int(value)


def _flatten_metrics(plan: SequencePlan, metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, values in metrics.items():
        if not isinstance(values, Mapping):
            continue
        rows.append(
            {
                "platform": plan.platform,
                "sequence": plan.sequence_name,
                "method": method,
                "AP": float(values["AP"]),
                "MR@100P": float(values["MR@100P"]),
                "run_root": str(plan.run_root),
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
        "backend": args.backend,
        "overwrite_dataset": bool(args.overwrite_dataset),
        "support_ensemble": bool(args.support_ensemble),
        "config": str(_selected_config_path(args)),
        "max_gt_delta_sec": float(args.max_gt_delta_sec),
        "sequences": summary_rows,
    }
    write_json(batch_root / "batch_summary.json", payload)
    write_csv(
        batch_root / "batch_summary.csv",
        summary_rows,
        [
            "platform",
            "sequence_name",
            "gt_label_source",
            "preprocess_status",
            "status",
            "query_limit",
            "cache_dir",
            "run_root",
            "error",
        ],
    )
    write_markdown_table(
        batch_root / "batch_summary.md",
        summary_rows,
        columns=[
            "platform",
            "sequence_name",
            "gt_label_source",
            "preprocess_status",
            "status",
            "query_limit",
            "run_root",
            "error",
        ],
    )
    write_csv(
        batch_root / "batch_metrics.csv",
        metric_rows,
        ["platform", "sequence", "method", "AP", "MR@100P", "run_root"],
    )
    write_markdown_table(
        batch_root / "batch_metrics.md",
        metric_rows,
        columns=["platform", "sequence", "method", "AP", "MR@100P"],
    )
    average_rows = average_score_sweep_rows(metric_rows) if metric_rows else []
    write_csv(
        batch_root / "batch_metrics_average.csv",
        average_rows,
        ["method", "sequence_count", "average_AP", "average_MR@100P"],
    )
    write_markdown_table(
        batch_root / "batch_metrics_average.md",
        average_rows,
        columns=["method", "sequence_count", "average_AP", "average_MR@100P"],
    )


def _gt_filename_for_sequence(sequence_name: str) -> str:
    match = re.fullmatch(r"Offroad0*(\d+)_[A-Za-z]+", sequence_name)
    if match is None:
        raise ValueError(
            "Cannot infer GEODE GT file from sequence name "
            f"{sequence_name!r}; expected e.g. Offroad05_beta"
        )
    return f"Offroad{int(match.group(1))}.txt"


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
