#!/usr/bin/env python3
"""Benchmark loop-verification efficiency on representative VPR caches.

The benchmark is online-oriented: it reports per-query and per-candidate
latency for the causal methods used in the paper main experiment.
Retrieval-only baselines are timed as separate scripts, while
residual/deformation methods share one SALAD+DA3+Sim3+PGO verifier run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


LOOPANYTHING_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = LOOPANYTHING_ROOT.parent
SRC_ROOT = LOOPANYTHING_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from robust_loop_verifier.io import (  # noqa: E402
    read_json,
    read_jsonl,
    read_yaml,
    write_json,
    write_yaml,
)
from robust_loop_verifier.score_sweep import compute_score_sweep  # noqa: E402


DEFAULT_SEQUENCE_SPECS = (
    ("handheld", "handheld_escalator00"),
    ("ugv", "ugv_parking01"),
)
DEFAULT_CACHE_ROOT = Path(
    "/data/datasets/FusionPortable/robust_loop_verifier_cache/FusionPortableV2"
)
DEFAULT_CONFIG = (
    LOOPANYTHING_ROOT
    / "configs"
    / "robust_loop_verifier"
    / "fusionportablev2_handheld.yaml"
)
DEFAULT_OUTPUT_ROOT = (
    LOOPANYTHING_ROOT / "workspace" / "efficiency_runs" / "fusionportablev2"
)

METHOD_DBOW2 = "ORB DBoW2 score only"
METHOD_NETVLAD = "NetVLAD score only"
METHOD_SALAD = "SALAD score only"
METHOD_ROVER = "ROVER deformation only"
METHOD_PGO_RESIDUAL = "PGO residual only"
METHOD_ABSOLUTE_GRAPH = "absolute_graph:def=0.5,res=0.25"
METHOD_QUERY_GATE = "query_gate_graph:def=0.5,res=0.25,margin=0"
ROBUST_METHODS = (
    METHOD_SALAD,
    METHOD_ROVER,
    METHOD_PGO_RESIDUAL,
    METHOD_ABSOLUTE_GRAPH,
    METHOD_QUERY_GATE,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root).resolve() / run_name
    sequence_specs = _parse_sequence_specs(args.sequence)
    rows: list[dict[str, Any]] = []

    for platform, sequence in sequence_specs:
        sequence_cache = Path(args.cache_root).resolve() / platform / sequence
        if not sequence_cache.is_dir():
            raise FileNotFoundError(f"sequence cache does not exist: {sequence_cache}")
        print(f"[sequence] {platform}/{sequence}")

        if not args.skip_retrieval_baselines:
            rows.append(
                _run_retrieval_baseline(
                    method=METHOD_DBOW2,
                    script=(
                        LOOPANYTHING_ROOT
                        / "baseline_scripts"
                        / "orb_dbow2_retrieval_pipeline.py"
                    ),
                    platform=platform,
                    sequence=sequence,
                    cache_root=Path(args.cache_root),
                    output_root=output_root / "retrieval_baselines" / "orb_dbow2",
                    run_name=f"{platform}_{sequence}",
                    top_k=args.top_k,
                    python_bin=args.python_bin,
                    extra_args=[],
                )
            )
            rows.append(
                _run_retrieval_baseline(
                    method=METHOD_NETVLAD,
                    script=(
                        LOOPANYTHING_ROOT
                        / "baseline_scripts"
                        / "netvlad_retrieval_pipeline.py"
                    ),
                    platform=platform,
                    sequence=sequence,
                    cache_root=Path(args.cache_root),
                    output_root=output_root / "retrieval_baselines" / "netvlad",
                    run_name=f"{platform}_{sequence}",
                    top_k=args.top_k,
                    python_bin=args.python_bin,
                    extra_args=["--device", args.device],
                )
            )

        if not args.skip_robust_verifier:
            robust_run_root = output_root / "robust_verifier" / platform / sequence
            config_path = _write_runtime_config(
                base_config=Path(args.config),
                output_root=output_root,
                platform=platform,
                sequence=sequence,
                top_k=args.top_k,
            )
            command = [
                args.python_bin,
                "-m",
                "robust_loop_verifier.cli",
                "run-cache",
                "--config",
                str(config_path),
                "--sequence-cache",
                str(sequence_cache),
                "--output-root",
                str(robust_run_root),
                "--query-limit",
                str(args.query_limit or _manifest_keyframe_count(sequence_cache)),
                "--backend",
                args.backend,
                "--collect-timing",
            ]
            elapsed = _run_command_timed(command, cwd=LOOPANYTHING_ROOT)
            rows.extend(
                build_robust_method_rows(
                    platform=platform,
                    sequence=sequence,
                    run_root=robust_run_root,
                    elapsed_wall_sec=elapsed,
                )
            )

        write_efficiency_outputs(output_root, rows)

    write_efficiency_outputs(output_root, rows)
    print(f"efficiency_summary_json={output_root / 'efficiency_summary.json'}")
    print(f"efficiency_summary_csv={output_root / 'efficiency_summary.csv'}")
    print(f"efficiency_summary_md={output_root / 'efficiency_summary.md'}")
    return 0


def build_robust_method_rows(
    *,
    platform: str,
    sequence: str,
    run_root: Path,
    elapsed_wall_sec: float,
) -> list[dict[str, Any]]:
    timing = read_json(Path(run_root) / "efficiency_timing.json")
    records = list(read_jsonl(Path(run_root) / "candidate_records.jsonl"))
    score_start = time.perf_counter()
    score_sweep = compute_score_sweep(records)
    score_wall_sec = max(0.0, time.perf_counter() - score_start)
    scores_by_name = {row["name"]: row for row in score_sweep["scores"]}

    rows = []
    for method in ROBUST_METHODS:
        if method not in scores_by_name:
            continue
        metrics = scores_by_name[method]
        rows.append(
            _method_runtime_row(
                platform=platform,
                sequence=sequence,
                method=method,
                metrics=metrics,
                elapsed_wall_sec=elapsed_wall_sec,
                timing=timing,
                runtime_group="SALAD+DA3+Sim3+PGO",
                additional_score_wall_sec=(
                    score_wall_sec
                    if method in {METHOD_ABSOLUTE_GRAPH, METHOD_QUERY_GATE}
                    else 0.0
                ),
            )
        )
    return rows


def write_efficiency_outputs(output_root: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    averages = _average_rows(rows)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rows": list(rows),
        "averages": averages,
    }
    write_json(output_root / "efficiency_summary.json", payload)
    _write_csv(output_root / "efficiency_summary.csv", list(rows))
    _write_csv(output_root / "efficiency_average.csv", averages)
    _write_markdown(output_root / "efficiency_summary.md", list(rows), averages)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--sequence",
        action="append",
        default=None,
        help=(
            "Representative sequence as platform/name. Repeat to add sequences. "
            "Defaults to handheld/handheld_escalator00 and ugv/ugv_parking01."
        ),
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--query-limit", type=int, default=None)
    parser.add_argument("--backend", default=os.environ.get("BACKEND", "real"))
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", sys.executable))
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument("--skip-retrieval-baselines", action="store_true")
    parser.add_argument("--skip-robust-verifier", action="store_true")
    return parser.parse_args(argv)


def _parse_sequence_specs(values: Sequence[str] | None) -> tuple[tuple[str, str], ...]:
    if not values:
        return DEFAULT_SEQUENCE_SPECS
    specs: list[tuple[str, str]] = []
    for value in values:
        if "/" not in value:
            raise ValueError(f"sequence must be formatted as platform/name, got: {value}")
        platform, sequence = value.split("/", 1)
        if not platform or not sequence:
            raise ValueError(f"sequence must be formatted as platform/name, got: {value}")
        specs.append((platform, sequence))
    return tuple(specs)


def _run_retrieval_baseline(
    *,
    method: str,
    script: Path,
    platform: str,
    sequence: str,
    cache_root: Path,
    output_root: Path,
    run_name: str,
    top_k: int,
    python_bin: str,
    extra_args: Sequence[str],
) -> dict[str, Any]:
    command = [
        python_bin,
        str(script),
        "--dataset",
        "fusionportablev2",
        "--cache-root",
        str(cache_root),
        "--platform",
        platform,
        "--sequence",
        sequence,
        "--top-k",
        str(top_k),
        "--output-root",
        str(output_root),
        "--run-name",
        run_name,
        *extra_args,
    ]
    elapsed = _run_command_timed(command, cwd=LOOPANYTHING_ROOT)
    metrics_path = output_root / "fusionportablev2" / run_name / "metrics.json"
    summary = read_json(metrics_path)
    if not summary["sequences"]:
        raise ValueError(f"{method} produced no sequence metrics: {metrics_path}")
    row = summary["sequences"][0]
    query_count = int(row.get("query_count", 0))
    candidate_count = int(row.get("candidate_count", 0))
    return {
        "platform": platform,
        "sequence": sequence,
        "method": method,
        "AP": float(row["metrics"]["AP"]),
        "MR@100P": float(row["metrics"]["MR@100P"]),
        "elapsed_wall_sec": float(elapsed),
        "query_count": query_count,
        "candidate_count": candidate_count,
        "mean_query_sec": _safe_divide(elapsed, query_count),
        "mean_candidate_sec": _safe_divide(elapsed, candidate_count),
        "p95_candidate_sec": 0.0,
        "additional_score_wall_sec": 0.0,
        "runtime_group": "retrieval-only",
        "descriptor_compute_sec": float(elapsed),
        "retrieval_search_sec": 0.0,
        "support_selection_sec": 0.0,
        "da3_triplet_sec": 0.0,
        "sim3_alignment_sec": 0.0,
        "pgo_sec": 0.0,
        "metrics_sec": 0.0,
    }


def _method_runtime_row(
    *,
    platform: str,
    sequence: str,
    method: str,
    metrics: Mapping[str, Any],
    elapsed_wall_sec: float,
    timing: Mapping[str, Any],
    runtime_group: str,
    additional_score_wall_sec: float,
) -> dict[str, Any]:
    query_count = int(timing.get("query_count", 0))
    candidate_count = int(timing.get("candidate_count", 0))
    component_totals = dict(timing.get("component_totals_sec", {}))
    candidate_total = dict(timing.get("per_candidate_sec", {}).get("total", {}))
    return {
        "platform": platform,
        "sequence": sequence,
        "method": method,
        "AP": float(metrics["AP"]),
        "MR@100P": float(metrics["MR@100P"]),
        "elapsed_wall_sec": float(elapsed_wall_sec),
        "query_count": query_count,
        "candidate_count": candidate_count,
        "mean_query_sec": _safe_divide(elapsed_wall_sec, query_count),
        "mean_candidate_sec": float(candidate_total.get("mean", 0.0)),
        "p95_candidate_sec": float(candidate_total.get("p95", 0.0)),
        "additional_score_wall_sec": float(additional_score_wall_sec),
        "runtime_group": runtime_group,
        "descriptor_compute_sec": float(component_totals.get("descriptor_compute", 0.0)),
        "retrieval_search_sec": float(component_totals.get("retrieval_search", 0.0)),
        "support_selection_sec": float(component_totals.get("support_selection", 0.0)),
        "da3_triplet_sec": float(component_totals.get("da3_triplet", 0.0)),
        "sim3_alignment_sec": float(component_totals.get("sim3_alignment", 0.0)),
        "pgo_sec": float(component_totals.get("pgo", 0.0)),
        "metrics_sec": float(component_totals.get("metrics", 0.0)),
    }


def _write_runtime_config(
    *,
    base_config: Path,
    output_root: Path,
    platform: str,
    sequence: str,
    top_k: int,
) -> Path:
    config = dict(read_yaml(base_config))
    config["platform"] = platform
    config["retrieval_top_k_main"] = int(top_k)
    support_ensemble = dict(config.get("support_ensemble", {}))
    support_ensemble["enabled"] = False
    config["support_ensemble"] = support_ensemble
    config_dir = output_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{platform}_{sequence}_runtime.yaml"
    write_yaml(path, config)
    return path


def _manifest_keyframe_count(sequence_cache: Path) -> int:
    manifest = read_json(Path(sequence_cache) / "manifest.json")
    return int(manifest["keyframe_count"])


def _run_command_timed(command: Sequence[str], cwd: Path) -> float:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = "src" if not existing_pythonpath else f"src:{existing_pythonpath}"
    print("[run] " + shlex.join([str(part) for part in command]))
    start = time.perf_counter()
    subprocess.run(
        [str(part) for part in command],
        cwd=Path(cwd),
        env=env,
        check=True,
    )
    return max(0.0, time.perf_counter() - start)


def _average_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["method"]), []).append(row)
    averages = []
    for method, method_rows in sorted(groups.items()):
        averages.append(
            {
                "method": method,
                "sequence_count": len(method_rows),
                "average_AP": _mean(row["AP"] for row in method_rows),
                "average_MR@100P": _mean(row["MR@100P"] for row in method_rows),
                "average_elapsed_wall_sec": _mean(
                    row["elapsed_wall_sec"] for row in method_rows
                ),
                "average_mean_query_sec": _mean(
                    row["mean_query_sec"] for row in method_rows
                ),
                "average_mean_candidate_sec": _mean(
                    row["mean_candidate_sec"] for row in method_rows
                ),
                "average_p95_candidate_sec": _mean(
                    row["p95_candidate_sec"] for row in method_rows
                ),
            }
        )
    return averages


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "platform",
        "sequence",
        "method",
        "AP",
        "MR@100P",
        "elapsed_wall_sec",
        "query_count",
        "candidate_count",
        "mean_query_sec",
        "mean_candidate_sec",
        "p95_candidate_sec",
        "additional_score_wall_sec",
        "runtime_group",
        "descriptor_compute_sec",
        "retrieval_search_sec",
        "support_selection_sec",
        "da3_triplet_sec",
        "sim3_alignment_sec",
        "pgo_sec",
        "metrics_sec",
        "sequence_count",
        "average_AP",
        "average_MR@100P",
        "average_elapsed_wall_sec",
        "average_mean_query_sec",
        "average_mean_candidate_sec",
        "average_p95_candidate_sec",
    ]
    used_columns = [
        column for column in columns if any(column in row for row in rows)
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=used_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in used_columns})


def _write_markdown(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    averages: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# Loop Verifier Efficiency",
        "",
        "| platform | sequence | method | AP | MR@100P | wall s | query ms | cand ms | "
        "p95 cand ms | runtime group |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {} | {} | {} | {:.4f} | {:.4f} | {:.3f} | {:.2f} | {:.2f} | {:.2f} | {} |".format(
                row["platform"],
                row["sequence"],
                row["method"],
                row["AP"],
                row["MR@100P"],
                row["elapsed_wall_sec"],
                1000.0 * row["mean_query_sec"],
                1000.0 * row["mean_candidate_sec"],
                1000.0 * row["p95_candidate_sec"],
                row["runtime_group"],
            )
        )
    lines.extend(
        [
            "",
            "## Average",
            "",
            "| method | sequences | AP | MR@100P | wall s | query ms | cand ms | p95 cand ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in averages:
        lines.append(
            "| {} | {} | {:.4f} | {:.4f} | {:.3f} | {:.2f} | {:.2f} | {:.2f} |".format(
                row["method"],
                row["sequence_count"],
                row["average_AP"],
                row["average_MR@100P"],
                row["average_elapsed_wall_sec"],
                1000.0 * row["average_mean_query_sec"],
                1000.0 * row["average_mean_candidate_sec"],
                1000.0 * row["average_p95_candidate_sec"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mean(values: Sequence[float] | Any) -> float:
    materialized = [float(value) for value in values]
    if not materialized:
        return 0.0
    return sum(materialized) / len(materialized)


def _safe_divide(numerator: float, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


if __name__ == "__main__":
    raise SystemExit(main())
