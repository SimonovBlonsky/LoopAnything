#!/usr/bin/env python3
"""Evaluate score files on a sealed ROVER-aligned benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

LOOPANYTHING_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = LOOPANYTHING_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from robust_loop_verifier.rover_evaluation import evaluate_methods, write_metrics_outputs


def parse_method_specs(
    values: Sequence[str],
    benchmark_root: Path | None = None,
) -> list[tuple[str, Path]]:
    """Parse NAME=PATH entries; evaluate_methods resolves relative paths once."""
    methods: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for value in values:
        separator_count = value.count("=")
        if separator_count == 0:
            raise ValueError("method entries must use NAME=PATH")
        if separator_count != 1:
            raise ValueError("method entries must contain exactly one '='")
        name, raw_path = value.split("=", 1)
        if not name or not raw_path:
            raise ValueError("method entries must use non-empty NAME=PATH")
        if name in seen:
            raise ValueError(f"duplicate method name: {name}")
        seen.add(name)
        path = Path(raw_path)
        methods.append((name, path))
    if not methods:
        raise ValueError("at least one --method NAME=PATH entry is required")
    return methods


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_root", type=Path)
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        help="Method score file as NAME=PATH; repeat for each method.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: BENCHMARK_ROOT/metrics).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    benchmark_root = args.benchmark_root
    method_files = parse_method_specs(args.method, benchmark_root)
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else benchmark_root / "metrics"
    )
    results = evaluate_methods(benchmark_root, method_files)
    write_metrics_outputs(output_dir, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
