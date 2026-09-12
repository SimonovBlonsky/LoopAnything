#!/usr/bin/env python3
"""Freeze the ROVER-aligned DBoW2 candidate benchmark."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Mapping, Sequence

LOOPANYTHING_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = LOOPANYTHING_ROOT / "src"
for import_path in (LOOPANYTHING_ROOT, SRC_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from baseline_scripts.orb_dbow2_retrieval_pipeline import OrbDbow2RetrievalBackend
from robust_loop_verifier.io import read_yaml, write_json, write_jsonl
from robust_loop_verifier.rover_benchmark import (
    BenchmarkPair,
    BenchmarkSequence,
    build_pairs_for_sequence,
    compute_recent_exclusion,
    read_sequence_keyframes,
    sample_eligible_queries,
    sha256_file,
    validate_pair_manifest,
    write_frozen_benchmark,
)


def validate_benchmark_config(config: Mapping[str, object]) -> list[BenchmarkSequence]:
    rows = config.get("sequences")
    if not isinstance(rows, list) or not rows:
        raise ValueError("sequences must be a non-empty list")
    if len(rows) != 10:
        raise ValueError("benchmark config must contain exactly 10 sequences")

    sequences: list[BenchmarkSequence] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("sequence rows must be mappings")
        sequences.append(
            BenchmarkSequence(
                dataset=str(row["dataset"]),
                platform=str(row["platform"]),
                sequence=str(row["sequence"]),
                cache=Path(str(row["cache"])),
            )
        )
    return sequences


def refuse_non_empty_output_root(output_root: Path) -> None:
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty output root: {output_root}")


def build_benchmark_manifest(
    *,
    config: Mapping[str, object],
    sequences: Sequence[BenchmarkSequence],
    helper_fingerprints: Mapping[str, str],
    recent_exclusion_by_sequence: Mapping[tuple[str, str, str], int],
    eligible_query_counts: Mapping[tuple[str, str, str], int],
    selected_query_counts: Mapping[tuple[str, str, str], int],
    pair_counts: Mapping[tuple[str, str, str], int],
    pair_manifest_sha256: str,
) -> dict[str, object]:
    sequence_rows = []
    cache_roots = []
    for sequence in sequences:
        key = (sequence.dataset, sequence.platform, sequence.sequence)
        cache_root = str(Path(sequence.cache).resolve())
        cache_roots.append(cache_root)
        sequence_rows.append(
            {
                "dataset": sequence.dataset,
                "platform": sequence.platform,
                "sequence": sequence.sequence,
                "cache": cache_root,
                "recent_exclusion_keyframes": int(recent_exclusion_by_sequence[key]),
                "eligible_query_count": int(eligible_query_counts[key]),
                "selected_query_count": int(selected_query_counts[key]),
                "pair_count": int(pair_counts[key]),
            }
        )

    return {
        "benchmark_version": config.get("benchmark_version", "benchmark_v1"),
        "sequence_count": len(sequences),
        "sequences": sequence_rows,
        "sequence_order": [
            f"{sequence.dataset}/{sequence.platform}/{sequence.sequence}"
            for sequence in sequences
        ],
        "cache_roots": cache_roots,
        "query_limit_per_sequence": int(config.get("query_limit_per_sequence", 40)),
        "retrieval_top_k": int(config.get("retrieval_top_k", 10)),
        "annotation_shuffle_seed": int(config.get("annotation_shuffle_seed", 20260610)),
        "netvlad_root": str(config.get("netvlad_root", "")),
        "verifier_configs": dict(config.get("verifier_configs", {})),
        "recent_exclusion_formula": "min(30, max(5, floor(0.08 * keyframe_count + 0.5)))",
        "pair_manifest_sha256": pair_manifest_sha256,
        "vocabulary_path": helper_fingerprints["vocabulary_path"],
        "vocabulary_sha256": helper_fingerprints["vocabulary_sha256"],
        "orb_slam3_git_commit": helper_fingerprints["orb_slam3_git_commit"],
        "helper_source_path": helper_fingerprints["helper_source_path"],
        "helper_source_sha256": helper_fingerprints["helper_source_sha256"],
        "helper_binary_path": helper_fingerprints["helper_binary_path"],
        "helper_binary_sha256": helper_fingerprints["helper_binary_sha256"],
        "dbow2_shared_library_path": helper_fingerprints["dbow2_shared_library_path"],
        "dbow2_shared_library_sha256": helper_fingerprints["dbow2_shared_library_sha256"],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rebuild-dbow2-helper", action="store_true")
    return parser


def _sequence_key(sequence: BenchmarkSequence) -> tuple[str, str, str]:
    return sequence.dataset, sequence.platform, sequence.sequence


def _helper_build_dir(config: Mapping[str, object], config_path: Path) -> Path:
    helper_build_dir = Path(str(config["helper_build_dir"]))
    if helper_build_dir.is_absolute():
        return helper_build_dir
    return config_path.resolve().parents[2] / helper_build_dir


def _reject_helper_build_dir_inside_output_root(
    helper_build_dir: Path,
    output_root: Path,
) -> None:
    helper_build_dir = Path(helper_build_dir).resolve()
    output_root = Path(output_root).resolve()
    if helper_build_dir == output_root:
        raise ValueError("helper_build_dir must not equal output_root")
    try:
        helper_build_dir.relative_to(output_root)
    except ValueError:
        return
    raise ValueError("helper_build_dir must not be inside output_root")


def _write_score_files(
    output_root: Path,
    pairs: Sequence[BenchmarkPair],
    helper_fingerprints: Mapping[str, str],
    *,
    pair_manifest_sha256: str,
) -> None:
    score_path = output_root / "scores" / "dbow2.jsonl"
    score_rows = [
        {"pair_id": pair.pair_id, "score": pair.dbow2_score, "status": "ok"} for pair in pairs
    ]
    write_jsonl(score_path, score_rows)
    write_json(
        output_root / "scores" / "dbow2.manifest.json",
        {
            "method": "ORB DBoW2",
            "evaluation_name": "DBoW2",
            "score_file": "scores/dbow2.jsonl",
            "pair_manifest_sha256": pair_manifest_sha256,
            "score_file_sha256": sha256_file(score_path),
            "source_commit": helper_fingerprints["orb_slam3_git_commit"],
            "command": "build_rover_aligned_benchmark.py",
            "helper_fingerprints": dict(helper_fingerprints),
        },
    )


def _write_pair_manifest_for_hash(output_root: Path, pairs: Sequence[BenchmarkPair]) -> str:
    write_jsonl(output_root / "benchmark_pairs.jsonl", (asdict(pair) for pair in pairs))
    return sha256_file(output_root / "benchmark_pairs.jsonl")


def build_rover_aligned_benchmark(
    *,
    config_path: Path,
    output_root: Path,
    rebuild_dbow2_helper: bool,
) -> dict[str, object]:
    config = read_yaml(config_path)
    sequences = validate_benchmark_config(config)
    refuse_non_empty_output_root(output_root)
    top_k = int(config.get("retrieval_top_k", 10))
    if top_k != 10:
        raise ValueError("retrieval_top_k must be 10 for benchmark_v1")

    orb = config.get("orb", {})
    if not isinstance(orb, Mapping):
        raise ValueError("orb config must be a mapping")
    helper_build_dir = _helper_build_dir(config, config_path)
    _reject_helper_build_dir_inside_output_root(helper_build_dir, output_root)
    backend = OrbDbow2RetrievalBackend(
        orb_slam3_root=Path(str(config["orb_slam3_root"])),
        vocabulary_path=Path(str(config["vocabulary"])),
        helper_build_dir=helper_build_dir,
        rebuild_helper=rebuild_dbow2_helper,
        nfeatures=int(orb["nfeatures"]),
        scale_factor=float(orb["scale_factor"]),
        nlevels=int(orb["nlevels"]),
        ini_fast=int(orb["ini_fast"]),
        min_fast=int(orb["min_fast"]),
    )
    helper_fingerprints = backend.helper_fingerprints()

    output_root = Path(output_root)
    all_pairs: list[BenchmarkPair] = []
    recent_exclusion_by_sequence: dict[tuple[str, str, str], int] = {}
    eligible_query_counts: dict[tuple[str, str, str], int] = {}
    selected_query_counts: dict[tuple[str, str, str], int] = {}
    pair_counts: dict[tuple[str, str, str], int] = {}
    query_limit = int(config.get("query_limit_per_sequence", 40))

    for sequence in sequences:
        keyframes = read_sequence_keyframes(sequence)
        recent_exclusion = compute_recent_exclusion(len(keyframes))
        retrieval_records = backend.retrieve(
            keyframes,
            top_k=top_k,
            recent_exclusion_keyframes=recent_exclusion,
        )
        eligible_queries = [
            query_idx
            for query_idx, record in sorted(retrieval_records.items())
            if len(record.candidates) == top_k
        ]
        selected_queries = sample_eligible_queries(eligible_queries, limit=query_limit)
        if not selected_queries:
            raise ValueError(
                "no selected queries for configured sequence "
                f"{sequence.dataset}/{sequence.platform}/{sequence.sequence}"
            )
        sequence_pairs = build_pairs_for_sequence(
            sequence,
            retrieval_records,
            selected_queries,
            recent_exclusion,
        )
        key = _sequence_key(sequence)
        recent_exclusion_by_sequence[key] = recent_exclusion
        eligible_query_counts[key] = len(eligible_queries)
        selected_query_counts[key] = len(selected_queries)
        pair_counts[key] = len(sequence_pairs)
        all_pairs.extend(sequence_pairs)

    validate_pair_manifest(all_pairs, recent_exclusion_by_sequence)
    pair_manifest_sha256 = _write_pair_manifest_for_hash(output_root, all_pairs)
    manifest = build_benchmark_manifest(
        config=config,
        sequences=sequences,
        helper_fingerprints=helper_fingerprints,
        recent_exclusion_by_sequence=recent_exclusion_by_sequence,
        eligible_query_counts=eligible_query_counts,
        selected_query_counts=selected_query_counts,
        pair_counts=pair_counts,
        pair_manifest_sha256=pair_manifest_sha256,
    )
    write_frozen_benchmark(output_root, all_pairs, manifest)
    _write_score_files(
        output_root,
        all_pairs,
        helper_fingerprints,
        pair_manifest_sha256=pair_manifest_sha256,
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = build_rover_aligned_benchmark(
        config_path=args.config,
        output_root=args.output_root,
        rebuild_dbow2_helper=args.rebuild_dbow2_helper,
    )
    print(json.dumps({"manifest": manifest["benchmark_version"], "output_root": str(args.output_root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
