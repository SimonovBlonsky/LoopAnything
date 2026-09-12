#!/usr/bin/env python3
"""Evaluate the AsterSLAM runtime hard gate on the frozen loop benchmark.

Each frozen candidate is evaluated independently.  The continuous ranking score
is the negative maximum normalized gate violation, so the runtime decision is
exactly ``score >= -1`` whenever all required measurements are valid.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


LOOPANYTHING_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = LOOPANYTHING_ROOT / "src"
if str(LOOPANYTHING_ROOT) not in sys.path:
    sys.path.insert(0, str(LOOPANYTHING_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from robust_loop_verification_scripts.retest_aster_runtime_loops_offline import (  # noqa: E402
    run_runtime_style_safety_pgo,
    select_local_nodes,
)
from robust_loop_verifier.pgo import trajectory_deformation_rmse  # noqa: E402


TABLE_SEQUENCE_ORDER = (
    "FusionPortableV2/handheld/handheld_escalator00",
    "FusionPortableV2/handheld/handheld_room00",
    "FusionPortableV2/ugv/ugv_campus01",
    "FusionPortableV2/ugv/ugv_parking01",
    "GEODE/Offroad/Offroad02_beta",
    "GEODE/Offroad/Offroad05_beta",
    "NTU-VIRAL/NTU-VIRAL/eee_01",
    "NTU-VIRAL/NTU-VIRAL/nya_02",
)

PARAMETER_KEY_BY_SEQUENCE = {
    "FusionPortableV2/handheld/handheld_escalator00": "esc0",
    "FusionPortableV2/handheld/handheld_room00": "room0",
    "FusionPortableV2/ugv/ugv_campus01": "camp1",
    "FusionPortableV2/ugv/ugv_parking01": "park1",
    "GEODE/Offroad/Offroad02_beta": "off2",
    "GEODE/Offroad/Offroad05_beta": "off5",
    "NTU-VIRAL/NTU-VIRAL/eee_01": "eee1",
    "NTU-VIRAL/NTU-VIRAL/nya_02": "nya2",
}

FIXED_THRESHOLDS = {
    "min_support_baseline_m": 0.3,
    "min_sim3_scale": 0.05,
    "max_sim3_scale": 20.0,
    "max_translation_norm_m": 80.0,
    "max_pgo_error_per_factor": 1.0,
    "max_odom_strain_chi2": 50.0,
    "max_loop_chi2": 25.0,
    "max_nodes": 80,
    "min_nodes": 4,
    "prior_sigmas": (1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-2),
    "odom_sigmas": (0.05, 0.05, 0.05, 0.10, 0.10, 0.10),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_root", type=Path)
    parser.add_argument("--table3-params", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-per-sequence", type=int, default=0)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_key(row: Mapping[str, Any]) -> str:
    return f"{row['dataset']}/{row['platform']}/{row['sequence']}"


def load_odom_cache(cache_root: Path) -> tuple[list[int], dict[int, np.ndarray]]:
    rows = read_jsonl(cache_root / "keyframes.jsonl")
    order: list[int] = []
    poses: dict[int, np.ndarray] = {}
    for row in rows:
        idx = int(row["idx"])
        pose = np.asarray(row["odom_pose"], dtype=np.float64).reshape(4, 4)
        if idx in poses:
            raise ValueError(f"duplicate keyframe {idx} in {cache_root}")
        if not np.all(np.isfinite(pose)):
            raise ValueError(f"non-finite odometry pose {idx} in {cache_root}")
        order.append(idx)
        poses[idx] = pose
    return order, poses


def threshold_config(
    sequence: str,
    table3_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    parameter_key = PARAMETER_KEY_BY_SEQUENCE[sequence]
    sequence_parameters = table3_parameters["sequences"][parameter_key]
    config = dict(FIXED_THRESHOLDS)
    config.update(
        {
            "parameter_key": parameter_key,
            "salad_min_score": float(sequence_parameters["salad_min_score"]),
            "max_sim3_alignment_rmse_m": float(
                sequence_parameters["da3_max_prior_alignment_rmse"]
            ),
            "max_sim3_direction_error_deg": float(
                sequence_parameters["da3_max_prior_direction_error_deg"]
            ),
            "max_deformation_rmse_m": float(
                sequence_parameters["da3_safety_gate_max_deformation_rmse"]
            ),
            "loop_rotation_sigma": float(sequence_parameters["da3_rotation_noise_sigma"]),
            "loop_translation_sigma": float(
                sequence_parameters["da3_translation_noise_sigma"]
            ),
        }
    )
    return config


def positive_ratio(threshold: float, value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return threshold / value


def upper_ratio(value: float, threshold: float, name: str) -> float:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value / threshold


def score_candidate(
    pair: Mapping[str, Any],
    geometry: Mapping[str, Any],
    salad_score: float,
    cache_order: list[int],
    odom_by_idx: dict[int, np.ndarray],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    pair_id = str(pair["pair_id"])
    base: dict[str, Any] = {
        "pair_id": pair_id,
        "sequence": sequence_key(pair),
        "query_idx": int(pair["query_idx"]),
        "candidate_idx": int(pair["candidate_idx"]),
        "rank": int(pair["rank"]),
        "status": "failed",
        "score": None,
        "accepted_at_runtime_threshold": False,
    }
    try:
        if geometry.get("factor_status") != "ok" or not geometry.get("sim3_valid"):
            raise ValueError("invalid_geometry_factor")
        loop_factor = np.asarray(geometry["estimated_relative_pose"], dtype=np.float64)
        if loop_factor.shape != (4, 4) or not np.all(np.isfinite(loop_factor)):
            raise ValueError("invalid_estimated_relative_pose")

        query_idx = int(pair["query_idx"])
        candidate_idx = int(pair["candidate_idx"])
        nodes = select_local_nodes(
            candidate_idx,
            query_idx,
            cache_order=cache_order,
            max_nodes=int(config["max_nodes"]),
        )
        if len(nodes) < int(config["min_nodes"]):
            raise ValueError("safety_gate_too_few_nodes")
        if nodes[0] != min(query_idx, candidate_idx) or nodes[-1] != max(
            query_idx, candidate_idx
        ):
            raise ValueError("safety_gate_missing_endpoint")

        loop_sigmas = (
            float(config["loop_rotation_sigma"]),
            float(config["loop_rotation_sigma"]),
            float(config["loop_rotation_sigma"]),
            float(config["loop_translation_sigma"]),
            float(config["loop_translation_sigma"]),
            float(config["loop_translation_sigma"]),
        )
        pgo = run_runtime_style_safety_pgo(
            nodes=nodes,
            odom_by_idx=odom_by_idx,
            query_idx=query_idx,
            candidate_idx=candidate_idx,
            loop_factor=loop_factor,
            accepted_history=[],
            include_accepted_loops=False,
            prior_sigmas=tuple(config["prior_sigmas"]),
            odom_sigmas=tuple(config["odom_sigmas"]),
            loop_sigmas=loop_sigmas,
        )
        if not pgo.converged:
            raise ValueError(pgo.failure_reason or "safety_gate_pgo_failed")

        deformation = float(
            trajectory_deformation_rmse(
                [odom_by_idx[idx] for idx in nodes],
                pgo.optimized_poses,
            )
        )
        if pgo.factor_count <= 0:
            raise ValueError("invalid_pgo_factor_count")
        error_per_factor = float(pgo.error_after / pgo.factor_count)
        support_baseline = float(geometry["support_baseline_m"])
        sim3_scale = float(geometry["sim3_scale"])
        sim3_rmse = float(geometry["sim3_support_alignment_residual_m"])
        sim3_direction = float(geometry["sim3_direction_error_deg"])
        translation_norm = float(geometry["estimated_translation_norm_m"])

        violations = {
            "salad_min_score": positive_ratio(
                float(config["salad_min_score"]), salad_score, "salad_score"
            ),
            "support_min_baseline": positive_ratio(
                float(config["min_support_baseline_m"]),
                support_baseline,
                "support_baseline_m",
            ),
            "sim3_min_scale": positive_ratio(
                float(config["min_sim3_scale"]), sim3_scale, "sim3_scale"
            ),
            "sim3_max_scale": upper_ratio(
                sim3_scale, float(config["max_sim3_scale"]), "sim3_scale"
            ),
            "sim3_alignment_rmse": upper_ratio(
                sim3_rmse,
                float(config["max_sim3_alignment_rmse_m"]),
                "sim3_alignment_rmse_m",
            ),
            "sim3_direction_error": upper_ratio(
                sim3_direction,
                float(config["max_sim3_direction_error_deg"]),
                "sim3_direction_error_deg",
            ),
            "translation_norm": upper_ratio(
                translation_norm,
                float(config["max_translation_norm_m"]),
                "translation_norm_m",
            ),
            "trajectory_deformation": upper_ratio(
                deformation,
                float(config["max_deformation_rmse_m"]),
                "trajectory_deformation_rmse_m",
            ),
            "pgo_error_per_factor": upper_ratio(
                error_per_factor,
                float(config["max_pgo_error_per_factor"]),
                "pgo_error_per_factor",
            ),
            "odom_strain_chi2": upper_ratio(
                float(pgo.odom_strain_chi2_after),
                float(config["max_odom_strain_chi2"]),
                "odom_strain_chi2",
            ),
            "loop_chi2": upper_ratio(
                float(pgo.loop_chi2_after),
                float(config["max_loop_chi2"]),
                "loop_chi2",
            ),
        }
        max_gate, max_violation = max(violations.items(), key=lambda item: item[1])
        score = -float(max_violation)
        base.update(
            {
                "status": "ok",
                "score": score,
                "accepted_at_runtime_threshold": bool(max_violation <= 1.0),
                "max_violation": float(max_violation),
                "max_violation_gate": max_gate,
                "violations": violations,
                "metrics": {
                    "salad_score": float(salad_score),
                    "support_baseline_m": support_baseline,
                    "sim3_scale": sim3_scale,
                    "sim3_alignment_rmse_m": sim3_rmse,
                    "sim3_direction_error_deg": sim3_direction,
                    "translation_norm_m": translation_norm,
                    "safety_node_count": len(nodes),
                    "safety_factor_count": int(pgo.factor_count),
                    "safety_pgo_error_before": float(pgo.error_before),
                    "safety_pgo_error_after": float(pgo.error_after),
                    "safety_pgo_error_per_factor_after": error_per_factor,
                    "safety_trajectory_deformation_rmse_m": deformation,
                    "safety_odom_strain_chi2_after": float(pgo.odom_strain_chi2_after),
                    "safety_loop_chi2_after": float(pgo.loop_chi2_after),
                },
            }
        )
    except (KeyError, TypeError, ValueError) as exc:
        base["failure_reason"] = str(exc)
    return base


def gv_metrics(labels: list[int], scores: list[float]) -> tuple[float, float]:
    ap = float(average_precision_score(labels, scores))
    precision, recall, _ = precision_recall_curve(labels, scores)
    mr100p = float(np.max(recall[precision == 1.0]))
    return ap, mr100p


def evaluate_rows(
    rows: list[dict[str, Any]],
    labels: Mapping[str, int],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    failure_sentinel = -math.pi / 2.0
    for sequence in TABLE_SEQUENCE_ORDER:
        sequence_rows = [row for row in rows if row["sequence"] == sequence]
        sequence_labels = [int(labels[row["pair_id"]]) for row in sequence_rows]
        successful_scores = [
            float(row["score"]) for row in sequence_rows if row["status"] == "ok"
        ]
        if successful_scores:
            minimum_success = min(successful_scores)
            failed_score = math.nextafter(minimum_success, -math.inf)
        else:
            failed_score = failure_sentinel
        sequence_scores = [
            float(row["score"]) if row["status"] == "ok" else failed_score
            for row in sequence_rows
        ]
        ap, mr100p = gv_metrics(sequence_labels, sequence_scores)
        positive_count = int(sum(sequence_labels))
        tp100p = int(round(mr100p * positive_count))
        accepted_rows = [row for row in sequence_rows if row["accepted_at_runtime_threshold"]]
        results[sequence] = {
            "candidate_count": len(sequence_rows),
            "positive_count": positive_count,
            "successful_score_count": len(successful_scores),
            "accepted_at_runtime_threshold_count": len(accepted_rows),
            "accepted_positive_count": sum(labels[row["pair_id"]] for row in accepted_rows),
            "AP": ap,
            "MR@100P": mr100p,
            "TP@100P": tp100p,
        }
    return {
        "sequence_order": list(TABLE_SEQUENCE_ORDER),
        "sequences": results,
        "macro_average": {
            "AP": float(np.mean([results[key]["AP"] for key in TABLE_SEQUENCE_ORDER])),
            "MR@100P": float(
                np.mean([results[key]["MR@100P"] for key in TABLE_SEQUENCE_ORDER])
            ),
        },
        "total": {
            "positive_count": sum(results[key]["positive_count"] for key in TABLE_SEQUENCE_ORDER),
            "TP@100P": sum(results[key]["TP@100P"] for key in TABLE_SEQUENCE_ORDER),
        },
    }


def write_metrics_csv(path: Path, metrics: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sequence",
                "candidate_count",
                "positive_count",
                "successful_score_count",
                "accepted_at_runtime_threshold_count",
                "accepted_positive_count",
                "AP",
                "MR@100P",
                "TP@100P",
            ]
        )
        for sequence in metrics["sequence_order"]:
            row = metrics["sequences"][sequence]
            writer.writerow(
                [
                    sequence,
                    row["candidate_count"],
                    row["positive_count"],
                    row["successful_score_count"],
                    row["accepted_at_runtime_threshold_count"],
                    row["accepted_positive_count"],
                    f"{row['AP']:.17g}",
                    f"{row['MR@100P']:.17g}",
                    row["TP@100P"],
                ]
            )


def main() -> int:
    args = parse_args()
    benchmark_root = args.benchmark_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = read_jsonl(benchmark_root / "benchmark_pairs.jsonl")
    geometry_rows = read_jsonl(benchmark_root / "geometry_predictions.jsonl")
    salad_rows = read_jsonl(benchmark_root / "scores" / "salad.jsonl")
    annotation_rows = read_jsonl(benchmark_root / "annotations.jsonl")
    benchmark_manifest = json.loads((benchmark_root / "manifest.json").read_text())
    table3_parameters = json.loads(args.table3_params.read_text())

    geometry_by_id = {row["pair_id"]: row for row in geometry_rows}
    salad_by_id = {row["pair_id"]: row for row in salad_rows}
    labels = {row["pair_id"]: int(row["label"]) for row in annotation_rows}
    cache_by_sequence = {
        f"{row['dataset']}/{row['platform']}/{row['sequence']}": Path(row["cache"])
        for row in benchmark_manifest["sequences"]
    }

    odometry: dict[str, tuple[list[int], dict[int, np.ndarray]]] = {}
    configs: dict[str, dict[str, Any]] = {}
    for sequence in TABLE_SEQUENCE_ORDER:
        odometry[sequence] = load_odom_cache(cache_by_sequence[sequence])
        configs[sequence] = threshold_config(sequence, table3_parameters)

    selected_counts: Counter[str] = Counter()
    score_rows: list[dict[str, Any]] = []
    selected_pairs = [pair for pair in pairs if sequence_key(pair) in TABLE_SEQUENCE_ORDER]
    for index, pair in enumerate(selected_pairs, start=1):
        sequence = sequence_key(pair)
        if args.limit_per_sequence and selected_counts[sequence] >= args.limit_per_sequence:
            continue
        selected_counts[sequence] += 1
        pair_id = pair["pair_id"]
        salad_row = salad_by_id[pair_id]
        salad_score = (
            float(salad_row["score"])
            if salad_row.get("status") == "ok" and salad_row.get("score") is not None
            else float("nan")
        )
        cache_order, odom_by_idx = odometry[sequence]
        score_rows.append(
            score_candidate(
                pair,
                geometry_by_id[pair_id],
                salad_score,
                cache_order,
                odom_by_idx,
                configs[sequence],
            )
        )
        if index % 100 == 0:
            print(f"scored {index}/{len(selected_pairs)}", flush=True)

    score_path = output_dir / "candidate_scores.jsonl"
    write_jsonl(score_path, score_rows)
    metrics = evaluate_rows(score_rows, labels)
    metrics_path = output_dir / "metrics_summary.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_metrics_csv(output_dir / "metrics_per_sequence.csv", metrics)

    failure_counts = Counter(
        row.get("failure_reason", "") for row in score_rows if row["status"] == "failed"
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "AsterSLAM hard-gate maximum normalized violation",
        "score_definition": "score = -max_i(normalized_gate_violation_i)",
        "runtime_acceptance_equivalence": "score >= -1 iff every numeric hard gate passes",
        "candidate_evaluation": (
            "Each candidate is independently added to the sampled odometry subgraph; accepted-loop "
            "history is excluded so scores remain fixed during the AP threshold sweep."
        ),
        "metric_definition": {
            "AP": "sklearn.metrics.average_precision_score(labels, scores)",
            "MR@100P": "max(recall[precision == 1.0]) from precision_recall_curve",
            "TP@100P": "MR@100P * positive_count",
        },
        "benchmark_root": str(benchmark_root),
        "sequence_order": list(TABLE_SEQUENCE_ORDER),
        "thresholds_by_sequence": configs,
        "input_sha256": {
            "benchmark_pairs.jsonl": sha256_file(benchmark_root / "benchmark_pairs.jsonl"),
            "geometry_predictions.jsonl": sha256_file(
                benchmark_root / "geometry_predictions.jsonl"
            ),
            "annotations.jsonl": sha256_file(benchmark_root / "annotations.jsonl"),
            "scores/salad.jsonl": sha256_file(benchmark_root / "scores" / "salad.jsonl"),
            "table3_loopanything_params.json": sha256_file(args.table3_params),
        },
        "output_sha256": {
            "candidate_scores.jsonl": sha256_file(score_path),
            "metrics_summary.json": sha256_file(metrics_path),
        },
        "candidate_count": len(score_rows),
        "failed_candidate_count": sum(row["status"] == "failed" for row in score_rows),
        "failure_reason_counts": dict(sorted(failure_counts.items())),
        "limit_per_sequence": args.limit_per_sequence,
        "command": sys.argv,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
