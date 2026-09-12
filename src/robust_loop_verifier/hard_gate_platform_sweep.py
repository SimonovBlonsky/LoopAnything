"""Platform-shared threshold sweep for the AsterSLAM maximum-violation gate."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .io import read_jsonl
from .platform_shared_sweep import (
    MAIN8_PLATFORM_GROUPS,
    ScoreSpec,
    choose_balanced_setting,
    evaluate_group_score,
)


GATE_NAMES = (
    "salad_min_score",
    "support_min_baseline",
    "sim3_min_scale",
    "sim3_max_scale",
    "sim3_alignment_rmse",
    "sim3_direction_error",
    "translation_norm",
    "trajectory_deformation",
    "pgo_error_per_factor",
    "odom_strain_chi2",
    "loop_chi2",
)

LOWER_BOUND_GATES = frozenset(
    {"salad_min_score", "support_min_baseline", "sim3_min_scale"}
)

PRIMARY_GRID_GATES = (
    "sim3_max_scale",
    "sim3_direction_error",
    "trajectory_deformation",
    "loop_chi2",
)

DEFAULT_THRESHOLD_MULTIPLIERS = (
    1.0 / 16.0,
    1.0 / 8.0,
    1.0 / 4.0,
    1.0 / 2.0,
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
)

BASE_THRESHOLDS_BY_GROUP: dict[str, dict[str, float]] = {
    "handheld": {
        "salad_min_score": 0.51,
        "support_min_baseline": 0.3,
        "sim3_min_scale": 0.05,
        "sim3_max_scale": 20.0,
        "sim3_alignment_rmse": 1.5,
        "sim3_direction_error": 10.0,
        "translation_norm": 80.0,
        "trajectory_deformation": 0.02,
        "pgo_error_per_factor": 1.0,
        "odom_strain_chi2": 50.0,
        "loop_chi2": 25.0,
    },
    "ugv": {
        "salad_min_score": 0.80,
        "support_min_baseline": 0.3,
        "sim3_min_scale": 0.05,
        "sim3_max_scale": 20.0,
        "sim3_alignment_rmse": 0.8,
        "sim3_direction_error": 5.0,
        "translation_norm": 80.0,
        "trajectory_deformation": 0.10,
        "pgo_error_per_factor": 1.0,
        "odom_strain_chi2": 50.0,
        "loop_chi2": 25.0,
    },
    "geode": {
        "salad_min_score": 0.84,
        "support_min_baseline": 0.3,
        "sim3_min_scale": 0.05,
        "sim3_max_scale": 20.0,
        "sim3_alignment_rmse": 0.55,
        "sim3_direction_error": 2.0,
        "translation_norm": 80.0,
        "trajectory_deformation": 0.055,
        "pgo_error_per_factor": 1.0,
        "odom_strain_chi2": 50.0,
        "loop_chi2": 25.0,
    },
    "ntu": {
        "salad_min_score": 0.62,
        "support_min_baseline": 0.3,
        "sim3_min_scale": 0.05,
        "sim3_max_scale": 20.0,
        "sim3_alignment_rmse": 1.0,
        "sim3_direction_error": 10.0,
        "translation_norm": 80.0,
        "trajectory_deformation": 0.12,
        "pgo_error_per_factor": 1.0,
        "odom_strain_chi2": 50.0,
        "loop_chi2": 25.0,
    },
}


def read_hard_gate_records(
    candidate_scores_path: Path,
    annotations_path: Path,
    *,
    allowed_sequence_keys: Sequence[str],
) -> list[dict[str, Any]]:
    labels = {str(row["pair_id"]): bool(int(row["label"])) for row in read_jsonl(annotations_path)}
    allowed = set(allowed_sequence_keys)
    records: list[dict[str, Any]] = []
    for row in read_jsonl(candidate_scores_path):
        sequence_key = str(row["sequence"])
        if sequence_key not in allowed:
            continue
        pair_id = str(row["pair_id"])
        metrics = row.get("metrics") if row.get("status") == "ok" else None
        record = {
            "pair_id": pair_id,
            "sequence_key": sequence_key,
            "query_idx": int(row["query_idx"]),
            "candidate_idx": int(row["candidate_idx"]),
            "label": labels[pair_id],
            "gate_features": _gate_features(metrics),
        }
        records.append(record)
    expected = {
        pair_id
        for pair_id, label in labels.items()
        if label in (False, True) and _pair_id_belongs_to_allowed_sequence(pair_id, allowed)
    }
    actual = {str(row["pair_id"]) for row in records}
    if expected != actual:
        raise ValueError(f"hard-gate score coverage mismatch: missing={sorted(expected - actual)[:5]}")
    return records


def score_hard_gate_records(
    records: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, float],
) -> list[float | None]:
    weights = _threshold_weights(thresholds)
    scores: list[float | None] = []
    for record in records:
        features = record.get("gate_features")
        if features is None:
            scores.append(None)
            continue
        violation = max(float(feature) * weight for feature, weight in zip(features, weights))
        scores.append(None if not math.isfinite(violation) else -violation)
    return scores


def evaluate_hard_gate_platform_shared_sweep(
    records: Sequence[Mapping[str, Any]],
    *,
    platform_groups: Mapping[str, Sequence[str]] = MAIN8_PLATFORM_GROUPS,
    threshold_multipliers: Sequence[float] = DEFAULT_THRESHOLD_MULTIPLIERS,
    mr_tolerance: float = 0.0,
) -> dict[str, Any]:
    group_rows: list[dict[str, Any]] = []
    selections: dict[str, dict[str, Any]] = {}
    selected_sequence_rows: list[dict[str, Any]] = []
    selected_score_rows: list[dict[str, Any]] = []

    for group_name, sequence_keys in platform_groups.items():
        group_records = [row for row in records if row["sequence_key"] in sequence_keys]
        base_thresholds = BASE_THRESHOLDS_BY_GROUP[group_name]
        rows = _primary_grid_rows(
            group_name,
            sequence_keys,
            group_records,
            base_thresholds,
            threshold_multipliers,
        )
        rows.extend(
            _coordinate_refinement_rows(
                group_name,
                sequence_keys,
                group_records,
                base_thresholds,
                rows,
                threshold_multipliers,
            )
        )
        group_rows.extend(rows)

        selection, sequence_rows, score_rows = _select_group_result(
            group_name,
            group_records,
            rows,
            mr_tolerance=mr_tolerance,
        )
        selections[group_name] = selection
        selected_sequence_rows.extend(sequence_rows)
        selected_score_rows.extend(score_rows)

    return {
        "group_rows": group_rows,
        "selections": selections,
        "selected_sequence_rows": selected_sequence_rows,
        "selected_aggregate": _aggregate_selected(selected_sequence_rows),
        "selected_score_rows": selected_score_rows,
        "mr_tolerance": mr_tolerance,
        "selection_used_for_table": "balanced",
        "threshold_multipliers": list(threshold_multipliers),
        "primary_grid_gates": list(PRIMARY_GRID_GATES),
        "sharing_scope": "platform",
    }


def evaluate_hard_gate_global_shared_sweep(
    records: Sequence[Mapping[str, Any]],
    *,
    sequence_keys: Sequence[str],
    threshold_multipliers: Sequence[float] = DEFAULT_THRESHOLD_MULTIPLIERS,
    mr_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Sweep one Hard Gate parameter set shared by every requested sequence."""

    group_name = "all"
    allowed = set(sequence_keys)
    group_records = [row for row in records if row["sequence_key"] in allowed]
    rows: list[dict[str, Any]] = []
    for base_thresholds in BASE_THRESHOLDS_BY_GROUP.values():
        base_rows = _primary_grid_rows(
            group_name,
            sequence_keys,
            group_records,
            base_thresholds,
            threshold_multipliers,
        )
        rows.extend(base_rows)
        rows.extend(
            _coordinate_refinement_rows(
                group_name,
                sequence_keys,
                group_records,
                base_thresholds,
                base_rows,
                threshold_multipliers,
            )
        )

    unique_rows = {str(row["name"]): row for row in rows}
    rows = list(unique_rows.values())
    selection, selected_sequence_rows, selected_score_rows = _select_group_result(
        group_name,
        group_records,
        rows,
        mr_tolerance=mr_tolerance,
    )
    return {
        "group_rows": rows,
        "selections": {group_name: selection},
        "selected_sequence_rows": selected_sequence_rows,
        "selected_aggregate": _aggregate_selected(selected_sequence_rows),
        "selected_score_rows": selected_score_rows,
        "mr_tolerance": mr_tolerance,
        "selection_used_for_table": "balanced",
        "threshold_multipliers": list(threshold_multipliers),
        "primary_grid_gates": list(PRIMARY_GRID_GATES),
        "base_profiles": list(BASE_THRESHOLDS_BY_GROUP),
        "sharing_scope": "global",
    }


def write_hard_gate_selected_outputs(output_dir: Path, result: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_thresholds = {
        group: selection["balanced"]["thresholds"]
        for group, selection in result["selections"].items()
    }
    with (output_dir / "hard_gate_selected_thresholds.json").open("w", encoding="utf-8") as handle:
        json.dump(selected_thresholds, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (output_dir / "hard_gate_selected_scores.jsonl").open("w", encoding="utf-8") as handle:
        for row in result["selected_score_rows"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _primary_grid_rows(
    group_name: str,
    sequence_keys: Sequence[str],
    records: Sequence[Mapping[str, Any]],
    base_thresholds: Mapping[str, float],
    multipliers: Sequence[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for values in itertools.product(multipliers, repeat=len(PRIMARY_GRID_GATES)):
        thresholds = dict(base_thresholds)
        for gate, multiplier in zip(PRIMARY_GRID_GATES, values):
            thresholds[gate] *= float(multiplier)
        rows.append(_evaluate_thresholds(group_name, sequence_keys, records, thresholds))
    return rows


def _coordinate_refinement_rows(
    group_name: str,
    sequence_keys: Sequence[str],
    records: Sequence[Mapping[str, Any]],
    base_thresholds: Mapping[str, float],
    grid_rows: Sequence[Mapping[str, Any]],
    multipliers: Sequence[float],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    starts = (
        max(grid_rows, key=_best_mr_key),
        max(grid_rows, key=_best_ap_key),
        max(grid_rows, key=_best_tp_key),
    )
    objectives = (_best_mr_key, _best_ap_key, _best_tp_key)
    tunable_gates = tuple(gate for gate in GATE_NAMES if gate != "salad_min_score")
    for start, objective in zip(starts, objectives):
        current = dict(start)
        for _ in range(3):
            improved = False
            for gate in tunable_gates:
                candidates = [current]
                for multiplier in multipliers:
                    if multiplier == 1.0:
                        continue
                    thresholds = dict(current["thresholds"])
                    thresholds[gate] = float(base_thresholds[gate]) * float(multiplier)
                    row = _evaluate_thresholds(group_name, sequence_keys, records, thresholds)
                    output.append(row)
                    candidates.append(row)
                selected = max(candidates, key=objective)
                if objective(selected) > objective(current):
                    current = dict(selected)
                    improved = True
            if not improved:
                break
    return output


def _evaluate_thresholds(
    group_name: str,
    sequence_keys: Sequence[str],
    records: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    name = _threshold_name(thresholds)
    row = evaluate_group_score(
        group_name=group_name,
        sequence_keys=sequence_keys,
        records=records,
        spec=ScoreSpec(name=name, scores=score_hard_gate_records(records, thresholds), kind="hard_gate"),
    )
    row["thresholds"] = dict(thresholds)
    return row


def _gate_features(metrics: Mapping[str, Any] | None) -> tuple[float, ...] | None:
    if metrics is None:
        return None
    salad = float(metrics["salad_score"])
    support = float(metrics["support_baseline_m"])
    scale = float(metrics["sim3_scale"])
    if salad <= 0.0 or support <= 0.0 or scale <= 0.0:
        return None
    features = (
        1.0 / salad,
        1.0 / support,
        1.0 / scale,
        scale,
        float(metrics["sim3_alignment_rmse_m"]),
        float(metrics["sim3_direction_error_deg"]),
        float(metrics["translation_norm_m"]),
        float(metrics["safety_trajectory_deformation_rmse_m"]),
        float(metrics["safety_pgo_error_per_factor_after"]),
        float(metrics["safety_odom_strain_chi2_after"]),
        float(metrics["safety_loop_chi2_after"]),
    )
    return features if all(math.isfinite(value) and value >= 0.0 for value in features) else None


def _threshold_weights(thresholds: Mapping[str, float]) -> np.ndarray:
    values = []
    for gate in GATE_NAMES:
        threshold = float(thresholds[gate])
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError(f"threshold {gate} must be finite and positive")
        values.append(threshold if gate in LOWER_BOUND_GATES else 1.0 / threshold)
    return np.asarray(values, dtype=np.float64)


def _threshold_name(thresholds: Mapping[str, float]) -> str:
    values = ",".join(f"{gate}={float(thresholds[gate]):.8g}" for gate in GATE_NAMES)
    return f"hard_gate_max_violation:{values}"


def _best_mr_key(row: Mapping[str, Any]) -> tuple[float, float, int, str]:
    return (
        float(row["macro_MR@100P"]),
        float(row["macro_AP"]),
        int(row["TP@100P"]),
        str(row["name"]),
    )


def _best_ap_key(row: Mapping[str, Any]) -> tuple[float, float, int, str]:
    return (
        float(row["macro_AP"]),
        float(row["macro_MR@100P"]),
        int(row["TP@100P"]),
        str(row["name"]),
    )


def _best_tp_key(row: Mapping[str, Any]) -> tuple[int, float, float, str]:
    return (
        int(row["TP@100P"]),
        float(row["macro_MR@100P"]),
        float(row["macro_AP"]),
        str(row["name"]),
    )


def _select_group_result(
    group_name: str,
    group_records: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    *,
    mr_tolerance: float,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    best_ap = max(rows, key=_best_ap_key)
    best_mr = max(rows, key=_best_mr_key)
    best_tp = max(rows, key=_best_tp_key)
    balanced = choose_balanced_setting(rows, mr_tolerance=mr_tolerance)
    selection = {
        "best_ap": dict(best_ap),
        "best_mr": dict(best_mr),
        "best_tp": dict(best_tp),
        "balanced": dict(balanced),
    }
    sequence_rows = []
    for row in balanced["sequences"]:
        selected = dict(row)
        selected.update(
            {
                "group": group_name,
                "selection": "balanced",
                "method": balanced["name"],
            }
        )
        sequence_rows.append(selected)
    score_rows = []
    selected_scores = score_hard_gate_records(group_records, balanced["thresholds"])
    for record, score in zip(group_records, selected_scores):
        score_rows.append(
            {
                "pair_id": record["pair_id"],
                "sequence": record["sequence_key"],
                "group": group_name,
                "score": score,
                "status": "ok" if score is not None else "failed",
            }
        )
    return selection, sequence_rows, score_rows


def _aggregate_selected(sequence_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    positives = sum(int(row["positive_count"]) for row in sequence_rows)
    tp = sum(int(row["TP@100P"]) for row in sequence_rows)
    return {
        "sequence_count": len(sequence_rows),
        "positive_count": positives,
        "macro_AP": sum(float(row["AP"]) for row in sequence_rows) / len(sequence_rows),
        "macro_MR@100P": sum(float(row["MR@100P"]) for row in sequence_rows)
        / len(sequence_rows),
        "TP@100P": tp,
        "micro_MR@100P": tp / positives if positives else 0.0,
    }


def _pair_id_belongs_to_allowed_sequence(pair_id: str, allowed: set[str]) -> bool:
    return any(sequence.replace("/", "_") in pair_id for sequence in allowed)
