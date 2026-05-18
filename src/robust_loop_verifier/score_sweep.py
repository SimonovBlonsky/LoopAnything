"""Post-hoc residual-aware score sweeps over candidate records."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from robust_loop_verifier.io import read_jsonl
from robust_loop_verifier.metrics import (
    assign_failure_worst_scores,
    average_precision,
    max_recall_at_100_precision,
)

DEFAULT_GRAPH_WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
DEFAULT_FUSION_WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def read_candidate_records(path: Path) -> list[Mapping[str, Any]]:
    return list(read_jsonl(Path(path)))


def compute_score_sweep(
    records: Sequence[Mapping[str, Any]],
    *,
    graph_weights: Sequence[float] = DEFAULT_GRAPH_WEIGHTS,
    fusion_weights: Sequence[float] = DEFAULT_FUSION_WEIGHTS,
) -> dict[str, Any]:
    labels = [_label(record) for record in records]
    score_rows: list[dict[str, Any]] = []

    score_rows.append(_metrics_row("SALAD score only", labels, _salad_scores(records)))
    score_rows.append(
        _metrics_row(
            "ROVER deformation only",
            labels,
            [
                _negative(_finite_float_or_none(record.get("trajectory_deformation_rmse")))
                for record in records
            ],
        )
    )
    score_rows.append(
        _metrics_row(
            "PGO residual only",
            labels,
            [_negative(_log1p_or_none(record.get("pgo_error_after"))) for record in records],
        )
    )
    support_ensemble_scores = [
        _finite_float_or_none(record.get("score_support_ensemble")) for record in records
    ]
    if any(score is not None for score in support_ensemble_scores):
        score_rows.append(
            _metrics_row(
                "DA3-ROVER++ support ensemble graph evidence",
                labels,
                support_ensemble_scores,
            )
        )

    graph_weights = _validated_weights(graph_weights, "graph_weights")
    fusion_weights = _validated_weights(fusion_weights, "fusion_weights")
    for deformation_weight in graph_weights:
        for residual_weight in graph_weights:
            if deformation_weight == 0.0 and residual_weight == 0.0:
                continue
            name = "raw_graph:def={},res={}".format(
                _format_weight(deformation_weight),
                _format_weight(residual_weight),
            )
            score_rows.append(
                _metrics_row(
                    name,
                    labels,
                    _raw_graph_scores(records, deformation_weight, residual_weight),
                )
            )

    normalized_features = _normalized_features(records)
    for salad_weight in fusion_weights:
        for deformation_weight in fusion_weights:
            for residual_weight in fusion_weights:
                if (
                    salad_weight == 0.0
                    and deformation_weight == 0.0
                    and residual_weight == 0.0
                ):
                    continue
                name = "z_fusion:salad={},def={},res={}".format(
                    _format_weight(salad_weight),
                    _format_weight(deformation_weight),
                    _format_weight(residual_weight),
                )
                score_rows.append(
                    _metrics_row(
                        name,
                        labels,
                        _z_fusion_scores(
                            normalized_features,
                            salad_weight,
                            deformation_weight,
                            residual_weight,
                        ),
                    )
                )

    return {
        "record_count": len(records),
        "positive_count": sum(labels),
        "scores": score_rows,
        "best_by_ap": _best_row(score_rows, "AP"),
        "best_by_mr": _best_row(score_rows, "MR@100P"),
    }


def write_score_sweep_markdown(path: Path, result: Mapping[str, Any], limit: int = 40) -> None:
    scores = list(result["scores"])
    ranked = sorted(scores, key=lambda row: (-row["AP"], -row["MR@100P"], row["name"]))
    lines = [
        "| method | AP | MR@100P |",
        "| --- | ---: | ---: |",
    ]
    for row in ranked[:limit]:
        lines.append("| {} | {:.4f} | {:.4f} |".format(row["name"], row["AP"], row["MR@100P"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metrics_row(
    name: str,
    labels: Sequence[bool],
    scores: Iterable[float | None],
) -> dict[str, Any]:
    fixed_scores = assign_failure_worst_scores(scores)
    return {
        "name": name,
        "AP": average_precision(labels, fixed_scores),
        "MR@100P": max_recall_at_100_precision(labels, fixed_scores),
    }


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


def _z_fusion_scores(
    features: Mapping[str, Sequence[float | None]],
    salad_weight: float,
    deformation_weight: float,
    residual_weight: float,
) -> list[float | None]:
    scores: list[float | None] = []
    for salad, deformation, residual in zip(
        features["salad"],
        features["deformation"],
        features["residual"],
    ):
        if salad is None or deformation is None or residual is None:
            scores.append(None)
            continue
        scores.append(
            salad_weight * salad
            - deformation_weight * deformation
            - residual_weight * residual
        )
    return scores


def _normalized_features(records: Sequence[Mapping[str, Any]]) -> dict[str, list[float | None]]:
    return {
        "salad": _z_scores(_salad_scores(records)),
        "deformation": _z_scores(
            [
                _finite_float_or_none(record.get("trajectory_deformation_rmse"))
                for record in records
            ]
        ),
        "residual": _z_scores(
            [_log1p_or_none(record.get("pgo_error_after")) for record in records]
        ),
    }


def _salad_scores(records: Sequence[Mapping[str, Any]]) -> list[float | None]:
    scores = []
    for record in records:
        value = record.get("score_salad", record.get("salad_score"))
        scores.append(_finite_float_or_none(value))
    return scores


def _z_scores(values: Sequence[float | None]) -> list[float | None]:
    finite_values = [float(value) for value in values if value is not None]
    if not finite_values:
        return [None for _ in values]
    mean = sum(finite_values) / len(finite_values)
    variance = sum((value - mean) ** 2 for value in finite_values) / len(finite_values)
    std = math.sqrt(variance)
    if std <= 0.0:
        return [0.0 if value is not None else None for value in values]
    return [None if value is None else (float(value) - mean) / std for value in values]


def _best_row(rows: Sequence[Mapping[str, Any]], metric: str) -> Mapping[str, Any]:
    best_index, best = max(
        enumerate(rows),
        key=lambda item: (item[1][metric], item[1]["MR@100P"], -item[0]),
    )
    return dict(best)


def _validated_weights(values: Sequence[float], field_name: str) -> tuple[float, ...]:
    weights = tuple(float(value) for value in values)
    if not weights:
        raise ValueError(f"{field_name} must not be empty")
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError(f"{field_name} must contain finite non-negative values")
    return weights


def _label(record: Mapping[str, Any]) -> bool:
    label = record.get("label")
    if type(label) is not bool:
        raise ValueError("candidate record label must be bool")
    return label


def _finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def _log1p_or_none(value: Any) -> float | None:
    value = _finite_float_or_none(value)
    if value is None or value < -1.0:
        return None
    return math.log1p(value)


def _negative(value: float | None) -> float | None:
    if value is None:
        return None
    return -float(value)


def _format_weight(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return "{:.6g}".format(value)
