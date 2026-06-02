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
DEFAULT_QUERY_GATE_MARGIN_WEIGHTS = (0.0, 0.5, 1.0, 2.0, 4.0)
RANK_PRODUCT_METHODS = (
    ("rank_product:res,def", ("residual", "deformation")),
    ("rank_product:salad,def", ("salad", "deformation")),
    ("rank_product:salad,res", ("salad", "residual")),
    ("rank_product:salad,res,def", ("salad", "residual", "deformation")),
)
PERCENTILE_PRODUCT_METHODS = (
    ("percentile_product:res,def", ("residual", "deformation")),
    ("percentile_product:salad,def", ("salad", "deformation")),
    ("percentile_product:salad,res", ("salad", "residual")),
    ("percentile_product:salad,res,def", ("salad", "residual", "deformation")),
)
QUERY_PERCENTILE_PRODUCT_METHODS = (
    ("query_percentile_product:res,def", ("residual", "deformation")),
    ("query_percentile_product:salad,def", ("salad", "deformation")),
    ("query_percentile_product:salad,res", ("salad", "residual")),
    (
        "query_percentile_product:salad,res,def",
        ("salad", "residual", "deformation"),
    ),
)
PERCENTILE_MIN_METHODS = (
    ("percentile_min:salad,res,def", ("salad", "residual", "deformation")),
)
HISTORY_PERCENTILE_PRODUCT_METHODS = (
    ("history_percentile_product:res,def", ("residual", "deformation")),
    ("history_percentile_product:salad,def", ("salad", "deformation")),
    ("history_percentile_product:salad,res", ("salad", "residual")),
    (
        "history_percentile_product:salad,res,def",
        ("salad", "residual", "deformation"),
    ),
)
PERCENTILE_EPS = 1e-6


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

    for name, signal_names in RANK_PRODUCT_METHODS:
        score_rows.append(
            _metrics_row(name, labels, _rank_product_scores(records, signal_names))
        )
    for name, signal_names in PERCENTILE_PRODUCT_METHODS:
        score_rows.append(
            _metrics_row(name, labels, _percentile_product_scores(records, signal_names))
        )
    for name, signal_names in QUERY_PERCENTILE_PRODUCT_METHODS:
        score_rows.append(
            _metrics_row(
                name,
                labels,
                _query_percentile_product_scores(records, signal_names),
            )
        )
    for name, signal_names in PERCENTILE_MIN_METHODS:
        score_rows.append(
            _metrics_row(name, labels, _percentile_min_scores(records, signal_names))
        )
    history_percentile_features = _history_percentile_features(records)
    for name, signal_names in HISTORY_PERCENTILE_PRODUCT_METHODS:
        score_rows.append(
            _metrics_row(
                name,
                labels,
                _percentile_product_scores_from_features(
                    history_percentile_features,
                    len(records),
                    signal_names,
                ),
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
            name = "absolute_graph:def={},res={}".format(
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
            for margin_weight in DEFAULT_QUERY_GATE_MARGIN_WEIGHTS:
                name = "query_gate_graph:def={},res={},margin={}".format(
                    _format_weight(deformation_weight),
                    _format_weight(residual_weight),
                    _format_weight(margin_weight),
                )
                score_rows.append(
                    _metrics_row(
                        name,
                        labels,
                        _query_gate_graph_scores(
                            records,
                            deformation_weight,
                            residual_weight,
                            margin_weight,
                        ),
                    )
                )

    normalized_features = _normalized_features(records)
    history_robust_z_features = _history_robust_z_features(records)
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
                name = "history_robust_z_fusion:salad={},def={},res={}".format(
                    _format_weight(salad_weight),
                    _format_weight(deformation_weight),
                    _format_weight(residual_weight),
                )
                score_rows.append(
                    _metrics_row(
                        name,
                        labels,
                        _z_fusion_scores(
                            history_robust_z_features,
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


def _query_gate_graph_scores(
    records: Sequence[Mapping[str, Any]],
    deformation_weight: float,
    residual_weight: float,
    margin_weight: float,
) -> list[float | None]:
    base_scores = _raw_graph_scores(records, deformation_weight, residual_weight)
    scores: list[float | None] = [None for _ in records]
    for indices in _query_groups(records):
        valid_scores = [
            float(base_scores[index])
            for index in indices
            if base_scores[index] is not None and math.isfinite(float(base_scores[index]))
        ]
        if not valid_scores:
            continue
        valid_scores.sort(reverse=True)
        best_score = valid_scores[0]
        second_score = valid_scores[1] if len(valid_scores) > 1 else best_score
        query_confidence = best_score + margin_weight * max(0.0, best_score - second_score)
        for index in indices:
            if base_scores[index] is None:
                continue
            scores[index] = float(base_scores[index]) + query_confidence
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


def _counterfactual_signals(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, list[float | None]]:
    return {
        "salad": _salad_scores(records),
        "residual": [
            _negative(_log1p_or_none(record.get("pgo_error_after"))) for record in records
        ],
        "deformation": [
            _negative(_finite_float_or_none(record.get("trajectory_deformation_rmse")))
            for record in records
        ],
    }


def _raw_calibration_features(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, list[float | None]]:
    return {
        "salad": _salad_scores(records),
        "deformation": [
            _finite_float_or_none(record.get("trajectory_deformation_rmse"))
            for record in records
        ],
        "residual": [_log1p_or_none(record.get("pgo_error_after")) for record in records],
    }


def _signal_larger_is_better(signal_name: str) -> bool:
    if signal_name in {"salad", "residual", "deformation"}:
        return True
    raise ValueError(f"unknown signal name: {signal_name}")


def _rank_product_scores(
    records: Sequence[Mapping[str, Any]],
    signal_names: Sequence[str],
) -> list[float | None]:
    signals = _counterfactual_signals(records)
    groups = _query_groups(records)
    scores: list[float | None] = [None for _ in records]
    for indices in groups:
        query_size = len(indices)
        ranks_by_signal = {
            signal_name: _query_signal_ranks(
                [signals[signal_name][index] for index in indices],
                larger_is_better=_signal_larger_is_better(signal_name),
                invalid_rank=query_size + 1,
            )
            for signal_name in signal_names
        }
        for local_offset, record_index in enumerate(indices):
            rank_sum = 0.0
            for signal_name in signal_names:
                rank_sum += math.log(float(ranks_by_signal[signal_name][local_offset]))
            scores[record_index] = -rank_sum
    return scores


def _percentile_product_scores(
    records: Sequence[Mapping[str, Any]],
    signal_names: Sequence[str],
) -> list[float | None]:
    percentiles = _percentile_features(records)
    return _percentile_product_scores_from_features(
        percentiles,
        len(records),
        signal_names,
    )


def _percentile_product_scores_from_features(
    percentiles: Mapping[str, Sequence[float]],
    record_count: int,
    signal_names: Sequence[str],
) -> list[float | None]:
    scores: list[float | None] = []
    for index in range(record_count):
        value = 0.0
        for signal_name in signal_names:
            value += math.log(PERCENTILE_EPS + percentiles[signal_name][index])
        scores.append(value)
    return scores


def _query_percentile_product_scores(
    records: Sequence[Mapping[str, Any]],
    signal_names: Sequence[str],
) -> list[float | None]:
    signals = _counterfactual_signals(records)
    groups = _query_groups(records)
    scores: list[float | None] = [None for _ in records]
    for indices in groups:
        percentiles_by_signal = {
            signal_name: _empirical_percentiles(
                [signals[signal_name][index] for index in indices]
            )
            for signal_name in signal_names
        }
        for local_offset, record_index in enumerate(indices):
            value = 0.0
            for signal_name in signal_names:
                value += math.log(
                    PERCENTILE_EPS
                    + percentiles_by_signal[signal_name][local_offset]
                )
            scores[record_index] = value
    return scores


def _history_percentile_product_scores(
    records: Sequence[Mapping[str, Any]],
    signal_names: Sequence[str],
) -> list[float | None]:
    return _percentile_product_scores_from_features(
        _history_percentile_features(records),
        len(records),
        signal_names,
    )


def _history_percentile_features(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, list[float]]:
    signals = _counterfactual_signals(records)
    histories: dict[str, list[float]] = {
        "salad": [],
        "residual": [],
        "deformation": [],
    }
    percentiles: dict[str, list[float]] = {
        signal_name: [0.0 for _ in records] for signal_name in histories
    }
    for indices in _query_groups(records):
        sorted_histories = {
            signal_name: sorted(history)
            for signal_name, history in histories.items()
        }
        for signal_name in histories:
            sorted_history = sorted_histories[signal_name]
            for index in indices:
                signal_value = signals[signal_name][index]
                if signal_value is None or not math.isfinite(float(signal_value)):
                    percentiles[signal_name][index] = 0.0
                    continue
                percentiles[signal_name][index] = _single_empirical_percentile_sorted(
                    sorted_history,
                    float(signal_value),
                    default=0.5,
                )
        for signal_name in histories:
            for index in indices:
                signal_value = signals[signal_name][index]
                if signal_value is not None and math.isfinite(float(signal_value)):
                    histories[signal_name].append(float(signal_value))
    return percentiles


def _history_robust_z_fusion_scores(
    records: Sequence[Mapping[str, Any]],
    salad_weight: float,
    deformation_weight: float,
    residual_weight: float,
) -> list[float | None]:
    return _z_fusion_scores(
        _history_robust_z_features(records),
        salad_weight,
        deformation_weight,
        residual_weight,
    )


def _history_robust_z_features(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, list[float | None]]:
    features = _raw_calibration_features(records)
    histories: dict[str, list[float]] = {
        "salad": [],
        "deformation": [],
        "residual": [],
    }
    normalized: dict[str, list[float | None]] = {
        signal_name: [None for _ in records] for signal_name in histories
    }
    for indices in _query_groups(records):
        stats_by_signal = {
            signal_name: _history_robust_stats(history)
            for signal_name, history in histories.items()
        }
        for signal_name, values in features.items():
            center, scale = stats_by_signal[signal_name]
            for index in indices:
                value = values[index]
                if value is None or not math.isfinite(float(value)):
                    continue
                normalized[signal_name][index] = _apply_history_robust_z(
                    float(value),
                    center,
                    scale,
                )
        for signal_name, values in features.items():
            for index in indices:
                value = values[index]
                if value is not None and math.isfinite(float(value)):
                    histories[signal_name].append(float(value))
    return normalized


def _percentile_min_scores(
    records: Sequence[Mapping[str, Any]],
    signal_names: Sequence[str],
) -> list[float | None]:
    percentiles = _percentile_features(records)
    scores: list[float | None] = []
    for index in range(len(records)):
        scores.append(min(percentiles[signal_name][index] for signal_name in signal_names))
    return scores


def _single_empirical_percentile(
    history: Sequence[float],
    value: float,
    *,
    default: float,
) -> float:
    valid = sorted(float(item) for item in history if math.isfinite(float(item)))
    if not valid:
        return float(default)
    return _single_empirical_percentile_sorted(valid, float(value), default=default)


def _single_empirical_percentile_sorted(
    valid: Sequence[float],
    value: float,
    *,
    default: float,
) -> float:
    if not valid:
        return float(default)
    return _count_less_equal(valid, float(value)) / float(len(valid))


def _percentile_features(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, list[float]]:
    signals = _counterfactual_signals(records)
    return {
        signal_name: _empirical_percentiles(values)
        for signal_name, values in signals.items()
    }


def _empirical_percentiles(values: Sequence[float | None]) -> list[float]:
    valid = sorted(
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    )
    if not valid:
        return [0.0 for _ in values]
    denominator = float(len(valid))
    percentiles: list[float] = []
    for value in values:
        if value is None or not math.isfinite(float(value)):
            percentiles.append(0.0)
            continue
        count_less_equal = _count_less_equal(valid, float(value))
        percentiles.append(count_less_equal / denominator)
    return percentiles


def _count_less_equal(sorted_values: Sequence[float], value: float) -> int:
    left = 0
    right = len(sorted_values)
    while left < right:
        middle = (left + right) // 2
        if sorted_values[middle] <= value:
            left = middle + 1
        else:
            right = middle
    return left


def _history_robust_z(value: float, history: Sequence[float]) -> float:
    center, scale = _history_robust_stats(history)
    return _apply_history_robust_z(value, center, scale)


def _history_robust_stats(history: Sequence[float]) -> tuple[float, float]:
    valid = sorted(float(item) for item in history if math.isfinite(float(item)))
    if len(valid) < 3:
        return 0.0, 0.0
    median = _median(valid)
    deviations = sorted(abs(item - median) for item in valid)
    mad = _median(deviations)
    scale = 1.4826 * mad
    if scale <= 1e-12:
        mean = sum(valid) / len(valid)
        variance = sum((item - mean) ** 2 for item in valid) / len(valid)
        scale = math.sqrt(variance)
        median = mean
    return median, scale


def _apply_history_robust_z(value: float, center: float, scale: float) -> float:
    if scale <= 1e-12:
        return 0.0
    return (float(value) - center) / scale


def _median(sorted_values: Sequence[float]) -> float:
    size = len(sorted_values)
    if size == 0:
        return 0.0
    middle = size // 2
    if size % 2:
        return float(sorted_values[middle])
    return 0.5 * (float(sorted_values[middle - 1]) + float(sorted_values[middle]))


def _query_groups(records: Sequence[Mapping[str, Any]]) -> list[list[int]]:
    groups_by_query: dict[int, list[int]] = {}
    fallback_query = -1
    for index, record in enumerate(records):
        query_idx = record.get("query_idx")
        if query_idx is None:
            query_idx = fallback_query
            fallback_query -= 1
        groups_by_query.setdefault(int(query_idx), []).append(index)
    return list(groups_by_query.values())


def _query_signal_ranks(
    values: Sequence[float | None],
    *,
    larger_is_better: bool,
    invalid_rank: int,
) -> list[int]:
    valid = [
        (index, float(value))
        for index, value in enumerate(values)
        if value is not None and math.isfinite(float(value))
    ]
    valid.sort(key=lambda item: item[1], reverse=larger_is_better)
    ranks = [invalid_rank for _ in values]
    current_rank = 1
    previous_value: float | None = None
    for offset, (index, value) in enumerate(valid):
        if previous_value is None or value != previous_value:
            current_rank = offset + 1
            previous_value = value
        ranks[index] = current_rank
    return ranks


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
