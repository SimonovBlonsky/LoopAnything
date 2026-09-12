"""Per-platform shared parameter sweeps for frozen loop-candidate records."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .io import read_jsonl
from .metrics import assign_failure_worst_scores, average_precision, max_recall_at_100_precision

MAIN8_PLATFORM_GROUPS: dict[str, tuple[str, ...]] = {
    "handheld": (
        "FusionPortableV2/handheld/handheld_escalator00",
        "FusionPortableV2/handheld/handheld_room00",
    ),
    "ugv": (
        "FusionPortableV2/ugv/ugv_campus01",
        "FusionPortableV2/ugv/ugv_parking01",
    ),
    "geode": (
        "GEODE/Offroad/Offroad02_beta",
        "GEODE/Offroad/Offroad05_beta",
    ),
    "ntu": (
        "NTU-VIRAL/NTU-VIRAL/eee_01",
        "NTU-VIRAL/NTU-VIRAL/nya_02",
    ),
}

DEFAULT_RATIOS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0)
DEFAULT_GRAPH_WEIGHTS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
DEFAULT_DIRECTION_WEIGHTS = (0.0, 0.1, 0.2, 0.5, 0.75, 1.0, 1.5, 2.0)
DEFAULT_SALAD_WEIGHTS = (0.5, 1.0, 1.5)
DEFAULT_MR_TOLERANCE = 0.01


@dataclass(frozen=True)
class ScoreSpec:
    name: str
    scores: Sequence[float | None]
    kind: str = "paper_safe"


def read_records_with_review_labels(
    candidate_records_path: Path,
    annotations_path: Path,
    *,
    allowed_sequence_keys: Iterable[str],
) -> list[dict[str, Any]]:
    labels_by_pair_id = {
        str(row["pair_id"]): bool(int(row["label"])) for row in read_jsonl(annotations_path)
    }
    allowed = set(allowed_sequence_keys)
    records: list[dict[str, Any]] = []
    for row in read_jsonl(candidate_records_path):
        sequence_key = _sequence_key(row)
        if sequence_key not in allowed:
            continue
        pair_id = str(row.get("pair_id", ""))
        if pair_id not in labels_by_pair_id:
            continue
        record = dict(row)
        record["sequence_key"] = sequence_key
        record["label"] = labels_by_pair_id[pair_id]
        records.append(record)

    missing = sorted(labels_by_pair_id.keys() - {str(row.get("pair_id", "")) for row in records})
    missing = [pair_id for pair_id in missing if _pair_id_sequence_key(pair_id) in allowed]
    if missing:
        raise ValueError(f"candidate records missing reviewed pairs: {missing[:5]}")
    return records


def generate_paper_safe_score_specs(
    records: Sequence[Mapping[str, Any]],
    *,
    ratios: Sequence[float] = DEFAULT_RATIOS,
    graph_weights: Sequence[float] = DEFAULT_GRAPH_WEIGHTS,
    direction_weights: Sequence[float] = DEFAULT_DIRECTION_WEIGHTS,
    salad_weights: Sequence[float] = DEFAULT_SALAD_WEIGHTS,
) -> list[ScoreSpec]:
    features_by_ratio = {
        ratio: _paper_safe_features(records, residual_ratio=float(ratio)) for ratio in ratios
    }
    specs = [
        ScoreSpec(
            name="baseline:SALAD",
            scores=[_finite_float_or_none(row.get("score_salad", row.get("salad_score"))) for row in records],
            kind="baseline",
        )
    ]
    for ratio in ratios:
        features = features_by_ratio[ratio]
        for salad_weight in salad_weights:
            for graph_weight in graph_weights:
                for direction_weight in direction_weights:
                    if graph_weight == 0.0 and direction_weight == 0.0:
                        continue
                    name = (
                        "salad_plus_seqz_graph_qpct_sim3dir:"
                        f"sw={_format_weight(salad_weight)},"
                        f"ratio={_format_weight(ratio)},"
                        f"ag={_format_weight(graph_weight)},"
                        f"ad={_format_weight(direction_weight)}"
                    )
                    scores = []
                    for salad, graph_z, direction_pct in zip(
                        features["salad"],
                        features["graph_z"],
                        features["direction_query_percentile"],
                    ):
                        if salad is None or graph_z is None or direction_pct is None:
                            scores.append(None)
                            continue
                        scores.append(
                            salad_weight * salad
                            + graph_weight * graph_z
                            + direction_weight * direction_pct
                        )
                    specs.append(ScoreSpec(name=name, scores=scores))
    return specs


def evaluate_group_score(
    *,
    group_name: str,
    sequence_keys: Sequence[str],
    records: Sequence[Mapping[str, Any]],
    spec: ScoreSpec,
) -> dict[str, Any]:
    if len(records) != len(spec.scores):
        raise ValueError("records and scores must have the same length")
    sequence_rows = []
    for sequence_key in sequence_keys:
        indices = [
            index for index, record in enumerate(records) if _sequence_key(record) == sequence_key
        ]
        if not indices:
            raise ValueError(f"no records for sequence_key={sequence_key}")
        labels = [bool(records[index]["label"]) for index in indices]
        scores = assign_failure_worst_scores(spec.scores[index] for index in indices)
        mr = max_recall_at_100_precision(labels, scores)
        positive_count = sum(labels)
        sequence_rows.append(
            {
                "sequence": sequence_key,
                "candidate_count": len(indices),
                "positive_count": positive_count,
                "AP": average_precision(labels, scores),
                "MR@100P": mr,
                "TP@100P": int(round(mr * positive_count)),
            }
        )
    positive_count = sum(int(row["positive_count"]) for row in sequence_rows)
    tp_count = sum(int(row["TP@100P"]) for row in sequence_rows)
    return {
        "group": group_name,
        "name": spec.name,
        "kind": spec.kind,
        "sequence_count": len(sequence_rows),
        "candidate_count": sum(int(row["candidate_count"]) for row in sequence_rows),
        "positive_count": positive_count,
        "macro_AP": sum(float(row["AP"]) for row in sequence_rows) / len(sequence_rows),
        "macro_MR@100P": sum(float(row["MR@100P"]) for row in sequence_rows)
        / len(sequence_rows),
        "TP@100P": tp_count,
        "micro_MR@100P": tp_count / positive_count if positive_count else 0.0,
        "sequences": sequence_rows,
    }


def evaluate_platform_shared_sweep(
    records: Sequence[Mapping[str, Any]],
    specs: Sequence[ScoreSpec],
    *,
    platform_groups: Mapping[str, Sequence[str]] = MAIN8_PLATFORM_GROUPS,
    mr_tolerance: float = DEFAULT_MR_TOLERANCE,
) -> dict[str, Any]:
    group_rows: list[dict[str, Any]] = []
    selections: dict[str, dict[str, Any]] = {}
    sequence_rows: list[dict[str, Any]] = []
    for group_name, sequence_keys in platform_groups.items():
        rows = [
            evaluate_group_score(
                group_name=group_name,
                sequence_keys=sequence_keys,
                records=records,
                spec=spec,
            )
            for spec in specs
        ]
        group_rows.extend(rows)
        best_ap = max(rows, key=lambda row: (row["macro_AP"], row["macro_MR@100P"], row["TP@100P"]))
        best_mr = max(rows, key=lambda row: (row["macro_MR@100P"], row["macro_AP"], row["TP@100P"]))
        best_tp = max(rows, key=lambda row: (row["TP@100P"], row["macro_MR@100P"], row["macro_AP"]))
        balanced = choose_balanced_setting(rows, mr_tolerance=mr_tolerance)
        selections[group_name] = {
            "best_ap": dict(best_ap),
            "best_mr": dict(best_mr),
            "best_tp": dict(best_tp),
            "balanced": dict(balanced),
        }
        sequence_rows.extend(_selected_sequence_rows(group_name, "balanced", balanced))

    aggregate = _aggregate_selected(sequence_rows)
    return {
        "group_rows": group_rows,
        "selections": selections,
        "selected_sequence_rows": sequence_rows,
        "selected_aggregate": aggregate,
        "mr_tolerance": mr_tolerance,
    }


def choose_balanced_setting(
    rows: Sequence[Mapping[str, Any]],
    *,
    mr_tolerance: float = DEFAULT_MR_TOLERANCE,
) -> Mapping[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty")
    best_mr = max(float(row["macro_MR@100P"]) for row in rows)
    candidates = [
        row
        for row in rows
        if float(row["macro_MR@100P"]) >= best_mr - float(mr_tolerance)
    ]
    return max(
        candidates,
        key=lambda row: (
            float(row["macro_AP"]),
            int(row["TP@100P"]),
            float(row["macro_MR@100P"]),
            str(row["name"]),
        ),
    )


def write_platform_shared_sweep_outputs(output_dir: Path, result: Mapping[str, Any]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "group_sweep_summary.csv",
        result["group_rows"],
        [
            "group",
            "name",
            "kind",
            "sequence_count",
            "candidate_count",
            "positive_count",
            "macro_AP",
            "macro_MR@100P",
            "TP@100P",
            "micro_MR@100P",
        ],
    )
    selection_rows = []
    for group_name, selected_by_kind in result["selections"].items():
        for selection_kind, row in selected_by_kind.items():
            selection_rows.append(
                {
                    "group": group_name,
                    "selection": selection_kind,
                    "name": row["name"],
                    "macro_AP": row["macro_AP"],
                    "macro_MR@100P": row["macro_MR@100P"],
                    "TP@100P": row["TP@100P"],
                    "micro_MR@100P": row["micro_MR@100P"],
                    "positive_count": row["positive_count"],
                }
            )
    _write_csv(
        output_dir / "selected_by_group.csv",
        selection_rows,
        [
            "group",
            "selection",
            "name",
            "macro_AP",
            "macro_MR@100P",
            "TP@100P",
            "micro_MR@100P",
            "positive_count",
        ],
    )
    _write_csv(
        output_dir / "selected_balanced_per_sequence.csv",
        result["selected_sequence_rows"],
        [
            "group",
            "selection",
            "method",
            "sequence",
            "candidate_count",
            "positive_count",
            "AP",
            "MR@100P",
            "TP@100P",
        ],
    )
    _write_json(output_dir / "selected_aggregate.json", result["selected_aggregate"])
    _write_markdown_report(output_dir / "report.md", result)


def _paper_safe_features(
    records: Sequence[Mapping[str, Any]],
    *,
    residual_ratio: float,
) -> dict[str, list[float | None]]:
    salad = [_finite_float_or_none(row.get("score_salad", row.get("salad_score"))) for row in records]
    graph_raw = []
    direction_raw = []
    for row in records:
        deformation = _finite_float_or_none(row.get("trajectory_deformation_rmse"))
        residual = _log1p_or_none(row.get("pgo_error_after"))
        direction = _finite_float_or_none(row.get("sim3_direction_error_deg"))
        if deformation is None or residual is None:
            graph_raw.append(None)
        else:
            graph_raw.append(-deformation - residual_ratio * residual)
        direction_raw.append(None if direction is None else -direction)
    return {
        "salad": salad,
        "graph_z": _sequence_z_scores(records, graph_raw),
        "direction_query_percentile": _query_percentiles(records, direction_raw),
    }


def _sequence_z_scores(
    records: Sequence[Mapping[str, Any]],
    values: Sequence[float | None],
) -> list[float | None]:
    output: list[float | None] = [None for _ in values]
    for indices in _groups_by(records, "sequence_key").values():
        valid = [
            float(values[index])
            for index in indices
            if values[index] is not None and math.isfinite(float(values[index]))
        ]
        if not valid:
            continue
        mean = sum(valid) / len(valid)
        variance = sum((value - mean) ** 2 for value in valid) / len(valid)
        std = math.sqrt(variance)
        for index in indices:
            value = values[index]
            if value is None or not math.isfinite(float(value)):
                continue
            output[index] = 0.0 if std <= 1e-12 else (float(value) - mean) / std
    return output


def _query_percentiles(
    records: Sequence[Mapping[str, Any]],
    values: Sequence[float | None],
) -> list[float | None]:
    output: list[float | None] = [None for _ in values]
    for indices in _query_groups(records).values():
        valid = sorted(
            float(values[index])
            for index in indices
            if values[index] is not None and math.isfinite(float(values[index]))
        )
        if not valid:
            continue
        for index in indices:
            value = values[index]
            if value is None or not math.isfinite(float(value)):
                continue
            output[index] = _count_less_equal(valid, float(value)) / len(valid)
    return output


def _selected_sequence_rows(
    group_name: str,
    selection_name: str,
    group_row: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for row in group_row["sequences"]:
        item = dict(row)
        item["group"] = group_name
        item["selection"] = selection_name
        item["method"] = group_row["name"]
        rows.append(item)
    return rows


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


def _groups_by(
    records: Sequence[Mapping[str, Any]],
    field_name: str,
) -> dict[Any, list[int]]:
    groups: dict[Any, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(record[field_name], []).append(index)
    return groups


def _query_groups(records: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], list[int]]:
    groups: dict[tuple[str, int], list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault((_sequence_key(record), int(record["query_idx"])), []).append(index)
    return groups


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


def _sequence_key(row: Mapping[str, Any]) -> str:
    sequence_key = row.get("sequence_key")
    if isinstance(sequence_key, str) and sequence_key:
        return sequence_key
    return "/".join(str(row[field]) for field in ("dataset", "platform", "sequence"))


def _pair_id_sequence_key(pair_id: str) -> str | None:
    if pair_id.startswith("FusionPortableV2_handheld_handheld_escalator00_"):
        return "FusionPortableV2/handheld/handheld_escalator00"
    if pair_id.startswith("FusionPortableV2_handheld_handheld_room00_"):
        return "FusionPortableV2/handheld/handheld_room00"
    if pair_id.startswith("FusionPortableV2_ugv_ugv_campus01_"):
        return "FusionPortableV2/ugv/ugv_campus01"
    if pair_id.startswith("FusionPortableV2_ugv_ugv_parking01_"):
        return "FusionPortableV2/ugv/ugv_parking01"
    if pair_id.startswith("GEODE_Offroad_Offroad02_beta_"):
        return "GEODE/Offroad/Offroad02_beta"
    if pair_id.startswith("GEODE_Offroad_Offroad05_beta_"):
        return "GEODE/Offroad/Offroad05_beta"
    if pair_id.startswith("NTU-VIRAL_NTU-VIRAL_eee_01_"):
        return "NTU-VIRAL/NTU-VIRAL/eee_01"
    if pair_id.startswith("NTU-VIRAL_NTU-VIRAL_nya_02_"):
        return "NTU-VIRAL/NTU-VIRAL/nya_02"
    return None


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


def _format_weight(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6g}"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_markdown_report(path: Path, result: Mapping[str, Any]) -> None:
    global_shared = result.get("sharing_scope") == "global"
    lines = [
        (
            "# Global-Shared LoopAnything Score Sweep"
            if global_shared
            else "# Platform-Shared LoopAnything Score Sweep"
        ),
        "",
        (
            "Balanced selection uses one parameter set across all evaluated sequences."
            if global_shared
            else "Balanced selection uses one parameter set per platform group."
        ),
        "It first keeps",
        f"settings within {result['mr_tolerance']:.3f} macro MR@100P of the group-best MR,",
        "then chooses the highest macro AP.",
        "",
        "## Selected Balanced Aggregate",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    aggregate = result["selected_aggregate"]
    lines.extend(
        [
            f"| Macro AP | {100.0 * aggregate['macro_AP']:.2f} |",
            f"| Macro MR@100P | {100.0 * aggregate['macro_MR@100P']:.2f} |",
            f"| TP@100P | {aggregate['TP@100P']} / {aggregate['positive_count']} |",
            "",
            "## Per-Group Selections",
            "",
            "| Group | Selection | AP | MR@100P | TP@100P | Parameters |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for group_name, selected_by_kind in result["selections"].items():
        for selection_name in ("best_ap", "best_mr", "best_tp", "balanced"):
            row = selected_by_kind[selection_name]
            lines.append(
                "| {} | {} | {:.2f} | {:.2f} | {} / {} | `{}` |".format(
                    group_name,
                    selection_name,
                    100.0 * row["macro_AP"],
                    100.0 * row["macro_MR@100P"],
                    row["TP@100P"],
                    row["positive_count"],
                    row["name"],
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
