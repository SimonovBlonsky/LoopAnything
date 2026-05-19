import json
import math
from pathlib import Path

from typer.testing import CliRunner

from robust_loop_verifier.cli import app


def _records():
    return [
        {
            "label": True,
            "score_salad": 0.90,
            "trajectory_deformation_rmse": 0.90,
            "pgo_error_after": 0.10517018598809245,
        },
        {
            "label": False,
            "score_salad": 0.95,
            "trajectory_deformation_rmse": 0.05,
            "pgo_error_after": 2.3201169227365472,
        },
        {
            "label": True,
            "score_salad": 0.80,
            "trajectory_deformation_rmse": 0.10,
            "pgo_error_after": 1.4596031111569499,
        },
        {
            "label": False,
            "score_salad": 0.10,
            "trajectory_deformation_rmse": 1.20,
            "pgo_error_after": 0.05127109637602412,
        },
    ]


def _write_records(path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in _records():
            handle.write(json.dumps(record))
            handle.write("\n")


def test_residual_aware_sweep_finds_graph_score_above_salad():
    from robust_loop_verifier.score_sweep import compute_score_sweep

    result = compute_score_sweep(
        _records(),
        graph_weights=(0.0, 1.0),
        fusion_weights=(0.0, 1.0),
    )

    metrics_by_name = {row["name"]: row for row in result["scores"]}

    assert metrics_by_name["SALAD score only"]["AP"] < 1.0
    assert metrics_by_name["raw_graph:def=1,res=1"]["AP"] == 1.0
    assert metrics_by_name["raw_graph:def=1,res=1"]["MR@100P"] == 1.0
    assert result["best_by_ap"]["AP"] == 1.0
    assert result["best_by_mr"]["MR@100P"] == 1.0


def test_score_sweep_assigns_failed_candidates_worst_scores():
    from robust_loop_verifier.score_sweep import compute_score_sweep

    records = _records() + [
        {
            "label": True,
            "score_salad": 0.70,
            "trajectory_deformation_rmse": None,
            "pgo_error_after": None,
        }
    ]

    result = compute_score_sweep(records, graph_weights=(1.0,), fusion_weights=(1.0,))

    for row in result["scores"]:
        assert math.isfinite(row["AP"])
        assert math.isfinite(row["MR@100P"])
    assert result["record_count"] == 5
    assert result["positive_count"] == 3


def test_score_sweep_reports_support_ensemble_score_when_present():
    from robust_loop_verifier.score_sweep import compute_score_sweep

    records = [
        {
            "label": True,
            "score_salad": 0.90,
            "trajectory_deformation_rmse": 0.90,
            "pgo_error_after": 0.10517018598809245,
            "score_support_ensemble": 0.96,
        },
        {
            "label": False,
            "score_salad": 0.95,
            "trajectory_deformation_rmse": 0.05,
            "pgo_error_after": 2.3201169227365472,
            "score_support_ensemble": 0.20,
        },
        {
            "label": True,
            "score_salad": 0.80,
            "trajectory_deformation_rmse": 0.10,
            "pgo_error_after": 1.4596031111569499,
            "score_support_ensemble": 0.75,
        },
    ]

    result = compute_score_sweep(records, graph_weights=(1.0,), fusion_weights=(1.0,))

    rows_by_name = {row["name"]: row for row in result["scores"]}
    support_ensemble_row = rows_by_name["DA3-ROVER++ support ensemble graph evidence"]
    assert math.isfinite(support_ensemble_row["AP"])
    assert math.isfinite(support_ensemble_row["MR@100P"])


def test_rank_product_res_def_reranks_within_each_query():
    from robust_loop_verifier.score_sweep import compute_score_sweep

    records = [
        {
            "query_idx": 10,
            "candidate_idx": 1,
            "label": False,
            "score_salad": 0.99,
            "trajectory_deformation_rmse": 2.0,
            "pgo_error_after": 6.0,
        },
        {
            "query_idx": 10,
            "candidate_idx": 2,
            "label": True,
            "score_salad": 0.70,
            "trajectory_deformation_rmse": 0.1,
            "pgo_error_after": 0.1,
        },
        {
            "query_idx": 11,
            "candidate_idx": 3,
            "label": False,
            "score_salad": 0.95,
            "trajectory_deformation_rmse": 1.5,
            "pgo_error_after": 4.0,
        },
        {
            "query_idx": 11,
            "candidate_idx": 4,
            "label": True,
            "score_salad": 0.60,
            "trajectory_deformation_rmse": 0.2,
            "pgo_error_after": 0.2,
        },
    ]

    result = compute_score_sweep(records, graph_weights=(1.0,), fusion_weights=(1.0,))
    rows = {row["name"]: row for row in result["scores"]}

    assert rows["SALAD score only"]["AP"] < 1.0
    assert rows["rank_product:res,def"]["AP"] == 1.0
    assert rows["rank_product:res,def"]["MR@100P"] == 1.0


def test_percentile_product_scores_global_candidate_distribution():
    from robust_loop_verifier.score_sweep import compute_score_sweep

    records = [
        {
            "query_idx": 10,
            "candidate_idx": 1,
            "label": False,
            "score_salad": 0.99,
            "trajectory_deformation_rmse": 2.0,
            "pgo_error_after": 6.0,
        },
        {
            "query_idx": 10,
            "candidate_idx": 2,
            "label": True,
            "score_salad": 0.70,
            "trajectory_deformation_rmse": 0.1,
            "pgo_error_after": 0.1,
        },
        {
            "query_idx": 11,
            "candidate_idx": 3,
            "label": False,
            "score_salad": 0.95,
            "trajectory_deformation_rmse": 1.5,
            "pgo_error_after": 4.0,
        },
        {
            "query_idx": 11,
            "candidate_idx": 4,
            "label": True,
            "score_salad": 0.60,
            "trajectory_deformation_rmse": 0.2,
            "pgo_error_after": 0.2,
        },
    ]

    result = compute_score_sweep(records, graph_weights=(1.0,), fusion_weights=(1.0,))
    rows = {row["name"]: row for row in result["scores"]}

    assert rows["percentile_product:res,def"]["AP"] == 1.0
    assert rows["percentile_product:res,def"]["MR@100P"] == 1.0
    assert rows["percentile_min:salad,res,def"]["AP"] < 1.0


def test_score_sweep_omits_support_ensemble_score_when_absent_or_all_none():
    from robust_loop_verifier.score_sweep import compute_score_sweep

    absent_result = compute_score_sweep(
        _records(),
        graph_weights=(1.0,),
        fusion_weights=(1.0,),
    )
    none_records = [
        {**record, "score_support_ensemble": None}
        for record in _records()
    ]
    none_result = compute_score_sweep(
        none_records,
        graph_weights=(1.0,),
        fusion_weights=(1.0,),
    )

    support_ensemble_name = "DA3-ROVER++ support ensemble graph evidence"
    assert support_ensemble_name not in [row["name"] for row in absent_result["scores"]]
    assert support_ensemble_name not in [row["name"] for row in none_result["scores"]]

    mixed_records = [
        {**record, "score_support_ensemble": None}
        for record in _records()
    ]
    mixed_records[0]["score_support_ensemble"] = 0.5
    mixed_result = compute_score_sweep(
        mixed_records,
        graph_weights=(1.0,),
        fusion_weights=(1.0,),
    )
    mixed_rows_by_name = {row["name"]: row for row in mixed_result["scores"]}
    assert math.isfinite(mixed_rows_by_name[support_ensemble_name]["AP"])


def test_sweep_scores_cli_writes_json_and_markdown(tmp_path):
    records_path = tmp_path / "candidate_records.jsonl"
    output_root = tmp_path / "sweep"
    _write_records(records_path)

    result = CliRunner().invoke(
        app,
        [
            "sweep-scores",
            "--candidate-records",
            str(records_path),
            "--output-root",
            str(output_root),
            "--graph-weights",
            "0,1",
            "--fusion-weights",
            "0,1",
        ],
    )

    assert result.exit_code == 0
    assert "best_ap=" in result.stdout
    sweep_json = output_root / "score_sweep.json"
    sweep_md = output_root / "score_sweep.md"
    assert sweep_json.is_file()
    assert sweep_md.is_file()
    payload = json.loads(sweep_json.read_text(encoding="utf-8"))
    assert payload["best_by_ap"]["AP"] == 1.0
    markdown = sweep_md.read_text(encoding="utf-8")
    assert "| raw_graph:def=1,res=1 |" in markdown
    assert "| rank_product:res,def |" in markdown
    assert "| percentile_product:res,def |" in markdown
