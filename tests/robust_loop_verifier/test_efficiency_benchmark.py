import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PY_SCRIPT = ROOT / "robust_loop_verification_scripts" / "benchmark_loop_verifier_efficiency.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("benchmark_loop_verifier_efficiency", PY_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def _candidate_records():
    return [
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


def test_default_efficiency_sequences_are_representative_subset():
    module = _load_script_module()

    assert module.DEFAULT_SEQUENCE_SPECS == (
        ("handheld", "handheld_escalator00"),
        ("ugv", "ugv_parking01"),
    )


def test_build_robust_method_rows_reports_shared_runtime_and_score_overhead(tmp_path: Path):
    module = _load_script_module()
    run_root = tmp_path / "run"
    _write_jsonl(run_root / "candidate_records.jsonl", _candidate_records())
    _write_json(
        run_root / "metrics.json",
        {
            "SALAD score only": {"AP": 0.25, "MR@100P": 0.0},
            "SALAD + DA3-ROVER full-prefix trajectory score": {
                "AP": 0.50,
                "MR@100P": 0.25,
            },
        },
    )
    _write_json(
        run_root / "efficiency_timing.json",
        {
            "candidate_count": 4,
            "query_count": 2,
            "component_totals_sec": {
                "descriptor_compute": 1.0,
                "retrieval_search": 0.2,
                "support_selection": 0.1,
                "da3_triplet": 8.0,
                "sim3_alignment": 0.3,
                "pgo": 2.0,
                "metrics": 0.05,
            },
            "per_candidate_sec": {
                "total": {
                    "count": 4,
                    "mean": 2.5,
                    "p50": 2.0,
                    "p90": 4.0,
                    "p95": 4.5,
                    "p99": 4.9,
                }
            },
        },
    )

    rows = module.build_robust_method_rows(
        platform="handheld",
        sequence="handheld_escalator00",
        run_root=run_root,
        elapsed_wall_sec=12.0,
    )

    rows_by_method = {row["method"]: row for row in rows}
    assert {
        "SALAD score only",
        "ROVER deformation only",
        "PGO residual only",
        "absolute_graph:def=0.5,res=0.25",
        "query_gate_graph:def=0.5,res=0.25,margin=0",
    } <= set(rows_by_method)
    query_gate = rows_by_method["query_gate_graph:def=0.5,res=0.25,margin=0"]
    assert query_gate["runtime_group"] == "SALAD+DA3+Sim3+PGO"
    assert query_gate["elapsed_wall_sec"] == 12.0
    assert query_gate["candidate_count"] == 4
    assert query_gate["mean_candidate_sec"] == 2.5
    assert query_gate["additional_score_wall_sec"] >= 0.0
    assert rows_by_method["ROVER deformation only"]["AP"] == 1.0


def test_write_efficiency_outputs_emits_json_csv_and_markdown(tmp_path: Path):
    module = _load_script_module()
    rows = [
        {
            "platform": "handheld",
            "sequence": "handheld_escalator00",
            "method": "SALAD score only",
            "AP": 0.8,
            "MR@100P": 0.2,
            "elapsed_wall_sec": 10.0,
            "query_count": 5,
            "candidate_count": 20,
            "mean_query_sec": 2.0,
            "mean_candidate_sec": 0.5,
            "p95_candidate_sec": 0.8,
            "additional_score_wall_sec": 0.0,
            "runtime_group": "SALAD+DA3+Sim3+PGO",
        }
    ]

    module.write_efficiency_outputs(tmp_path, rows)

    assert (tmp_path / "efficiency_summary.json").is_file()
    assert (tmp_path / "efficiency_summary.csv").is_file()
    assert (tmp_path / "efficiency_summary.md").is_file()
    payload = json.loads((tmp_path / "efficiency_summary.json").read_text(encoding="utf-8"))
    assert payload["rows"] == rows
    assert payload["averages"][0]["method"] == "SALAD score only"
