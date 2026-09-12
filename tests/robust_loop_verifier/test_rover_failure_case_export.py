import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from robust_loop_verification_scripts.export_rover_aligned_failure_cases import (
    run_rover_aligned_failure_case_export,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 30), color=color).save(path)


def test_rover_aligned_failure_export_attaches_labels_and_exports_regressions(
    tmp_path: Path,
):
    benchmark_root = tmp_path / "benchmark"
    cache_root = tmp_path / "cache"
    output_root = tmp_path / "failure_cases"
    for idx, color in {
        1: (255, 0, 0),
        2: (0, 255, 0),
        3: (0, 0, 255),
        4: (255, 255, 0),
        10: (255, 0, 255),
    }.items():
        _write_image(cache_root / "images" / f"{idx:06d}.png", color)
    _write_jsonl(
        cache_root / "keyframes.jsonl",
        [
            {"idx": idx, "image_path": f"images/{idx:06d}.png"}
            for idx in (1, 2, 3, 4, 10)
        ],
    )
    _write_json(
        benchmark_root / "manifest.json",
        {
            "sequences": [
                {
                    "dataset": "UnitDataset",
                    "platform": "unit",
                    "sequence": "unit_seq",
                    "cache": str(cache_root),
                }
            ]
        },
    )
    _write_jsonl(
        benchmark_root / "candidate_records.jsonl",
        [
            {
                "pair_id": "false_salad",
                "dataset": "UnitDataset",
                "platform": "unit",
                "sequence": "unit_seq",
                "sequence_key": "UnitDataset/unit/unit_seq",
                "query_idx": 10,
                "candidate_idx": 1,
                "support_idx": 2,
                "rank": 1,
                "score_salad": 0.99,
                "trajectory_deformation_rmse": 0.01,
                "pgo_error_after": 0.0,
            },
            {
                "pair_id": "true_regression",
                "dataset": "UnitDataset",
                "platform": "unit",
                "sequence": "unit_seq",
                "sequence_key": "UnitDataset/unit/unit_seq",
                "query_idx": 10,
                "candidate_idx": 3,
                "support_idx": 2,
                "rank": 2,
                "score_salad": 0.95,
                "trajectory_deformation_rmse": 10.0,
                "pgo_error_after": 100.0,
            },
            {
                "pair_id": "true_good_graph",
                "dataset": "UnitDataset",
                "platform": "unit",
                "sequence": "unit_seq",
                "sequence_key": "UnitDataset/unit/unit_seq",
                "query_idx": 10,
                "candidate_idx": 4,
                "support_idx": 2,
                "rank": 3,
                "score_salad": 0.10,
                "trajectory_deformation_rmse": 0.02,
                "pgo_error_after": 0.0,
            },
        ],
    )
    _write_jsonl(
        benchmark_root / "annotations.jsonl",
        [
            {"pair_id": "false_salad", "label": 0},
            {"pair_id": "true_regression", "label": 1},
            {"pair_id": "true_good_graph", "label": 1},
        ],
    )

    run_rover_aligned_failure_case_export(
        benchmark_root,
        output_root=output_root,
        methods=("SALAD score only",),
        target_method="query_gate_graph:def=0.5,res=0.25,margin=0",
        reference_method="SALAD score only",
        max_false_positives_per_method=1,
        max_regressions_per_sequence=1,
    )

    false_positive_dir = (
        output_root
        / "UnitDataset"
        / "unit"
        / "unit_seq"
        / "false_positives"
        / "salad_score_only"
        / "rank001_q000010_c000001_s000002"
    )
    assert (false_positive_dir / "query.png").is_file()
    assert (false_positive_dir / "candidate.png").is_file()
    assert (false_positive_dir / "support.png").is_file()
    assert (false_positive_dir / "overview.png").is_file()
    false_positive_record = json.loads(
        (false_positive_dir / "record.json").read_text(encoding="utf-8")
    )
    assert false_positive_record["label"] is False
    assert false_positive_record["case_type"] == "false_positive"

    regression_root = (
        output_root
        / "UnitDataset"
        / "unit"
        / "unit_seq"
        / "regressions"
        / "query_gate_graph_def_0_5_res_0_25_margin_0_vs_salad_score_only"
    )
    regression_dirs = list(regression_root.glob("refrank002_targetrank*_q000010_c000003_s000002"))
    assert regression_dirs
    regression_record = json.loads(
        (regression_dirs[0] / "record.json").read_text(encoding="utf-8")
    )
    assert regression_record["label"] is True
    assert regression_record["case_type"] == "target_regression"
    assert regression_record["reference_method"] == "SALAD score only"
    assert regression_record["target_method"] == "query_gate_graph:def=0.5,res=0.25,margin=0"
    assert regression_record["rank_drop"] > 0

    summary = (output_root / "summary.md").read_text(encoding="utf-8")
    assert "false_positive" in summary
    assert "target_regression" in summary


def test_rover_aligned_failure_export_script_help_runs_directly():
    repo_root = Path(__file__).resolve().parents[2]
    script = (
        repo_root
        / "robust_loop_verification_scripts"
        / "export_rover_aligned_failure_cases.py"
    )

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--target-method" in result.stdout
