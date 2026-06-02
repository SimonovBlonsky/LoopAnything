import json
from pathlib import Path

from PIL import Image

from robust_loop_verification_scripts.failure_case_analysis import (
    DEFAULT_METHODS,
    SequenceSpec,
    _build_arg_parser,
    _parse_methods,
    run_failure_case_analysis,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color=color).save(path)


def test_failure_case_analysis_exports_ranked_false_positive_visualizations(tmp_path: Path):
    cache_root = tmp_path / "cache"
    run_root = tmp_path / "run"
    output_root = tmp_path / "failure_analysis"

    _write_jsonl(
        cache_root / "keyframes.jsonl",
        [
            {"idx": 1, "image_path": "images/000001.png"},
            {"idx": 2, "image_path": "images/000002.png"},
            {"idx": 3, "image_path": "images/000003.png"},
            {"idx": 10, "image_path": "images/000010.png"},
        ],
    )
    _write_image(cache_root / "images/000001.png", (255, 0, 0))
    _write_image(cache_root / "images/000002.png", (0, 255, 0))
    _write_image(cache_root / "images/000003.png", (0, 0, 255))
    _write_image(cache_root / "images/000010.png", (255, 255, 0))

    _write_jsonl(
        run_root / "candidate_records.jsonl",
        [
            {
                "query_idx": 10,
                "candidate_idx": 1,
                "support_idx": 3,
                "rank": 1,
                "label": False,
                "score_salad": 0.9,
                "trajectory_deformation_rmse": 0.05,
                "pgo_error_after": 0.0,
            },
            {
                "query_idx": 10,
                "candidate_idx": 2,
                "support_idx": 3,
                "rank": 2,
                "label": True,
                "score_salad": 0.8,
                "trajectory_deformation_rmse": 0.5,
                "pgo_error_after": 0.0,
            },
        ],
    )

    run_failure_case_analysis(
        [
            SequenceSpec(
                dataset="UnitDataset",
                platform="unit",
                sequence="unit_seq",
                run_root=run_root,
                cache_root=cache_root,
            )
        ],
        output_root=output_root,
        max_failures_per_method=1,
        methods=("absolute_graph:def=0.5,res=0.25",),
    )

    summary = (output_root / "summary.md").read_text(encoding="utf-8")
    assert "UnitDataset" in summary
    assert "unit_seq" in summary
    assert "absolute_graph:def=0.5,res=0.25" in summary
    assert "1" in summary

    method_dir = (
        output_root
        / "UnitDataset"
        / "unit"
        / "unit_seq"
        / "absolute_graph_def_0_5_res_0_25"
    )
    false_positive_dirs = list(method_dir.glob("rank001_q000010_c000001_s000003"))
    assert false_positive_dirs
    false_positive_dir = false_positive_dirs[0]
    assert (false_positive_dir / "query.png").is_file()
    assert (false_positive_dir / "candidate.png").is_file()
    assert (false_positive_dir / "support.png").is_file()
    assert (false_positive_dir / "overview.png").is_file()
    exported_record = json.loads((false_positive_dir / "record.json").read_text())
    assert exported_record["label"] is False
    assert exported_record["method_score"] > -0.1


def test_failure_case_analysis_cli_keeps_default_method_names_with_commas():
    args = _build_arg_parser().parse_args([])

    assert args.methods is None
    assert args.method is None
    assert _parse_methods(args.methods, args.method) == DEFAULT_METHODS
