import json
from pathlib import Path

import pytest


def _write_ate_summary(path: Path, rmse: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "APE w.r.t. translation part (m)",
                "       max\t0.900000",
                f"      rmse\t{rmse:.6f}",
                "       std\t0.050000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_discover_aster_slam_ate_sequences_filters_by_rmse(tmp_path):
    from robust_loop_verifier.batch import discover_aster_slam_ate_sequences

    root = tmp_path / "fusionportable_loop_dataset"
    _write_ate_summary(root / "handheld" / "handheld_ok" / "raw" / "evo_ape_summary.txt", 0.05)
    _write_ate_summary(root / "handheld" / "handheld_bad" / "raw" / "evo_ape_summary.txt", 0.20)
    _write_ate_summary(root / "legged" / "legged_ok" / "raw" / "evo_ape_summary.txt", 0.099)
    _write_ate_summary(root / "ugv" / "ugv_ok" / "raw" / "evo_ape_summary.txt", 0.01)

    selected = discover_aster_slam_ate_sequences(
        root,
        platforms=("handheld", "legged"),
        ate_rmse_threshold_m=0.1,
    )

    assert [(item.platform, item.sequence_name, item.ate_rmse_m) for item in selected] == [
        ("handheld", "handheld_ok", 0.05),
        ("legged", "legged_ok", 0.099),
    ]


def test_discover_aster_slam_ate_sequences_reports_missing_rmse(tmp_path):
    from robust_loop_verifier.batch import discover_aster_slam_ate_sequences

    root = tmp_path / "fusionportable_loop_dataset"
    path = root / "handheld" / "broken" / "raw" / "evo_ape_summary.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("APE summary without rmse\n", encoding="utf-8")

    with pytest.raises(ValueError, match="rmse"):
        discover_aster_slam_ate_sequences(root, platforms=("handheld",), ate_rmse_threshold_m=0.1)


def test_average_score_sweep_rows_groups_by_method():
    from robust_loop_verifier.batch import average_score_sweep_rows

    rows = [
        {
            "platform": "handheld",
            "sequence": "a",
            "method": "SALAD score only",
            "AP": 0.8,
            "MR@100P": 0.1,
        },
        {
            "platform": "handheld",
            "sequence": "b",
            "method": "SALAD score only",
            "AP": 0.6,
            "MR@100P": 0.3,
        },
        {
            "platform": "legged",
            "sequence": "c",
            "method": "PGO residual only",
            "AP": 1.0,
            "MR@100P": 0.5,
        },
    ]

    averages = average_score_sweep_rows(rows)

    assert averages == [
        {
            "method": "PGO residual only",
            "sequence_count": 1,
            "average_AP": 1.0,
            "average_MR@100P": 0.5,
        },
        {
            "method": "SALAD score only",
            "sequence_count": 2,
            "average_AP": 0.7,
            "average_MR@100P": 0.2,
        },
    ]


def test_read_batch_summary_run_roots(tmp_path):
    from robust_loop_verifier.batch import read_batch_summary_run_roots

    summary = {
        "sequences": [
            {
                "platform": "handheld",
                "sequence_name": "a",
                "run_root": str(tmp_path / "runs" / "a"),
                "ate_rmse_m": 0.05,
            },
            {
                "platform": "legged",
                "sequence_name": "b",
                "run_root": str(tmp_path / "runs" / "b"),
                "ate_rmse_m": 0.08,
            },
        ]
    }
    path = tmp_path / "batch_summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")

    records = read_batch_summary_run_roots(path)

    assert [(record.platform, record.sequence_name, record.run_root) for record in records] == [
        ("handheld", "a", tmp_path / "runs" / "a"),
        ("legged", "b", tmp_path / "runs" / "b"),
    ]
