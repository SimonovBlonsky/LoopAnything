from typer.testing import CliRunner

from robust_loop_verifier.cli import app
from robust_loop_verifier.io import write_yaml


def _combined_output(result):
    return result.stdout + result.stderr


def _pose_at_x(x):
    return [
        1.0,
        0.0,
        0.0,
        x,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]


def _write_tiny_sequence_cache(cache_dir, keyframe_count=6):
    import json

    cache_dir.mkdir(parents=True)
    (cache_dir / "images").mkdir()
    (cache_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_name": "unit",
                "platform": "tiny",
                "sequence_name": "sequence",
                "keyframe_count": keyframe_count,
            }
        ),
        encoding="utf-8",
    )
    with (cache_dir / "keyframes.jsonl").open("w", encoding="utf-8") as handle:
        for idx in range(keyframe_count):
            image_path = f"images/{idx:06d}.png"
            (cache_dir / image_path).write_bytes(b"not used by mock backends")
            handle.write(
                json.dumps(
                    {
                        "idx": idx,
                        "timestamp": float(idx),
                        "image_path": image_path,
                        "odom_pose": _pose_at_x(float(idx)),
                        "gt_pose": _pose_at_x(0.1 if idx == keyframe_count - 1 else float(idx)),
                    }
                )
            )
            handle.write("\n")
    with (cache_dir / "positives.jsonl").open("w", encoding="utf-8") as handle:
        for idx in range(keyframe_count):
            positives = [0] if idx == keyframe_count - 1 else []
            handle.write(json.dumps({"query_idx": idx, "positive_indices": positives}))
            handle.write("\n")


def test_cli_help():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "preprocess-fusionportable" in result.stdout
    assert "run-mock" in result.stdout
    assert "run-cache" in result.stdout


def test_preprocess_fusionportable_help():
    result = CliRunner().invoke(app, ["preprocess-fusionportable", "--help"])
    assert result.exit_code == 0
    assert "--config" in result.stdout
    assert "--raw-dir" in result.stdout
    assert "--gt-trajectory-file" in result.stdout


def test_run_mock_success(tmp_path):
    output_root = tmp_path / "mock-run"

    result = CliRunner().invoke(app, ["run-mock", "--output-root", str(output_root)])

    assert result.exit_code == 0
    assert "candidate_count=8" in result.stdout
    assert (output_root / "candidate_records.jsonl").is_file()
    assert (output_root / "metrics.json").is_file()
    assert (output_root / "metrics.md").is_file()


def test_run_mock_non_empty_output_root_fails_cleanly(tmp_path):
    output_root = tmp_path / "mock-run"
    output_root.mkdir()
    (output_root / "existing.txt").write_text("existing", encoding="utf-8")

    result = CliRunner().invoke(app, ["run-mock", "--output-root", str(output_root)])
    output = _combined_output(result)

    assert result.exit_code != 0
    assert "run_root must be empty or absent" in output
    assert "Traceback" not in output
    assert not isinstance(result.exception, ValueError)


def test_run_cache_help():
    result = CliRunner().invoke(app, ["run-cache", "--help"])
    assert result.exit_code == 0
    assert "--config" in result.stdout
    assert "--sequence-cache" in result.stdout
    assert "--output-root" in result.stdout
    assert "--query-limit" in result.stdout
    assert "--backend" in result.stdout


def test_run_cache_mock_success(tmp_path):
    sequence_cache = tmp_path / "cache"
    output_root = tmp_path / "run"
    config_path = tmp_path / "config.yaml"
    _write_tiny_sequence_cache(sequence_cache)
    write_yaml(
        config_path,
        {
            "dataset_name": "unit",
            "platform": "tiny",
            "input_root": str(tmp_path / "input"),
            "output_root": str(tmp_path / "cache-root"),
            "gt_root": str(tmp_path / "gt"),
            "positive_radius_m": 0.5,
            "recent_exclusion_keyframes": 1,
            "retrieval_top_k_main": 2,
            "retrieval_top_k_ablations": [1],
            "support_window": 4,
            "support_count": 1,
            "min_support_baseline_m": 0.3,
            "pgo_noise": {
                "prior_sigmas": [0.01, 0.01, 0.01, 0.1, 0.1, 0.1],
                "odom_sigmas": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                "loop_sigmas": [0.1, 0.1, 0.1, 1.0, 1.0, 1.0],
            },
            "da3": {
                "process_res": 504,
                "ref_view_strategy": "first",
            },
        },
    )

    result = CliRunner().invoke(
        app,
        [
            "run-cache",
            "--config",
            str(config_path),
            "--sequence-cache",
            str(sequence_cache),
            "--output-root",
            str(output_root),
            "--query-limit",
            "6",
            "--backend",
            "mock",
        ],
    )

    assert result.exit_code == 0
    assert "candidate_count=" in result.stdout
    assert (output_root / "candidate_records.jsonl").is_file()
    assert (output_root / "metrics.json").is_file()
