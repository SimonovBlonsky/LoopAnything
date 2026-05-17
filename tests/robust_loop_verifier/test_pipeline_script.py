import json
import os
import subprocess
import sys
from pathlib import Path

from robust_loop_verifier.io import write_yaml


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
            (cache_dir / image_path).write_bytes(b"not used by mock backend")
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


def _write_config(path, tmp_path):
    write_yaml(
        path,
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


def test_run_robust_loop_verifier_pipeline_script_prints_metrics(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "robust_loop_verification_scripts" / "run_robust_loop_verifier_pipeline.sh"
    cache_dir = tmp_path / "cache"
    output_root = tmp_path / "run"
    config_path = tmp_path / "config.yaml"
    _write_tiny_sequence_cache(cache_dir)
    _write_config(config_path, tmp_path)

    env = {
        **os.environ,
        "PYTHON_BIN": sys.executable,
        "CONFIG": str(config_path),
        "BACKEND": "mock",
        "QUERY_LIMIT": "6",
        "OUTPUT_ROOT": str(output_root),
    }
    result = subprocess.run(
        [str(script), str(cache_dir)],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "candidate_count=" in result.stdout
    assert "| method | AP | MR@100P |" in result.stdout
    assert (output_root / "candidate_records.jsonl").is_file()
    assert (output_root / "metrics.json").is_file()
    assert (output_root / "metrics.md").is_file()
