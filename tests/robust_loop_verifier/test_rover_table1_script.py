from __future__ import annotations

import subprocess
from pathlib import Path


def test_rover_aligned_table1_script_dry_run_lists_expected_commands():
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "robust_loop_verification_scripts" / "run_rover_aligned_table1_experiment.sh"

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--benchmark-root",
            "workspace/rover_aligned_benchmark/benchmark_v1",
            "--device",
            "cpu",
            "--backend",
            "mock",
            "--dry-run",
        ],
        cwd=repo_root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
    assert "score_rover_aligned_benchmark.py" in result.stdout
    assert "--method netvlad" in result.stdout
    assert "--method salad" in result.stdout
    assert "--method verifier" in result.stdout
    assert "evaluate_rover_aligned_benchmark.py" in result.stdout
    assert "DBoW2=scores/dbow2.jsonl" in result.stdout
    assert "NetVLAD=scores/netvlad.jsonl" in result.stdout
    assert "SALAD=scores/salad.jsonl" in result.stdout
    assert "ROVER-like=scores/rover_like.jsonl" in result.stdout
    assert "LoopAnything=scores/loopanything.jsonl" in result.stdout


def test_rover_aligned_table1_script_dry_run_can_include_loftr(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "robust_loop_verification_scripts" / "run_rover_aligned_table1_experiment.sh"
    benchmark_root = tmp_path / "benchmark"
    (benchmark_root / "scores").mkdir(parents=True)
    (benchmark_root / "scores" / "dbow2.jsonl").write_text("", encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--benchmark-root",
            str(benchmark_root),
            "--device",
            "cpu",
            "--backend",
            "mock",
            "--include-loftr",
            "--loftr-python",
            "/tmp/loftr-python",
            "--loftr-root",
            "/tmp/loftr-root",
            "--loftr-ckpt",
            "/tmp/loftr.ckpt",
            "--dry-run",
        ],
        cwd=repo_root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
    assert "score_loftr_rover_aligned_benchmark.py" in result.stdout
    assert "--loftr-root /tmp/loftr-root" in result.stdout
    assert "--ckpt-path /tmp/loftr.ckpt" in result.stdout
    assert "LoFTR=scores/loftr.jsonl" in result.stdout


def test_rover_aligned_table1_script_dry_run_can_include_dust3r(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "robust_loop_verification_scripts" / "run_rover_aligned_table1_experiment.sh"
    benchmark_root = tmp_path / "benchmark"
    (benchmark_root / "scores").mkdir(parents=True)
    (benchmark_root / "scores" / "dbow2.jsonl").write_text("", encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--benchmark-root",
            str(benchmark_root),
            "--device",
            "cpu",
            "--backend",
            "mock",
            "--include-dust3r",
            "--dust3r-python",
            "/tmp/dust3r-python",
            "--dust3r-root",
            "/tmp/dust3r-root",
            "--dust3r-ckpt",
            "/tmp/dust3r.pth",
            "--dry-run",
        ],
        cwd=repo_root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
    assert "score_dust3r_rover_aligned_benchmark.py" in result.stdout
    assert "--dust3r-root /tmp/dust3r-root" in result.stdout
    assert "--ckpt-path /tmp/dust3r.pth" in result.stdout
    assert "DUSt3R=scores/dust3r.jsonl" in result.stdout


def test_rover_aligned_table1_script_dry_run_can_include_mast3r(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "robust_loop_verification_scripts" / "run_rover_aligned_table1_experiment.sh"
    benchmark_root = tmp_path / "benchmark"
    (benchmark_root / "scores").mkdir(parents=True)
    (benchmark_root / "scores" / "dbow2.jsonl").write_text("", encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--benchmark-root",
            str(benchmark_root),
            "--device",
            "cpu",
            "--backend",
            "mock",
            "--include-mast3r",
            "--mast3r-python",
            "/tmp/mast3r-python",
            "--mast3r-root",
            "/tmp/mast3r-root",
            "--mast3r-ckpt",
            "/tmp/mast3r.pth",
            "--dry-run",
        ],
        cwd=repo_root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
    assert "score_mast3r_rover_aligned_benchmark.py" in result.stdout
    assert "--mast3r-root /tmp/mast3r-root" in result.stdout
    assert "--ckpt-path /tmp/mast3r.pth" in result.stdout
    assert "MAST3R=scores/mast3r.jsonl" in result.stdout


def test_rover_aligned_table1_script_requires_existing_dbow2_score(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "robust_loop_verification_scripts" / "run_rover_aligned_table1_experiment.sh"
    benchmark_root = tmp_path / "benchmark"
    benchmark_root.mkdir()

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--benchmark-root",
            str(benchmark_root),
            "--dry-run",
        ],
        cwd=repo_root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode != 0
    assert "missing frozen DBoW2 score file" in result.stderr
