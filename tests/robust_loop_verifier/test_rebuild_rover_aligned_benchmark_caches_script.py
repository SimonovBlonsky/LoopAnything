import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "robust_loop_verification_scripts"
    / "rebuild_rover_aligned_benchmark_caches.sh"
)


def test_rebuild_rover_aligned_benchmark_caches_dry_run_lists_exact_target_sequences():
    result = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    expected_sequences = {
        "handheld_escalator00",
        "handheld_room00",
        "handheld_room01",
        "ugv_campus01",
        "ugv_parking01",
        "Offroad02_beta",
        "Offroad05_beta",
        "eee_01",
        "eee_02",
        "nya_02",
    }
    assert result.stdout.count("[preprocess]") == len(expected_sequences)
    for sequence in expected_sequences:
        assert sequence in result.stdout
    assert "run-cache" not in result.stdout
    assert "build_da3_geometry_annotation" not in result.stdout
    assert "--max-gt-delta-sec 0.1" in result.stdout
    assert "[validate] source loop datasets" in result.stdout
    assert "[validate] rebuilt camera-frame caches" in result.stdout


def test_rebuild_rover_aligned_benchmark_caches_help_documents_scope():
    result = subprocess.run(
        ["bash", str(SCRIPT), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "10 frozen benchmark sequences" in result.stdout
    assert "--dry-run" in result.stdout
    assert "--check-only" in result.stdout
    assert "does not run DA3" in result.stdout
