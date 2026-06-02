import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PY_SCRIPT = ROOT / "robust_loop_verification_scripts" / "ntu_viral_robust_loop_verifier_batch_run.py"


def test_ntu_viral_batch_run_uses_raw_trajectory_keyframes_for_labels(tmp_path):
    loop_root = tmp_path / "ntu_viral_loop_dataset"
    gt_root = tmp_path / "processed_gt_tum"
    cache_root = tmp_path / "cache"
    batch_root = tmp_path / "batch"
    raw_dir = loop_root / "NTU-VIRAL" / "eee_01" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "trajectory_keyframes.txt").write_text(
        "1.0 0 0 0 0 0 0 1\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    result = subprocess.run(
        [
            str(PY_SCRIPT),
            "--repo-root",
            str(ROOT),
            "--loop-dataset-root",
            str(loop_root),
            "--gt-data-root",
            str(gt_root),
            "--cache-root",
            str(cache_root),
            "--batch-output-root",
            str(batch_root),
            "--sequences",
            "eee_01",
            "--run-id",
            "unit",
            "--dry-run",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((batch_root / "batch_summary.json").read_text(encoding="utf-8"))
    row = summary["sequences"][0]
    assert row["gt_label_source"] == "aster_slam_trajectory_keyframes"
    assert row["gt_trajectory_file"] == str(raw_dir / "trajectory_keyframes.txt")
