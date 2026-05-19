import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PY_SCRIPT = ROOT / "robust_loop_verification_scripts" / "geode_robust_loop_verifier_batch_run.py"
SH_SCRIPT = ROOT / "robust_loop_verification_scripts" / "geode_robust_loop_verifier_batch_run.sh"


def _write_raw_sequence(root: Path, platform: str, sequence_name: str) -> None:
    raw = root / platform / sequence_name / "raw"
    raw.mkdir(parents=True)
    (raw / "trajectory_keyframes.txt").write_text(
        "1.0 0 0 0 0 0 0 1\n",
        encoding="utf-8",
    )


def _write_gt_sequence(gt_root: Path, filename: str) -> None:
    gt_root.mkdir(parents=True)
    (gt_root / filename).write_text("1.0 0 0 0 0 0 0 1\n", encoding="utf-8")


def _write_cache_manifest(cache_dir: Path, keyframe_count: int) -> None:
    cache_dir.mkdir(parents=True)
    (cache_dir / "manifest.json").write_text(
        json.dumps({"keyframe_count": keyframe_count}),
        encoding="utf-8",
    )
    (cache_dir / "keyframes.jsonl").write_text("", encoding="utf-8")
    (cache_dir / "positives.jsonl").write_text("", encoding="utf-8")


def test_geode_batch_run_script_help_mentions_support_ensemble_and_overwrite():
    result = subprocess.run(
        ["bash", str(SH_SCRIPT), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "geode_robust_loop_verifier_batch_run.sh" in result.stdout
    assert "--overwrite_dataset" in result.stdout
    assert "--no-support-ensemble" in result.stdout


def test_geode_batch_run_dry_run_defaults_to_offroad05_beta_with_support_ensemble(tmp_path):
    loop_root = tmp_path / "geode_loop_dataset"
    gt_root = tmp_path / "data" / "offroad"
    cache_root = tmp_path / "cache"
    output_base = tmp_path / "runs"
    batch_root = tmp_path / "batch"

    _write_raw_sequence(loop_root, "Offroad", "Offroad05_beta")
    _write_gt_sequence(gt_root, "Offroad5.txt")
    _write_cache_manifest(
        cache_root / "GEODE" / "Offroad" / "Offroad05_beta",
        keyframe_count=12,
    )

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
            "--output-base",
            str(output_base),
            "--batch-output-root",
            str(batch_root),
            "--run-id",
            "unit",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((batch_root / "batch_summary.json").read_text(encoding="utf-8"))
    assert summary["dataset_name"] == "GEODE"
    assert summary["support_ensemble"] is True
    assert summary["max_gt_delta_sec"] == 0.1
    assert summary["config"].endswith("geode_offroad_support_ensemble.yaml")
    rows = summary["sequences"]
    assert len(rows) == 1
    row = rows[0]
    assert row["platform"] == "Offroad"
    assert row["sequence_name"] == "Offroad05_beta"
    assert row["gt_trajectory_file"].endswith("Offroad5.txt")
    assert row["gt_label_source"] == "external_gt_trajectory"
    assert row["preprocess_status"] == "skipped_existing"
    assert row["query_limit"] == 12


def test_geode_batch_run_overwrite_dataset_forces_preprocess(tmp_path):
    loop_root = tmp_path / "geode_loop_dataset"
    gt_root = tmp_path / "data" / "offroad"
    cache_root = tmp_path / "cache"
    batch_root = tmp_path / "batch"

    _write_raw_sequence(loop_root, "Offroad", "Offroad05_beta")
    _write_gt_sequence(gt_root, "Offroad5.txt")
    _write_cache_manifest(
        cache_root / "GEODE" / "Offroad" / "Offroad05_beta",
        keyframe_count=12,
    )

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
            "--run-id",
            "unit",
            "--dry-run",
            "--overwrite_dataset",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((batch_root / "batch_summary.json").read_text(encoding="utf-8"))
    row = summary["sequences"][0]
    assert summary["overwrite_dataset"] is True
    assert row["preprocess_status"] == "pending"
