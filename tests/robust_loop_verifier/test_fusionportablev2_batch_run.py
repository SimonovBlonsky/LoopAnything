import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PY_SCRIPT = (
    ROOT
    / "robust_loop_verification_scripts"
    / "fusionportablev2_robust_loop_verifier_batch_run.py"
)
SH_SCRIPT = (
    ROOT
    / "robust_loop_verification_scripts"
    / "fusionportablev2_robust_loop_verifier_batch_run.sh"
)


def _write_ate_sequence(root: Path, platform: str, sequence_name: str, rmse: float) -> None:
    raw = root / platform / sequence_name / "raw"
    raw.mkdir(parents=True)
    (raw / "evo_ape_summary.txt").write_text(f"rmse\t{rmse}\n", encoding="utf-8")
    (raw / "trajectory_keyframes.txt").write_text(
        "1.0 0 0 0 0 0 0 1\n",
        encoding="utf-8",
    )


def _write_raw_sequence(root: Path, platform: str, sequence_name: str) -> None:
    (root / platform / sequence_name / "raw").mkdir(parents=True)


def _write_gt_sequence(root: Path, platform: str, sequence_name: str) -> None:
    sequence_root = root / platform / sequence_name
    sequence_root.mkdir(parents=True)
    (sequence_root / f"{sequence_name}.txt").write_text(
        "1.0 0 0 0 0 0 0 1\n",
        encoding="utf-8",
    )


def _write_cache_manifest(cache_dir: Path, keyframe_count: int) -> None:
    cache_dir.mkdir(parents=True)
    (cache_dir / "manifest.json").write_text(
        json.dumps({"keyframe_count": keyframe_count}),
        encoding="utf-8",
    )
    (cache_dir / "keyframes.jsonl").write_text("", encoding="utf-8")
    (cache_dir / "positives.jsonl").write_text("", encoding="utf-8")


def test_fusionportablev2_batch_run_script_help_mentions_overwrite_dataset():
    result = subprocess.run(
        ["bash", str(SH_SCRIPT), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "fusionportablev2_robust_loop_verifier_batch_run.sh" in result.stdout
    assert "--overwrite_dataset" in result.stdout
    assert "--support-ensemble" in result.stdout


def test_fusionportablev2_batch_run_dry_run_selects_handheld_and_ugv(tmp_path):
    loop_root = tmp_path / "fusionportable_loop_dataset"
    gt_root = tmp_path / "gt"
    cache_root = tmp_path / "cache"
    output_base = tmp_path / "runs"
    batch_root = tmp_path / "batch"

    _write_ate_sequence(loop_root, "handheld", "handheld_escalator00", 0.09)
    _write_ate_sequence(loop_root, "handheld", "handheld_underground00", 1.20)
    _write_raw_sequence(loop_root, "ugv", "ugv_campus00")
    _write_raw_sequence(loop_root, "ugv", "ugv_parking00")
    _write_gt_sequence(gt_root, "ugv", "ugv_campus00")
    _write_gt_sequence(gt_root, "ugv", "ugv_parking00")

    existing_cache = cache_root / "FusionPortableV2" / "handheld" / "handheld_escalator00"
    _write_cache_manifest(existing_cache, keyframe_count=12)

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
            "--platforms",
            "handheld,ugv",
            "--ugv-sequences",
            "ugv_campus00,ugv_parking00",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((batch_root / "batch_summary.json").read_text(encoding="utf-8"))
    rows = summary["sequences"]
    assert [(row["platform"], row["sequence_name"]) for row in rows] == [
        ("handheld", "handheld_escalator00"),
        ("ugv", "ugv_campus00"),
        ("ugv", "ugv_parking00"),
    ]
    handheld = rows[0]
    assert handheld["ate_rmse_m"] == 0.09
    assert handheld["gt_label_source"] == "aster_slam_trajectory_keyframes"
    assert handheld["preprocess_status"] == "skipped_existing"
    assert handheld["query_limit"] == 12
    assert rows[1]["gt_label_source"] == "aster_slam_trajectory_keyframes"
    assert rows[1]["gt_trajectory_file"].endswith(
        "fusionportable_loop_dataset/ugv/ugv_campus00/raw/trajectory_keyframes.txt"
    )
    assert rows[1]["preprocess_status"] == "pending"
    assert rows[2]["gt_label_source"] == "aster_slam_trajectory_keyframes"
    assert rows[2]["preprocess_status"] == "pending"
    assert summary["overwrite_dataset"] is False
    assert summary["support_ensemble"] is False
    assert summary["config"].endswith("fusionportablev2_handheld.yaml")


def test_fusionportablev2_batch_run_overwrite_dataset_forces_preprocess(tmp_path):
    loop_root = tmp_path / "fusionportable_loop_dataset"
    gt_root = tmp_path / "gt"
    cache_root = tmp_path / "cache"
    batch_root = tmp_path / "batch"

    _write_ate_sequence(loop_root, "handheld", "handheld_escalator00", 0.09)
    existing_cache = cache_root / "FusionPortableV2" / "handheld" / "handheld_escalator00"
    _write_cache_manifest(existing_cache, keyframe_count=12)

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
            "--platforms",
            "handheld",
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


def test_fusionportablev2_batch_run_support_ensemble_is_explicit_opt_in(tmp_path):
    loop_root = tmp_path / "fusionportable_loop_dataset"
    gt_root = tmp_path / "gt"
    cache_root = tmp_path / "cache"
    batch_root = tmp_path / "batch"

    _write_ate_sequence(loop_root, "handheld", "handheld_escalator00", 0.09)
    existing_cache = cache_root / "FusionPortableV2" / "handheld" / "handheld_escalator00"
    _write_cache_manifest(existing_cache, keyframe_count=12)

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
            "--platforms",
            "handheld",
            "--dry-run",
            "--support-ensemble",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((batch_root / "batch_summary.json").read_text(encoding="utf-8"))
    assert summary["support_ensemble"] is True
