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


def _write_candidate_records(run_root: Path) -> None:
    run_root.mkdir(parents=True)
    records = [
        {
            "query_idx": 10,
            "candidate_idx": 1,
            "label": False,
            "score_salad": 0.99,
            "trajectory_deformation_rmse": 2.0,
            "pgo_error_after": 6.0,
        },
        {
            "query_idx": 10,
            "candidate_idx": 2,
            "label": True,
            "score_salad": 0.70,
            "trajectory_deformation_rmse": 0.1,
            "pgo_error_after": 0.1,
        },
        {
            "query_idx": 11,
            "candidate_idx": 3,
            "label": False,
            "score_salad": 0.95,
            "trajectory_deformation_rmse": 1.5,
            "pgo_error_after": 4.0,
        },
        {
            "query_idx": 11,
            "candidate_idx": 4,
            "label": True,
            "score_salad": 0.60,
            "trajectory_deformation_rmse": 0.2,
            "pgo_error_after": 0.2,
        },
    ]
    with (run_root / "candidate_records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


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


def test_geode_batch_run_resolves_nested_gt_layout(tmp_path):
    loop_root = tmp_path / "geode_loop_dataset"
    gt_root = tmp_path / "data" / "offroad"
    cache_root = tmp_path / "cache"
    batch_root = tmp_path / "batch"

    _write_raw_sequence(loop_root, "Offroad", "Offroad01_beta")
    _write_gt_sequence(gt_root / "Offroad1", "Offroad1.txt")
    _write_cache_manifest(
        cache_root / "GEODE" / "Offroad" / "Offroad01_beta",
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
            "--sequences",
            "Offroad01_beta",
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
    row = summary["sequences"][0]
    assert row["sequence_name"] == "Offroad01_beta"
    assert row["gt_trajectory_file"].endswith("Offroad1/Offroad1.txt")


def test_geode_batch_merges_selected_score_sweep_metrics(tmp_path):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("geode_batch", PY_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["geode_batch"] = module
    spec.loader.exec_module(module)

    plan = module.SequencePlan(
        platform="Offroad",
        sequence_name="Offroad05_beta",
        raw_dir=tmp_path / "raw",
        gt_trajectory_file=tmp_path / "Offroad5.txt",
        gt_label_source="external_gt_trajectory",
        cache_dir=tmp_path / "cache",
        run_root=tmp_path / "run",
    )
    _write_candidate_records(plan.run_root)
    metrics = {
        "SALAD score only": {
            "AP": 0.1,
            "MR@100P": 0.2,
        }
    }

    rows, merged_metrics = module._metrics_with_selected_score_sweep(plan, metrics)
    methods = {row["method"] for row in rows}

    assert "SALAD score only" in methods
    assert "absolute_graph:def=0.5,res=0.25" in methods
    assert "query_gate_graph:def=0.5,res=0.25,margin=0" in methods
    assert "percentile_product:res,def" in methods
    assert "z_fusion:salad=0,def=4,res=2" in methods
    assert "absolute_graph:def=0.5,res=0.25" in merged_metrics
    assert "query_gate_graph:def=0.5,res=0.25,margin=0" in merged_metrics
    assert "percentile_product:res,def" in merged_metrics
    assert "z_fusion:salad=0,def=4,res=2" in merged_metrics
    assert (plan.run_root / "score_sweep" / "score_sweep.json").is_file()
    assert (plan.run_root / "score_sweep" / "score_sweep.md").is_file()
