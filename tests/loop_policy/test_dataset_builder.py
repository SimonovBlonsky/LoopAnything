import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import loop_policy.dataset_builder as dataset_builder
from loop_policy.da3_runner import Da3GroupResult, Da3Runner, MockDa3Runner
from loop_policy.dataset_builder import (
    _pose_map,
    audit_causal_leakage,
    build_sequence_cache,
    raw_dir_for_sequence,
    write_root_manifests,
)
from loop_policy.geometry import make_transform
from loop_policy.retrieval import PrecomputedDescriptorExtractor
from loop_policy.schema import KeyframeRecord, LoopPolicyDatasetConfig, PoseRecord


def _descriptor_extractor_for(raw_dir):
    image_paths = [str(raw_dir / "keyframe_images" / f"{idx:06d}.jpg") for idx in range(6)]
    descriptors_by_path = {
        image_paths[0]: np.array([1.0, 0.0]),
        image_paths[1]: np.array([0.0, 1.0]),
        image_paths[2]: np.array([0.8, 0.2]),
        image_paths[3]: np.array([0.7, 0.3]),
        image_paths[4]: np.array([0.1, 0.9]),
        image_paths[5]: np.array([1.0, 0.0]),
    }
    return PrecomputedDescriptorExtractor(descriptors_by_path)


class DegenerateDa3Runner(Da3Runner):
    def run(self, groups):
        results = []
        for group in groups:
            view_count = len(group.keyframe_indices)
            poses = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], view_count, axis=0)
            results.append(
                Da3GroupResult(
                    keyframe_indices=list(group.keyframe_indices),
                    camera_poses=poses,
                    depth_conf_medians=[1.0] * view_count,
                    valid_depth_ratios=[1.0] * view_count,
                )
            )
        return results


def test_build_sequence_cache_writes_causal_outputs(synthetic_raw_sequence, tmp_path):
    config = LoopPolicyDatasetConfig(
        dataset_root=str(synthetic_raw_sequence.parents[2]),
        output_root=str(tmp_path / "loop_policy_dataset"),
        sequences=("handheld_room01",),
        exclude_recent_keyframes=1,
        support_window=4,
        min_support_baseline_m=0.5,
        retrieval_pool_size=3,
        runtime_top_k=2,
    )

    summary = build_sequence_cache(
        raw_dir=synthetic_raw_sequence,
        config=config,
        descriptor_extractor=_descriptor_extractor_for(synthetic_raw_sequence),
        da3_runner=MockDa3Runner(),
    )

    sequence_dir = tmp_path / "loop_policy_dataset" / "handheld" / "handheld_room01"
    assert summary["sequence"] == "handheld_room01"
    assert (sequence_dir / "sequence_index.json").is_file()
    assert (sequence_dir / "descriptors.npz").is_file()
    assert (sequence_dir / "retrieval_topk.jsonl").is_file()
    assert (sequence_dir / "support_selection.jsonl").is_file()
    assert (sequence_dir / "candidate_features.jsonl").is_file()
    assert (sequence_dir / "sequence_summary.json").is_file()

    feature_lines = (sequence_dir / "candidate_features.jsonl").read_text().splitlines()
    assert len(feature_lines) > 0
    assert summary["retrieval_candidate_count"] > 0
    assert summary["valid_support_count"] >= len(feature_lines)
    assert summary["da3_success_count"] == len(feature_lines)
    assert summary["causal_leakage_audit_passed"] is True

    for line in feature_lines:
        record = json.loads(line)
        assert record["causal"] is True
        assert record["candidate_timestamp"] < record["query_timestamp"]
        for timestamp in record["selected_support_timestamps"]:
            assert timestamp < record["query_timestamp"]


def test_build_sequence_cache_writes_visualization_record_tree(
    synthetic_raw_sequence,
    tmp_path,
):
    config = LoopPolicyDatasetConfig(
        dataset_root=str(synthetic_raw_sequence.parents[2]),
        output_root=str(tmp_path / "loop_policy_dataset"),
        sequences=("handheld_room01",),
        exclude_recent_keyframes=1,
        support_window=4,
        min_support_baseline_m=0.5,
        retrieval_pool_size=3,
        runtime_top_k=2,
        write_visualization_records=True,
    )

    build_sequence_cache(
        raw_dir=synthetic_raw_sequence,
        config=config,
        descriptor_extractor=_descriptor_extractor_for(synthetic_raw_sequence),
        da3_runner=MockDa3Runner(),
    )

    sequence_dir = tmp_path / "loop_policy_dataset" / "handheld" / "handheld_room01"
    record = json.loads(
        (sequence_dir / "candidate_features.jsonl").read_text().splitlines()[0]
    )
    label_path = [
        "new_precondition_valid"
        if record["precondition_valid"]
        else "new_precondition_invalid",
        "sim3_quality_good"
        if record["labels"]["sim3_quality_good"]
        else "sim3_quality_bad",
        "odom_consistent_loose"
        if record["labels"]["odom_consistent_loose"]
        else "odom_consistent_not_loose",
        "safe_loop_factor_v1"
        if record["safe_loop_factor_v1"]
        else "safe_loop_factor_negative",
    ]
    record_dir = (
        sequence_dir
        / "visual_records"
        / Path(*label_path)
        / f"q{record['query_idx']:06d}_c{record['candidate_idx']:06d}_s"
        f"{record['selected_support_indices'][0]:06d}"
    )

    assert (record_dir / "query.png").is_file()
    assert (record_dir / "candidate.png").is_file()
    assert (record_dir / f"support_{record['selected_support_indices'][0]:06d}.png").is_file()
    payload = json.loads((record_dir / "record.json").read_text())
    assert payload["feature_record"]["query_idx"] == record["query_idx"]
    assert payload["feature_record"]["candidate_idx"] == record["candidate_idx"]
    assert payload["image_paths"]["query"].endswith(f"{record['query_idx']:06d}.jpg")
    assert payload["image_paths"]["candidate"].endswith(f"{record['candidate_idx']:06d}.jpg")


def test_build_sequence_cache_emits_negative_record_for_degenerate_da3(
    synthetic_raw_sequence,
    tmp_path,
):
    config = LoopPolicyDatasetConfig(
        dataset_root=str(synthetic_raw_sequence.parents[2]),
        output_root=str(tmp_path / "loop_policy_dataset"),
        sequences=("handheld_room01",),
        exclude_recent_keyframes=1,
        support_window=4,
        min_support_baseline_m=0.5,
        retrieval_pool_size=3,
        runtime_top_k=2,
    )

    summary = build_sequence_cache(
        raw_dir=synthetic_raw_sequence,
        config=config,
        descriptor_extractor=_descriptor_extractor_for(synthetic_raw_sequence),
        da3_runner=DegenerateDa3Runner(),
    )

    feature_path = (
        tmp_path
        / "loop_policy_dataset"
        / "handheld"
        / "handheld_room01"
        / "candidate_features.jsonl"
    )
    feature_text = feature_path.read_text()
    assert "NaN" not in feature_text
    assert "Infinity" not in feature_text
    assert "-Infinity" not in feature_text
    records = [json.loads(line) for line in feature_text.splitlines()]

    assert records
    assert summary["negative_reasons"]["invalid_support_baseline"] > 0
    rejected_records = [
        record for record in records if record["negative_reason"] == "invalid_support_baseline"
    ]
    assert rejected_records
    for record in rejected_records:
        assert record["precondition_valid"] is False
        assert record["safe_loop_factor_v1"] is False
        assert np.isfinite(record["x_geom"]).all()
        assert np.isfinite(list(record["metrics"].values())).all()


def test_build_sequence_cache_resolves_relative_image_paths(synthetic_raw_sequence, tmp_path):
    relative_lines = []
    for idx in range(6):
        relative_lines.append(
            json.dumps(
                {
                    "keyframe_idx": idx,
                    "timestamp": 100.0 + idx,
                    "trajectory_idx": idx,
                    "image_path": f"keyframe_images/{idx:06d}.jpg",
                }
            )
        )
    (synthetic_raw_sequence / "keyframes_with_images.jsonl").write_text(
        "\n".join(relative_lines) + "\n",
        encoding="utf-8",
    )

    class PathCheckingDa3Runner(MockDa3Runner):
        def run(self, groups):
            for group in groups:
                for image_path in group.image_paths:
                    assert Path(image_path).is_absolute()
                    assert Path(image_path).is_file()
            return super().run(groups)

    config = LoopPolicyDatasetConfig(
        dataset_root=str(synthetic_raw_sequence.parents[2]),
        output_root=str(tmp_path / "loop_policy_dataset"),
        sequences=("handheld_room01",),
        exclude_recent_keyframes=1,
        support_window=4,
        min_support_baseline_m=0.5,
        retrieval_pool_size=3,
        runtime_top_k=2,
    )

    summary = build_sequence_cache(
        raw_dir=synthetic_raw_sequence,
        config=config,
        descriptor_extractor=_descriptor_extractor_for(synthetic_raw_sequence),
        da3_runner=PathCheckingDa3Runner(),
    )

    assert summary["causal_leakage_audit_passed"] is True


def test_audit_causal_leakage_rejects_future_candidate():
    records = [
        {
            "query_timestamp": 10.0,
            "candidate_timestamp": 11.0,
            "selected_support_timestamps": [5.0],
        }
    ]

    assert audit_causal_leakage(records) is False


def test_audit_causal_leakage_accepts_historical_candidate_and_supports():
    records = [
        {
            "query_timestamp": 10.0,
            "candidate_timestamp": 8.0,
            "selected_support_timestamps": [5.0, 7.0],
        }
    ]

    assert audit_causal_leakage(records) is True


def test_pose_map_uses_trajectory_idx_when_present():
    sequence = SimpleNamespace(
        keyframes=[
            KeyframeRecord(
                keyframe_idx=10,
                timestamp=100.0,
                image_path="10.jpg",
                trajectory_idx=2,
            ),
            KeyframeRecord(
                keyframe_idx=11,
                timestamp=101.0,
                image_path="11.jpg",
                trajectory_idx=0,
            ),
        ],
        trajectory=[
            PoseRecord(
                timestamp=101.0,
                position=(1.0, 0.0, 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            ),
            PoseRecord(
                timestamp=102.0,
                position=(2.0, 0.0, 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            ),
            PoseRecord(
                timestamp=100.0,
                position=(3.0, 0.0, 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            ),
        ],
        t_camera_lidar=np.eye(4),
    )

    poses = _pose_map(sequence)

    np.testing.assert_allclose(
        poses[10],
        make_transform(np.eye(3), np.array([3.0, 0.0, 0.0])),
    )
    np.testing.assert_allclose(
        poses[11],
        make_transform(np.eye(3), np.array([1.0, 0.0, 0.0])),
    )


def test_pose_map_rejects_invalid_trajectory_mapping():
    sequence = SimpleNamespace(
        keyframes=[
            KeyframeRecord(keyframe_idx=0, timestamp=100.0, image_path="0.jpg"),
            KeyframeRecord(keyframe_idx=1, timestamp=101.0, image_path="1.jpg"),
        ],
        trajectory=[
            PoseRecord(
                timestamp=100.0,
                position=(0.0, 0.0, 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            ),
        ],
        t_camera_lidar=np.eye(4),
    )

    with pytest.raises(ValueError, match="trajectory length must match keyframes"):
        _pose_map(sequence)

    sequence = SimpleNamespace(
        keyframes=[
            KeyframeRecord(keyframe_idx=0, timestamp=100.0, image_path="0.jpg", trajectory_idx=2),
        ],
        trajectory=[
            PoseRecord(
                timestamp=100.0,
                position=(0.0, 0.0, 0.0),
                quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            ),
        ],
        t_camera_lidar=np.eye(4),
    )

    with pytest.raises(ValueError, match="trajectory_idx out of range"):
        _pose_map(sequence)


def test_build_sequence_cache_passes_all_selected_supports_to_sim3(
    synthetic_raw_sequence,
    tmp_path,
    monkeypatch,
):
    forwarded_support_counts = []
    selected_support_counts = []
    original_align = dataset_builder.align_da3_poses_with_candidate_support_prior

    def record_support_count(*args, **kwargs):
        forwarded_support_counts.append(len(kwargs["da3_support_poses"]))
        selected_support_counts.append(len(kwargs["odom_support_poses"]))
        return original_align(*args, **kwargs)

    monkeypatch.setattr(
        dataset_builder,
        "align_da3_poses_with_candidate_support_prior",
        record_support_count,
    )
    config = LoopPolicyDatasetConfig(
        dataset_root=str(synthetic_raw_sequence.parents[2]),
        output_root=str(tmp_path / "loop_policy_dataset"),
        sequences=("handheld_room01",),
        exclude_recent_keyframes=0,
        support_window=4,
        support_count=2,
        min_support_baseline_m=0.5,
        retrieval_pool_size=3,
        runtime_top_k=2,
    )

    build_sequence_cache(
        raw_dir=synthetic_raw_sequence,
        config=config,
        descriptor_extractor=_descriptor_extractor_for(synthetic_raw_sequence),
        da3_runner=MockDa3Runner(),
    )

    assert forwarded_support_counts
    assert forwarded_support_counts == selected_support_counts
    assert max(forwarded_support_counts) == 2


def test_write_root_manifests_records_thresholds(tmp_path):
    config = LoopPolicyDatasetConfig(
        dataset_root="/dataset",
        output_root=str(tmp_path),
        sequences=("handheld_room01",),
        retrieval_pool_size=50,
        runtime_top_k=4,
    )
    summaries = [
        {
            "sequence": "handheld_room01",
            "platform": "handheld",
            "query_count": 6,
            "causal_leakage_audit_passed": True,
        }
    ]

    write_root_manifests(config, summaries)

    manifest = json.loads((tmp_path / "dataset_manifest.json").read_text())
    lines = (tmp_path / "sequence_summaries.jsonl").read_text().splitlines()
    assert manifest["schema_version"] == "loop_policy_dataset_v1"
    assert manifest["causal"] is True
    assert manifest["settings"]["runtime_top_k"] == 4
    assert json.loads(lines[0])["sequence"] == "handheld_room01"


def test_write_root_manifests_rejects_nonfinite_json(tmp_path):
    config = LoopPolicyDatasetConfig(
        dataset_root="/dataset",
        output_root=str(tmp_path),
        sequences=("handheld_room01",),
        loose_trans_thr_m=float("nan"),
    )

    with pytest.raises(ValueError, match="Out of range float values"):
        write_root_manifests(
            config,
            [
                {
                    "sequence": "handheld_room01",
                    "causal_leakage_audit_passed": True,
                }
            ],
        )


def test_raw_dir_for_sequence_uses_platform_prefix():
    root = Path("/data/datasets/FusionPortable/fusionportable_loop_dataset")

    assert raw_dir_for_sequence(root, "vehicle_campus00") == (
        root / "vehicle" / "vehicle_campus00" / "raw"
    )
    assert raw_dir_for_sequence(root, "ugv_parking00") == (
        root / "ugv" / "ugv_parking00" / "raw"
    )


def test_main_writes_failure_manifest_for_unsupported_sequence_platform(tmp_path, monkeypatch):
    def unexpected_build_sequence_cache(*args, **kwargs):
        raise AssertionError("unsupported platform should fail before building sequence cache")

    monkeypatch.setattr(dataset_builder, "build_sequence_cache", unexpected_build_sequence_cache)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataset_builder.py",
            "--dataset-root",
            "/dataset",
            "--output-root",
            str(tmp_path),
            "--sequences",
            "drone_room00",
            "--salad-checkpoint",
            str(tmp_path / "unused.ckpt"),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        dataset_builder.main()

    assert exc_info.value.code == 1
    manifest = json.loads((tmp_path / "dataset_manifest.json").read_text())
    summaries = [
        json.loads(line)
        for line in (tmp_path / "sequence_summaries.jsonl").read_text().splitlines()
    ]
    assert manifest["schema_version"] == "loop_policy_dataset_v1"
    assert manifest["sequence_count"] == 1
    assert summaries == [
        {
            "sequence": "drone_room00",
            "platform": "drone",
            "failed": True,
            "error": "ValueError(\"unsupported FusionPortable sequence platform in "
            "'drone_room00'\")",
            "causal_leakage_audit_passed": False,
        }
    ]


def test_main_writes_success_manifest_with_cli_backends(tmp_path, monkeypatch):
    constructed = {}

    class FakeDescriptorExtractor:
        def __init__(self, checkpoint, device, image_size, batch_size):
            constructed["descriptor"] = {
                "checkpoint": checkpoint,
                "device": device,
                "image_size": image_size,
                "batch_size": batch_size,
            }

    class FakeDa3Runner:
        def __init__(self, model_name, device, process_res, extrinsics_are_c2w, ref_view_strategy):
            constructed["da3"] = {
                "model_name": model_name,
                "device": device,
                "process_res": process_res,
                "extrinsics_are_c2w": extrinsics_are_c2w,
                "ref_view_strategy": ref_view_strategy,
            }

    def fake_build_sequence_cache(raw_dir, config, descriptor_extractor, da3_runner):
        constructed["build"] = {
            "raw_dir": raw_dir,
            "config": config,
            "descriptor_extractor": descriptor_extractor,
            "da3_runner": da3_runner,
        }
        return {
            "sequence": "handheld_room01",
            "platform": "handheld",
            "query_count": 2,
            "causal_leakage_audit_passed": True,
        }

    monkeypatch.setattr(dataset_builder, "DinoSaladDescriptorExtractor", FakeDescriptorExtractor)
    monkeypatch.setattr(dataset_builder, "DepthAnything3Runner", FakeDa3Runner)
    monkeypatch.setattr(dataset_builder, "build_sequence_cache", fake_build_sequence_cache)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dataset_builder.py",
            "--dataset-root",
            "/dataset",
            "--output-root",
            str(tmp_path),
            "--sequences",
            "handheld_room01",
            "--salad-checkpoint",
            str(tmp_path / "fake.ckpt"),
            "--salad-device",
            "cpu",
            "--salad-image-size",
            "112",
            "224",
            "--salad-batch-size",
            "3",
            "--da3-model",
            "fake-da3",
            "--da3-device",
            "cpu",
            "--da3-process-res",
            "256",
            "--write-visualization-records",
        ],
    )

    dataset_builder.main()

    assert constructed["descriptor"] == {
        "checkpoint": tmp_path / "fake.ckpt",
        "device": "cpu",
        "image_size": (112, 224),
        "batch_size": 3,
    }
    assert constructed["da3"] == {
        "model_name": "fake-da3",
        "device": "cpu",
        "process_res": 256,
        "extrinsics_are_c2w": False,
        "ref_view_strategy": "first",
    }
    assert constructed["build"]["raw_dir"] == (
        Path("/dataset") / "handheld" / "handheld_room01" / "raw"
    )
    assert constructed["build"]["config"].sequences == ("handheld_room01",)
    assert constructed["build"]["config"].write_visualization_records is True
    manifest = json.loads((tmp_path / "dataset_manifest.json").read_text())
    summaries = [
        json.loads(line)
        for line in (tmp_path / "sequence_summaries.jsonl").read_text().splitlines()
    ]
    assert manifest["sequence_count"] == 1
    assert summaries[0]["sequence"] == "handheld_room01"
    assert summaries[0]["causal_leakage_audit_passed"] is True
