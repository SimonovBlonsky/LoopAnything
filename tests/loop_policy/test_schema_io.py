import json
from pathlib import Path
from types import MappingProxyType

import pytest

from loop_policy.io import (
    _read_keyframes,
    load_aster_raw_sequence,
    read_jsonl,
    read_tum_trajectory,
    write_jsonl,
)
from loop_policy.schema import (
    CandidateFeatureRecord,
    KeyframeRecord,
    LoopPolicyDatasetConfig,
    RetrievalCandidate,
    RetrievalRecord,
    SupportDecision,
    dataclass_to_json_dict,
)


def test_loop_policy_dataset_config_defaults_are_causal():
    config = LoopPolicyDatasetConfig(
        dataset_root="/dataset",
        output_root="/loop_policy_dataset",
        sequences=("handheld_room01",),
    )

    assert config.causal is True
    assert config.runtime_top_k == 4
    assert config.retrieval_pool_size >= config.runtime_top_k


def test_schema_public_names_describe_dataset_builder_role():
    import loop_policy.schema as schema

    old_config_name = "Stage" + "0Config"
    old_schema_version = "stage" + "0_v1"

    assert schema.SCHEMA_VERSION == "loop_policy_dataset_v1"
    assert not hasattr(schema, old_config_name)

    source = Path(schema.__file__).read_text(encoding="utf-8")
    assert old_config_name not in source
    assert old_schema_version not in source


def test_json_roundtrip_keeps_required_audit_fields():
    record = RetrievalRecord(
        sequence="handheld_room01",
        query_idx=10,
        query_timestamp=100.0,
        causal=True,
        database_max_idx=4,
        database_max_timestamp=94.0,
        retrieval_db_size=2,
        candidates=[
            RetrievalCandidate(
                rank=1,
                keyframe_idx=4,
                timestamp=94.0,
                score=0.91,
                runtime_topk=True,
            )
        ],
    )

    payload = json.loads(json.dumps(dataclass_to_json_dict(record)))

    assert payload["causal"] is True
    assert payload["query_idx"] == 10
    assert payload["database_max_idx"] == 4
    assert payload["candidates"][0]["runtime_topk"] is True


def test_candidate_feature_record_requires_32_geom_features():
    feature = CandidateFeatureRecord(
        sequence="handheld_room01",
        query_idx=10,
        query_timestamp=100.0,
        candidate_source="retrieval_topk",
        candidate_idx=4,
        candidate_timestamp=94.0,
        causal=True,
        database_max_idx=4,
        database_max_timestamp=94.0,
        retrieval_db_size=2,
        support_snapshot_max_idx=5,
        support_snapshot_max_timestamp=95.0,
        selected_support_indices=[3],
        selected_support_timestamps=[93.0],
        support_count=1,
        precondition_valid=True,
        negative_reason=None,
        x_geom=[0.0] * 32,
        safe_loop_factor_v1=False,
        labels={"sim3_quality_good": False, "odom_consistent_loose": True},
        metrics={"support_align_rmse": 0.3},
    )

    payload = dataclass_to_json_dict(feature)

    assert len(payload["x_geom"]) == 32
    assert payload["selected_support_indices"] == [3]
    assert payload["labels"]["sim3_quality_good"] is False


def test_candidate_feature_record_rejects_wrong_geom_feature_count():
    with pytest.raises(ValueError, match="x_geom must contain 32 values"):
        CandidateFeatureRecord(
            sequence="handheld_room01",
            query_idx=10,
            query_timestamp=100.0,
            candidate_source="retrieval_topk",
            candidate_idx=4,
            candidate_timestamp=94.0,
            causal=True,
            database_max_idx=4,
            database_max_timestamp=94.0,
            retrieval_db_size=2,
            support_snapshot_max_idx=5,
            support_snapshot_max_timestamp=95.0,
            selected_support_indices=[3],
            selected_support_timestamps=[93.0],
            support_count=1,
            precondition_valid=True,
            negative_reason=None,
            x_geom=[0.0] * 31,
            safe_loop_factor_v1=False,
            labels={"sim3_quality_good": False, "odom_consistent_loose": True},
            metrics={"support_align_rmse": 0.3},
        )


def test_dataclass_to_json_dict_recurses_into_mapping_raw_values():
    record = KeyframeRecord(
        keyframe_idx=7,
        timestamp=123.0,
        image_path=None,
        trajectory_idx=2,
        raw=MappingProxyType(
            {
                "path": Path("frames/000007.png"),
                "tuple_value": (Path("a.txt"), [Path("b.txt")]),
                "nested": MappingProxyType({"inner": Path("c.txt")}),
            }
        ),
    )

    payload = dataclass_to_json_dict(record)

    assert payload["raw"] == {
        "path": "frames/000007.png",
        "tuple_value": ["a.txt", ["b.txt"]],
        "nested": {"inner": "c.txt"},
    }


def test_support_decision_records_no_valid_support_reason():
    decision = SupportDecision(
        sequence="handheld_room01",
        query_idx=10,
        query_timestamp=100.0,
        candidate_idx=4,
        candidate_timestamp=94.0,
        causal=True,
        support_snapshot_max_idx=4,
        support_snapshot_max_timestamp=94.0,
        selected_support_indices=[],
        selected_support_timestamps=[],
        selected_support_baselines=[],
        support_count=0,
        rejected=True,
        rejection_reason="no_valid_support",
    )

    assert dataclass_to_json_dict(decision)["rejection_reason"] == "no_valid_support"


def test_read_jsonl_and_write_jsonl_roundtrip(tmp_path):
    path = tmp_path / "nested" / "records.jsonl"
    write_jsonl(path, [{"a": 1}, {"b": 2}])

    assert read_jsonl(path) == [{"a": 1}, {"b": 2}]


def test_write_jsonl_serializes_keyframe_dataclass(tmp_path):
    path = tmp_path / "records.jsonl"
    record = KeyframeRecord(
        keyframe_idx=3,
        timestamp=12.5,
        image_path="frames/000003.jpg",
        trajectory_idx=None,
        raw={"source_path": Path("frames/000003.jpg")},
    )

    write_jsonl(path, [record])

    assert read_jsonl(path) == [
        {
            "image_path": "frames/000003.jpg",
            "keyframe_idx": 3,
            "raw": {"source_path": "frames/000003.jpg"},
            "timestamp": 12.5,
            "trajectory_idx": None,
        }
    ]


def test_load_aster_raw_sequence_reads_required_files(synthetic_raw_sequence):
    sequence = load_aster_raw_sequence(synthetic_raw_sequence)

    assert sequence.sequence_name == "handheld_room01"
    assert sequence.platform == "handheld"
    assert sequence.loop_closure_enabled is False
    assert sequence.meta["sequence_name"] == "handheld_room01"
    assert len(sequence.keyframes) == 6
    assert sequence.keyframes[0].keyframe_idx == 0
    assert sequence.keyframes[5].timestamp == 105.0
    assert sequence.t_camera_lidar.shape == (4, 4)


def test_load_aster_raw_sequence_accepts_flat_camera_lidar_matrix(synthetic_raw_sequence):
    meta_path = synthetic_raw_sequence / "sequence_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["T_camera_lidar"] = [
        value for row in meta["T_camera_lidar"] for value in row
    ]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    sequence = load_aster_raw_sequence(synthetic_raw_sequence)

    assert sequence.t_camera_lidar.shape == (4, 4)
    assert sequence.t_camera_lidar[0, 3] == 0.1


def test_read_tum_trajectory_parses_xyzw_quaternion(synthetic_raw_sequence):
    poses = read_tum_trajectory(synthetic_raw_sequence / "trajectory.txt")

    assert len(poses) == 6
    assert poses[2].timestamp == 102.0
    assert poses[2].position == (2.0, 0.0, 0.0)
    assert poses[2].quaternion_xyzw == (0.0, 0.0, 0.0, 1.0)


def test_read_tum_trajectory_reports_path_and_line_for_wrong_field_count(tmp_path):
    path = tmp_path / "bad_trajectory.txt"
    path.write_text("1.0 0 0 0 0 0 1\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        read_tum_trajectory(path)

    message = str(exc_info.value)
    assert f"{path}:1" in message
    assert "expected 8 TUM fields, got 7" in message


def test_read_tum_trajectory_reports_path_line_and_fields_for_invalid_float(tmp_path):
    path = tmp_path / "bad_float_trajectory.txt"
    path.write_text("1.0 0 nope 0 0 0 0 1\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        read_tum_trajectory(path)

    message = str(exc_info.value)
    assert f"{path}:1" in message
    assert "1.0 0 nope 0 0 0 0 1" in message
    assert "nope" in message


def test_read_keyframes_accepts_idx_fallback_and_trajectory_idx_optional(tmp_path):
    path = tmp_path / "keyframes.jsonl"
    write_jsonl(
        path,
        [
            {"idx": 4, "timestamp": 10.0, "trajectory_idx": None},
            {"idx": 5, "timestamp": 11.0, "trajectory_idx": "8"},
        ],
    )

    records = _read_keyframes(path)

    assert records[0].keyframe_idx == 4
    assert records[0].trajectory_idx is None
    assert records[1].keyframe_idx == 5
    assert records[1].trajectory_idx == 8


def test_read_keyframes_reports_path_and_line_when_keyframe_id_missing(tmp_path):
    path = tmp_path / "keyframes.jsonl"
    write_jsonl(path, [{"timestamp": 10.0, "trajectory_idx": None}])

    with pytest.raises(ValueError) as exc_info:
        _read_keyframes(path)

    message = str(exc_info.value)
    assert f"{path}:1" in message
    assert "keyframe_idx or idx" in message


def test_read_keyframes_reports_path_and_line_for_invalid_trajectory_idx(tmp_path):
    path = tmp_path / "keyframes.jsonl"
    write_jsonl(path, [{"idx": 4, "timestamp": 10.0, "trajectory_idx": "bad"}])

    with pytest.raises(ValueError) as exc_info:
        _read_keyframes(path)

    message = str(exc_info.value)
    assert f"{path}:1" in message
    assert "trajectory_idx" in message
    assert "bad" in message
