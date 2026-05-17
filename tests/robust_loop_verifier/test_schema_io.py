from pathlib import Path

import pytest

from robust_loop_verifier.io import (
    read_json,
    read_jsonl,
    read_yaml,
    write_json,
    write_jsonl,
    write_yaml,
)
from robust_loop_verifier.schema import RobustLoopVerifierConfig


def _valid_config(tmp_path: Path) -> dict:
    return {
        "dataset_name": "FusionPortableV2",
        "platform": "handheld",
        "input_root": "/data/datasets/FusionPortable/fusionportable_loop_dataset",
        "output_root": str(tmp_path / "cache"),
        "gt_root": str(tmp_path / "gt"),
        "positive_radius_m": 2.0,
        "recent_exclusion_keyframes": 50,
        "retrieval_top_k_main": 10,
        "retrieval_top_k_ablations": [5, 20],
        "support_window": 4,
        "support_count": 1,
        "min_support_baseline_m": 0.3,
        "pgo_noise": {
            "prior_sigmas": [0.01, 0.01, 0.01, 0.1, 0.1, 0.1],
            "odom_sigmas": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
            "loop_sigmas": [0.1, 0.1, 0.1, 1.0, 1.0, 1.0],
        },
        "da3": {"process_res": 504, "ref_view_strategy": "first"},
    }


def test_config_requires_gt_fields(tmp_path: Path):
    path = tmp_path / "config.yaml"
    data = _valid_config(tmp_path)
    del data["gt_root"]
    del data["positive_radius_m"]
    del data["recent_exclusion_keyframes"]
    write_yaml(path, data)

    with pytest.raises(ValueError, match="positive_radius_m"):
        RobustLoopVerifierConfig.from_yaml(path)


def test_config_rejects_non_positive_positive_radius(tmp_path: Path):
    path = tmp_path / "config.yaml"
    data = _valid_config(tmp_path)
    data["positive_radius_m"] = 0.0
    write_yaml(path, data)

    with pytest.raises(ValueError, match="positive_radius_m must be positive"):
        RobustLoopVerifierConfig.from_yaml(path)


def test_config_rejects_negative_recent_exclusion(tmp_path: Path):
    path = tmp_path / "config.yaml"
    data = _valid_config(tmp_path)
    data["recent_exclusion_keyframes"] = -1
    write_yaml(path, data)

    with pytest.raises(ValueError, match="recent_exclusion_keyframes must be non-negative"):
        RobustLoopVerifierConfig.from_yaml(path)


def test_config_defaults_positive_max_rotation_to_45_degrees(tmp_path: Path):
    path = tmp_path / "config.yaml"
    write_yaml(path, _valid_config(tmp_path))

    config = RobustLoopVerifierConfig.from_yaml(path)

    assert config.positive_max_rotation_deg == 45.0


def test_config_rejects_invalid_positive_max_rotation(tmp_path: Path):
    path = tmp_path / "config.yaml"
    data = _valid_config(tmp_path)
    data["positive_max_rotation_deg"] = -1.0
    write_yaml(path, data)

    with pytest.raises(ValueError, match="positive_max_rotation_deg"):
        RobustLoopVerifierConfig.from_yaml(path)


def test_config_rejects_pgo_vector_with_wrong_length(tmp_path: Path):
    path = tmp_path / "config.yaml"
    data = _valid_config(tmp_path)
    data["pgo_noise"]["prior_sigmas"] = [0.01, 0.01, 0.01, 0.1, 0.1]
    write_yaml(path, data)

    with pytest.raises(ValueError, match="prior_sigmas"):
        RobustLoopVerifierConfig.from_yaml(path)


def test_config_rejects_da3_process_res_below_minimum(tmp_path: Path):
    path = tmp_path / "config.yaml"
    data = _valid_config(tmp_path)
    data["da3"]["process_res"] = 223
    write_yaml(path, data)

    with pytest.raises(ValueError, match="process_res"):
        RobustLoopVerifierConfig.from_yaml(path)


def test_config_rejects_da3_ref_view_strategy_other_than_first(tmp_path: Path):
    path = tmp_path / "config.yaml"
    data = _valid_config(tmp_path)
    data["da3"]["ref_view_strategy"] = "nearest"
    write_yaml(path, data)

    with pytest.raises(ValueError, match="ref_view_strategy"):
        RobustLoopVerifierConfig.from_yaml(path)


def test_jsonl_roundtrip(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    rows = [{"idx": 1, "name": "a"}, {"idx": 2, "name": "b"}]

    write_jsonl(path, rows)

    assert list(read_jsonl(path)) == rows


def test_write_jsonl_rejects_non_mapping_row(tmp_path: Path):
    path = tmp_path / "rows.jsonl"

    with pytest.raises(ValueError, match="JSONL rows must be mappings"):
        write_jsonl(path, [["not", "mapping"]])  # type: ignore[list-item]


def test_json_roundtrip(tmp_path: Path):
    path = tmp_path / "manifest.json"
    manifest = {"dataset_name": "demo", "frame_count": 2}

    write_json(path, manifest)

    assert read_json(path) == manifest


def test_write_json_rejects_non_mapping_root(tmp_path: Path):
    path = tmp_path / "records.json"

    with pytest.raises(ValueError, match="JSON root must be a mapping"):
        write_json(path, ["not", "a", "mapping"])  # type: ignore[arg-type]


def test_yaml_roundtrip(tmp_path: Path):
    path = tmp_path / "config.yaml"
    data = {"dataset_name": "FusionPortableV2", "keyframe_count": 2}

    write_yaml(path, data)

    assert read_yaml(path) == data


def test_read_yaml_empty_file_returns_empty_mapping(tmp_path: Path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")

    assert read_yaml(path) == {}


def test_write_yaml_rejects_non_mapping_root(tmp_path: Path):
    path = tmp_path / "config.yaml"

    with pytest.raises(ValueError, match="YAML root must be a mapping"):
        write_yaml(path, ["not", "a", "mapping"])  # type: ignore[arg-type]
