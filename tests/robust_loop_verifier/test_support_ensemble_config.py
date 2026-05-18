from pathlib import Path

import pytest

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


def test_support_ensemble_explicit_values_parse(tmp_path: Path):
    data = _valid_config(tmp_path)
    data["support_ensemble"] = {
        "enabled": True,
        "support_count": 4,
        "sigma_rot_floor": 0.06,
        "sigma_trans_floor": 0.3,
        "covariance_scale": 1.5,
        "c_align": 1.2,
        "c_consensus": 2.4,
        "lambda_dir": 0.5,
        "robust_iterations": 5,
    }

    config = RobustLoopVerifierConfig.from_mapping(data)

    assert config.support_ensemble.enabled is True
    assert config.support_ensemble.support_count == 4
    assert config.support_ensemble.sigma_rot_floor == 0.06
    assert config.support_ensemble.sigma_trans_floor == 0.3
    assert config.support_ensemble.covariance_scale == 1.5
    assert config.support_ensemble.c_align == 1.2
    assert config.support_ensemble.c_consensus == 2.4
    assert config.support_ensemble.lambda_dir == 0.5
    assert config.support_ensemble.robust_iterations == 5


def test_support_ensemble_missing_section_defaults_to_disabled(tmp_path: Path):
    config = RobustLoopVerifierConfig.from_mapping(_valid_config(tmp_path))

    assert config.support_ensemble.enabled is False
    assert config.support_ensemble.support_count == 1
    assert config.support_ensemble.sigma_rot_floor == 0.05
    assert config.support_ensemble.sigma_trans_floor == 0.25
    assert config.support_ensemble.covariance_scale == 1.0
    assert config.support_ensemble.c_align == 1.0
    assert config.support_ensemble.c_consensus == 2.0
    assert config.support_ensemble.lambda_dir == 1.0
    assert config.support_ensemble.robust_iterations == 3


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("enabled", "false"),
        ("enabled", "true"),
        ("support_count", 0),
        ("support_count", True),
        ("support_count", float("inf")),
        ("support_count", float("nan")),
        ("support_count", 1.9),
        ("support_count", "not-a-number"),
        ("sigma_rot_floor", 0.0),
        ("sigma_rot_floor", True),
        ("sigma_rot_floor", float("inf")),
        ("sigma_rot_floor", "not-a-number"),
        ("sigma_trans_floor", 0.0),
        ("sigma_trans_floor", True),
        ("sigma_trans_floor", float("inf")),
        ("sigma_trans_floor", "not-a-number"),
        ("covariance_scale", 0.0),
        ("covariance_scale", True),
        ("covariance_scale", float("inf")),
        ("covariance_scale", "not-a-number"),
        ("c_align", 0.0),
        ("c_align", True),
        ("c_align", float("inf")),
        ("c_align", "not-a-number"),
        ("c_consensus", 0.0),
        ("c_consensus", True),
        ("c_consensus", float("inf")),
        ("c_consensus", "not-a-number"),
        ("lambda_dir", -0.1),
        ("lambda_dir", True),
        ("lambda_dir", float("inf")),
        ("lambda_dir", "not-a-number"),
        ("robust_iterations", 0),
        ("robust_iterations", True),
        ("robust_iterations", float("inf")),
        ("robust_iterations", float("nan")),
        ("robust_iterations", 1.9),
        ("robust_iterations", "not-a-number"),
    ],
)
def test_support_ensemble_rejects_invalid_values(tmp_path: Path, field: str, value: object):
    data = _valid_config(tmp_path)
    data["support_ensemble"] = {
        "enabled": True,
        "support_count": 4,
        "sigma_rot_floor": 0.05,
        "sigma_trans_floor": 0.25,
        "covariance_scale": 1.0,
        "c_align": 1.0,
        "c_consensus": 2.0,
        "lambda_dir": 1.0,
        "robust_iterations": 3,
    }
    data["support_ensemble"][field] = value

    with pytest.raises(ValueError, match=r"support_ensemble\.{}".format(field)):
        RobustLoopVerifierConfig.from_mapping(data)


def test_support_ensemble_enabled_yaml_config_parses():
    config = RobustLoopVerifierConfig.from_yaml(
        Path("configs/robust_loop_verifier/fusionportablev2_handheld_support_ensemble.yaml")
    )

    assert config.retrieval_top_k_main == 10
    assert config.support_count == 1
    assert config.support_ensemble.enabled is True
    assert config.support_ensemble.support_count == 4


def test_support_ensemble_disabled_yaml_config_parses():
    config = RobustLoopVerifierConfig.from_yaml(
        Path("configs/robust_loop_verifier/fusionportablev2_handheld.yaml")
    )

    assert config.retrieval_top_k_main == 10
    assert config.support_count == 1
    assert config.support_ensemble.enabled is False
    assert config.support_ensemble.support_count == 1
