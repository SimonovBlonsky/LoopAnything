import pytest

from loop_policy.labels import X_GEOM_FIELDS, build_x_geom, compute_safe_loop_factor_v1
from loop_policy.schema import LoopPolicyDatasetConfig


def test_compute_safe_loop_factor_requires_all_boolean_parts():
    config = LoopPolicyDatasetConfig(
        dataset_root="/dataset",
        output_root="/loop_policy_dataset",
        sequences=("s",),
    )
    metrics = {
        "abs_log_sim3_scale": 0.1,
        "support_align_rmse": 0.2,
        "direction_error_deg": 10.0,
        "aligned_vs_odom_rot_residual_deg": 5.0,
        "aligned_vs_odom_trans_residual_norm": 0.3,
    }

    label = compute_safe_loop_factor_v1(precondition_valid=True, metrics=metrics, config=config)

    assert label["sim3_quality_good"] is True
    assert label["odom_consistent_loose"] is True
    assert label["safe_loop_factor_v1"] is True


def test_compute_safe_loop_factor_does_not_gate_on_abs_log_sim3_scale():
    config = LoopPolicyDatasetConfig(
        dataset_root="/dataset",
        output_root="/loop_policy_dataset",
        sequences=("s",),
    )
    metrics = {
        "abs_log_sim3_scale": 3.0,
        "support_align_rmse": 0.2,
        "direction_error_deg": 10.0,
        "aligned_vs_odom_rot_residual_deg": 5.0,
        "aligned_vs_odom_trans_residual_norm": 0.3,
    }

    label = compute_safe_loop_factor_v1(precondition_valid=True, metrics=metrics, config=config)

    assert label["sim3_quality_good"] is True
    assert label["odom_consistent_loose"] is True
    assert label["safe_loop_factor_v1"] is True


def test_compute_safe_loop_factor_fails_precondition():
    config = LoopPolicyDatasetConfig(
        dataset_root="/dataset",
        output_root="/loop_policy_dataset",
        sequences=("s",),
    )

    label = compute_safe_loop_factor_v1(precondition_valid=False, metrics={}, config=config)

    assert label["safe_loop_factor_v1"] is False
    assert label["sim3_quality_good"] is False


def test_compute_safe_loop_factor_fails_missing_metrics_with_valid_precondition():
    config = LoopPolicyDatasetConfig(
        dataset_root="/dataset",
        output_root="/loop_policy_dataset",
        sequences=("s",),
    )

    label = compute_safe_loop_factor_v1(precondition_valid=True, metrics={}, config=config)

    assert label == {
        "sim3_quality_good": False,
        "odom_consistent_loose": False,
        "safe_loop_factor_v1": False,
    }


def _valid_x_geom_values():
    return {
        "rank_norm": 0.25,
        "salad_score_qc": 0.9,
        "salad_score_qs": 0.7,
        "salad_score_cs": 0.8,
        "salad_score_qc_minus_top1": 0.0,
        "salad_score_qc_minus_topk": 0.2,
        "da3_rot_qc_deg": 1.0,
        "da3_trans_qc_norm": 2.0,
        "da3_rot_cs_deg": 3.0,
        "da3_trans_cs_norm": 4.0,
        "da3_rot_qs_deg": 5.0,
        "da3_trans_qs_norm": 6.0,
        "odom_rot_qc_deg": 7.0,
        "odom_trans_qc_norm": 8.0,
        "odom_rot_cs_deg": 9.0,
        "odom_trans_cs_norm": 10.0,
        "support_baseline": 11.0,
        "sim3_scale": 1.2,
        "abs_log_sim3_scale": 0.18,
        "support_align_rmse": 0.1,
        "direction_error_deg": 12.0,
        "aligned_vs_odom_rot_residual_deg": 13.0,
        "aligned_vs_odom_trans_residual_norm": 14.0,
        "aligned_loop_rot_deg": 15.0,
        "aligned_loop_trans_norm": 16.0,
        "q_depth_conf_median": 0.91,
        "c_depth_conf_median": 0.92,
        "s_depth_conf_median": 0.93,
        "q_valid_depth_ratio": 0.81,
        "c_valid_depth_ratio": 0.82,
        "s_valid_depth_ratio": 0.83,
        "min_depth_conf_median": 0.91,
    }


def test_build_x_geom_uses_primary_support_and_has_32_values():
    assert X_GEOM_FIELDS == (
        "rank_norm",
        "salad_score_qc",
        "salad_score_qs",
        "salad_score_cs",
        "salad_score_qc_minus_top1",
        "salad_score_qc_minus_topk",
        "da3_rot_qc_deg",
        "da3_trans_qc_norm",
        "da3_rot_cs_deg",
        "da3_trans_cs_norm",
        "da3_rot_qs_deg",
        "da3_trans_qs_norm",
        "odom_rot_qc_deg",
        "odom_trans_qc_norm",
        "odom_rot_cs_deg",
        "odom_trans_cs_norm",
        "support_baseline",
        "sim3_scale",
        "abs_log_sim3_scale",
        "support_align_rmse",
        "direction_error_deg",
        "aligned_vs_odom_rot_residual_deg",
        "aligned_vs_odom_trans_residual_norm",
        "aligned_loop_rot_deg",
        "aligned_loop_trans_norm",
        "q_depth_conf_median",
        "c_depth_conf_median",
        "s_depth_conf_median",
        "q_valid_depth_ratio",
        "c_valid_depth_ratio",
        "s_valid_depth_ratio",
        "min_depth_conf_median",
    )

    x_geom = build_x_geom(_valid_x_geom_values())

    assert x_geom == [
        0.25,
        0.9,
        0.7,
        0.8,
        0.0,
        0.2,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        1.2,
        0.18,
        0.1,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
        0.91,
        0.92,
        0.93,
        0.81,
        0.82,
        0.83,
        0.91,
    ]


def test_build_x_geom_rejects_missing_field():
    values = _valid_x_geom_values()
    del values["support_baseline"]

    with pytest.raises(ValueError, match="missing x_geom field: support_baseline"):
        build_x_geom(values)


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_build_x_geom_rejects_nonfinite_field(value):
    values = _valid_x_geom_values()
    values["sim3_scale"] = value

    with pytest.raises(ValueError, match="x_geom field must be finite: sim3_scale"):
        build_x_geom(values)
