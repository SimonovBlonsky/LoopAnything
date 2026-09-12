import json

import numpy as np
import pytest


def _rotation_z(angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    return np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _rotation_axis_angle(axis, angle_deg):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    angle_rad = np.deg2rad(angle_deg)
    cross_matrix = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return (
        np.eye(3)
        + np.sin(angle_rad) * cross_matrix
        + (1.0 - np.cos(angle_rad)) * (cross_matrix @ cross_matrix)
    )


def _transform(rotation=None, translation=(0.0, 0.0, 0.0)):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.eye(3) if rotation is None else rotation
    transform[:3, 3] = translation
    return transform


def _evaluate(
    gt_relative_pose,
    estimated_relative_pose,
    *,
    factor_status="ok",
    thresholds=None,
):
    from robust_loop_verifier.geometry_annotation import (
        GeometryLabelThresholds,
        evaluate_metric_loop_factor,
    )

    gt_query_c2w = _transform(_rotation_z(90.0), (1.0, 2.0, 3.0))
    gt_candidate_c2w = gt_query_c2w @ gt_relative_pose
    return evaluate_metric_loop_factor(
        gt_query_c2w,
        gt_candidate_c2w,
        estimated_relative_pose,
        factor_status=factor_status,
        thresholds=GeometryLabelThresholds() if thresholds is None else thresholds,
    )


def test_exact_estimate_uses_pose_between_convention_and_is_automatic_positive():
    gt_relative_pose = _transform(_rotation_z(30.0), (2.0, -1.0, 0.5))

    result = _evaluate(gt_relative_pose, gt_relative_pose.copy())

    np.testing.assert_allclose(result["gt_relative_pose"], gt_relative_pose)
    np.testing.assert_allclose(result["estimated_relative_pose"], gt_relative_pose)
    np.testing.assert_allclose(result["gt_rotation_axis"], [0.0, 0.0, 1.0])
    assert result["gt_rotation_angle_deg"] == pytest.approx(30.0)
    np.testing.assert_allclose(result["estimated_rotation_axis"], [0.0, 0.0, 1.0])
    assert result["estimated_rotation_angle_deg"] == pytest.approx(30.0)
    np.testing.assert_allclose(result["gt_translation"], [2.0, -1.0, 0.5])
    assert result["gt_translation_norm_m"] == pytest.approx(np.sqrt(5.25))
    np.testing.assert_allclose(result["estimated_translation"], [2.0, -1.0, 0.5])
    assert result["estimated_translation_norm_m"] == pytest.approx(np.sqrt(5.25))
    assert result["translation_error_m"] == pytest.approx(0.0, abs=1e-12)
    assert result["rotation_error_deg"] == pytest.approx(0.0, abs=1e-12)
    assert result["translation_direction_error_deg"] == pytest.approx(0.0, abs=1e-12)
    assert result["factor_status"] == "ok"
    assert result["automatic_label"] == 1
    assert result["automatic_label_reason"] == "accepted"
    json.dumps(result, allow_nan=False)


def test_exact_pi_rotation_reports_correct_non_coordinate_axis():
    expected_axis = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )
    relative_pose = _transform(rotation, (1.0, 0.0, 0.0))

    result = _evaluate(relative_pose, relative_pose)

    reported_axis = np.asarray(result["gt_rotation_axis"])
    assert abs(float(np.dot(reported_axis, expected_axis))) == pytest.approx(1.0)
    assert result["gt_rotation_angle_deg"] == pytest.approx(180.0)


def test_small_rotation_reports_numerically_stable_axis_and_angle():
    expected_axis = np.array([1.0, 2.0, -3.0]) / np.sqrt(14.0)
    angle_deg = 1e-7
    relative_pose = _transform(
        _rotation_axis_angle(expected_axis, angle_deg),
        (1.0, 0.0, 0.0),
    )

    result = _evaluate(relative_pose, relative_pose)

    np.testing.assert_allclose(result["gt_rotation_axis"], expected_axis, atol=1e-9)
    assert result["gt_rotation_angle_deg"] == pytest.approx(angle_deg, abs=1e-12)


def test_twenty_degree_rotation_error_is_rejected():
    gt_relative_pose = _transform(translation=(2.0, 0.0, 0.0))
    estimated = _transform(_rotation_z(20.0), (2.0, 0.0, 0.0))

    result = _evaluate(gt_relative_pose, estimated)

    assert result["rotation_error_deg"] == pytest.approx(20.0)
    assert result["automatic_label"] == 0
    assert result["automatic_label_reason"] == "rotation_error_exceeds_threshold"


def test_two_meter_metric_translation_error_is_rejected():
    gt_relative_pose = _transform(translation=(2.0, 0.0, 0.0))
    estimated = _transform(translation=(4.0, 0.0, 0.0))

    result = _evaluate(gt_relative_pose, estimated)

    assert result["translation_error_m"] == pytest.approx(2.0)
    assert result["automatic_label"] == 0
    assert result["automatic_label_reason"] == "translation_error_exceeds_threshold"


def test_translation_threshold_scales_with_gt_baseline():
    gt_relative_pose = _transform(translation=(10.0, 0.0, 0.0))
    estimated = _transform(translation=(11.5, 0.0, 0.0))

    result = _evaluate(gt_relative_pose, estimated)

    assert result["translation_error_m"] == pytest.approx(1.5)
    assert result["effective_translation_error_threshold_m"] == pytest.approx(2.0)
    assert result["automatic_label"] == 1
    assert result["automatic_label_reason"] == "accepted"


def test_translation_threshold_uses_one_meter_floor_for_short_baseline():
    gt_relative_pose = _transform(translation=(2.0, 0.0, 0.0))
    estimated = _transform(translation=(3.1, 0.0, 0.0))

    result = _evaluate(gt_relative_pose, estimated)

    assert result["translation_error_m"] == pytest.approx(1.1)
    assert result["effective_translation_error_threshold_m"] == pytest.approx(1.0)
    assert result["automatic_label"] == 0
    assert result["automatic_label_reason"] == "translation_error_exceeds_threshold"


def test_translation_threshold_uses_five_meter_cap_for_large_baseline():
    gt_relative_pose = _transform(translation=(100.0, 0.0, 0.0))
    estimated = _transform(translation=(105.0, 0.0, 0.0))

    result = _evaluate(gt_relative_pose, estimated)

    assert result["translation_error_m"] == pytest.approx(5.0)
    assert result["effective_translation_error_threshold_m"] == pytest.approx(5.0)
    assert result["automatic_label"] == 1
    assert result["automatic_label_reason"] == "accepted"


def test_short_gt_baseline_does_not_reject_degenerate_translation_direction():
    gt_relative_pose = _transform(translation=(0.25, 0.0, 0.0))
    estimated = _transform()

    result = _evaluate(gt_relative_pose, estimated)

    assert result["gt_translation_norm_m"] == pytest.approx(0.25)
    assert result["translation_direction_error_deg"] is None
    assert result["automatic_label"] == 1
    assert result["automatic_label_reason"] == "accepted"
    json.dumps(result, allow_nan=False)


def test_long_gt_baseline_rejects_translation_direction_error_over_threshold():
    angle_rad = np.deg2rad(25.0)
    gt_relative_pose = _transform(translation=(2.0, 0.0, 0.0))
    estimated = _transform(translation=(2.0 * np.cos(angle_rad), 2.0 * np.sin(angle_rad), 0.0))

    result = _evaluate(gt_relative_pose, estimated)

    assert result["gt_translation_norm_m"] == pytest.approx(2.0)
    assert result["translation_error_m"] < 1.0
    assert result["translation_direction_error_deg"] == pytest.approx(25.0)
    assert result["automatic_label"] == 0
    assert result["automatic_label_reason"] == "translation_direction_error_exceeds_threshold"


def test_custom_threshold_accepts_exact_translation_boundary():
    from robust_loop_verifier.geometry_annotation import GeometryLabelThresholds

    gt_relative_pose = _transform(translation=(2.0, 0.0, 0.0))
    estimated = _transform(translation=(2.4, 0.0, 0.0))

    result = _evaluate(
        gt_relative_pose,
        estimated,
        thresholds=GeometryLabelThresholds(
            min_translation_error_m=0.4,
            max_translation_error_m=0.4,
        ),
    )

    assert result["translation_error_m"] == pytest.approx(0.4)
    assert result["automatic_label"] == 1
    assert result["automatic_label_reason"] == "accepted"


def test_custom_thresholds_accept_exact_rotation_and_direction_boundaries():
    from robust_loop_verifier.geometry_annotation import GeometryLabelThresholds

    angle_rad = np.deg2rad(0.4)
    gt_relative_pose = _transform(translation=(2.0, 0.0, 0.0))
    estimated = _transform(
        _rotation_z(0.4),
        (2.0 * np.cos(angle_rad), 2.0 * np.sin(angle_rad), 0.0),
    )
    thresholds = GeometryLabelThresholds(
        min_translation_error_m=0.4,
        max_translation_error_m=0.4,
        max_rotation_error_deg=0.4,
        max_translation_direction_error_deg=0.4,
    )

    result = _evaluate(gt_relative_pose, estimated, thresholds=thresholds)

    assert result["rotation_error_deg"] == pytest.approx(0.4)
    assert result["translation_direction_error_deg"] == pytest.approx(0.4)
    assert result["automatic_label"] == 1
    assert result["automatic_label_reason"] == "accepted"


@pytest.mark.parametrize(
    "estimated,expected_reason",
    [
        (
            _transform(translation=(2.40001, 0.0, 0.0)),
            "translation_error_exceeds_threshold",
        ),
        (
            _transform(_rotation_z(0.40001), (2.0, 0.0, 0.0)),
            "rotation_error_exceeds_threshold",
        ),
        (
            _transform(
                translation=(
                    2.0 * np.cos(np.deg2rad(0.40001)),
                    2.0 * np.sin(np.deg2rad(0.40001)),
                    0.0,
                )
            ),
            "translation_direction_error_exceeds_threshold",
        ),
    ],
)
def test_custom_thresholds_reject_values_just_over_boundary(estimated, expected_reason):
    from robust_loop_verifier.geometry_annotation import GeometryLabelThresholds

    gt_relative_pose = _transform(translation=(2.0, 0.0, 0.0))
    thresholds = GeometryLabelThresholds(
        min_translation_error_m=0.4,
        max_translation_error_m=0.4,
        max_rotation_error_deg=0.4,
        max_translation_direction_error_deg=0.4,
    )

    result = _evaluate(gt_relative_pose, estimated, thresholds=thresholds)

    assert result["automatic_label"] == 0
    assert result["automatic_label_reason"] == expected_reason


def test_min_direction_baseline_equality_requires_direction_threshold():
    from robust_loop_verifier.geometry_annotation import GeometryLabelThresholds

    gt_relative_pose = _transform(translation=(0.3, 0.4, 0.0))
    thresholds = GeometryLabelThresholds(min_direction_baseline_m=0.5)

    result = _evaluate(gt_relative_pose, _transform(), thresholds=thresholds)

    assert result["gt_translation_norm_m"] == pytest.approx(0.5)
    assert result["automatic_label"] == 0
    assert result["automatic_label_reason"] == "translation_direction_unavailable"


def test_custom_translation_threshold_changes_label():
    from robust_loop_verifier.geometry_annotation import GeometryLabelThresholds

    gt_relative_pose = _transform(translation=(2.0, 0.0, 0.0))
    estimated = _transform(translation=(3.5, 0.0, 0.0))

    result = _evaluate(
        gt_relative_pose,
        estimated,
        thresholds=GeometryLabelThresholds(
            min_translation_error_m=2.0,
            max_translation_error_m=2.0,
        ),
    )

    assert result["translation_error_m"] == pytest.approx(1.5)
    assert result["automatic_label"] == 1
    assert result["automatic_label_reason"] == "accepted"


def test_missing_estimate_is_negative_with_explicit_reason():
    result = _evaluate(_transform(translation=(1.0, 0.0, 0.0)), None)

    assert result["estimated_relative_pose"] is None
    assert result["estimated_rotation_axis"] is None
    assert result["estimated_rotation_angle_deg"] is None
    assert result["estimated_translation"] is None
    assert result["estimated_translation_norm_m"] is None
    assert result["translation_error_m"] is None
    assert result["rotation_error_deg"] is None
    assert result["translation_direction_error_deg"] is None
    assert result["automatic_label"] == 0
    assert result["automatic_label_reason"] == "missing_estimate"
    json.dumps(result, allow_nan=False)


def test_invalid_estimate_is_negative_with_explicit_reason():
    invalid_estimate = np.eye(4)
    invalid_estimate[0, 0] = np.nan

    result = _evaluate(_transform(translation=(1.0, 0.0, 0.0)), invalid_estimate)

    assert result["estimated_relative_pose"] is None
    assert result["automatic_label"] == 0
    assert result["automatic_label_reason"] == "invalid_estimate"
    json.dumps(result, allow_nan=False)


def test_estimate_with_scaled_rotation_block_is_invalid():
    invalid_estimate = np.eye(4)
    invalid_estimate[0, 0] = 1.000003

    result = _evaluate(_transform(translation=(1.0, 0.0, 0.0)), invalid_estimate)

    assert result["estimated_relative_pose"] is None
    assert result["automatic_label"] == 0
    assert result["automatic_label_reason"] == "invalid_estimate"


def test_estimate_with_overflowing_python_int_is_invalid():
    invalid_estimate = np.eye(4, dtype=object)
    invalid_estimate[0, 3] = 10**1000

    result = _evaluate(_transform(translation=(1.0, 0.0, 0.0)), invalid_estimate)

    assert result["estimated_relative_pose"] is None
    assert result["automatic_label"] == 0
    assert result["automatic_label_reason"] == "invalid_estimate"


def test_complex_estimate_is_invalid_without_discarding_imaginary_part():
    invalid_estimate = np.eye(4, dtype=np.complex128)
    invalid_estimate[0, 3] = 1.0 + 2.0j

    result = _evaluate(_transform(translation=(1.0, 0.0, 0.0)), invalid_estimate)

    assert result["estimated_relative_pose"] is None
    assert result["automatic_label"] == 0
    assert result["automatic_label_reason"] == "invalid_estimate"


def test_gt_pose_with_overflowing_python_int_raises_value_error():
    from robust_loop_verifier.geometry_annotation import (
        GeometryLabelThresholds,
        evaluate_metric_loop_factor,
    )

    invalid_gt_query = np.eye(4, dtype=object)
    invalid_gt_query[0, 3] = 10**1000

    with pytest.raises(ValueError, match="gt_query_c2w"):
        evaluate_metric_loop_factor(
            invalid_gt_query,
            np.eye(4),
            np.eye(4),
            factor_status="ok",
            thresholds=GeometryLabelThresholds(),
        )


def test_complex_gt_pose_raises_value_error():
    from robust_loop_verifier.geometry_annotation import (
        GeometryLabelThresholds,
        evaluate_metric_loop_factor,
    )

    invalid_gt_query = np.eye(4, dtype=np.complex128)
    invalid_gt_query[0, 3] = 1.0 + 2.0j

    with pytest.raises(ValueError, match="gt_query_c2w"):
        evaluate_metric_loop_factor(
            invalid_gt_query,
            np.eye(4),
            np.eye(4),
            factor_status="ok",
            thresholds=GeometryLabelThresholds(),
        )


def test_non_ok_factor_status_is_negative_even_for_an_exact_estimate():
    relative_pose = _transform(translation=(1.0, 0.0, 0.0))

    result = _evaluate(relative_pose, relative_pose, factor_status="failed")

    assert result["automatic_label"] == 0
    assert result["automatic_label_reason"] == "factor_status_not_ok"


@pytest.mark.parametrize(
    "field,value",
    [
        ("min_translation_error_m", -1.0),
        ("max_translation_error_m", -1.0),
        ("translation_error_scale_ratio", np.inf),
        ("max_rotation_error_deg", np.inf),
        ("max_translation_direction_error_deg", np.nan),
        ("min_direction_baseline_m", -0.1),
    ],
)
def test_thresholds_reject_non_finite_or_negative_values(field, value):
    from robust_loop_verifier.geometry_annotation import GeometryLabelThresholds

    with pytest.raises(ValueError, match=field):
        GeometryLabelThresholds(**{field: value})


def test_threshold_with_overflowing_python_int_raises_value_error():
    from robust_loop_verifier.geometry_annotation import GeometryLabelThresholds

    with pytest.raises(ValueError, match="max_translation_error_m"):
        GeometryLabelThresholds(max_translation_error_m=10**1000)


def test_translation_threshold_range_must_be_ordered():
    from robust_loop_verifier.geometry_annotation import GeometryLabelThresholds

    with pytest.raises(ValueError, match="min_translation_error_m"):
        GeometryLabelThresholds(
            min_translation_error_m=2.0,
            max_translation_error_m=1.0,
        )


def test_invalid_gt_pose_is_rejected():
    from robust_loop_verifier.geometry_annotation import (
        GeometryLabelThresholds,
        evaluate_metric_loop_factor,
    )

    invalid_gt_query = np.eye(4)
    invalid_gt_query[3, 0] = 1.0

    with pytest.raises(ValueError, match="gt_query_c2w"):
        evaluate_metric_loop_factor(
            invalid_gt_query,
            np.eye(4),
            np.eye(4),
            factor_status="ok",
            thresholds=GeometryLabelThresholds(),
        )
