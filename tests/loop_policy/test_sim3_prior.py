import math

import numpy as np
import pytest

from loop_policy.geometry import make_transform
from loop_policy.geometry import quaternion_xyzw_to_matrix
from loop_policy.sim3_prior import (
    Sim3PriorConfig,
    align_da3_poses_with_candidate_support_prior,
)


_SUPPORT_LENGTH_MESSAGE = "DA3 and odom supports must have the same positive length"


def _pose(x):
    return make_transform(np.eye(3), np.array([x, 0.0, 0.0]))


def _pose_with_rotation(rotation, translation):
    return make_transform(rotation, np.asarray(translation, dtype=np.float64))


def test_sim3_prior_recovers_scale_from_candidate_support_baseline():
    da3 = {
        "query": _pose(4.0),
        "candidate": _pose(0.0),
        "supports": [_pose(2.0)],
    }
    odom = {
        "query": _pose(8.0),
        "candidate": _pose(0.0),
        "supports": [_pose(4.0)],
    }

    result = align_da3_poses_with_candidate_support_prior(
        da3_query_pose=da3["query"],
        da3_candidate_pose=da3["candidate"],
        da3_support_poses=da3["supports"],
        odom_query_pose=odom["query"],
        odom_candidate_pose=odom["candidate"],
        odom_support_poses=odom["supports"],
        config=Sim3PriorConfig(),
    )

    assert result.accepted is True
    assert math.isclose(result.sim3_scale, 2.0, rel_tol=1e-6)
    assert math.isclose(result.aligned_loop.translation_norm, 8.0, rel_tol=1e-6)
    assert result.rejection_reason is None


def test_sim3_prior_rejects_zero_da3_support_baseline():
    result = align_da3_poses_with_candidate_support_prior(
        da3_query_pose=_pose(1.0),
        da3_candidate_pose=_pose(0.0),
        da3_support_poses=[_pose(0.0)],
        odom_query_pose=_pose(1.0),
        odom_candidate_pose=_pose(0.0),
        odom_support_poses=[_pose(1.0)],
        config=Sim3PriorConfig(),
    )

    assert result.accepted is False
    assert result.rejection_reason == "invalid_support_baseline"


def test_sim3_prior_requires_same_positive_support_count():
    with pytest.raises(ValueError, match=_SUPPORT_LENGTH_MESSAGE):
        align_da3_poses_with_candidate_support_prior(
            da3_query_pose=_pose(1.0),
            da3_candidate_pose=_pose(0.0),
            da3_support_poses=[],
            odom_query_pose=_pose(1.0),
            odom_candidate_pose=_pose(0.0),
            odom_support_poses=[],
            config=Sim3PriorConfig(),
        )

    with pytest.raises(ValueError, match=_SUPPORT_LENGTH_MESSAGE):
        align_da3_poses_with_candidate_support_prior(
            da3_query_pose=_pose(1.0),
            da3_candidate_pose=_pose(0.0),
            da3_support_poses=[_pose(1.0)],
            odom_query_pose=_pose(1.0),
            odom_candidate_pose=_pose(0.0),
            odom_support_poses=[_pose(1.0), _pose(2.0)],
            config=Sim3PriorConfig(),
        )


def test_sim3_prior_uses_median_scale_and_reports_alignment_metrics():
    result = align_da3_poses_with_candidate_support_prior(
        da3_query_pose=_pose(3.0),
        da3_candidate_pose=_pose(0.0),
        da3_support_poses=[_pose(1.0), _pose(2.0), _pose(4.0)],
        odom_query_pose=_pose(9.0),
        odom_candidate_pose=_pose(1.0),
        odom_support_poses=[_pose(4.0), _pose(7.0), _pose(9.0)],
        config=Sim3PriorConfig(max_support_align_rmse_m=10.0),
    )

    assert result.accepted is True
    assert math.isclose(result.sim3_scale, 3.0, rel_tol=1e-6)
    assert math.isclose(result.abs_log_sim3_scale, abs(math.log(3.0)), rel_tol=1e-6)
    assert math.isclose(result.support_align_rmse, math.sqrt(16.0 / 3.0), rel_tol=1e-6)
    assert math.isclose(result.aligned_vs_odom.translation_norm, 1.0, rel_tol=1e-6)
    np.testing.assert_allclose(result.aligned_query_pose[:3, 3], [10.0, 0.0, 0.0])


def test_sim3_prior_rejects_large_support_alignment_rmse_before_direction_error():
    result = align_da3_poses_with_candidate_support_prior(
        da3_query_pose=_pose(1.0),
        da3_candidate_pose=_pose(0.0),
        da3_support_poses=[_pose(1.0), _pose(2.0)],
        odom_query_pose=_pose(1.0),
        odom_candidate_pose=_pose(0.0),
        odom_support_poses=[_pose(100.0), _pose(-100.0)],
        config=Sim3PriorConfig(max_sim3_scale=200.0, max_support_align_rmse_m=1.0),
    )

    assert result.accepted is False
    assert result.rejection_reason == "support_align_rmse_too_large"


def test_sim3_prior_rotates_candidate_local_offsets_before_scaling():
    da3_candidate_rotation = quaternion_xyzw_to_matrix(
        (0.0, 0.0, -math.sin(math.pi / 4.0), math.cos(math.pi / 4.0))
    )
    da3 = {
        "query": _pose_with_rotation(da3_candidate_rotation, [4.0, 0.0, 0.0]),
        "candidate": _pose_with_rotation(da3_candidate_rotation, [0.0, 0.0, 0.0]),
        "supports": [_pose_with_rotation(da3_candidate_rotation, [2.0, 0.0, 0.0])],
    }
    odom = {
        "query": _pose_with_rotation(np.eye(3), [10.0, 4.0, 0.0]),
        "candidate": _pose_with_rotation(np.eye(3), [10.0, 0.0, 0.0]),
        "supports": [_pose_with_rotation(np.eye(3), [10.0, 2.0, 0.0])],
    }

    result = align_da3_poses_with_candidate_support_prior(
        da3_query_pose=da3["query"],
        da3_candidate_pose=da3["candidate"],
        da3_support_poses=da3["supports"],
        odom_query_pose=odom["query"],
        odom_candidate_pose=odom["candidate"],
        odom_support_poses=odom["supports"],
        config=Sim3PriorConfig(),
    )

    assert result.accepted is True
    np.testing.assert_allclose(result.aligned_query_pose[:3, 3], [10.0, 4.0, 0.0], atol=1e-9)
    assert math.isclose(result.direction_error_deg, 0.0, abs_tol=1e-9)


def test_sim3_prior_rejects_nonpositive_sim3_scale():
    result = align_da3_poses_with_candidate_support_prior(
        da3_query_pose=_pose(1.0),
        da3_candidate_pose=_pose(0.0),
        da3_support_poses=[_pose(1.0)],
        odom_query_pose=_pose(1.0),
        odom_candidate_pose=_pose(0.0),
        odom_support_poses=[_pose(0.0)],
        config=Sim3PriorConfig(max_direction_error_deg=180.0),
    )

    assert result.accepted is False
    assert result.rejection_reason == "invalid_sim3_scale"
    assert result.abs_log_sim3_scale == float("inf")


def test_sim3_prior_rejects_nonfinite_input_pose():
    da3_query = _pose(1.0)
    da3_query[0, 3] = np.nan

    result = align_da3_poses_with_candidate_support_prior(
        da3_query_pose=da3_query,
        da3_candidate_pose=_pose(0.0),
        da3_support_poses=[_pose(1.0)],
        odom_query_pose=_pose(1.0),
        odom_candidate_pose=_pose(0.0),
        odom_support_poses=[_pose(1.0)],
        config=Sim3PriorConfig(),
    )

    assert result.accepted is False
    assert result.rejection_reason == "sim3_alignment_failed"
    assert np.isfinite(
        [
            result.sim3_scale,
            result.abs_log_sim3_scale,
            result.support_align_rmse,
            result.direction_error_deg,
            result.aligned_loop.rotation_deg,
            result.aligned_loop.translation_norm,
            result.aligned_vs_odom.rotation_deg,
            result.aligned_vs_odom.translation_norm,
        ]
    ).all()
    assert np.isfinite(result.aligned_query_pose).all()


def test_sim3_prior_does_not_reject_scale_outside_aster_slam_bounds():
    result = align_da3_poses_with_candidate_support_prior(
        da3_query_pose=_pose(1.0),
        da3_candidate_pose=_pose(0.0),
        da3_support_poses=[_pose(1.0)],
        odom_query_pose=_pose(1000.0),
        odom_candidate_pose=_pose(0.0),
        odom_support_poses=[_pose(1000.0)],
        config=Sim3PriorConfig(max_support_align_rmse_m=2000.0),
    )

    assert result.accepted is True
    assert result.rejection_reason is None
    assert result.sim3_scale == 1000.0


def test_sim3_prior_rejects_zero_odom_direction_as_invalid_scale_first():
    result = align_da3_poses_with_candidate_support_prior(
        da3_query_pose=_pose(1.0),
        da3_candidate_pose=_pose(0.0),
        da3_support_poses=[_pose(1.0)],
        odom_query_pose=_pose(1.0),
        odom_candidate_pose=_pose(1.0),
        odom_support_poses=[_pose(1.0)],
        config=Sim3PriorConfig(max_direction_error_deg=45.0),
    )

    assert result.accepted is False
    assert result.direction_error_deg == 180.0
    assert result.rejection_reason == "invalid_sim3_scale"
