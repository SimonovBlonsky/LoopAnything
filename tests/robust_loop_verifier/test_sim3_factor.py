import numpy as np


def _pose_at_x(x):
    pose = np.eye(4)
    pose[0, 3] = x
    return pose


def _pose_at_xyz(x, y, z):
    pose = np.eye(4)
    pose[:3, 3] = [x, y, z]
    return pose


def test_align_triplet_to_candidate_support_recovers_metric_loop_factor():
    from robust_loop_verifier.sim3_factor import align_triplet_to_candidate_support

    da3_c2w = np.stack([_pose_at_x(0.5), _pose_at_x(0.0), _pose_at_x(0.25)])
    odom_candidate_c2w = _pose_at_x(10.0)
    odom_support_c2w = _pose_at_x(11.0)

    result = align_triplet_to_candidate_support(
        da3_c2w,
        odom_candidate_c2w,
        odom_support_c2w,
    )

    assert result.valid
    assert result.rejection_reason is None
    np.testing.assert_allclose(result.sim3_scale, 4.0)
    np.testing.assert_allclose(result.loop_factor[:3, 3], [-2.0, 0.0, 0.0])
    np.testing.assert_allclose(result.loop_factor[:3, :3], np.eye(3))
    assert result.support_alignment_residual_m < 1e-8
    assert result.direction_error_deg < 1e-8


def test_align_triplet_to_candidate_support_reports_direction_mismatch_diagnostics():
    from robust_loop_verifier.sim3_factor import align_triplet_to_candidate_support

    da3_c2w = np.stack(
        [
            _pose_at_xyz(0.5, 0.0, 0.0),
            _pose_at_xyz(0.0, 0.0, 0.0),
            _pose_at_xyz(0.0, 1.0, 0.0),
        ]
    )
    odom_candidate_c2w = _pose_at_xyz(10.0, 0.0, 0.0)
    odom_support_c2w = _pose_at_xyz(11.0, 0.0, 0.0)

    result = align_triplet_to_candidate_support(
        da3_c2w,
        odom_candidate_c2w,
        odom_support_c2w,
    )

    assert result.valid
    np.testing.assert_allclose(result.sim3_scale, 1.0)
    assert result.support_alignment_residual_m > 1.0
    np.testing.assert_allclose(result.direction_error_deg, 90.0)


def test_align_triplet_to_candidate_support_rejects_zero_da3_support_baseline():
    from robust_loop_verifier.sim3_factor import align_triplet_to_candidate_support

    da3_c2w = np.stack([_pose_at_x(0.5), _pose_at_x(0.0), _pose_at_x(0.0)])
    odom_candidate_c2w = _pose_at_x(10.0)
    odom_support_c2w = _pose_at_x(11.0)

    result = align_triplet_to_candidate_support(
        da3_c2w,
        odom_candidate_c2w,
        odom_support_c2w,
    )

    assert not result.valid
    assert result.loop_factor is None
    assert result.rejection_reason == "invalid_da3_support_baseline"


def test_align_triplet_to_candidate_support_rejects_zero_odom_support_baseline():
    from robust_loop_verifier.sim3_factor import align_triplet_to_candidate_support

    da3_c2w = np.stack([_pose_at_x(0.5), _pose_at_x(0.0), _pose_at_x(0.25)])
    odom_candidate_c2w = _pose_at_x(10.0)
    odom_support_c2w = _pose_at_x(10.0)

    result = align_triplet_to_candidate_support(
        da3_c2w,
        odom_candidate_c2w,
        odom_support_c2w,
    )

    assert not result.valid
    assert result.loop_factor is None
    assert result.rejection_reason == "invalid_odom_support_baseline"


def test_align_triplet_to_candidate_support_rejects_non_finite_pose():
    from robust_loop_verifier.sim3_factor import align_triplet_to_candidate_support

    da3_c2w = np.stack([_pose_at_x(0.5), _pose_at_x(0.0), _pose_at_x(0.25)])
    odom_candidate_c2w = _pose_at_x(10.0)
    odom_candidate_c2w[0, 3] = np.nan
    odom_support_c2w = _pose_at_x(11.0)

    result = align_triplet_to_candidate_support(
        da3_c2w,
        odom_candidate_c2w,
        odom_support_c2w,
    )

    assert not result.valid
    assert result.loop_factor is None
    assert result.rejection_reason == "invalid_pose"


def test_align_triplet_to_candidate_support_rejects_malformed_pose():
    from robust_loop_verifier.sim3_factor import align_triplet_to_candidate_support

    result = align_triplet_to_candidate_support(
        object(),
        _pose_at_x(10.0),
        _pose_at_x(11.0),
    )

    assert not result.valid
    assert result.loop_factor is None
    assert result.rejection_reason == "invalid_pose"


def test_align_triplet_to_candidate_support_allows_large_finite_scale():
    from robust_loop_verifier.sim3_factor import align_triplet_to_candidate_support

    da3_c2w = np.stack([_pose_at_x(0.5), _pose_at_x(0.0), _pose_at_x(1e-6)])
    odom_candidate_c2w = _pose_at_x(10.0)
    odom_support_c2w = _pose_at_x(11.0)

    result = align_triplet_to_candidate_support(
        da3_c2w,
        odom_candidate_c2w,
        odom_support_c2w,
    )

    assert result.valid
    np.testing.assert_allclose(result.sim3_scale, 1e6)
    assert result.rejection_reason is None
