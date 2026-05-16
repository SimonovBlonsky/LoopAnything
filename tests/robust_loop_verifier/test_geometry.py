import numpy as np
import pytest


def test_quaternion_identity_to_rotation_matrix():
    from robust_loop_verifier.geometry import rotation_matrix_from_quat_xyzw

    rotation = rotation_matrix_from_quat_xyzw([0, 0, 0, 1])

    np.testing.assert_allclose(rotation, np.eye(3))


def test_pose_between_translation():
    from robust_loop_verifier.geometry import invert_transform, make_transform, pose_between

    rotation = np.eye(3)
    a_c2w = make_transform(rotation, [1, 0, 0])
    b_c2w = make_transform(rotation, [3.5, 0, 0])

    relative = pose_between(a_c2w, b_c2w)

    np.testing.assert_allclose(relative[:3, 3], [2.5, 0, 0])
    np.testing.assert_allclose(invert_transform(invert_transform(a_c2w)), a_c2w)


def test_pose_between_rejects_invalid_second_pose():
    from robust_loop_verifier.geometry import pose_between

    a_c2w = np.eye(4)
    invalid_rotation = np.eye(4)
    invalid_rotation[0, 0] = 2.0
    invalid_bottom_row = np.eye(4)
    invalid_bottom_row[3, 0] = 1.0

    with pytest.raises(ValueError, match="SE3|rotation"):
        pose_between(a_c2w, invalid_rotation)

    with pytest.raises(ValueError, match="SE3|bottom row"):
        pose_between(a_c2w, invalid_bottom_row)


def test_invert_transform_rejects_non_se3_matrix():
    from robust_loop_verifier.geometry import invert_transform

    transform = np.eye(4)
    transform[0, 0] = 2.0

    with pytest.raises(ValueError, match="SE3|rotation"):
        invert_transform(transform)


def test_invert_transform_rejects_non_finite_transform():
    from robust_loop_verifier.geometry import invert_transform

    transform = np.eye(4)
    transform[0, 3] = np.nan

    with pytest.raises(ValueError):
        invert_transform(transform)


def test_sim3_align_points_recovers_scale_rotation_translation():
    from robust_loop_verifier.geometry import sim3_align_points

    src_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    scale = 3.0
    translation = np.array([5.0, -2.0, 1.0])
    dst_points = scale * (src_points @ rotation.T) + translation

    alignment = sim3_align_points(src_points, dst_points)
    aligned_points = alignment.scale * (src_points @ alignment.rotation.T) + alignment.translation

    np.testing.assert_allclose(alignment.scale, scale)
    np.testing.assert_allclose(alignment.rotation, rotation)
    np.testing.assert_allclose(alignment.translation, translation)
    np.testing.assert_allclose(aligned_points, dst_points)
    assert alignment.rmse < 1e-8


def test_sim3_align_points_rejects_mismatched_shapes():
    from robust_loop_verifier.geometry import sim3_align_points

    src_points = np.zeros((3, 3))
    dst_points = np.zeros((4, 3))

    with pytest.raises(ValueError, match="matching shapes"):
        sim3_align_points(src_points, dst_points)


def test_sim3_align_points_rejects_degenerate_source():
    from robust_loop_verifier.geometry import sim3_align_points

    src_points = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
        ]
    )
    dst_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    with pytest.raises(ValueError, match="Source points are degenerate"):
        sim3_align_points(src_points, dst_points)


def test_sim3_align_points_rejects_collapsed_destination():
    from robust_loop_verifier.geometry import sim3_align_points

    src_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    dst_points = np.array(
        [
            [5.0, -2.0, 1.0],
            [5.0, -2.0, 1.0],
            [5.0, -2.0, 1.0],
        ]
    )

    with pytest.raises(ValueError, match="destination|degenerate"):
        sim3_align_points(src_points, dst_points)


def test_sim3_align_points_rejects_collinear_points():
    from robust_loop_verifier.geometry import sim3_align_points

    non_collinear_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    collinear_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ]
    )

    with pytest.raises(ValueError, match="collinear|rank|degenerate"):
        sim3_align_points(collinear_points, non_collinear_points)

    with pytest.raises(ValueError, match="collinear|rank|degenerate"):
        sim3_align_points(non_collinear_points, collinear_points)
