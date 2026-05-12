import math

import numpy as np
import pytest

from loop_policy.geometry import (
    camera_center_baseline,
    camera_pose_from_lidar_pose,
    invert_transform,
    make_transform,
    pose_residual,
    quaternion_xyzw_to_matrix,
    relative_transform,
)


def test_quaternion_xyzw_to_matrix_identity():
    rotation = quaternion_xyzw_to_matrix((0.0, 0.0, 0.0, 1.0))

    np.testing.assert_allclose(rotation, np.eye(3))


def test_relative_transform_and_residual_translation():
    a_world = make_transform(np.eye(3), [0.0, 0.0, 0.0])
    b_world = make_transform(np.eye(3), [3.0, 4.0, 0.0])

    relative = relative_transform(a_world, b_world)
    residual = pose_residual(a_world, b_world)

    np.testing.assert_allclose(relative[:3, 3], [3.0, 4.0, 0.0])
    assert residual.rotation_deg == 0.0
    assert residual.translation_norm == 5.0


def test_pose_residual_rotation_degrees():
    reference = make_transform(np.eye(3), [0.0, 0.0, 0.0])
    z_quarter_turn = quaternion_xyzw_to_matrix(
        (0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0))
    )
    estimate = make_transform(z_quarter_turn, [0.0, 0.0, 0.0])

    residual = pose_residual(reference, estimate)

    assert math.isclose(residual.rotation_deg, 90.0)


def test_camera_pose_from_lidar_pose_uses_inverse_t_camera_lidar():
    t_world_lidar = make_transform(np.eye(3), [10.0, 0.0, 0.0])
    t_camera_lidar = make_transform(np.eye(3), [0.5, 0.0, 0.0])

    t_world_camera = camera_pose_from_lidar_pose(t_world_lidar, t_camera_lidar)

    np.testing.assert_allclose(t_world_camera[:3, 3], [9.5, 0.0, 0.0])


def test_camera_pose_from_lidar_pose_uses_rotated_extrinsic_inverse():
    t_world_lidar = make_transform(np.eye(3), [10.0, 0.0, 0.0])
    z_quarter_turn = quaternion_xyzw_to_matrix(
        (0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0))
    )
    t_camera_lidar = make_transform(z_quarter_turn, [0.5, 2.0, 0.0])

    t_world_camera = camera_pose_from_lidar_pose(t_world_lidar, t_camera_lidar)
    expected = t_world_lidar @ invert_transform(t_camera_lidar)

    np.testing.assert_allclose(t_world_camera, expected, atol=1e-12)
    np.testing.assert_allclose(t_world_camera[:3, 3], [8.0, 0.5, 0.0], atol=1e-12)


def test_camera_center_baseline():
    a_world_camera = make_transform(np.eye(3), [1.0, 2.0, 3.0])
    b_world_camera = make_transform(np.eye(3), [1.0, 6.0, 3.0])

    assert camera_center_baseline(a_world_camera, b_world_camera) == 4.0
    np.testing.assert_allclose(invert_transform(np.eye(4)), np.eye(4))


@pytest.mark.parametrize(
    ("a_world", "b_world", "message"),
    [
        (np.eye(3), np.eye(4), "a_world must be a 4x4 transform"),
        (np.eye(4), np.eye(3), "b_world must be a 4x4 transform"),
    ],
)
def test_relative_transform_rejects_malformed_transform_inputs(a_world, b_world, message):
    with pytest.raises(ValueError, match=message):
        relative_transform(a_world, b_world)


@pytest.mark.parametrize(
    ("a_world_camera", "b_world_camera", "message"),
    [
        (np.eye(3), np.eye(4), "a_world_camera must be a 4x4 transform"),
        (np.eye(4), np.eye(3), "b_world_camera must be a 4x4 transform"),
    ],
)
def test_camera_center_baseline_rejects_malformed_transform_inputs(
    a_world_camera, b_world_camera, message
):
    with pytest.raises(ValueError, match=message):
        camera_center_baseline(a_world_camera, b_world_camera)


@pytest.mark.parametrize(
    ("t_world_lidar", "t_camera_lidar", "message"),
    [
        (np.eye(3), np.eye(4), "t_world_lidar must be a 4x4 transform"),
        (np.eye(4), np.eye(3), "t_camera_lidar must be a 4x4 transform"),
    ],
)
def test_camera_pose_from_lidar_pose_rejects_malformed_transform_inputs(
    t_world_lidar, t_camera_lidar, message
):
    with pytest.raises(ValueError, match=message):
        camera_pose_from_lidar_pose(t_world_lidar, t_camera_lidar)
