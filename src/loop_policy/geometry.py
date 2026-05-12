from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PoseResidual:
    rotation_deg: float
    translation_norm: float


def _as_transform(transform, name: str) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 transform")
    return transform


def quaternion_xyzw_to_matrix(q_xyzw) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError("q_xyzw must contain 4 values")

    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("q_xyzw must have a positive norm")

    q = q / norm
    if q[3] < 0.0:
        q = -q

    x, y, z, w = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def make_transform(rotation, translation) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("rotation must be a 3x3 matrix")
    if translation.shape != (3,):
        raise ValueError("translation must contain 3 values")

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def invert_transform(transform) -> np.ndarray:
    transform = _as_transform(transform, "transform")

    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def relative_transform(a_world, b_world) -> np.ndarray:
    a_world = _as_transform(a_world, "a_world")
    b_world = _as_transform(b_world, "b_world")
    return invert_transform(a_world) @ b_world


def rotation_angle_deg(rotation) -> float:
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("rotation must be a 3x3 matrix")

    cos_angle = (float(np.trace(rotation)) - 1.0) * 0.5
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def pose_residual(reference, estimate) -> PoseResidual:
    reference = _as_transform(reference, "reference")
    estimate = _as_transform(estimate, "estimate")
    residual = relative_transform(reference, estimate)
    return PoseResidual(
        rotation_deg=rotation_angle_deg(residual[:3, :3]),
        translation_norm=float(np.linalg.norm(residual[:3, 3])),
    )


def camera_pose_from_lidar_pose(t_world_lidar, t_camera_lidar) -> np.ndarray:
    t_world_lidar = _as_transform(t_world_lidar, "t_world_lidar")
    t_camera_lidar = _as_transform(t_camera_lidar, "t_camera_lidar")
    return t_world_lidar @ invert_transform(t_camera_lidar)


def camera_center_baseline(a_world_camera, b_world_camera) -> float:
    a_world_camera = _as_transform(a_world_camera, "a_world_camera")
    b_world_camera = _as_transform(b_world_camera, "b_world_camera")
    return float(np.linalg.norm(a_world_camera[:3, 3] - b_world_camera[:3, 3]))
