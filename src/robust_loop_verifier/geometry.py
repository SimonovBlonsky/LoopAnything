from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_ROTATION_ATOL = 1e-7


@dataclass(frozen=True)
class Sim3Alignment:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    rmse: float


def rotation_matrix_from_quat_xyzw(quat_xyzw) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float64)
    if quat.shape != (4,):
        raise ValueError("Quaternion must have shape (4,)")
    if not np.all(np.isfinite(quat)):
        raise ValueError("Quaternion must contain only finite values")

    norm = np.linalg.norm(quat)
    if norm <= 0.0:
        raise ValueError("Quaternion norm must be positive")

    x, y, z, w = quat / norm
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
        raise ValueError("Rotation must have shape (3, 3)")
    if translation.shape != (3,):
        raise ValueError("Translation must have shape (3,)")

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def so3_log(rotation) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("Rotation must have shape (3, 3)")
    if not np.all(np.isfinite(rotation)):
        raise ValueError("Rotation must contain only finite values")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=_ROTATION_ATOL):
        raise ValueError("Rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=_ROTATION_ATOL):
        raise ValueError("Rotation must have determinant 1")

    cos_angle = (np.trace(rotation) - 1.0) * 0.5
    angle = float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    if angle < 1e-12:
        return np.zeros(3, dtype=np.float64)

    skew_vector = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    sin_angle = np.sin(angle)
    if abs(sin_angle) > 1e-8:
        return angle * skew_vector / (2.0 * sin_angle)

    axis = np.sqrt(np.maximum((np.diag(rotation) + 1.0) * 0.5, 0.0))
    axis[0] = np.copysign(axis[0], rotation[2, 1] - rotation[1, 2])
    axis[1] = np.copysign(axis[1], rotation[0, 2] - rotation[2, 0])
    axis[2] = np.copysign(axis[2], rotation[1, 0] - rotation[0, 1])
    axis_norm = np.linalg.norm(axis)
    if axis_norm <= 0.0:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        axis = axis / axis_norm
    return angle * axis


def se3_log(transform) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    invert_transform(transform)
    residual = np.empty(6, dtype=np.float64)
    residual[:3] = so3_log(transform[:3, :3])
    residual[3:] = transform[:3, 3]
    return residual


def _project_rotation(rotation) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("Rotation must have shape (3, 3)")
    if not np.all(np.isfinite(rotation)):
        raise ValueError("Rotation must contain only finite values")

    u_matrix, _, vt_matrix = np.linalg.svd(rotation)
    projected = u_matrix @ vt_matrix
    if np.linalg.det(projected) < 0.0:
        u_matrix[:, -1] *= -1.0
        projected = u_matrix @ vt_matrix
    return projected


def weighted_se3_mean(poses, weights, iterations: int = 3) -> np.ndarray:
    try:
        iterations_float = float(iterations)
    except (TypeError, ValueError) as exc:
        raise ValueError("iterations must be a finite positive integer") from exc
    if (
        not np.isfinite(iterations_float)
        or not iterations_float.is_integer()
        or iterations_float <= 0.0
    ):
        raise ValueError("iterations must be a finite positive integer")
    _iterations_count = int(iterations_float)

    if len(poses) == 0:
        raise ValueError("poses must be non-empty")

    weights_array = np.asarray(weights, dtype=np.float64)
    if weights_array.shape != (len(poses),):
        raise ValueError("weights length must match poses")
    if not np.all(np.isfinite(weights_array)):
        raise ValueError("weights must contain only finite values")
    if np.any(weights_array < 0.0):
        raise ValueError("weights must be non-negative")

    weight_sum = float(np.sum(weights_array))
    if weight_sum <= 0.0:
        raise ValueError("weights must include at least one positive value")
    normalized_weights = weights_array / weight_sum

    pose_matrices = []
    for pose in poses:
        pose_matrix = np.asarray(pose, dtype=np.float64)
        invert_transform(pose_matrix)
        pose_matrices.append(pose_matrix)

    translations = np.stack([pose[:3, 3] for pose in pose_matrices], axis=0)
    rotations = np.stack([pose[:3, :3] for pose in pose_matrices], axis=0)
    mean_translation = np.sum(normalized_weights[:, None] * translations, axis=0)
    mean_rotation = _project_rotation(
        np.sum(normalized_weights[:, None, None] * rotations, axis=0)
    )
    return make_transform(mean_rotation, mean_translation)


def invert_transform(transform) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("Transform must have shape (4, 4)")
    if not np.all(np.isfinite(transform)):
        raise ValueError("SE3 transform must contain only finite values")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=_ROTATION_ATOL):
        raise ValueError("SE3 transform bottom row must be [0, 0, 0, 1]")

    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=_ROTATION_ATOL):
        raise ValueError("SE3 rotation block must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=_ROTATION_ATOL):
        raise ValueError("SE3 rotation block must have determinant 1")

    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return inverse


def pose_between(a_c2w, b_c2w) -> np.ndarray:
    b_c2w = np.asarray(b_c2w, dtype=np.float64)
    invert_transform(b_c2w)
    return invert_transform(a_c2w) @ b_c2w


def sim3_align_points(src_points, dst_points) -> Sim3Alignment:
    src = np.asarray(src_points, dtype=np.float64)
    dst = np.asarray(dst_points, dtype=np.float64)
    if src.shape != dst.shape:
        raise ValueError("Source and destination points must have matching shapes")
    if src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("Points must have shape (N, 3)")
    if src.shape[0] < 3:
        raise ValueError("At least 3 point pairs are required")
    if not np.all(np.isfinite(src)) or not np.all(np.isfinite(dst)):
        raise ValueError("Points must contain only finite values")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    src_variance = np.mean(np.sum(src_centered * src_centered, axis=1))
    if src_variance <= 0.0:
        raise ValueError("Source points are degenerate")
    dst_variance = np.mean(np.sum(dst_centered * dst_centered, axis=1))
    if dst_variance <= 0.0:
        raise ValueError("Destination points are degenerate")
    if np.linalg.matrix_rank(src_centered) < 2:
        raise ValueError("Source points are collinear or rank-degenerate")
    if np.linalg.matrix_rank(dst_centered) < 2:
        raise ValueError("Destination points are collinear or rank-degenerate")

    covariance = (dst_centered.T @ src_centered) / src.shape[0]
    u_matrix, singular_values, vt_matrix = np.linalg.svd(covariance)
    correction = np.ones(3, dtype=np.float64)
    if np.linalg.det(u_matrix @ vt_matrix) < 0.0:
        correction[-1] = -1.0

    rotation = u_matrix @ np.diag(correction) @ vt_matrix
    rotation[np.abs(rotation) < 1e-15] = 0.0
    scale = float(np.sum(singular_values * correction) / src_variance)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Recovered Sim3 scale must be finite and positive")
    translation = dst_mean - scale * (rotation @ src_mean)
    translation[np.abs(translation) < 1e-15] = 0.0
    aligned = scale * (src @ rotation.T) + translation
    rmse = float(np.sqrt(np.mean(np.sum((aligned - dst) ** 2, axis=1))))

    return Sim3Alignment(
        scale=scale,
        rotation=rotation,
        translation=translation,
        rmse=rmse,
    )
