from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from .geometry import invert_transform, pose_between

_DIRECTION_NORM_EPS = 1e-12
_AXIS_ZERO_ANGLE_RAD = 1e-12
_AXIS_NEAR_PI_RAD = 1e-7
_SE3_ATOL = 1e-7
_TRANSLATION_COMPARE_ATOL_M = 1e-9
_ANGLE_COMPARE_ATOL_DEG = 1e-9
_BASELINE_COMPARE_ATOL_M = 1e-9


@dataclass(frozen=True)
class GeometryLabelThresholds:
    min_translation_error_m: float = 1.0
    max_translation_error_m: float = 5.0
    translation_error_scale_ratio: float = 0.2
    max_rotation_error_deg: float = 15.0
    max_translation_direction_error_deg: float = 20.0
    min_direction_baseline_m: float = 0.5

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            try:
                value = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{field.name} must be a finite non-negative number") from exc
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{field.name} must be a finite non-negative number")
            object.__setattr__(self, field.name, value)
        if self.min_translation_error_m > self.max_translation_error_m:
            raise ValueError("min_translation_error_m must not exceed max_translation_error_m")

    def translation_error_threshold_m(self, gt_translation_norm_m: float) -> float:
        scaled_threshold = self.translation_error_scale_ratio * gt_translation_norm_m
        return float(
            np.clip(
                scaled_threshold,
                self.min_translation_error_m,
                self.max_translation_error_m,
            )
        )


def _validated_transform(transform: np.ndarray, name: str) -> np.ndarray:
    if np.iscomplexobj(transform):
        raise ValueError(f"{name} must be a real-valued SE(3) transform")
    try:
        matrix = np.asarray(transform, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite SE(3) transform") from exc

    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(
        matrix[3],
        [0.0, 0.0, 0.0, 1.0],
        atol=_SE3_ATOL,
        rtol=0.0,
    ):
        raise ValueError(f"{name} bottom row must be [0, 0, 0, 1]")

    rotation = matrix[:3, :3]
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=_SE3_ATOL,
        rtol=0.0,
    ):
        raise ValueError(f"{name} rotation block must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=_SE3_ATOL, rtol=0.0):
        raise ValueError(f"{name} rotation block must have determinant 1")
    return matrix


def _axis_angle(rotation: np.ndarray) -> tuple[list[float] | None, float]:
    skew_vector = 0.5 * np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    sin_angle = float(np.linalg.norm(skew_vector))
    cos_angle = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle_rad = float(np.arctan2(sin_angle, cos_angle))
    if angle_rad <= _AXIS_ZERO_ANGLE_RAD:
        return None, 0.0

    if np.pi - angle_rad <= _AXIS_NEAR_PI_RAD:
        symmetric_rotation = 0.5 * (rotation + rotation.T)
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric_rotation)
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        if sin_angle > _AXIS_ZERO_ANGLE_RAD and np.dot(axis, skew_vector) < 0.0:
            axis = -axis
    else:
        axis = skew_vector / sin_angle
    axis = axis / np.linalg.norm(axis)
    return axis.tolist(), float(np.rad2deg(angle_rad))


def _translation_direction_error_deg(
    gt_translation: np.ndarray,
    estimated_translation: np.ndarray,
) -> float | None:
    gt_norm = float(np.linalg.norm(gt_translation))
    estimated_norm = float(np.linalg.norm(estimated_translation))
    if gt_norm <= _DIRECTION_NORM_EPS or estimated_norm <= _DIRECTION_NORM_EPS:
        return None
    cosine = float(np.dot(gt_translation, estimated_translation) / (gt_norm * estimated_norm))
    return float(np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _exceeds(value: float, threshold: float, tolerance: float) -> bool:
    return value > threshold + tolerance


def _is_below(value: float, threshold: float, tolerance: float) -> bool:
    return value < threshold - tolerance


def evaluate_metric_loop_factor(
    gt_query_c2w: np.ndarray,
    gt_candidate_c2w: np.ndarray,
    estimated_query_to_candidate: np.ndarray | None,
    *,
    factor_status: str,
    thresholds: GeometryLabelThresholds,
) -> dict[str, object]:
    if not isinstance(thresholds, GeometryLabelThresholds):
        raise ValueError("thresholds must be a GeometryLabelThresholds instance")
    if not isinstance(factor_status, str):
        raise ValueError("factor_status must be a string")

    gt_query = _validated_transform(gt_query_c2w, "gt_query_c2w")
    gt_candidate = _validated_transform(gt_candidate_c2w, "gt_candidate_c2w")
    gt_relative = pose_between(gt_query, gt_candidate)
    gt_axis, gt_angle_deg = _axis_angle(gt_relative[:3, :3])
    gt_translation = gt_relative[:3, 3]
    gt_translation_norm = float(np.linalg.norm(gt_translation))
    translation_error_threshold = thresholds.translation_error_threshold_m(gt_translation_norm)

    result: dict[str, object] = {
        "gt_relative_pose": gt_relative.tolist(),
        "estimated_relative_pose": None,
        "gt_rotation_axis": gt_axis,
        "gt_rotation_angle_deg": gt_angle_deg,
        "estimated_rotation_axis": None,
        "estimated_rotation_angle_deg": None,
        "gt_translation": gt_translation.tolist(),
        "gt_translation_norm_m": gt_translation_norm,
        "estimated_translation": None,
        "estimated_translation_norm_m": None,
        "translation_error_m": None,
        "effective_translation_error_threshold_m": translation_error_threshold,
        "rotation_error_deg": None,
        "translation_direction_error_deg": None,
        "factor_status": factor_status,
        "automatic_label": 0,
        "automatic_label_reason": "missing_estimate",
    }

    if estimated_query_to_candidate is None:
        return result

    try:
        estimated = _validated_transform(
            estimated_query_to_candidate,
            "estimated_query_to_candidate",
        )
    except ValueError:
        result["automatic_label_reason"] = "invalid_estimate"
        return result

    estimated_axis, estimated_angle_deg = _axis_angle(estimated[:3, :3])
    estimated_translation = estimated[:3, 3]
    estimated_translation_norm = float(np.linalg.norm(estimated_translation))
    error_transform = invert_transform(gt_relative) @ estimated
    _, rotation_error_deg = _axis_angle(error_transform[:3, :3])
    translation_error = float(np.linalg.norm(error_transform[:3, 3]))
    direction_error_deg = _translation_direction_error_deg(
        gt_translation,
        estimated_translation,
    )

    result.update(
        {
            "estimated_relative_pose": estimated.tolist(),
            "estimated_rotation_axis": estimated_axis,
            "estimated_rotation_angle_deg": estimated_angle_deg,
            "estimated_translation": estimated_translation.tolist(),
            "estimated_translation_norm_m": estimated_translation_norm,
            "translation_error_m": translation_error,
            "rotation_error_deg": rotation_error_deg,
            "translation_direction_error_deg": direction_error_deg,
        }
    )

    if factor_status != "ok":
        reason = "factor_status_not_ok"
    elif _exceeds(
        translation_error,
        translation_error_threshold,
        _TRANSLATION_COMPARE_ATOL_M,
    ):
        reason = "translation_error_exceeds_threshold"
    elif _exceeds(
        rotation_error_deg,
        thresholds.max_rotation_error_deg,
        _ANGLE_COMPARE_ATOL_DEG,
    ):
        reason = "rotation_error_exceeds_threshold"
    elif not _is_below(
        gt_translation_norm,
        thresholds.min_direction_baseline_m,
        _BASELINE_COMPARE_ATOL_M,
    ):
        if direction_error_deg is None:
            reason = "translation_direction_unavailable"
        elif _exceeds(
            direction_error_deg,
            thresholds.max_translation_direction_error_deg,
            _ANGLE_COMPARE_ATOL_DEG,
        ):
            reason = "translation_direction_error_exceeds_threshold"
        else:
            reason = "accepted"
    else:
        reason = "accepted"

    result["automatic_label_reason"] = reason
    result["automatic_label"] = int(reason == "accepted")
    return result
