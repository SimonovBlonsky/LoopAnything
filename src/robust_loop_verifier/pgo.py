from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from robust_loop_verifier.geometry import invert_transform, pose_between, se3_log, sim3_align_points


@dataclass(frozen=True)
class PgoNoise:
    prior_sigmas: tuple[float, float, float, float, float, float]
    odom_sigmas: tuple[float, float, float, float, float, float]
    loop_sigmas: tuple[float, float, float, float, float, float]

    @classmethod
    def default_for_tests(cls) -> "PgoNoise":
        return cls(
            prior_sigmas=(1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6),
            odom_sigmas=(0.05, 0.05, 0.05, 0.05, 0.05, 0.05),
            loop_sigmas=(0.05, 0.05, 0.05, 0.05, 0.05, 0.05),
        )


@dataclass(frozen=True)
class PgoResult:
    converged: bool
    optimized_poses: list[np.ndarray]
    error_before: float
    error_after: float
    failure_reason: str | None
    loop_chi2_after: float | None = None
    odom_chi2_before: float | None = None
    odom_chi2_after: float | None = None
    odom_strain_chi2_after: float | None = None


def run_full_prefix_pgo(
    prefix_indices: Sequence[int],
    odom_poses: Sequence[np.ndarray],
    loop_from_idx: int,
    loop_to_idx: int,
    loop_factor: np.ndarray,
    noise: PgoNoise,
    loop_sigmas_override: Sequence[float] | None = None,
) -> PgoResult:
    original_poses = [_as_pose_matrix(pose) for pose in odom_poses]
    loop_factor = _as_pose_matrix(loop_factor)
    failure = _validate_inputs(
        prefix_indices,
        original_poses,
        loop_from_idx,
        loop_to_idx,
        loop_factor,
        noise,
    )
    if failure is not None:
        return _failed_result(original_poses, failure)

    if loop_sigmas_override is None:
        loop_sigmas = np.asarray(noise.loop_sigmas, dtype=np.float64)
    else:
        try:
            loop_sigmas = np.asarray(loop_sigmas_override, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            return _failed_result(
                original_poses,
                f"loop_sigmas_override must contain numeric values: {exc}",
            )
        failure = _validate_sigmas("loop_sigmas_override", loop_sigmas)
        if failure is not None:
            return _failed_result(original_poses, failure)

    odom_measurements = [
        pose_between(original_poses[position], original_poses[position + 1])
        for position in range(len(original_poses) - 1)
    ]

    try:
        import gtsam

        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()
        key_by_index = {
            prefix_idx: gtsam.symbol("x", position)
            for position, prefix_idx in enumerate(prefix_indices)
        }

        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.asarray(noise.prior_sigmas, dtype=np.float64)
        )
        odom_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.asarray(noise.odom_sigmas, dtype=np.float64)
        )
        loop_noise = gtsam.noiseModel.Diagonal.Sigmas(loop_sigmas)

        for prefix_idx, pose in zip(prefix_indices, original_poses):
            initial.insert(key_by_index[prefix_idx], _to_gtsam_pose3(gtsam, pose))

        first_key = key_by_index[prefix_indices[0]]
        graph.add(
            gtsam.PriorFactorPose3(
                first_key,
                _to_gtsam_pose3(gtsam, original_poses[0]),
                prior_noise,
            )
        )

        for position in range(len(original_poses) - 1):
            left_key = key_by_index[prefix_indices[position]]
            right_key = key_by_index[prefix_indices[position + 1]]
            odom_factor = odom_measurements[position]
            graph.add(
                gtsam.BetweenFactorPose3(
                    left_key,
                    right_key,
                    _to_gtsam_pose3(gtsam, odom_factor),
                    odom_noise,
                )
            )

        graph.add(
            gtsam.BetweenFactorPose3(
                key_by_index[loop_from_idx],
                key_by_index[loop_to_idx],
                _to_gtsam_pose3(gtsam, loop_factor),
                loop_noise,
            )
        )

        error_before = float(graph.error(initial))
        if not np.isfinite(error_before):
            return _failed_result(
                original_poses,
                "non-finite error_before after graph construction",
            )

        params = gtsam.LevenbergMarquardtParams()
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
        optimized_values = optimizer.optimize()
        error_after = float(graph.error(optimized_values))
        if not np.isfinite(error_after):
            return _failed_result(original_poses, "non-finite error_after after optimization")

        optimized_poses = [
            np.asarray(
                optimized_values.atPose3(key_by_index[prefix_idx]).matrix(),
                dtype=np.float64,
            )
            for prefix_idx in prefix_indices
        ]
        failure = _validate_pose_sequence("optimized_poses", optimized_poses)
        if failure is not None:
            return _failed_result(original_poses, failure)

        odom_chi2_before = _odom_chi2(
            original_poses,
            odom_measurements,
            noise.odom_sigmas,
        )
        odom_chi2_after = _odom_chi2(
            optimized_poses,
            odom_measurements,
            noise.odom_sigmas,
        )
        loop_chi2_after = _between_chi2(
            optimized_poses[prefix_indices.index(loop_from_idx)],
            optimized_poses[prefix_indices.index(loop_to_idx)],
            loop_factor,
            loop_sigmas,
        )
        odom_strain_chi2_after = max(0.0, odom_chi2_after - odom_chi2_before)
    except Exception as exc:
        return _failed_result(original_poses, f"{type(exc).__name__}: {exc}")

    return PgoResult(
        converged=True,
        optimized_poses=optimized_poses,
        error_before=error_before,
        error_after=error_after,
        failure_reason=None,
        loop_chi2_after=loop_chi2_after,
        odom_chi2_before=odom_chi2_before,
        odom_chi2_after=odom_chi2_after,
        odom_strain_chi2_after=odom_strain_chi2_after,
    )


def trajectory_deformation_rmse(
    original_poses: Sequence[np.ndarray],
    optimized_poses: Sequence[np.ndarray],
) -> float:
    original = _translations(original_poses)
    optimized = _translations(optimized_poses)
    if original.shape != optimized.shape:
        raise ValueError("Original and optimized pose sequences must have matching lengths")
    if original.shape[0] == 0:
        return 0.0

    try:
        return sim3_align_points(optimized, original).rmse
    except ValueError:
        return _rank_degenerate_sim3_rmse(optimized, original)


def _validate_inputs(
    prefix_indices: Sequence[int],
    odom_poses: Sequence[np.ndarray],
    loop_from_idx: int,
    loop_to_idx: int,
    loop_factor: np.ndarray,
    noise: PgoNoise,
) -> str | None:
    if len(prefix_indices) != len(odom_poses):
        return "prefix_indices and odom_poses length mismatch"
    if len(prefix_indices) == 0:
        return "prefix_indices must not be empty"
    if len(set(prefix_indices)) != len(prefix_indices):
        return "prefix_indices must be unique"
    if loop_from_idx not in prefix_indices:
        return "loop_from_idx is not in prefix_indices"
    if loop_to_idx not in prefix_indices:
        return "loop_to_idx is not in prefix_indices"
    if loop_from_idx == loop_to_idx:
        return "loop_from_idx and loop_to_idx must differ"

    failure = _validate_pose_sequence("odom_poses", odom_poses)
    if failure is not None:
        return failure
    failure = _validate_se3("loop_factor", loop_factor)
    if failure is not None:
        return failure
    failure = _validate_noise(noise)
    if failure is not None:
        return failure
    return None


def _as_pose_matrix(pose) -> np.ndarray:
    return np.asarray(pose, dtype=np.float64).copy()


def _to_gtsam_pose3(gtsam, transform: np.ndarray):
    transform = np.asarray(transform, dtype=np.float64)
    return gtsam.Pose3(gtsam.Rot3(transform[:3, :3]), transform[:3, 3])


def _validate_pose_sequence(name: str, poses: Sequence[np.ndarray]) -> str | None:
    for index, pose in enumerate(poses):
        failure = _validate_se3(f"{name}[{index}]", pose)
        if failure is not None:
            return failure
    return None


def _validate_se3(name: str, transform: np.ndarray) -> str | None:
    try:
        invert_transform(transform)
    except ValueError as exc:
        return f"{name}: {exc}"
    return None


def _validate_noise(noise: PgoNoise) -> str | None:
    for name in ("prior_sigmas", "odom_sigmas", "loop_sigmas"):
        try:
            sigmas = np.asarray(getattr(noise, name), dtype=np.float64)
        except (TypeError, ValueError) as exc:
            return f"{name} must contain numeric values: {exc}"
        failure = _validate_sigmas(name, sigmas)
        if failure is not None:
            return failure
    return None


def _validate_sigmas(name: str, sigmas: np.ndarray) -> str | None:
    if sigmas.shape != (6,):
        return f"{name} must contain exactly six values"
    if not np.all(np.isfinite(sigmas)):
        return f"{name} must contain only finite values"
    if not np.all(sigmas > 0.0):
        return f"{name} must contain only positive values"
    return None


def _between_chi2(
    left: np.ndarray,
    right: np.ndarray,
    measured: np.ndarray,
    sigmas: Sequence[float],
) -> float:
    predicted = pose_between(left, right)
    residual_transform = invert_transform(measured) @ predicted
    residual = se3_log(residual_transform)
    sigma_array = np.asarray(sigmas, dtype=np.float64)
    normalized = residual / sigma_array
    return float(np.dot(normalized, normalized))


def _odom_chi2(
    poses: Sequence[np.ndarray],
    odom_measurements: Sequence[np.ndarray],
    sigmas: Sequence[float],
) -> float:
    return float(
        sum(
            _between_chi2(poses[position], poses[position + 1], measurement, sigmas)
            for position, measurement in enumerate(odom_measurements)
        )
    )


def _failed_result(original_poses: Sequence[np.ndarray], reason: str) -> PgoResult:
    return PgoResult(
        converged=False,
        optimized_poses=[np.asarray(pose, dtype=np.float64).copy() for pose in original_poses],
        error_before=float("inf"),
        error_after=float("inf"),
        failure_reason=reason,
    )


def _translations(poses: Sequence[np.ndarray]) -> np.ndarray:
    return np.asarray(
        [np.asarray(pose, dtype=np.float64)[:3, 3] for pose in poses],
        dtype=np.float64,
    )


def _rank_degenerate_sim3_rmse(src: np.ndarray, dst: np.ndarray) -> float:
    if not np.all(np.isfinite(src)) or not np.all(np.isfinite(dst)):
        raise ValueError("Points must contain only finite values")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    src_variance = np.mean(np.sum(src_centered * src_centered, axis=1))

    if src_variance <= 0.0:
        aligned = src + (dst_mean - src_mean)
        return float(np.sqrt(np.mean(np.sum((aligned - dst) ** 2, axis=1))))

    covariance = (dst_centered.T @ src_centered) / src.shape[0]
    u_matrix, singular_values, vt_matrix = np.linalg.svd(covariance)
    correction = np.ones(3, dtype=np.float64)
    if np.linalg.det(u_matrix @ vt_matrix) < 0.0:
        correction[-1] = -1.0

    rotation = u_matrix @ np.diag(correction) @ vt_matrix
    scale = float(np.sum(singular_values * correction) / src_variance)
    if not np.isfinite(scale) or scale <= 0.0:
        aligned = src + (dst_mean - src_mean)
        return float(np.sqrt(np.mean(np.sum((aligned - dst) ** 2, axis=1))))

    translation = dst_mean - scale * (rotation @ src_mean)
    aligned = scale * (src @ rotation.T) + translation
    return float(np.sqrt(np.mean(np.sum((aligned - dst) ** 2, axis=1))))
