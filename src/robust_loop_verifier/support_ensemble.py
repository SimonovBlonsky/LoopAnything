from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from robust_loop_verifier.geometry import invert_transform, se3_log, weighted_se3_mean


@dataclass(frozen=True)
class SupportEnsembleConfig:
    sigma_rot_floor: float
    sigma_trans_floor: float
    covariance_scale: float
    c_align: float
    c_consensus: float
    lambda_dir: float
    robust_iterations: int

    @staticmethod
    def default() -> "SupportEnsembleConfig":
        return SupportEnsembleConfig(
            sigma_rot_floor=0.05,
            sigma_trans_floor=0.25,
            covariance_scale=1.0,
            c_align=1.0,
            c_consensus=2.0,
            lambda_dir=1.0,
            robust_iterations=3,
        )


@dataclass(frozen=True)
class SupportLoopFactor:
    support_idx: int
    loop_factor: np.ndarray
    support_alignment_residual_m: float
    direction_error_deg: float
    candidate_support_baseline_m: float


@dataclass(frozen=True)
class SupportEnsembleResult:
    valid: bool
    loop_factor_mean: np.ndarray | None
    loop_sigmas: tuple[float, float, float, float, float, float]
    support_weights: dict[int, float]
    support_residual_norms: dict[int, float]
    effective_support_count: float
    sigma_rot: float
    sigma_trans: float
    uncertainty_logdet_penalty: float
    rejection_reason: str | None = None


def cauchy_weight(normalized_error) -> float:
    error = abs(float(normalized_error))
    if not math.isfinite(error):
        return 0.0
    return 1.0 / (1.0 + error * error)


def huber_weight(normalized_error) -> float:
    error = abs(float(normalized_error))
    if not math.isfinite(error):
        return 0.0
    if error <= 1.0:
        return 1.0
    return 1.0 / error


def aggregate_support_loop_factors(
    factors, config: SupportEnsembleConfig
) -> SupportEnsembleResult:
    config_error = _config_rejection_reason(config)
    if config_error is not None:
        return _invalid_result_from_values(config, config_error)

    factor_list = list(factors)
    if not factor_list:
        return _invalid_result(config, "no_valid_support_loop_factors")

    loop_factors = []
    support_indices = set()
    for factor in factor_list:
        if factor.support_idx in support_indices:
            return _invalid_result(config, "duplicate_support_idx")
        support_indices.add(factor.support_idx)
        try:
            loop_factor = np.asarray(factor.loop_factor, dtype=np.float64)
            invert_transform(loop_factor)
        except (AttributeError, TypeError, ValueError):
            return _invalid_result(config, "invalid_loop_factor")
        loop_factors.append(loop_factor)

    try:
        alignment_weights = np.array(
            [_alignment_weight(factor, config) for factor in factor_list],
            dtype=np.float64,
        )
    except (AttributeError, TypeError, ValueError):
        return _invalid_result(config, "invalid_support_metric")

    if not np.all(np.isfinite(alignment_weights)):
        return _invalid_result(config, "invalid_support_metric")

    mean = _weighted_mean_with_uniform_fallback(loop_factors, alignment_weights)
    final_weights = alignment_weights.copy()

    for _ in range(config.robust_iterations):
        residuals = _se3_residuals(mean, loop_factors)
        normalized_residuals = np.array(
            [_normalized_residual_norm(residual, config) for residual in residuals],
            dtype=np.float64,
        )
        consensus_weights = np.array(
            [
                huber_weight(normalized / config.c_consensus)
                for normalized in normalized_residuals
            ],
            dtype=np.float64,
        )
        candidate_weights = alignment_weights * consensus_weights
        final_weights = candidate_weights
        mean = _weighted_mean_with_uniform_fallback(loop_factors, candidate_weights)

    residuals = _se3_residuals(mean, loop_factors)
    residual_norms = np.array(
        [_normalized_residual_norm(residual, config) for residual in residuals],
        dtype=np.float64,
    )
    weights_for_moments = _positive_or_uniform_weights(final_weights)
    sigma_rot = (
        math.sqrt(_weighted_component_mse(residuals[:, :3], weights_for_moments))
        * config.covariance_scale
        + config.sigma_rot_floor
    )
    sigma_trans = (
        math.sqrt(_weighted_component_mse(residuals[:, 3:], weights_for_moments))
        * config.covariance_scale
        + config.sigma_trans_floor
    )
    penalty = _uncertainty_logdet_penalty(sigma_rot, sigma_trans, config)
    support_weights = {
        factor.support_idx: float(weight)
        for factor, weight in zip(factor_list, weights_for_moments)
    }
    support_residual_norms = {
        factor.support_idx: float(norm) for factor, norm in zip(factor_list, residual_norms)
    }

    return SupportEnsembleResult(
        valid=True,
        loop_factor_mean=mean,
        loop_sigmas=(sigma_rot, sigma_rot, sigma_rot, sigma_trans, sigma_trans, sigma_trans),
        support_weights=support_weights,
        support_residual_norms=support_residual_norms,
        effective_support_count=_effective_support_count(weights_for_moments),
        sigma_rot=sigma_rot,
        sigma_trans=sigma_trans,
        uncertainty_logdet_penalty=penalty,
    )


def graph_evidence_nll(
    loop_chi2_after, odom_strain_chi2_after, uncertainty_logdet_penalty
) -> float:
    terms = (
        float(loop_chi2_after),
        float(odom_strain_chi2_after),
        float(uncertainty_logdet_penalty),
    )
    if not all(math.isfinite(term) for term in terms):
        return math.inf
    return 0.5 * sum(terms)


def _config_rejection_reason(config: SupportEnsembleConfig) -> str | None:
    positive_fields = (
        "sigma_rot_floor",
        "sigma_trans_floor",
        "covariance_scale",
        "c_align",
        "c_consensus",
    )
    for field_name in positive_fields:
        value = getattr(config, field_name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0.0
        ):
            return f"invalid_config_{field_name}"

    lambda_dir = config.lambda_dir
    if (
        isinstance(lambda_dir, bool)
        or not isinstance(lambda_dir, (int, float))
        or not math.isfinite(lambda_dir)
        or lambda_dir < 0.0
    ):
        return "invalid_config_lambda_dir"

    robust_iterations = config.robust_iterations
    if (
        isinstance(robust_iterations, bool)
        or not isinstance(robust_iterations, int)
        or robust_iterations <= 0
    ):
        return "invalid_config_robust_iterations"
    return None


def _invalid_result(config: SupportEnsembleConfig, rejection_reason: str) -> SupportEnsembleResult:
    sigma_rot = float(config.sigma_rot_floor)
    sigma_trans = float(config.sigma_trans_floor)
    return _invalid_result_with_sigmas(sigma_rot, sigma_trans, rejection_reason)


def _invalid_result_from_values(
    config: SupportEnsembleConfig, rejection_reason: str
) -> SupportEnsembleResult:
    sigma_rot = _safe_positive_float(config.sigma_rot_floor, 0.0)
    sigma_trans = _safe_positive_float(config.sigma_trans_floor, 0.0)
    return _invalid_result_with_sigmas(sigma_rot, sigma_trans, rejection_reason)


def _invalid_result_with_sigmas(
    sigma_rot: float, sigma_trans: float, rejection_reason: str
) -> SupportEnsembleResult:
    return SupportEnsembleResult(
        valid=False,
        loop_factor_mean=None,
        loop_sigmas=(sigma_rot, sigma_rot, sigma_rot, sigma_trans, sigma_trans, sigma_trans),
        support_weights={},
        support_residual_norms={},
        effective_support_count=0.0,
        sigma_rot=sigma_rot,
        sigma_trans=sigma_trans,
        uncertainty_logdet_penalty=0.0,
        rejection_reason=rejection_reason,
    )


def _alignment_weight(factor: SupportLoopFactor, config: SupportEnsembleConfig) -> float:
    residual_m = float(factor.support_alignment_residual_m)
    direction_error_deg = float(factor.direction_error_deg)
    baseline_m = float(factor.candidate_support_baseline_m)
    if not math.isfinite(residual_m) or residual_m < 0.0:
        raise ValueError("support_alignment_residual_m must be finite and non-negative")
    if not math.isfinite(baseline_m) or baseline_m <= 0.0:
        raise ValueError("candidate_support_baseline_m must be finite and positive")
    if not math.isfinite(direction_error_deg) or direction_error_deg < 0.0:
        raise ValueError("direction_error_deg must be finite and non-negative")
    direction_rad = math.radians(direction_error_deg)
    alignment_error = residual_m / max(baseline_m, 1e-9) + config.lambda_dir * (
        2.0 * math.sin(direction_rad * 0.5)
    )
    return cauchy_weight(alignment_error / config.c_align)


def _safe_positive_float(value, fallback: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return fallback
    value_float = float(value)
    if not math.isfinite(value_float) or value_float < 0.0:
        return fallback
    return value_float


def _weighted_mean_with_uniform_fallback(loop_factors, weights: np.ndarray) -> np.ndarray:
    if float(np.sum(weights)) > 0.0:
        return weighted_se3_mean(loop_factors, weights)
    return weighted_se3_mean(loop_factors, np.ones(len(loop_factors), dtype=np.float64))


def _positive_or_uniform_weights(weights: np.ndarray) -> np.ndarray:
    if float(np.sum(weights)) > 0.0:
        return weights
    return np.ones(weights.shape, dtype=np.float64)


def _se3_residuals(mean: np.ndarray, loop_factors) -> np.ndarray:
    mean_inverse = invert_transform(mean)
    return np.stack([se3_log(mean_inverse @ loop_factor) for loop_factor in loop_factors], axis=0)


def _normalized_residual_norm(residual: np.ndarray, config: SupportEnsembleConfig) -> float:
    rot = residual[:3] / config.sigma_rot_floor
    trans = residual[3:] / config.sigma_trans_floor
    return float(math.sqrt(float(np.sum(rot * rot) + np.sum(trans * trans))))


def _weighted_component_mse(components: np.ndarray, weights: np.ndarray) -> float:
    normalized_weights = weights / float(np.sum(weights))
    return float(np.sum(normalized_weights[:, None] * components * components) / 3.0)


def _uncertainty_logdet_penalty(
    sigma_rot: float, sigma_trans: float, config: SupportEnsembleConfig
) -> float:
    penalty = 3.0 * math.log((sigma_rot * sigma_rot) / (config.sigma_rot_floor**2))
    penalty += 3.0 * math.log((sigma_trans * sigma_trans) / (config.sigma_trans_floor**2))
    if -1e-12 < penalty < 0.0:
        return 0.0
    return max(0.0, penalty)


def _effective_support_count(weights: np.ndarray) -> float:
    weight_sum = float(np.sum(weights))
    squared_sum = float(np.sum(weights * weights))
    if weight_sum <= 0.0 or squared_sum <= 0.0:
        return 0.0
    return (weight_sum * weight_sum) / squared_sum
