import math

import numpy as np

from robust_loop_verifier.geometry import make_transform
from robust_loop_verifier.support_ensemble import (
    SupportEnsembleConfig,
    SupportLoopFactor,
    aggregate_support_loop_factors,
    cauchy_weight,
    graph_evidence_nll,
    huber_weight,
)


def _factor(support_idx: int, translation, alignment_residual_m: float = 0.0):
    return SupportLoopFactor(
        support_idx=support_idx,
        loop_factor=make_transform(np.eye(3), translation),
        support_alignment_residual_m=alignment_residual_m,
        direction_error_deg=0.0,
        candidate_support_baseline_m=1.0,
    )


def test_robust_weight_functions_are_bounded_and_piecewise():
    assert cauchy_weight(0.0) == 1.0
    assert 0.0 < cauchy_weight(1e9) < 1e-12
    assert huber_weight(0.5) == 1.0
    assert huber_weight(4.0) == 0.25


def test_support_ensemble_downweights_translation_outlier_and_reports_uncertainty():
    config = SupportEnsembleConfig(
        sigma_rot_floor=0.05,
        sigma_trans_floor=0.25,
        covariance_scale=1.0,
        c_align=1.0,
        c_consensus=1.0,
        lambda_dir=1.0,
        robust_iterations=3,
    )
    factors = [
        _factor(1, [0.0, 0.0, 0.0]),
        _factor(2, [0.08, 0.0, 0.0]),
        _factor(3, [9.0, 0.0, 0.0], alignment_residual_m=20.0),
    ]

    result = aggregate_support_loop_factors(factors, config)

    assert result.valid is True
    assert result.rejection_reason is None
    assert result.loop_factor_mean is not None
    assert result.loop_factor_mean[:3, 3][0] < 0.25
    assert result.support_weights[3] < result.support_weights[1]
    assert result.effective_support_count > 1.5
    assert result.loop_sigmas == (
        result.sigma_rot,
        result.sigma_rot,
        result.sigma_rot,
        result.sigma_trans,
        result.sigma_trans,
        result.sigma_trans,
    )
    assert result.sigma_rot >= config.sigma_rot_floor
    assert result.sigma_trans >= config.sigma_trans_floor
    assert result.uncertainty_logdet_penalty >= 0.0
    assert set(result.support_residual_norms) == {1, 2, 3}


def test_single_support_returns_floor_covariance():
    config = SupportEnsembleConfig.default()

    result = aggregate_support_loop_factors([_factor(7, [1.0, 2.0, 3.0])], config)

    assert result.valid is True
    np.testing.assert_allclose(result.loop_factor_mean[:3, 3], [1.0, 2.0, 3.0])
    assert result.effective_support_count == 1.0
    assert result.sigma_rot == config.sigma_rot_floor
    assert result.sigma_trans == config.sigma_trans_floor
    assert result.loop_sigmas == (
        config.sigma_rot_floor,
        config.sigma_rot_floor,
        config.sigma_rot_floor,
        config.sigma_trans_floor,
        config.sigma_trans_floor,
        config.sigma_trans_floor,
    )
    assert result.uncertainty_logdet_penalty == 0.0


def test_all_zero_final_weights_use_uniform_fallback_for_effective_support_count():
    config = SupportEnsembleConfig.default()
    factors = [
        _factor(1, [0.0, 0.0, 0.0], alignment_residual_m=1e200),
        _factor(2, [2.0, 0.0, 0.0], alignment_residual_m=1e200),
        _factor(3, [4.0, 0.0, 0.0], alignment_residual_m=1e200),
    ]

    result = aggregate_support_loop_factors(factors, config)

    assert result.valid is True
    assert result.effective_support_count == 3.0
    assert result.support_weights == {1: 1.0, 2: 1.0, 3: 1.0}


def test_graph_evidence_nll_averages_finite_terms():
    assert graph_evidence_nll(2, 4, 6) == 6.0


def test_support_ensemble_rejects_empty_factor_list():
    result = aggregate_support_loop_factors([], SupportEnsembleConfig.default())

    assert result.valid is False
    assert result.rejection_reason == "no_valid_support_loop_factors"
    assert result.loop_factor_mean is None


def test_support_ensemble_rejects_invalid_loop_factor_without_crashing():
    invalid_factor = SupportLoopFactor(
        support_idx=1,
        loop_factor=np.eye(3),
        support_alignment_residual_m=0.0,
        direction_error_deg=0.0,
        candidate_support_baseline_m=1.0,
    )

    result = aggregate_support_loop_factors([invalid_factor], SupportEnsembleConfig.default())

    assert result.valid is False
    assert result.rejection_reason == "invalid_loop_factor"


def test_support_ensemble_rejects_invalid_config_without_crashing():
    valid_factor = _factor(1, [0.0, 0.0, 0.0])
    invalid_configs = [
        SupportEnsembleConfig(
            sigma_rot_floor=0.05,
            sigma_trans_floor=0.25,
            covariance_scale=1.0,
            c_align=1.0,
            c_consensus=2.0,
            lambda_dir=1.0,
            robust_iterations=3.0,
        ),
        SupportEnsembleConfig(
            sigma_rot_floor=0.05,
            sigma_trans_floor=0.25,
            covariance_scale=1.0,
            c_align=1.0,
            c_consensus=2.0,
            lambda_dir=1.0,
            robust_iterations="3",
        ),
        SupportEnsembleConfig(
            sigma_rot_floor=True,
            sigma_trans_floor=0.25,
            covariance_scale=1.0,
            c_align=1.0,
            c_consensus=2.0,
            lambda_dir=1.0,
            robust_iterations=3,
        ),
        SupportEnsembleConfig(
            sigma_rot_floor=0.05,
            sigma_trans_floor=0.25,
            covariance_scale=1.0,
            c_align=1.0,
            c_consensus=2.0,
            lambda_dir=1.0,
            robust_iterations=True,
        ),
    ]

    for config in invalid_configs:
        result = aggregate_support_loop_factors([valid_factor], config)

        assert result.valid is False
        assert "invalid_config" in result.rejection_reason


def test_support_ensemble_rejects_invalid_support_metrics():
    invalid_factors = [
        SupportLoopFactor(
            support_idx=1,
            loop_factor=make_transform(np.eye(3), [0.0, 0.0, 0.0]),
            support_alignment_residual_m=-0.1,
            direction_error_deg=0.0,
            candidate_support_baseline_m=1.0,
        ),
        SupportLoopFactor(
            support_idx=2,
            loop_factor=make_transform(np.eye(3), [0.0, 0.0, 0.0]),
            support_alignment_residual_m=0.0,
            direction_error_deg=0.0,
            candidate_support_baseline_m=0.0,
        ),
        SupportLoopFactor(
            support_idx=3,
            loop_factor=make_transform(np.eye(3), [0.0, 0.0, 0.0]),
            support_alignment_residual_m=0.0,
            direction_error_deg=0.0,
            candidate_support_baseline_m=-1.0,
        ),
        SupportLoopFactor(
            support_idx=4,
            loop_factor=make_transform(np.eye(3), [0.0, 0.0, 0.0]),
            support_alignment_residual_m=0.0,
            direction_error_deg=-1.0,
            candidate_support_baseline_m=1.0,
        ),
    ]

    for factor in invalid_factors:
        result = aggregate_support_loop_factors([factor], SupportEnsembleConfig.default())

        assert result.valid is False
        assert "invalid_support_metric" in result.rejection_reason


def test_support_ensemble_rejects_duplicate_support_indices():
    factors = [
        _factor(1, [0.0, 0.0, 0.0]),
        _factor(1, [1.0, 0.0, 0.0]),
    ]

    result = aggregate_support_loop_factors(factors, SupportEnsembleConfig.default())

    assert result.valid is False
    assert "duplicate_support_idx" in result.rejection_reason


def test_graph_evidence_nll_returns_infinity_for_non_finite_input():
    assert math.isinf(graph_evidence_nll(float("nan"), 4.0, 6.0))
    assert math.isinf(graph_evidence_nll(2.0, float("inf"), 6.0))
