from __future__ import annotations

import math
from numbers import Real
from typing import Any, Dict, List

from loop_policy.schema import LoopPolicyDatasetConfig, X_GEOM_DIM


X_GEOM_FIELDS = (
    "rank_norm",
    "salad_score_qc",
    "salad_score_qs",
    "salad_score_cs",
    "salad_score_qc_minus_top1",
    "salad_score_qc_minus_topk",
    "da3_rot_qc_deg",
    "da3_trans_qc_norm",
    "da3_rot_cs_deg",
    "da3_trans_cs_norm",
    "da3_rot_qs_deg",
    "da3_trans_qs_norm",
    "odom_rot_qc_deg",
    "odom_trans_qc_norm",
    "odom_rot_cs_deg",
    "odom_trans_cs_norm",
    "support_baseline",
    "sim3_scale",
    "abs_log_sim3_scale",
    "support_align_rmse",
    "direction_error_deg",
    "aligned_vs_odom_rot_residual_deg",
    "aligned_vs_odom_trans_residual_norm",
    "aligned_loop_rot_deg",
    "aligned_loop_trans_norm",
    "q_depth_conf_median",
    "c_depth_conf_median",
    "s_depth_conf_median",
    "q_valid_depth_ratio",
    "c_valid_depth_ratio",
    "s_valid_depth_ratio",
    "min_depth_conf_median",
)


def _finite(value: Any) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def build_x_geom(values: Dict[str, Any]) -> List[float]:
    x_geom: List[float] = []
    for field in X_GEOM_FIELDS:
        if field not in values:
            raise ValueError(f"missing x_geom field: {field}")
        value = values[field]
        if not _finite(value):
            raise ValueError(f"x_geom field must be finite: {field}")
        x_geom.append(float(value))

    if len(x_geom) != X_GEOM_DIM:
        raise ValueError(f"x_geom must contain {X_GEOM_DIM} values")
    return x_geom


def compute_safe_loop_factor_v1(
    precondition_valid: bool,
    metrics: Dict[str, Any],
    config: LoopPolicyDatasetConfig,
) -> Dict[str, bool]:
    support_align_rmse = metrics.get("support_align_rmse")
    direction_error_deg = metrics.get("direction_error_deg")
    sim3_quality_good = (
        _finite(support_align_rmse)
        and _finite(direction_error_deg)
        and support_align_rmse <= config.support_align_rmse_thr
        and direction_error_deg <= config.direction_error_thr_deg
    )

    aligned_vs_odom_rot_residual_deg = metrics.get("aligned_vs_odom_rot_residual_deg")
    aligned_vs_odom_trans_residual_norm = metrics.get("aligned_vs_odom_trans_residual_norm")
    odom_consistent_loose = (
        _finite(aligned_vs_odom_rot_residual_deg)
        and _finite(aligned_vs_odom_trans_residual_norm)
        and aligned_vs_odom_rot_residual_deg <= config.loose_rot_thr_deg
        and aligned_vs_odom_trans_residual_norm <= config.loose_trans_thr_m
    )

    safe_loop_factor_v1 = bool(precondition_valid and sim3_quality_good and odom_consistent_loose)
    return {
        "sim3_quality_good": bool(sim3_quality_good),
        "odom_consistent_loose": bool(odom_consistent_loose),
        "safe_loop_factor_v1": safe_loop_factor_v1,
    }
