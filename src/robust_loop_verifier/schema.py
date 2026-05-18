from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

import numpy as np

from robust_loop_verifier.io import read_yaml


@dataclass(frozen=True)
class PgoNoiseConfig:
    prior_sigmas: List[float]
    odom_sigmas: List[float]
    loop_sigmas: List[float]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PgoNoiseConfig":
        return cls(
            prior_sigmas=_six_floats(_required(data, "prior_sigmas"), "prior_sigmas"),
            odom_sigmas=_six_floats(_required(data, "odom_sigmas"), "odom_sigmas"),
            loop_sigmas=_six_floats(_required(data, "loop_sigmas"), "loop_sigmas"),
        )


@dataclass(frozen=True)
class Da3RuntimeConfig:
    process_res: int
    ref_view_strategy: str

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Da3RuntimeConfig":
        process_res = int(_required(data, "process_res"))
        if process_res < 224:
            raise ValueError("process_res must be >= 224")

        ref_view_strategy = str(_required(data, "ref_view_strategy"))
        if ref_view_strategy != "first":
            raise ValueError("ref_view_strategy must be 'first'")

        return cls(process_res=process_res, ref_view_strategy=ref_view_strategy)


@dataclass(frozen=True)
class SupportEnsembleSettings:
    enabled: bool = False
    support_count: int = 1
    sigma_rot_floor: float = 0.05
    sigma_trans_floor: float = 0.25
    covariance_scale: float = 1.0
    c_align: float = 1.0
    c_consensus: float = 2.0
    lambda_dir: float = 1.0
    robust_iterations: int = 3

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SupportEnsembleSettings":
        enabled = _bool(data.get("enabled", False), "support_ensemble.enabled")
        support_count = _positive_int(
            data.get("support_count", 1), "support_ensemble.support_count"
        )
        sigma_rot_floor = _float(
            data.get("sigma_rot_floor", 0.05), "support_ensemble.sigma_rot_floor"
        )
        sigma_trans_floor = _float(
            data.get("sigma_trans_floor", 0.25), "support_ensemble.sigma_trans_floor"
        )
        covariance_scale = _float(
            data.get("covariance_scale", 1.0), "support_ensemble.covariance_scale"
        )
        c_align = _float(data.get("c_align", 1.0), "support_ensemble.c_align")
        c_consensus = _float(data.get("c_consensus", 2.0), "support_ensemble.c_consensus")
        lambda_dir = _float(data.get("lambda_dir", 1.0), "support_ensemble.lambda_dir")
        robust_iterations = _positive_int(
            data.get("robust_iterations", 3), "support_ensemble.robust_iterations"
        )

        _validate_positive_finite(sigma_rot_floor, "support_ensemble.sigma_rot_floor")
        _validate_positive_finite(sigma_trans_floor, "support_ensemble.sigma_trans_floor")
        _validate_positive_finite(covariance_scale, "support_ensemble.covariance_scale")
        _validate_positive_finite(c_align, "support_ensemble.c_align")
        _validate_positive_finite(c_consensus, "support_ensemble.c_consensus")
        if not np.isfinite(lambda_dir) or lambda_dir < 0.0:
            raise ValueError("support_ensemble.lambda_dir must be finite and non-negative")

        return cls(
            enabled=enabled,
            support_count=support_count,
            sigma_rot_floor=sigma_rot_floor,
            sigma_trans_floor=sigma_trans_floor,
            covariance_scale=covariance_scale,
            c_align=c_align,
            c_consensus=c_consensus,
            lambda_dir=lambda_dir,
            robust_iterations=robust_iterations,
        )


@dataclass(frozen=True)
class RobustLoopVerifierConfig:
    dataset_name: str
    platform: str
    input_root: Path
    output_root: Path
    gt_root: Path
    positive_radius_m: float
    positive_max_rotation_deg: float
    recent_exclusion_keyframes: int
    retrieval_top_k_main: int
    retrieval_top_k_ablations: List[int]
    support_window: int
    support_count: int
    min_support_baseline_m: float
    support_ensemble: SupportEnsembleSettings
    pgo_noise: PgoNoiseConfig
    da3: Da3RuntimeConfig

    @classmethod
    def from_yaml(cls, path: Path) -> "RobustLoopVerifierConfig":
        return cls.from_mapping(read_yaml(path))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RobustLoopVerifierConfig":
        positive_radius_m = float(_required(data, "positive_radius_m"))
        positive_max_rotation_deg = float(data.get("positive_max_rotation_deg", 45.0))
        recent_exclusion_keyframes = int(_required(data, "recent_exclusion_keyframes"))
        if positive_radius_m <= 0.0:
            raise ValueError("positive_radius_m must be positive")
        if (
            not np.isfinite(positive_max_rotation_deg)
            or positive_max_rotation_deg < 0.0
            or positive_max_rotation_deg > 180.0
        ):
            raise ValueError("positive_max_rotation_deg must be in [0, 180]")
        if recent_exclusion_keyframes < 0:
            raise ValueError("recent_exclusion_keyframes must be non-negative")

        return cls(
            dataset_name=str(_required(data, "dataset_name")),
            platform=str(_required(data, "platform")),
            input_root=Path(str(_required(data, "input_root"))),
            output_root=Path(str(_required(data, "output_root"))),
            gt_root=Path(str(_required(data, "gt_root"))),
            positive_radius_m=positive_radius_m,
            positive_max_rotation_deg=positive_max_rotation_deg,
            recent_exclusion_keyframes=recent_exclusion_keyframes,
            retrieval_top_k_main=int(_required(data, "retrieval_top_k_main")),
            retrieval_top_k_ablations=[
                int(value) for value in _required(data, "retrieval_top_k_ablations")
            ],
            support_window=int(_required(data, "support_window")),
            support_count=int(_required(data, "support_count")),
            min_support_baseline_m=float(_required(data, "min_support_baseline_m")),
            support_ensemble=SupportEnsembleSettings.from_mapping(
                _mapping(data.get("support_ensemble", {}), "support_ensemble")
            ),
            pgo_noise=PgoNoiseConfig.from_mapping(
                _mapping(_required(data, "pgo_noise"), "pgo_noise")
            ),
            da3=Da3RuntimeConfig.from_mapping(_mapping(_required(data, "da3"), "da3")),
        )


@dataclass(frozen=True)
class KeyframeRecord:
    idx: int
    timestamp: float
    image_path: Optional[str]
    odom_pose: np.ndarray
    gt_pose: np.ndarray


@dataclass(frozen=True)
class PositiveRecord:
    query_idx: int
    positive_indices: List[int]


def _required(data: Mapping[str, Any], key: str) -> Any:
    if key not in data:
        raise ValueError("Missing required field: {}".format(key))
    return data[key]


def _mapping(value: Any, key: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("{} must be a mapping".format(key))
    return value


def _six_floats(value: Any, key: str) -> List[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("{} must contain six floats".format(key))
    if len(value) != 6:
        raise ValueError("{} must contain six floats".format(key))
    return [float(item) for item in value]


def _validate_positive_finite(value: float, key: str) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("{} must be finite and positive".format(key))


def _positive_int(value: Any, key: str) -> int:
    if isinstance(value, bool):
        raise ValueError("{} must be a positive integer".format(key))
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError("{} must be a positive integer".format(key))
    if not np.isfinite(parsed) or not parsed.is_integer() or parsed <= 0.0:
        raise ValueError("{} must be a positive integer".format(key))
    return int(parsed)


def _float(value: Any, key: str) -> float:
    if isinstance(value, bool):
        raise ValueError("{} must be a float".format(key))
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError("{} must be a float".format(key))


def _bool(value: Any, key: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError("{} must be a bool".format(key))
    return value
