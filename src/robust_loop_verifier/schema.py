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
class RobustLoopVerifierConfig:
    dataset_name: str
    platform: str
    input_root: Path
    output_root: Path
    gt_root: Path
    positive_radius_m: float
    recent_exclusion_keyframes: int
    retrieval_top_k_main: int
    retrieval_top_k_ablations: List[int]
    support_window: int
    support_count: int
    min_support_baseline_m: float
    pgo_noise: PgoNoiseConfig
    da3: Da3RuntimeConfig

    @classmethod
    def from_yaml(cls, path: Path) -> "RobustLoopVerifierConfig":
        return cls.from_mapping(read_yaml(path))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RobustLoopVerifierConfig":
        positive_radius_m = float(_required(data, "positive_radius_m"))
        recent_exclusion_keyframes = int(_required(data, "recent_exclusion_keyframes"))
        if positive_radius_m <= 0.0:
            raise ValueError("positive_radius_m must be positive")
        if recent_exclusion_keyframes < 0:
            raise ValueError("recent_exclusion_keyframes must be non-negative")

        return cls(
            dataset_name=str(_required(data, "dataset_name")),
            platform=str(_required(data, "platform")),
            input_root=Path(str(_required(data, "input_root"))),
            output_root=Path(str(_required(data, "output_root"))),
            gt_root=Path(str(_required(data, "gt_root"))),
            positive_radius_m=positive_radius_m,
            recent_exclusion_keyframes=recent_exclusion_keyframes,
            retrieval_top_k_main=int(_required(data, "retrieval_top_k_main")),
            retrieval_top_k_ablations=[
                int(value) for value in _required(data, "retrieval_top_k_ablations")
            ],
            support_window=int(_required(data, "support_window")),
            support_count=int(_required(data, "support_count")),
            min_support_baseline_m=float(_required(data, "min_support_baseline_m")),
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
