from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple


SCHEMA_VERSION = "loop_policy_dataset_v1"
X_GEOM_DIM = 32


@dataclass(frozen=True)
class LoopPolicyDatasetConfig:
    dataset_root: str
    output_root: str
    sequences: Tuple[str, ...]
    gt_root: Optional[str] = None
    causal: bool = True
    exclude_recent_keyframes: int = 30
    support_window: int = 20
    support_count: int = 1
    min_support_baseline_m: float = 0.3
    retrieval_pool_size: int = 50
    runtime_top_k: int = 4
    write_visualization_records: bool = False
    write_empty_queries: bool = False
    query_limit: Optional[int] = None
    abs_log_sim3_scale_thr: float = 0.4
    support_align_rmse_thr: float = 1.0
    direction_error_thr_deg: float = 45.0
    loose_rot_thr_deg: float = 30.0
    loose_trans_thr_m: float = 5.0

    def __post_init__(self) -> None:
        if self.retrieval_pool_size < self.runtime_top_k:
            raise ValueError("retrieval_pool_size must be >= runtime_top_k")
        if not self.causal:
            raise ValueError("Loop policy dataset cache requires causal=True")


@dataclass(frozen=True)
class KeyframeRecord:
    keyframe_idx: int
    timestamp: float
    image_path: Optional[str]
    trajectory_idx: Optional[int] = None
    raw: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class PoseRecord:
    timestamp: float
    position: Tuple[float, float, float]
    quaternion_xyzw: Tuple[float, float, float, float]


@dataclass(frozen=True)
class RetrievalCandidate:
    rank: int
    keyframe_idx: int
    timestamp: float
    score: float
    runtime_topk: bool


@dataclass(frozen=True)
class RetrievalRecord:
    sequence: str
    query_idx: int
    query_timestamp: float
    causal: bool
    database_max_idx: Optional[int]
    database_max_timestamp: Optional[float]
    retrieval_db_size: int
    candidates: List[RetrievalCandidate]


@dataclass(frozen=True)
class SupportDecision:
    sequence: str
    query_idx: int
    query_timestamp: float
    candidate_idx: int
    candidate_timestamp: float
    causal: bool
    support_snapshot_max_idx: Optional[int]
    support_snapshot_max_timestamp: Optional[float]
    selected_support_indices: List[int]
    selected_support_timestamps: List[float]
    selected_support_baselines: List[float]
    support_count: int
    rejected: bool
    rejection_reason: Optional[str]


@dataclass(frozen=True)
class CandidateFeatureRecord:
    sequence: str
    query_idx: int
    query_timestamp: float
    candidate_source: str
    candidate_idx: int
    candidate_timestamp: float
    causal: bool
    database_max_idx: Optional[int]
    database_max_timestamp: Optional[float]
    retrieval_db_size: int
    support_snapshot_max_idx: Optional[int]
    support_snapshot_max_timestamp: Optional[float]
    selected_support_indices: List[int]
    selected_support_timestamps: List[float]
    support_count: int
    precondition_valid: bool
    negative_reason: Optional[str]
    x_geom: List[float]
    safe_loop_factor_v1: bool
    labels: Dict[str, Any]
    metrics: Dict[str, Any]

    def __post_init__(self) -> None:
        if len(self.x_geom) != X_GEOM_DIM:
            raise ValueError(f"x_geom must contain {X_GEOM_DIM} values")


def dataclass_to_json_dict(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: dataclass_to_json_dict(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [dataclass_to_json_dict(item) for item in value]
    if isinstance(value, tuple):
        return [dataclass_to_json_dict(item) for item in value]
    if isinstance(value, MappingABC):
        return {str(key): dataclass_to_json_dict(item) for key, item in value.items()}
    return value
