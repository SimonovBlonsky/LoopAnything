from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from loop_policy.schema import KeyframeRecord, RetrievalCandidate, RetrievalRecord


@dataclass(frozen=True)
class DescriptorCache:
    keyframe_idx: np.ndarray
    timestamps: np.ndarray
    descriptors: np.ndarray
    normalized: bool

    def __post_init__(self) -> None:
        if self.keyframe_idx.ndim != 1:
            raise ValueError("keyframe_idx must be a 1D array")
        if self.timestamps.ndim != 1:
            raise ValueError("timestamps must be a 1D array")
        if self.descriptors.ndim != 2:
            raise ValueError("descriptors must be a 2D array")
        if not (
            len(self.keyframe_idx) == len(self.timestamps) == self.descriptors.shape[0]
        ):
            raise ValueError("descriptor cache row count mismatch")
        if len(np.unique(self.keyframe_idx)) != len(self.keyframe_idx):
            raise ValueError("keyframe_idx must be unique")

    def index_map(self) -> Dict[int, int]:
        return {int(keyframe_idx): row for row, keyframe_idx in enumerate(self.keyframe_idx)}


class DescriptorExtractor:
    def extract(self, image_paths: List[Path]) -> np.ndarray:
        raise NotImplementedError


class PrecomputedDescriptorExtractor(DescriptorExtractor):
    def __init__(self, descriptors_by_path: Dict[str, np.ndarray]):
        self.descriptors_by_path = descriptors_by_path

    def extract(self, image_paths: List[Path]) -> np.ndarray:
        return np.stack(
            [np.asarray(self.descriptors_by_path[str(image_path)]) for image_path in image_paths],
            axis=0,
        )


def normalize_descriptors(descriptors: np.ndarray) -> np.ndarray:
    values = np.asarray(descriptors, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0.0)


def save_descriptor_cache(path: Path, cache: DescriptorCache) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        keyframe_idx=cache.keyframe_idx,
        timestamps=cache.timestamps,
        descriptors=cache.descriptors,
        normalized=np.asarray([cache.normalized], dtype=np.bool_),
    )


def load_descriptor_cache(path: Path) -> DescriptorCache:
    with np.load(path) as data:
        normalized_values = np.asarray(data["normalized"], dtype=np.bool_).reshape(-1)
        if normalized_values.size != 1:
            raise ValueError("normalized must contain exactly one value")
        return DescriptorCache(
            keyframe_idx=data["keyframe_idx"],
            timestamps=data["timestamps"],
            descriptors=data["descriptors"],
            normalized=bool(normalized_values[0]),
        )


def causal_retrieval_database(
    keyframes: List[KeyframeRecord],
    query: KeyframeRecord,
    exclude_recent_keyframes: int,
) -> List[KeyframeRecord]:
    return [
        record
        for record in keyframes
        if record.timestamp < query.timestamp
        and abs(record.keyframe_idx - query.keyframe_idx) > exclude_recent_keyframes
        and record.image_path is not None
    ]


def rank_causal_topk(
    sequence: str,
    query: KeyframeRecord,
    keyframes: List[KeyframeRecord],
    descriptors: DescriptorCache,
    retrieval_pool_size: int,
    runtime_top_k: int,
    exclude_recent_keyframes: int,
) -> RetrievalRecord:
    if not descriptors.normalized:
        raise ValueError("descriptor cache must be normalized")

    descriptor_rows = descriptors.index_map()
    query_row = descriptor_rows.get(query.keyframe_idx)
    if query_row is None:
        raise ValueError(f"missing query descriptor for keyframe_idx={query.keyframe_idx}")

    database = causal_retrieval_database(
        keyframes=keyframes,
        query=query,
        exclude_recent_keyframes=exclude_recent_keyframes,
    )
    query_descriptor = descriptors.descriptors[query_row]
    scored = []
    for record in database:
        candidate_row = descriptor_rows.get(record.keyframe_idx)
        if candidate_row is None:
            continue
        score = float(np.dot(query_descriptor, descriptors.descriptors[candidate_row]))
        scored.append((score, record))

    scored.sort(key=lambda item: (-item[0], item[1].keyframe_idx))
    candidates = [
        RetrievalCandidate(
            rank=rank,
            keyframe_idx=record.keyframe_idx,
            timestamp=record.timestamp,
            score=score,
            runtime_topk=rank <= runtime_top_k,
        )
        for rank, (score, record) in enumerate(scored[:retrieval_pool_size], start=1)
    ]

    return RetrievalRecord(
        sequence=sequence,
        query_idx=query.keyframe_idx,
        query_timestamp=query.timestamp,
        causal=True,
        database_max_idx=max((record.keyframe_idx for record in database), default=None),
        database_max_timestamp=max((record.timestamp for record in database), default=None),
        retrieval_db_size=len(database),
        candidates=candidates,
    )
