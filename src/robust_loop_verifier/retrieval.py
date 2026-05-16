from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class DescriptorSet:
    keyframe_indices: List[int]
    descriptors: np.ndarray


@dataclass(frozen=True)
class RetrievalCandidate:
    query_idx: int
    candidate_idx: int
    rank: int
    score: float


@dataclass(frozen=True)
class RetrievalRecord:
    query_idx: int
    candidates: List[RetrievalCandidate]


def _l2_normalize(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Descriptor norm must be finite and positive")
    return vector / norm


def _validate_non_negative_int(value, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def retrieve_historical_topk(
    query_idx: int,
    descriptors: DescriptorSet,
    top_k: int,
    recent_exclusion_keyframes: int,
) -> RetrievalRecord:
    top_k = _validate_non_negative_int(top_k, "top_k")
    recent_exclusion_keyframes = _validate_non_negative_int(
        recent_exclusion_keyframes,
        "recent_exclusion_keyframes",
    )

    keyframe_indices = list(descriptors.keyframe_indices)
    descriptor_matrix = np.asarray(descriptors.descriptors, dtype=np.float64)

    if descriptor_matrix.ndim != 2:
        raise ValueError("Descriptors must have shape (N, D)")
    if descriptor_matrix.shape[0] != len(keyframe_indices):
        raise ValueError("Descriptor row count must match keyframe indices")
    if query_idx not in keyframe_indices:
        raise ValueError(f"Query keyframe index {query_idx} is missing")

    query_row = keyframe_indices.index(query_idx)
    query_descriptor = _l2_normalize(descriptor_matrix[query_row])
    exclusion_threshold = query_idx - recent_exclusion_keyframes

    scored_candidates = []
    for candidate_idx, descriptor in zip(keyframe_indices, descriptor_matrix):
        if candidate_idx == query_idx:
            continue
        if candidate_idx >= exclusion_threshold:
            continue

        candidate_descriptor = _l2_normalize(descriptor)
        score = float(np.dot(query_descriptor, candidate_descriptor))
        scored_candidates.append((score, candidate_idx))

    scored_candidates.sort(key=lambda item: (-item[0], item[1]))
    candidates = [
        RetrievalCandidate(
            query_idx=query_idx,
            candidate_idx=candidate_idx,
            rank=rank,
            score=score,
        )
        for rank, (score, candidate_idx) in enumerate(scored_candidates[:top_k], start=1)
    ]
    return RetrievalRecord(query_idx=query_idx, candidates=candidates)
