from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


TRIPLET_VIEW_ROLES = ("query", "candidate", "support")


@dataclass(frozen=True)
class Da3Triplet:
    image_paths: Tuple[str, str, str]
    keyframe_indices: Tuple[int, int, int]
    view_roles: Tuple[str, str, str]


@dataclass(frozen=True)
class Da3TripletResult:
    predicted_c2w: np.ndarray
    keyframe_indices: Tuple[int, int, int]
    view_roles: Tuple[str, str, str]


def build_da3_triplet(
    query_image,
    candidate_image,
    support_image,
    query_idx: int,
    candidate_idx: int,
    support_idx: int,
) -> Da3Triplet:
    return Da3Triplet(
        image_paths=(str(query_image), str(candidate_image), str(support_image)),
        keyframe_indices=(int(query_idx), int(candidate_idx), int(support_idx)),
        view_roles=TRIPLET_VIEW_ROLES,
    )


class MockDa3Runner:
    """Interface-compatible DA3 runner with c2w output and no model dependency."""

    def run_triplet(self, triplet: Da3Triplet) -> Da3TripletResult:
        predicted_c2w = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], 3, axis=0)
        predicted_c2w[1, 0, 3] = 0.25
        predicted_c2w[2, 0, 3] = 0.5
        return Da3TripletResult(
            predicted_c2w=predicted_c2w,
            keyframe_indices=triplet.keyframe_indices,
            view_roles=triplet.view_roles,
        )
