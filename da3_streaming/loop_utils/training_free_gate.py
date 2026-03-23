"""Training-free utilities for loop pair scoring."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def relative_pose_from_extrinsics(extrinsics_a: np.ndarray, extrinsics_b: np.ndarray) -> np.ndarray:
    """Return the relative pose that maps coordinates from ``b`` into ``a`` space."""

    extrinsics_a = np.asarray(extrinsics_a, dtype=np.float32)
    extrinsics_b = np.asarray(extrinsics_b, dtype=np.float32)
    if extrinsics_a.shape != (4, 4) or extrinsics_b.shape != (4, 4):
        raise ValueError("extrinsics_a and extrinsics_b must both have shape (4, 4)")
    return extrinsics_a @ np.linalg.inv(extrinsics_b)


def combine_loop_scores(
    desc_sim: float,
    pair_sim: float,
    overlap: float,
    weights: Sequence[float] = (0.5, 0.25, 0.25),
) -> float:
    """Combine descriptor, pair-token, and overlap scores into a clipped scalar."""

    if len(weights) != 3:
        raise ValueError("weights must contain exactly three values")
    score = weights[0] * desc_sim + weights[1] * pair_sim + weights[2] * overlap
    return float(np.clip(score, 0.0, 1.0))


def estimate_overlap_score(
    depth_a: np.ndarray,
    depth_b: np.ndarray,
    intr_a: np.ndarray,
    intr_b: np.ndarray,
    rel_pose: np.ndarray,
    stride: int = 16,
) -> float:
    """Estimate geometric overlap with a cheap sampled reprojection consistency ratio."""

    depth_a = np.asarray(depth_a, dtype=np.float32)
    depth_b = np.asarray(depth_b, dtype=np.float32)
    intr_a = np.asarray(intr_a, dtype=np.float32)
    intr_b = np.asarray(intr_b, dtype=np.float32)
    rel_pose = np.asarray(rel_pose, dtype=np.float32)

    if depth_a.ndim != 2 or depth_b.ndim != 2:
        raise ValueError("depth_a and depth_b must be 2D arrays")
    if intr_a.shape != (3, 3) or intr_b.shape != (3, 3):
        raise ValueError("intr_a and intr_b must both have shape (3, 3)")
    if rel_pose.shape != (4, 4):
        raise ValueError("rel_pose must have shape (4, 4)")
    if stride <= 0:
        raise ValueError("stride must be positive")

    score_ab = _sampled_reprojection_ratio(depth_a, depth_b, intr_a, intr_b, rel_pose, stride)
    score_ba = _sampled_reprojection_ratio(
        depth_b, depth_a, intr_b, intr_a, np.linalg.inv(rel_pose), stride
    )
    return float(np.clip(0.5 * (score_ab + score_ba), 0.0, 1.0))


def _sampled_reprojection_ratio(
    depth_src: np.ndarray,
    depth_dst: np.ndarray,
    intr_src: np.ndarray,
    intr_dst: np.ndarray,
    rel_pose: np.ndarray,
    stride: int,
) -> float:
    height, width = depth_src.shape
    ys, xs = np.mgrid[0:height:stride, 0:width:stride]
    if ys.size == 0:
        return 0.0

    z = depth_src[ys, xs]
    valid_src = np.isfinite(z) & (z > 1e-6)
    if not np.any(valid_src):
        return 0.0

    xs = xs[valid_src].astype(np.float32)
    ys = ys[valid_src].astype(np.float32)
    z = z[valid_src]

    x = (xs - intr_src[0, 2]) * z / intr_src[0, 0]
    y = (ys - intr_src[1, 2]) * z / intr_src[1, 1]
    points_src = np.stack([x, y, z, np.ones_like(z)], axis=0)
    points_dst = rel_pose @ points_src

    z_dst = points_dst[2]
    valid_dst = np.isfinite(z_dst) & (z_dst > 1e-6)
    if not np.any(valid_dst):
        return 0.0

    points_dst = points_dst[:, valid_dst]
    z_dst = z_dst[valid_dst]

    u = points_dst[0] / z_dst * intr_dst[0, 0] + intr_dst[0, 2]
    v = points_dst[1] / z_dst * intr_dst[1, 1] + intr_dst[1, 2]

    u_round = np.rint(u).astype(np.int64)
    v_round = np.rint(v).astype(np.int64)
    in_bounds = (u_round >= 0) & (u_round < depth_dst.shape[1]) & (v_round >= 0) & (v_round < depth_dst.shape[0])
    if not np.any(in_bounds):
        return 0.0

    u_round = u_round[in_bounds]
    v_round = v_round[in_bounds]
    z_dst = z_dst[in_bounds]
    z_measured = depth_dst[v_round, u_round]
    valid_measured = np.isfinite(z_measured) & (z_measured > 1e-6)
    if not np.any(valid_measured):
        return 0.0

    z_dst = z_dst[valid_measured]
    z_measured = z_measured[valid_measured]
    depth_tol = np.maximum(0.05, 0.15 * z_measured)
    consistent = np.abs(z_dst - z_measured) <= depth_tol
    return float(np.mean(consistent))


__all__ = [
    "combine_loop_scores",
    "estimate_overlap_score",
    "relative_pose_from_extrinsics",
]
