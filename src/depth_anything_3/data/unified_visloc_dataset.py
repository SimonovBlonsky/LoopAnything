from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def compute_pose_distance(pose_a: np.ndarray, pose_b: np.ndarray, alpha: float = 0.5) -> float:
    """Compute weighted pose distance (translation L2 + rotation angle).

    Args:
        pose_a, pose_b: [4, 4] pose matrices
        alpha: weight for translation (1-alpha for rotation)

    Returns:
        Weighted distance: alpha * trans_L2 + (1-alpha) * rot_angle_deg
    """
    trans_dist = np.linalg.norm(pose_a[:3, 3] - pose_b[:3, 3])
    R_rel = pose_a[:3, :3].T @ pose_b[:3, :3]
    trace = np.clip((np.trace(R_rel) - 1) / 2, -1.0, 1.0)
    rot_angle_deg = np.degrees(np.arccos(trace))
    return alpha * trans_dist + (1 - alpha) * rot_angle_deg


def sample_candidates(
    query_idx: int,
    all_entries: list[dict],
    num_candidates: int,
    pos_threshold: float = 1.0,
    neg_threshold: float = 3.0,
    pos_ratio: float = 0.7,
    distance_alpha: float = 0.5,
) -> list[int]:
    """Sample positive and negative candidates based on pose distance.

    Args:
        query_idx: index of query in all_entries
        all_entries: list of dicts with "pose" key
        num_candidates: total number of candidates to sample
        pos_threshold: max pose distance for positive candidates
        neg_threshold: min pose distance for negative candidates
        pos_ratio: fraction of candidates that should be positive
        distance_alpha: weight for translation vs rotation in pose distance

    Returns:
        List of candidate indices into all_entries.
    """
    query_pose = all_entries[query_idx]["pose"]
    num_pos = int(num_candidates * pos_ratio)
    num_neg = num_candidates - num_pos

    positives = []
    negatives = []
    for i, entry in enumerate(all_entries):
        if i == query_idx:
            continue
        dist = compute_pose_distance(query_pose, entry["pose"], alpha=distance_alpha)
        if dist < pos_threshold:
            positives.append(i)
        elif dist > neg_threshold:
            negatives.append(i)

    rng = np.random.default_rng()

    # Sample positives
    if len(positives) >= num_pos:
        sampled_pos = rng.choice(positives, size=num_pos, replace=False).tolist()
    elif len(positives) > 0:
        sampled_pos = rng.choice(positives, size=num_pos, replace=True).tolist()
    else:
        sampled_pos = []

    # Sample negatives
    if len(negatives) >= num_neg:
        sampled_neg = rng.choice(negatives, size=num_neg, replace=False).tolist()
    elif len(negatives) > 0:
        sampled_neg = rng.choice(negatives, size=num_neg, replace=True).tolist()
    else:
        sampled_neg = []

    candidates = sampled_pos + sampled_neg

    # Fill any shortage from all remaining entries
    if len(candidates) < num_candidates:
        remaining = [i for i in range(len(all_entries)) if i != query_idx and i not in candidates]
        need = num_candidates - len(candidates)
        if remaining:
            extra = rng.choice(remaining, size=min(need, len(remaining)), replace=False).tolist()
            candidates.extend(extra)

    return candidates[:num_candidates]


class UnifiedVislocDataset(Dataset):
    """Training dataset for unified visual localization pipeline.

    Loads query + sampled candidates from a single scene. Each item returns
    a query image with GT pose and K candidate images with GT poses.
    """

    def __init__(
        self,
        entries: list[dict],
        num_candidates: int = 10,
        pos_threshold: float = 1.0,
        neg_threshold: float = 3.0,
        pos_ratio: float = 0.7,
        distance_alpha: float = 0.5,
        image_size: tuple[int, int] = (504, 504),
    ):
        self.entries = entries
        self.num_candidates = num_candidates
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.pos_ratio = pos_ratio
        self.distance_alpha = distance_alpha
        self.image_size = image_size

    def __len__(self):
        return len(self.entries)

    def _load_image(self, path: str) -> torch.Tensor:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.image_size)
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1)  # [3, H, W]

    def __getitem__(self, idx):
        query_entry = self.entries[idx]
        candidate_indices = sample_candidates(
            query_idx=idx,
            all_entries=self.entries,
            num_candidates=self.num_candidates,
            pos_threshold=self.pos_threshold,
            neg_threshold=self.neg_threshold,
            pos_ratio=self.pos_ratio,
            distance_alpha=self.distance_alpha,
        )

        query_image = self._load_image(query_entry["image_path"])
        query_pose = torch.from_numpy(query_entry["pose"]).float()

        candidate_images = []
        candidate_poses = []
        for ci in candidate_indices:
            cand = self.entries[ci]
            candidate_images.append(self._load_image(cand["image_path"]))
            candidate_poses.append(torch.from_numpy(cand["pose"]).float())

        return {
            "query_image": query_image,             # [3, H, W]
            "query_pose": query_pose,               # [4, 4]
            "candidate_images": torch.stack(candidate_images),   # [K, 3, H, W]
            "candidate_poses": torch.stack(candidate_poses),     # [K, 4, 4]
        }
