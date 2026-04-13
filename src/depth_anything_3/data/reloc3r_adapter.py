"""Adapter to use reloc3r training datasets with the DA3 backbone.

Wraps reloc3r's BaseStereoViewDataset subclasses so that:
- Images are resized to 504x504 with ImageNet normalization (DA3 convention).
- Ground truth pairwise relative poses are computed from c2w matrices.
- The output is a flat dict suitable for RelPoseHead training.

Usage (real datasets, on server):
    from depth_anything_3.data.reloc3r_adapter import build_training_dataset
    dataset = build_training_dataset(epoch=0)

Usage (local smoke test, no data required):
    from depth_anything_3.data.reloc3r_adapter import DummyStereoDataset
    dataset = DummyStereoDataset(length=100)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tvf
from torch.utils.data import ConcatDataset, Dataset

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

REPO_ROOT = Path(__file__).resolve().parents[5]  # NeurIPS26/
RELOC3R_ROOT = REPO_ROOT / "reloc3r"

# DA3 training target size.
DA3_TARGET_SIZE = (504, 504)  # (H, W)


def _ensure_reloc3r_imports():
    """Add reloc3r paths so its dataset classes are importable."""
    for p in [str(RELOC3R_ROOT), str(RELOC3R_ROOT / "reloc3r")]:
        if p not in sys.path:
            sys.path.insert(0, p)


def _imagenet_normalize(img: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet normalization to a [C, H, W] tensor in [0, 1]."""
    return (img - IMAGENET_MEAN) / IMAGENET_STD


def _compute_relative_pose(c2w_a: np.ndarray, c2w_b: np.ndarray) -> np.ndarray:
    """Compute the relative pose from view A to view B.

    Returns inv(c2w_b) @ c2w_a, i.e. the transform that maps points in A's
    camera frame to B's camera frame.
    """
    return np.linalg.inv(c2w_b.astype(np.float64)) @ c2w_a.astype(np.float64)


# ---------------------------------------------------------------------------
# DA3 color-jitter transform (replaces reloc3r's ColorJitter + ImgNorm)
# ---------------------------------------------------------------------------
DA3ColorJitter = tvf.Compose([
    tvf.ColorJitter(0.5, 0.5, 0.5, 0.1),
    tvf.ToTensor(),
])

DA3ToTensor = tvf.ToTensor()


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------
class DA3StereoViewAdapter(Dataset):
    """Wraps a reloc3r stereo-view dataset for DA3 RelPoseHead training.

    The underlying reloc3r dataset returns ``[view1, view2]`` where each view
    is a dict with at least ``img`` (tensor) and ``camera_pose`` (4x4 c2w).

    This adapter:
    1. Resizes each image to 504x504.
    2. Applies ImageNet normalization.
    3. Computes bidirectional relative poses.
    """

    def __init__(self, reloc3r_dataset, target_size: tuple[int, int] = DA3_TARGET_SIZE):
        self.dataset = reloc3r_dataset
        self.target_h, self.target_w = target_size

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int):
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def __getitem__(self, idx) -> dict[str, Any]:
        views = self.dataset[idx]
        view1, view2 = views[0], views[1]

        img1 = self._process_image(view1["img"])
        img2 = self._process_image(view2["img"])

        c2w_1 = np.asarray(view1["camera_pose"], dtype=np.float64)
        c2w_2 = np.asarray(view2["camera_pose"], dtype=np.float64)

        rel_2to1 = _compute_relative_pose(c2w_2, c2w_1)  # inv(c2w_1) @ c2w_2
        rel_1to2 = _compute_relative_pose(c2w_1, c2w_2)  # inv(c2w_2) @ c2w_1

        return {
            "img1": img1,
            "img2": img2,
            "c2w_1": torch.from_numpy(c2w_1).float(),
            "c2w_2": torch.from_numpy(c2w_2).float(),
            "rel_pose_2to1": torch.from_numpy(rel_2to1).float(),
            "rel_pose_1to2": torch.from_numpy(rel_1to2).float(),
        }

    def _process_image(self, img: torch.Tensor) -> torch.Tensor:
        """Resize to target and apply ImageNet normalization."""
        if img.ndim == 3:
            img = img.unsqueeze(0)
        img = F.interpolate(img, size=(self.target_h, self.target_w), mode="bilinear", align_corners=False)
        img = img.squeeze(0)
        img = _imagenet_normalize(img)
        return img


# ---------------------------------------------------------------------------
# Dummy dataset for local smoke testing (no real data needed)
# ---------------------------------------------------------------------------
class DummyStereoDataset(Dataset):
    """Synthetic stereo pairs with random SE(3) poses for local testing."""

    def __init__(self, length: int = 100, target_size: tuple[int, int] = DA3_TARGET_SIZE):
        self.length = length
        self.target_h, self.target_w = target_size

    def __len__(self) -> int:
        return self.length

    def set_epoch(self, epoch: int):
        pass

    @staticmethod
    def _random_se3(rng: np.random.RandomState | None = None) -> np.ndarray:
        """Generate a random valid SE(3) matrix."""
        from scipy.spatial.transform import Rotation

        R = Rotation.random(random_state=rng).as_matrix()
        t = (rng.randn(3) if rng is not None else np.random.randn(3)) * 0.5
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = R
        pose[:3, 3] = t
        return pose

    def __getitem__(self, idx) -> dict[str, Any]:
        gen = torch.Generator().manual_seed(idx)

        img1 = torch.rand(3, self.target_h, self.target_w, generator=gen)
        img2 = torch.rand(3, self.target_h, self.target_w, generator=gen)
        img1 = _imagenet_normalize(img1)
        img2 = _imagenet_normalize(img2)

        rng = np.random.RandomState(idx)
        c2w_1 = self._random_se3(rng)
        c2w_2 = self._random_se3(rng)
        rel_2to1 = _compute_relative_pose(c2w_2, c2w_1)
        rel_1to2 = _compute_relative_pose(c2w_1, c2w_2)

        return {
            "img1": img1.float(),
            "img2": img2.float(),
            "c2w_1": torch.from_numpy(c2w_1).float(),
            "c2w_2": torch.from_numpy(c2w_2).float(),
            "rel_pose_2to1": torch.from_numpy(rel_2to1).float(),
            "rel_pose_1to2": torch.from_numpy(rel_1to2).float(),
        }


# ---------------------------------------------------------------------------
# Dataset builder (for real training on server)
# ---------------------------------------------------------------------------
def _build_reloc3r_dataset(dataset_str: str, use_augmentation: bool = True):
    """Construct a single reloc3r dataset from a descriptor string.

    The string follows reloc3r's convention, e.g.:
        "50_000 @ Co3d(split='train', resolution=(512, 384))"

    The transform is overridden to produce [0,1] tensors with optional
    color jitter (instead of reloc3r's ImgNorm / ColorJitter).
    """
    _ensure_reloc3r_imports()
    from reloc3r.datasets import (  # noqa: F401
        ARKitScenes, BlendedMVS, Co3d, DL3DV, MegaDepth, RealEstate, ScanNetpp,
        ScanNet1500, MegaDepth_valid,
    )

    transform = DA3ColorJitter if use_augmentation else DA3ToTensor
    # Inject our transform into the string before eval.
    # Replace any existing transform= argument.
    import re
    if "transform=" in dataset_str:
        dataset_str = re.sub(r"transform=\w+", f"transform=transform", dataset_str)
    elif dataset_str.rstrip().endswith(")"):
        dataset_str = dataset_str.rstrip()[:-1] + ", transform=transform)"
    return eval(dataset_str)


TRAIN_DATASET_STR = (
    "50_000 @ Co3d(split='train', resolution=(512, 384), transform=transform)"
    " + 50_000 @ ScanNetpp(split='train', resolution=(512, 384), transform=transform)"
    " + 50_000 @ ARKitScenes(split='train', resolution=(512, 384), transform=transform)"
    " + 50_000 @ BlendedMVS(split='train', resolution=(512, 384), transform=transform)"
    " + 50_000 @ MegaDepth(split='train', resolution=(512, 384), transform=transform)"
    " + 50_000 @ DL3DV(split='train', resolution=(512, 384), transform=transform)"
    " + 50_000 @ RealEstate(split='train', resolution=(512, 384), transform=transform)"
)

TEST_DATASET_STR = (
    "1_000 @ ScanNet1500(resolution=(512, 384), seed=777, transform=transform)"
)


def build_training_dataset(epoch: int = 0, use_augmentation: bool = True) -> DA3StereoViewAdapter:
    """Build the full 350K-pair training dataset wrapped for DA3."""
    _ensure_reloc3r_imports()
    from reloc3r.datasets import (  # noqa: F401
        ARKitScenes, BlendedMVS, Co3d, DL3DV, MegaDepth, RealEstate, ScanNetpp,
    )
    transform = DA3ColorJitter if use_augmentation else DA3ToTensor
    dataset = eval(TRAIN_DATASET_STR)
    dataset.set_epoch(epoch)
    return DA3StereoViewAdapter(dataset)


def build_test_dataset() -> DA3StereoViewAdapter:
    """Build the validation dataset (ScanNet1500) wrapped for DA3."""
    _ensure_reloc3r_imports()
    from reloc3r.datasets import ScanNet1500  # noqa: F401
    transform = DA3ToTensor  # no augmentation for validation
    dataset = eval(TEST_DATASET_STR)
    dataset.set_epoch(0)
    return DA3StereoViewAdapter(dataset)
