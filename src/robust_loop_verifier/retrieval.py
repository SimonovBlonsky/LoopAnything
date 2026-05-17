from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, List, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class DescriptorSet:
    keyframe_indices: List[int]
    descriptors: np.ndarray


@dataclass(frozen=True)
class SaladDescriptorBackendConfig:
    salad_repo: Path
    checkpoint_path: Path
    device: str = "cuda"
    backbone: str = "dinov2_vitb14"
    batch_size: int = 32


class DescriptorBackend(Protocol):
    def compute(self, image_paths: Sequence[str], keyframe_indices: Sequence[int]) -> DescriptorSet:
        ...


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


def _normalize_descriptor_matrix(descriptors) -> np.ndarray:
    descriptor_matrix = np.asarray(descriptors, dtype=np.float64)
    if descriptor_matrix.ndim != 2:
        raise ValueError("SALAD descriptors must have shape (N, D)")
    norms = np.linalg.norm(descriptor_matrix, axis=1)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0.0):
        raise ValueError("SALAD descriptor norm must be finite and positive")
    normalized = descriptor_matrix / norms[:, None]
    if not np.all(np.isfinite(normalized)):
        raise ValueError("SALAD descriptors must contain only finite values")
    return normalized


def _strip_checkpoint_prefixes(state_dict: Mapping) -> dict:
    prefixes = ("model.", "module.", "net.")
    cleaned = {}
    for key, value in state_dict.items():
        cleaned_key = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if cleaned_key.startswith(prefix):
                    cleaned_key = cleaned_key[len(prefix) :]
                    changed = True
        cleaned[cleaned_key] = value
    return cleaned


def _build_local_salad_model(salad_repo: Path, backbone: str):
    salad_repo = Path(salad_repo)
    module_names = (
        "vpr_model",
        "models",
        "models.backbones",
        "models.backbones.dinov2",
    )
    saved_modules = {name: sys.modules.get(name) for name in module_names}
    saved_path = list(sys.path)
    try:
        for name in module_names:
            sys.modules.pop(name, None)
        sys.path.insert(0, str(salad_repo))
        vpr_module = importlib.import_module("vpr_model")
        dinov2_module = importlib.import_module("models.backbones.dinov2")
        dinov2_archs = dinov2_module.DINOV2_ARCHS
        if backbone not in dinov2_archs:
            raise ValueError(
                f"Parameter `backbone` is set to {backbone} but it must be one of "
                f"{list(dinov2_archs.keys())}"
            )
        return vpr_module.VPRModel(
            backbone_arch=backbone,
            backbone_config={
                "num_trainable_blocks": 4,
                "return_token": True,
                "norm_layer": True,
            },
            agg_arch="SALAD",
            agg_config={
                "num_channels": dinov2_archs[backbone],
                "num_clusters": 64,
                "cluster_dim": 128,
                "token_dim": 256,
            },
        )
    finally:
        sys.path[:] = saved_path
        for name in module_names:
            if saved_modules[name] is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved_modules[name]


class SaladDescriptorBackend:
    """Lazy local DINO-SALAD descriptor backend."""

    def __init__(self, config: SaladDescriptorBackendConfig):
        self.config = config
        self._model = None
        self._transform = None

    def _load_model(self):
        if self._model is None:
            import torch

            model = _build_local_salad_model(self.config.salad_repo, self.config.backbone)
            checkpoint = torch.load(
                self.config.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            state_dict = (
                checkpoint.get("state_dict", checkpoint)
                if isinstance(checkpoint, dict)
                else checkpoint
            )
            state_dict = _strip_checkpoint_prefixes(state_dict)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                raise ValueError(
                    f"SALAD checkpoint mismatch: missing={missing}, unexpected={unexpected}"
                )
            self._model = model.eval().to(self.config.device)
        return self._model

    def _load_transform(self):
        if self._transform is None:
            from torchvision import transforms

            self._transform = transforms.Compose(
                [
                    transforms.Resize((322, 322)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
                ]
            )
        return self._transform

    def compute(self, image_paths: Sequence[str], keyframe_indices: Sequence[int]) -> DescriptorSet:
        if len(image_paths) != len(keyframe_indices):
            raise ValueError("image_paths and keyframe_indices must have matching lengths")
        if not image_paths:
            raise ValueError("image_paths must not be empty")
        if self.config.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        import torch
        from PIL import Image

        model = self._load_model()
        transform = self._load_transform()
        descriptors = []
        with torch.inference_mode():
            for start in range(0, len(image_paths), self.config.batch_size):
                batch_paths = image_paths[start : start + self.config.batch_size]
                batch_images = []
                for image_path in batch_paths:
                    with Image.open(image_path) as image:
                        batch_images.append(transform(image.convert("RGB")))
                batch = torch.stack(batch_images).to(self.config.device, non_blocking=True)
                batch_descriptors = model(batch).detach().cpu().numpy()
                descriptors.append(batch_descriptors)

        descriptor_matrix = (
            np.concatenate(descriptors, axis=0)
            if descriptors
            else np.empty((0, 0), dtype=np.float64)
        )
        if descriptor_matrix.shape[0] != len(keyframe_indices):
            raise ValueError("SALAD descriptor row count must match keyframe indices")
        descriptor_matrix = _normalize_descriptor_matrix(descriptor_matrix)
        return DescriptorSet(
            keyframe_indices=[int(index) for index in keyframe_indices],
            descriptors=descriptor_matrix,
        )


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
