from __future__ import annotations

import math
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


class DinoSaladDescriptorExtractor(DescriptorExtractor):
    def __init__(
        self,
        checkpoint: Path,
        device: str = "cuda",
        image_size: tuple[int, int] = (336, 336),
        batch_size: int = 16,
    ):
        self.checkpoint = Path(checkpoint)
        self.device = device
        self.image_size = image_size
        self.batch_size = batch_size
        self._model = None

    def _build_model(self):
        import torch

        try:
            from models.helper import get_model
        except ImportError:
            get_model = None

        if get_model is not None:
            return get_model(
                "dinov2_vitb14",
                num_channels=768,
                num_clusters=64,
                cluster_dim=128,
                token_dim=256,
            )

        from models.helper import get_aggregator, get_backbone

        class InferenceOnlySaladModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = get_backbone(
                    "dinov2_vitb14",
                    {
                        "num_trainable_blocks": 4,
                        "return_token": True,
                        "norm_layer": True,
                    },
                )
                self.aggregator = get_aggregator(
                    "SALAD",
                    {
                        "num_channels": 768,
                        "num_clusters": 64,
                        "cluster_dim": 128,
                        "token_dim": 256,
                    },
                )

            def forward(self, x):
                return self.aggregator(self.backbone(x))

        return InferenceOnlySaladModel()

    def _checkpoint_load_candidates(self, state_dict: Dict[str, object]):
        yield "unprefixed", state_dict
        for prefix in ("model.", "module.", "net."):
            prefixed = {
                key.removeprefix(prefix): value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
            if prefixed:
                yield prefix.rstrip("."), prefixed

    def _required_checkpoint_namespaces(self, model_state: Dict[str, object]) -> tuple[str, ...]:
        return tuple(
            namespace
            for namespace in ("backbone", "aggregator")
            if any(key.startswith(f"{namespace}.") for key in model_state)
        )

    def _load_checkpoint_state_dict(self, model, state_dict: Dict[str, object]) -> None:
        model_state = model.state_dict()
        total = len(model_state)
        if total == 0:
            raise ValueError("DINO-SALAD model exposes no state_dict parameters")

        best_label = ""
        best_matched: Dict[str, object] = {}
        best_namespace_count = -1
        for label, candidate in self._checkpoint_load_candidates(state_dict):
            matched = {
                key: value
                for key, value in candidate.items()
                if key in model_state
                and hasattr(value, "shape")
                and tuple(value.shape) == tuple(model_state[key].shape)
            }
            namespace_count = sum(
                any(key.startswith(f"{namespace}.") for key in matched)
                for namespace in self._required_checkpoint_namespaces(model_state)
            )
            if len(matched) > len(best_matched) or (
                len(matched) == len(best_matched) and namespace_count > best_namespace_count
            ):
                best_label = label
                best_matched = matched
                best_namespace_count = namespace_count

        matched_count = len(best_matched)
        if matched_count == 0:
            raise ValueError(
                "DINO-SALAD checkpoint did not match any model parameters "
                f"(matched 0/{total}): {self.checkpoint}"
            )

        minimum_matches = total if total <= 4 else math.ceil(total * 0.8)
        required_namespaces = self._required_checkpoint_namespaces(model_state)
        missing_namespaces = [
            namespace
            for namespace in required_namespaces
            if not any(key.startswith(f"{namespace}.") for key in best_matched)
        ]
        if matched_count < minimum_matches or missing_namespaces:
            namespace_message = (
                f"; missing namespaces: {', '.join(missing_namespaces)}"
                if missing_namespaces
                else ""
            )
            raise ValueError(
                "DINO-SALAD checkpoint coverage is too low "
                f"for candidate '{best_label}': matched {matched_count}/{total} "
                f"model parameters, require at least {minimum_matches}/{total}"
                f"{namespace_message}: {self.checkpoint}"
            )

        model.load_state_dict(best_matched, strict=False)

    def _load_model(self):
        if self._model is not None:
            return self._model
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)

        import sys

        import torch

        repo = Path(__file__).resolve().parents[2]
        salad_root = repo / "da3_streaming" / "loop_utils" / "salad"
        if str(salad_root) not in sys.path:
            sys.path.insert(0, str(salad_root))

        model = self._build_model()
        checkpoint = torch.load(self.checkpoint, map_location="cpu")
        state_dict = (
            checkpoint.get("state_dict", checkpoint)
            if isinstance(checkpoint, dict)
            else checkpoint
        )
        self._load_checkpoint_state_dict(model, state_dict)
        model = model.to(self.device).eval()
        self._model = model
        return model

    def _transform(self):
        import torchvision.transforms as transforms

        return transforms.Compose(
            [
                transforms.Resize(
                    self.image_size,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def extract(self, image_paths: List[Path]) -> np.ndarray:
        import torch
        from PIL import Image

        model = self._load_model()
        transform = self._transform()
        descriptors = []
        device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        autocast_enabled = device_type == "cuda"
        for start in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[start : start + self.batch_size]
            images = []
            for path in batch_paths:
                with Image.open(path) as image:
                    images.append(transform(image.convert("RGB")))
            batch = torch.stack(images, dim=0).to(self.device)
            with torch.no_grad():
                with torch.autocast(
                    device_type=device_type,
                    dtype=torch.float16,
                    enabled=autocast_enabled,
                ):
                    descriptors.append(model(batch).detach().cpu().numpy())
        return np.concatenate(descriptors, axis=0)


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
