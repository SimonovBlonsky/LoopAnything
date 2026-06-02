from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Sequence, Tuple

import numpy as np

from robust_loop_verifier.geometry import invert_transform


TRIPLET_VIEW_ROLES = ("query", "candidate", "support")
DEFAULT_DA3_MODEL_NAME = "depth-anything/DA3-LARGE-1.1"


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


@dataclass(frozen=True)
class RealDa3RunnerConfig:
    model_name: str = DEFAULT_DA3_MODEL_NAME
    device: str = "cuda"
    process_res: int = 504
    process_res_method: str = "upper_bound_resize"
    ref_view_strategy: str = "first"
    triplet_batch_size: int = 4
    extrinsics_are_c2w: bool = False
    local_files_only: bool = True
    cache_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.process_res < 224:
            raise ValueError("process_res must be at least 224")
        if self.ref_view_strategy != "first":
            raise ValueError("RealDa3Runner requires ref_view_strategy='first'")
        if self.triplet_batch_size <= 0:
            raise ValueError("triplet_batch_size must be positive")
        if self.cache_dir is not None:
            object.__setattr__(self, "cache_dir", Path(self.cache_dir).expanduser())


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


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _to_numpy(value: Any) -> np.ndarray:
    if value is None:
        raise ValueError("DA3 output is missing extrinsics")
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _as_4x4_pose_array(extrinsics) -> np.ndarray:
    poses = _to_numpy(extrinsics)
    if poses.ndim < 3 or poses.shape[-2] not in (3, 4) or poses.shape[-1] != 4:
        raise ValueError(
            "DA3 extrinsics must have trailing shape (3,4) or (4,4)"
        )

    if poses.shape[-2] == 4:
        batch = poses.copy()
    else:
        batch = np.broadcast_to(
            np.eye(4, dtype=np.float64),
            poses.shape[:-2] + (4, 4),
        ).copy()
        batch[..., :3, :] = poses

    if not np.all(np.isfinite(batch)):
        raise ValueError("DA3 extrinsics must contain only finite values")
    if not np.allclose(batch[..., 3, :], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-7):
        raise ValueError("DA3 extrinsics bottom row must be [0, 0, 0, 1]")
    rotations = batch[..., :3, :3].reshape(-1, 3, 3)
    batch[..., :3, :3] = np.stack(
        [_project_rotation_to_so3(rotation) for rotation in rotations],
        axis=0,
    ).reshape(batch.shape[:-2] + (3, 3))
    return batch


def _as_4x4_batch(extrinsics) -> np.ndarray:
    batch = _as_4x4_pose_array(extrinsics)
    if batch.ndim == 4 and batch.shape[0] == 1:
        batch = batch[0]
    if batch.ndim != 3:
        raise ValueError(
            "DA3 extrinsics must have shape (N,3,4), (N,4,4), "
            "(1,N,3,4), or (1,N,4,4)"
        )
    return batch


def convert_da3_batched_extrinsics_to_c2w(
    extrinsics,
    extrinsics_are_c2w: bool,
) -> np.ndarray:
    pose_batches = _as_4x4_pose_array(extrinsics)
    if pose_batches.ndim == 3:
        pose_batches = pose_batches[None]
    if pose_batches.ndim != 4:
        raise ValueError(
            "DA3 batched extrinsics must have shape (B,N,3,4) or (B,N,4,4)"
        )
    if extrinsics_are_c2w:
        return pose_batches
    return np.stack(
        [
            np.stack([invert_transform(pose) for pose in pose_batch], axis=0)
            for pose_batch in pose_batches
        ],
        axis=0,
    )


def convert_da3_extrinsics_to_c2w(extrinsics, extrinsics_are_c2w: bool) -> np.ndarray:
    poses = _as_4x4_batch(extrinsics)
    if extrinsics_are_c2w:
        return poses
    return np.stack([invert_transform(pose) for pose in poses], axis=0)


def _project_rotation_to_so3(rotation) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    u_matrix, _, vt_matrix = np.linalg.svd(rotation)
    projected = u_matrix @ vt_matrix
    if np.linalg.det(projected) < 0.0:
        u_matrix[:, -1] *= -1.0
        projected = u_matrix @ vt_matrix
    return projected


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

    def run_triplets(self, triplets: Sequence[Da3Triplet]) -> list[Da3TripletResult]:
        return [self.run_triplet(triplet) for triplet in triplets]


class RealDa3Runner:
    """Lazy real Depth Anything 3 triplet runner returning camera-to-world poses."""

    def __init__(self, config: RealDa3RunnerConfig, model=None):
        self.config = config
        self._model = model

    def _load_model(self):
        if self._model is None:
            from depth_anything_3.api import DepthAnything3

            model_name_or_path = resolve_local_da3_model_name_or_path(
                self.config.model_name,
                cache_dir=self.config.cache_dir,
                require_local=self.config.local_files_only,
            )
            self._model = (
                DepthAnything3.from_pretrained(
                    model_name_or_path,
                    local_files_only=self.config.local_files_only,
                )
                .to(self.config.device)
                .eval()
            )
        return self._model

    def run_triplet(self, triplet: Da3Triplet) -> Da3TripletResult:
        return self.run_triplets([triplet])[0]

    def run_triplets(self, triplets: Sequence[Da3Triplet]) -> list[Da3TripletResult]:
        from PIL import Image
        import torch

        triplets = list(triplets)
        if not triplets:
            return []
        model = self._load_model()
        preprocessed_triplets = []
        for triplet in triplets:
            images = []
            for image_path in triplet.image_paths:
                with Image.open(image_path) as image:
                    images.append(image.convert("RGB").copy())

            imgs_cpu, _, _ = model.input_processor(
                images,
                extrinsics=None,
                intrinsics=None,
                process_res=self.config.process_res,
                process_res_method=self.config.process_res_method,
                num_workers=1,
                print_progress=False,
                sequential=True,
                desc=None,
            )
            if imgs_cpu.ndim == 5 and imgs_cpu.shape[0] == 1:
                imgs_cpu = imgs_cpu[0]
            if imgs_cpu.ndim != 4 or imgs_cpu.shape[0] != len(triplet.image_paths):
                raise ValueError("DA3 preprocessed triplet must have shape (3,C,H,W)")
            preprocessed_triplets.append(imgs_cpu)

        image_batch = torch.stack(preprocessed_triplets, dim=0).to(
            self.config.device,
            non_blocking=True,
        ).float()

        output = model.forward(
            image_batch,
            extrinsics=None,
            intrinsics=None,
            export_feat_layers=[],
            infer_gs=False,
            use_ray_pose=False,
            ref_view_strategy=self.config.ref_view_strategy,
        )
        predicted_c2w_batches = convert_da3_batched_extrinsics_to_c2w(
            _field(output, "extrinsics"),
            extrinsics_are_c2w=self.config.extrinsics_are_c2w,
        )
        if predicted_c2w_batches.shape != (len(triplets), 3, 4, 4):
            raise ValueError("DA3 batched c2w poses must have shape (B,3,4,4)")
        if not np.all(np.isfinite(predicted_c2w_batches)):
            raise ValueError("DA3 c2w poses must contain only finite values")

        return [
            Da3TripletResult(
                predicted_c2w=predicted_c2w,
                keyframe_indices=triplet.keyframe_indices,
                view_roles=triplet.view_roles,
            )
            for triplet, predicted_c2w in zip(triplets, predicted_c2w_batches)
        ]


def resolve_local_da3_model_name_or_path(
    model_name_or_path: str,
    *,
    cache_dir: Path | None = None,
    require_local: bool = False,
) -> str:
    """Return a local HF snapshot path when it is already cached.

    Passing the snapshot directory to ``from_pretrained`` avoids proxy/network
    code paths entirely. If ``require_local`` is true and the snapshot cannot be
    resolved, fail before handing a repo id to Hugging Face.
    """

    model_path = Path(model_name_or_path).expanduser()
    if model_path.exists():
        return str(model_path)
    if "/" not in model_name_or_path:
        if require_local:
            raise FileNotFoundError(
                f"local DA3 snapshot/path was not found: {model_name_or_path}"
            )
        return model_name_or_path

    repo_cache = _huggingface_repo_cache_dir(model_name_or_path, cache_dir=cache_dir)
    snapshots_dir = repo_cache / "snapshots"
    if not snapshots_dir.is_dir():
        if require_local:
            raise FileNotFoundError(
                f"local DA3 snapshot for {model_name_or_path} was not found under {repo_cache}"
            )
        return model_name_or_path

    ref_path = repo_cache / "refs" / "main"
    if ref_path.is_file():
        revision = ref_path.read_text(encoding="utf-8").strip()
        if revision:
            snapshot = snapshots_dir / revision
            if snapshot.is_dir():
                return str(snapshot)

    snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
    if not snapshots:
        if require_local:
            raise FileNotFoundError(
                f"local DA3 snapshot for {model_name_or_path} was not found under {repo_cache}"
            )
        return model_name_or_path
    return str(snapshots[-1])


def _huggingface_repo_cache_dir(model_name: str, *, cache_dir: Path | None) -> Path:
    cache_root = Path(cache_dir).expanduser() if cache_dir is not None else _default_hf_hub_cache()
    return cache_root / f"models--{model_name.replace('/', '--')}"


def _default_hf_hub_cache() -> Path:
    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub_cache:
        return Path(hub_cache).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"
