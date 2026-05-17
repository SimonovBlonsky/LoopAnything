from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Tuple

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
    extrinsics_are_c2w: bool = False
    local_files_only: bool = True
    cache_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.process_res < 224:
            raise ValueError("process_res must be at least 224")
        if self.ref_view_strategy != "first":
            raise ValueError("RealDa3Runner requires ref_view_strategy='first'")
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


def _as_4x4_batch(extrinsics) -> np.ndarray:
    poses = _to_numpy(extrinsics)
    if poses.ndim == 4 and poses.shape[0] == 1:
        poses = poses[0]
    if poses.ndim != 3 or poses.shape[1] not in (3, 4) or poses.shape[2] != 4:
        raise ValueError(
            "DA3 extrinsics must have shape (N,3,4), (N,4,4), "
            "(1,N,3,4), or (1,N,4,4)"
        )

    if poses.shape[1] == 4:
        batch = poses.copy()
    else:
        batch = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], poses.shape[0], axis=0)
        batch[:, :3, :] = poses

    if not np.all(np.isfinite(batch)):
        raise ValueError("DA3 extrinsics must contain only finite values")
    if not np.allclose(batch[:, 3, :], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-7):
        raise ValueError("DA3 extrinsics bottom row must be [0, 0, 0, 1]")
    batch[:, :3, :3] = np.stack(
        [_project_rotation_to_so3(rotation) for rotation in batch[:, :3, :3]],
        axis=0,
    )
    return batch


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
        from PIL import Image

        model = self._load_model()
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
        if imgs_cpu.ndim == 4:
            image_batch = imgs_cpu[None].to(self.config.device, non_blocking=True).float()
        elif imgs_cpu.ndim == 5 and imgs_cpu.shape[0] == 1:
            image_batch = imgs_cpu.to(self.config.device, non_blocking=True).float()
        else:
            raise ValueError(
                "DA3 preprocessed images must have shape (N,C,H,W) or (1,N,C,H,W)"
            )

        output = model.forward(
            image_batch,
            extrinsics=None,
            intrinsics=None,
            export_feat_layers=[],
            infer_gs=False,
            use_ray_pose=False,
            ref_view_strategy=self.config.ref_view_strategy,
        )
        predicted_c2w = convert_da3_extrinsics_to_c2w(
            _field(output, "extrinsics"),
            extrinsics_are_c2w=self.config.extrinsics_are_c2w,
        )
        if predicted_c2w.shape[0] != len(triplet.image_paths):
            raise ValueError("DA3 extrinsics count must match triplet image count")
        if not np.all(np.isfinite(predicted_c2w)):
            raise ValueError("DA3 c2w poses must contain only finite values")

        return Da3TripletResult(
            predicted_c2w=predicted_c2w,
            keyframe_indices=triplet.keyframe_indices,
            view_roles=triplet.view_roles,
        )


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
