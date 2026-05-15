from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from loop_policy.geometry import invert_transform
from loop_policy.schema import KeyframeRecord


@dataclass(frozen=True)
class Da3Group:
    keyframe_indices: List[int]
    image_paths: List[str]


@dataclass(frozen=True)
class Da3GroupResult:
    keyframe_indices: List[int]
    camera_poses: np.ndarray
    depth_conf_medians: List[float]
    valid_depth_ratios: List[float]


def build_da3_group(
    query: KeyframeRecord,
    candidate: KeyframeRecord,
    supports: Sequence[KeyframeRecord],
) -> Da3Group:
    keyframes = [query, candidate, *supports]
    if any(keyframe.image_path is None for keyframe in keyframes):
        raise ValueError("DA3 group requires image_path for query, candidate, and supports")
    return Da3Group(
        keyframe_indices=[keyframe.keyframe_idx for keyframe in keyframes],
        image_paths=[str(keyframe.image_path) for keyframe in keyframes],
    )


class Da3Runner:
    def run(self, groups: List[Da3Group]) -> List[Da3GroupResult]:
        raise NotImplementedError


class MockDa3Runner(Da3Runner):
    def run(self, groups: List[Da3Group]) -> List[Da3GroupResult]:
        results: List[Da3GroupResult] = []
        for group in groups:
            view_count = len(group.keyframe_indices)
            poses = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], view_count, axis=0)
            for view_idx in range(view_count):
                poses[view_idx, 0, 3] = float(view_idx)
            results.append(
                Da3GroupResult(
                    keyframe_indices=list(group.keyframe_indices),
                    camera_poses=poses,
                    depth_conf_medians=[1.0] * view_count,
                    valid_depth_ratios=[1.0] * view_count,
                )
            )
        return results


def _as_44(extrinsics: np.ndarray) -> np.ndarray:
    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    if extrinsics.ndim != 3:
        raise ValueError(
            f"expected extrinsics with shape [N, 4, 4] or [N, 3, 4], got {extrinsics.shape}"
        )
    if extrinsics.shape[-2:] == (4, 4):
        return extrinsics
    if extrinsics.shape[-2:] == (3, 4):
        padded = np.repeat(
            np.eye(4, dtype=np.float64)[None, :, :],
            extrinsics.shape[0],
            axis=0,
        )
        padded[:, :3, :4] = extrinsics
        return padded
    raise ValueError(
        f"expected extrinsics with shape [N, 4, 4] or [N, 3, 4], got {extrinsics.shape}"
    )


def _confidence_stats(prediction, view_count: int) -> tuple[List[float], List[float]]:
    if prediction.conf is None:
        return [1.0] * view_count, [1.0] * view_count

    if len(prediction.conf) != view_count:
        raise ValueError(
            "DA3 prediction confidence count mismatch: "
            f"expected {view_count}, got {len(prediction.conf)}"
        )

    medians: List[float] = []
    ratios: List[float] = []
    for conf in prediction.conf:
        conf = np.asarray(conf, dtype=np.float64)
        finite = np.isfinite(conf)
        if not finite.any():
            medians.append(0.0)
            ratios.append(0.0)
            continue

        positive = finite & (conf > 0.0)
        positive_values = conf[positive]
        medians.append(float(np.median(positive_values)) if positive_values.size else 0.0)
        ratios.append(float(positive.mean()))
    return medians, ratios


class DepthAnything3Runner(Da3Runner):
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        extrinsics_are_c2w: bool = False,
        ref_view_strategy: str = "first",
    ):
        self.model_name = model_name
        self.device = device
        self.process_res = process_res
        self.process_res_method = process_res_method
        self.extrinsics_are_c2w = extrinsics_are_c2w
        self.ref_view_strategy = ref_view_strategy
        self._model = None

    def _load_model(self):
        if self._model is None:
            from depth_anything_3.api import DepthAnything3

            self._model = DepthAnything3.from_pretrained(self.model_name).to(self.device).eval()
        return self._model

    def run(self, groups: List[Da3Group]) -> List[Da3GroupResult]:
        model = self._load_model()
        results: List[Da3GroupResult] = []
        for group in groups:
            prediction = model.inference(
                group.image_paths,
                align_to_input_ext_scale=False,
                process_res=self.process_res,
                process_res_method=self.process_res_method,
                export_format="mini_npz",
                export_dir=None,
                ref_view_strategy=self.ref_view_strategy,
            )
            if prediction.extrinsics is None:
                raise RuntimeError("DA3 prediction did not include extrinsics")

            poses = _as_44(prediction.extrinsics)
            if poses.shape[0] != len(group.keyframe_indices):
                raise ValueError(
                    "DA3 prediction extrinsics count mismatch: "
                    f"expected {len(group.keyframe_indices)}, got {poses.shape[0]}"
                )
            if not np.isfinite(poses).all():
                raise ValueError("DA3 prediction extrinsics contain non-finite values")
            if not self.extrinsics_are_c2w:
                poses = np.stack([invert_transform(pose) for pose in poses], axis=0)

            medians, ratios = _confidence_stats(prediction, len(group.keyframe_indices))
            results.append(
                Da3GroupResult(
                    keyframe_indices=list(group.keyframe_indices),
                    camera_poses=poses,
                    depth_conf_medians=medians,
                    valid_depth_ratios=ratios,
                )
            )
        return results
