from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS = Image.LANCZOS


@dataclass(frozen=True)
class PairSample:
    image1: str
    image2: str
    pose1_c2w: np.ndarray
    pose2_c2w: np.ndarray
    dataset: str
    pair_id: str
    intrinsics1: np.ndarray | None = None
    intrinsics2: np.ndarray | None = None

    @property
    def gt_pose2to1(self) -> np.ndarray:
        """Relative pose from camera-2 coordinates to camera-1 coordinates."""
        return np.linalg.inv(self.pose1_c2w) @ self.pose2_c2w


def _colmap_to_opencv_intrinsics(K: np.ndarray) -> np.ndarray:
    K = K.copy()
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    return K


def _opencv_to_colmap_intrinsics(K: np.ndarray) -> np.ndarray:
    K = K.copy()
    K[0, 2] += 0.5
    K[1, 2] += 0.5
    return K


def _camera_matrix_of_crop(
    input_camera_matrix: np.ndarray,
    input_resolution,
    output_resolution,
    scaling: float = 1.0,
    offset_factor: float = 0.5,
    offset=None,
) -> np.ndarray:
    margins = np.asarray(input_resolution) * scaling - output_resolution
    assert np.all(margins >= 0.0)
    if offset is None:
        offset = offset_factor * margins

    output_camera_matrix_colmap = _opencv_to_colmap_intrinsics(input_camera_matrix)
    output_camera_matrix_colmap[:2, :] *= scaling
    output_camera_matrix_colmap[:2, 2] -= offset
    return _colmap_to_opencv_intrinsics(output_camera_matrix_colmap)


def _bbox_from_intrinsics_in_out(
    input_camera_matrix: np.ndarray,
    output_camera_matrix: np.ndarray,
    output_resolution,
) -> tuple[int, int, int, int]:
    out_width, out_height = output_resolution
    l, t = np.int32(np.round(input_camera_matrix[:2, 2] - output_camera_matrix[:2, 2]))
    return (l, t, l + out_width, t + out_height)


def _crop_image(
    image: Image.Image,
    camera_intrinsics: np.ndarray,
    crop_bbox: tuple[int, int, int, int],
) -> tuple[Image.Image, np.ndarray]:
    l, t, r, b = crop_bbox
    image = image.crop((l, t, r, b))

    camera_intrinsics = camera_intrinsics.copy()
    camera_intrinsics[0, 2] -= l
    camera_intrinsics[1, 2] -= t
    return image, camera_intrinsics


def _rescale_image(
    image: Image.Image,
    camera_intrinsics: np.ndarray,
    output_resolution,
) -> tuple[Image.Image, np.ndarray]:
    input_resolution = np.array(image.size)
    output_resolution = np.array(output_resolution)

    scale_final = max(output_resolution / image.size) + 1e-8
    output_resolution = np.floor(input_resolution * scale_final).astype(int)
    image = image.resize(tuple(output_resolution), resample=LANCZOS)

    camera_intrinsics = _camera_matrix_of_crop(
        camera_intrinsics,
        input_resolution,
        output_resolution,
        scaling=scale_final,
    )
    return image, camera_intrinsics


def _crop_resize_if_necessary(
    image: Image.Image,
    intrinsics: np.ndarray,
    resolution: tuple[int, int],
    info: str | None = None,
) -> tuple[Image.Image, np.ndarray]:
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    W, H = image.size
    cx, cy = intrinsics[:2, 2].round().astype(int)
    min_margin_x = min(cx, W - cx)
    min_margin_y = min(cy, H - cy)
    assert min_margin_x > W / 5, f"Bad principal point in view={info}"
    assert min_margin_y > H / 5, f"Bad principal point in view={info}"

    crop_bbox = (cx - min_margin_x, cy - min_margin_y, cx + min_margin_x, cy + min_margin_y)
    image, intrinsics = _crop_image(image, intrinsics, crop_bbox)

    W, H = image.size
    if H > 1.1 * W:
        resolution = resolution[::-1]

    image, intrinsics = _rescale_image(image, intrinsics, np.array(resolution))

    intrinsics2 = _camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
    crop_bbox = _bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
    image, intrinsics2 = _crop_image(image, intrinsics, crop_bbox)
    return image, intrinsics2


class ScanNet1500Pairs:
    """
    Lightweight reader for reloc3r/data/scannet1500.

    Notes:
    - Pair definitions come from test.npz['name'].
    - GT pose files are loaded directly from the extracted ScanNet1500 folder.
    - Pose files are treated as camera-to-world transforms, matching reloc3r's eval logic.
    """

    def __init__(self, data_root: str | Path):
        self.data_root = Path(data_root)
        self.pairs_path = self.data_root / "test.npz"
        self.scenes_root = self.data_root / "scannet_test_1500"
        if not self.pairs_path.exists():
            raise FileNotFoundError(f"Cannot find pair file: {self.pairs_path}")
        if not self.scenes_root.exists():
            raise FileNotFoundError(
                f"Cannot find extracted ScanNet1500 folder: {self.scenes_root}. "
                "Expected reloc3r/data/scannet1500/scannet_test_1500/"
            )

        with np.load(self.pairs_path) as data:
            self.pair_names = data["name"]

    def __len__(self) -> int:
        return len(self.pair_names)

    def __getitem__(self, idx: int) -> PairSample:
        scene_name, scene_sub_name, name1, name2 = self.pair_names[idx]
        scene_dir = self.scenes_root / f"scene{int(scene_name):04d}_{int(scene_sub_name):02d}"

        image1 = scene_dir / "color" / f"{int(name1)}.jpg"
        image2 = scene_dir / "color" / f"{int(name2)}.jpg"
        pose1 = scene_dir / "pose" / f"{int(name1)}.txt"
        pose2 = scene_dir / "pose" / f"{int(name2)}.txt"

        if not image1.exists() or not image2.exists():
            raise FileNotFoundError(f"Missing image pair at index {idx}: {image1}, {image2}")
        if not pose1.exists() or not pose2.exists():
            raise FileNotFoundError(f"Missing pose pair at index {idx}: {pose1}, {pose2}")

        pair_id = f"scene{int(scene_name):04d}_{int(scene_sub_name):02d}/{int(name1)}-{int(name2)}"
        return PairSample(
            image1=str(image1),
            image2=str(image2),
            pose1_c2w=np.loadtxt(pose1).astype(np.float32),
            pose2_c2w=np.loadtxt(pose2).astype(np.float32),
            dataset="scannet1500",
            pair_id=pair_id,
        )


class MegaDepth1500Pairs:
    """
    Lightweight reader for reloc3r/data/megadepth1500.

    Notes:
    - Pair definitions come from megadepth_test_pairs.txt.
    - GT camera poses follow reloc3r/reloc3r/datasets/megadepth_valid.py exactly:
      camera_pose = inv(metadata[view_idx]['pose']).
    - For DA3 batching, MegaDepth images can be preprocessed with reloc3r-style
      crop-resize around the principal point into a fixed square resolution.
    """

    def __init__(self, data_root: str | Path):
        self.data_root = Path(data_root)
        self.meta_path = self.data_root / "megadepth_meta_test.npz"
        self.pairs_path = self.data_root / "megadepth_test_pairs.txt"
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Cannot find metadata file: {self.meta_path}")
        if not self.pairs_path.exists():
            raise FileNotFoundError(f"Cannot find pair list: {self.pairs_path}")

        self.metadata = np.load(self.meta_path, allow_pickle=True)
        self.pairs = []
        with self.pairs_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                image1, image2 = line.split()
                self.pairs.append((image1, image2))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> PairSample:
        image1_rel, image2_rel = self.pairs[idx]
        image1 = self.data_root / image1_rel
        image2 = self.data_root / image2_rel
        if not image1.exists() or not image2.exists():
            raise FileNotFoundError(f"Missing image pair at index {idx}: {image1}, {image2}")

        meta1 = self.metadata[image1_rel].item()
        meta2 = self.metadata[image2_rel].item()
        intrinsics1 = np.asarray(meta1["intrinsic"], dtype=np.float32)
        intrinsics2 = np.asarray(meta2["intrinsic"], dtype=np.float32)
        pose1_c2w = np.linalg.inv(np.asarray(meta1["pose"], dtype=np.float32)).astype(np.float32)
        pose2_c2w = np.linalg.inv(np.asarray(meta2["pose"], dtype=np.float32)).astype(np.float32)

        return PairSample(
            image1=str(image1),
            image2=str(image2),
            pose1_c2w=pose1_c2w,
            pose2_c2w=pose2_c2w,
            dataset="megadepth1500",
            pair_id=f"{image1_rel} {image2_rel}",
            intrinsics1=intrinsics1,
            intrinsics2=intrinsics2,
        )

    def _prepare_image_for_da3(
        self,
        image_path: str,
        intrinsics: np.ndarray,
        resolution: tuple[int, int],
    ) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        image, _ = _crop_resize_if_necessary(image, intrinsics.copy(), resolution, info=image_path)
        return np.asarray(image)

    def prepare_pair_images_for_da3(
        self,
        sample: PairSample,
        resolution: tuple[int, int],
    ) -> list[np.ndarray]:
        assert sample.intrinsics1 is not None and sample.intrinsics2 is not None
        return [
            self._prepare_image_for_da3(sample.image1, sample.intrinsics1, resolution),
            self._prepare_image_for_da3(sample.image2, sample.intrinsics2, resolution),
        ]
