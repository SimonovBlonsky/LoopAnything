#!/usr/bin/env python3
"""Evaluate Reloc3r-512 on KITTI visual localization.

Uses the exact same preprocessing as reloc3r/eval_visloc.py:
- NetVLAD retrieval with BGR min-max normalized images (same as SevenScenesRetrieval)
- Reloc3r relpose with reloc3r-style crop + ImgNorm (same as SevenScenesRelpose)
- Motion averaging with Reloc3rVisloc (same solver)

This ensures fair comparison with our DA3 eval_training_free_visloc.py on KITTI.

Usage:
    cd LoopAnything/.worktrees/unified_pipeline1.1
    PYTHONPATH=src python scripts/eval_reloc3r_kitti.py \
        --data-root data/kitti_visloc \
        --scene 00 --topk 10
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Bootstrap paths — walk up to find NeurIPS26 repo root (contains reloc3r/).
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = "/home/chenguyuan/code/NeurIPS26"
RELOC3R_ROOT = "/home/chenguyuan/code/NeurIPS26/reloc3r"
for p in [str(REPO_ROOT), str(RELOC3R_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from netvlad_image_retrieval.netvlad import NetVLAD
from reloc3r.reloc3r_relpose import Reloc3rRelpose, setup_reloc3r_relpose_model, inference_relpose
from reloc3r.reloc3r_visloc import Reloc3rVisloc
from reloc3r.utils.metric import get_rot_err
from utils.image import imread_cv2, ImgNorm


# ---------------------------------------------------------------------------
# Inline cropping utilities (from datasets/utils/cropping.py + geometry.py)
# Avoids importing `datasets` package which conflicts with HuggingFace on some envs.
# ---------------------------------------------------------------------------
try:
    import PIL.Image
    _lanczos = getattr(PIL.Image.Resampling, "LANCZOS", PIL.Image.LANCZOS)
except AttributeError:
    _lanczos = PIL.Image.LANCZOS


def _colmap_to_opencv_intrinsics(K):
    K = K.copy()
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    return K

def _opencv_to_colmap_intrinsics(K):
    K = K.copy()
    K[0, 2] += 0.5
    K[1, 2] += 0.5
    return K

def _camera_matrix_of_crop(input_camera_matrix, input_resolution, output_resolution, scaling=1, offset_factor=0.5, offset=None):
    margins = np.asarray(input_resolution) * scaling - output_resolution
    if offset is None:
        offset = offset_factor * margins
    output_camera_matrix_colmap = _opencv_to_colmap_intrinsics(input_camera_matrix)
    output_camera_matrix_colmap[:2, :] *= scaling
    output_camera_matrix_colmap[:2, 2] -= offset
    return _colmap_to_opencv_intrinsics(output_camera_matrix_colmap)

def _bbox_from_intrinsics_in_out(input_camera_matrix, output_camera_matrix, output_resolution):
    out_width, out_height = output_resolution
    l, t = np.int32(np.round(input_camera_matrix[:2, 2] - output_camera_matrix[:2, 2]))
    return (l, t, l + out_width, t + out_height)

def _crop_image(image, camera_intrinsics, crop_bbox):
    if not isinstance(image, PIL.Image.Image):
        image = PIL.Image.fromarray(image)
    l, t, r, b = crop_bbox
    image = image.crop((l, t, r, b))
    camera_intrinsics = camera_intrinsics.copy()
    camera_intrinsics[0, 2] -= l
    camera_intrinsics[1, 2] -= t
    return image, camera_intrinsics

def _rescale_image(image, camera_intrinsics, output_resolution):
    if not isinstance(image, PIL.Image.Image):
        image = PIL.Image.fromarray(image)
    input_resolution = np.array(image.size)
    output_resolution = np.array(output_resolution)
    scale_final = max(output_resolution / image.size) + 1e-8
    scaled_res = np.floor(input_resolution * scale_final).astype(int)
    image = image.resize(tuple(scaled_res), resample=_lanczos)
    camera_intrinsics = _camera_matrix_of_crop(camera_intrinsics, input_resolution, scaled_res, scaling=scale_final)
    return image, camera_intrinsics


# ---------------------------------------------------------------------------
# KITTI data loading (same format as prepare_kitti_visloc.py output)
# ---------------------------------------------------------------------------
def load_kitti_entries(data_root: str, scene: str, split: str):
    """Load frame entries from KITTI visloc format."""
    scene_dir = Path(data_root) / scene
    frames_dir = scene_dir / "frames"
    split_file = scene_dir / ("TrainSplit.txt" if split == "train" else "TestSplit.txt")
    intrinsics = np.loadtxt(str(scene_dir / "intrinsics.txt")).astype(np.float32)

    entries = []
    with open(split_file) as f:
        for line in f:
            fid = line.strip()
            if not fid:
                continue
            img_path = str(frames_dir / f"{fid}.color.png")
            pose_path = str(frames_dir / f"{fid}.pose.txt")
            if os.path.exists(img_path) and os.path.exists(pose_path):
                pose = np.loadtxt(pose_path).astype(np.float32)
                entries.append({
                    "frame_id": fid,
                    "image_path": img_path,
                    "pose": pose,
                    "intrinsics": intrinsics.copy(),
                })
    return entries


# ---------------------------------------------------------------------------
# Retrieval: NetVLAD with BGR + min-max (same as reloc3r SevenScenesRetrieval)
# ---------------------------------------------------------------------------
def load_image_for_netvlad(image_path: str, device):
    """Load image for NetVLAD: BGR, min-max [0,1]. Same as reloc3r's load_image."""
    color = cv2.imread(image_path)  # BGR, uint8
    data = {}
    data["image"] = torch.Tensor(color).permute(2, 0, 1)[None].to(device)
    data["image"] = (data["image"] - data["image"].min()) / (data["image"].max() - data["image"].min())
    return data


@torch.no_grad()
def build_netvlad_descriptors(entries, netvlad_model, device):
    """Extract NetVLAD descriptors for all entries."""
    descs = []
    for entry in tqdm(entries, desc="NetVLAD descriptors"):
        data = load_image_for_netvlad(entry["image_path"], device)
        desc = netvlad_model(data)["global_descriptor"]
        descs.append(desc.cpu().squeeze())
    return torch.stack(descs)  # [N, D]


@torch.no_grad()
def retrieve_topk(query_desc, db_descs, topk, device):
    """Retrieve top-K database indices for a single query."""
    sim = torch.einsum("d,nd->n", query_desc.to(device), db_descs.to(device))
    values, indices = torch.topk(sim, k=min(topk, len(db_descs)))
    return indices.cpu().numpy()


# ---------------------------------------------------------------------------
# Relpose: reloc3r-style crop + ImgNorm (same as BaseStereoViewDataset)
# ---------------------------------------------------------------------------
def crop_resize_reloc3r(image_path: str, intrinsics: np.ndarray,
                        resolution=(512, 384)) -> tuple:
    """Apply reloc3r's _crop_resize_if_necessary. Returns (PIL image, intrinsics)."""
    import PIL.Image
    img = imread_cv2(image_path)  # RGB numpy
    if not isinstance(img, PIL.Image.Image):
        img = PIL.Image.fromarray(img)

    K = copy.deepcopy(intrinsics)
    W, H = img.size
    cx, cy = K[:2, 2].round().astype(int)
    min_margin_x = min(cx, W - cx)
    min_margin_y = min(cy, H - cy)
    l, t = cx - min_margin_x, cy - min_margin_y
    r, b = cx + min_margin_x, cy + min_margin_y
    img, K = _crop_image(img, K, (l, t, r, b))

    W, H = img.size
    res = resolution
    assert res[0] >= res[1]
    if H > 1.1 * W:
        res = res[::-1]

    img, K = _rescale_image(img, K, np.array(res))
    K2 = _camera_matrix_of_crop(K, img.size, res, offset_factor=0.5)
    crop_bbox = _bbox_from_intrinsics_in_out(K, K2, res)
    img, K2 = _crop_image(img, K, crop_bbox)

    return img, K2


def prepare_reloc3r_pair(query_entry, db_entry, resolution=(512, 384)):
    """Prepare a view pair for reloc3r inference.

    Returns (view1=db, view2=query) matching reloc3r's convention.
    Each view has 'img' (tensor), 'camera_pose' (4x4), etc.
    """
    views = []
    for entry in [db_entry, query_entry]:  # [db, query] = [view1, view2]
        pil_img, _ = crop_resize_reloc3r(
            entry["image_path"], entry["intrinsics"], resolution,
        )
        true_shape = np.int32((pil_img.size[1], pil_img.size[0]))
        img_tensor = ImgNorm(pil_img)  # ToTensor + Normalize(0.5, 0.5, 0.5) → [-1, 1]

        views.append({
            "img": img_tensor,
            "camera_pose": torch.from_numpy(entry["pose"]).float(),
            "camera_intrinsics": torch.from_numpy(entry["intrinsics"]).float(),
            "true_shape": torch.from_numpy(true_shape),
            "dataset": "KITTI",
            "label": entry["frame_id"],
            "instance": entry["frame_id"],
        })
    return views


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Reloc3r on KITTI visloc")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Path to kitti_visloc/ (output of prepare_kitti_visloc.py)")
    parser.add_argument("--scene", type=str, required=True, help="Sequence id (e.g. 00)")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--model", type=str, default="Reloc3rRelpose(img_size=512)")
    parser.add_argument("--resolution", type=int, nargs=2, default=[512, 384],
                        help="Reloc3r crop resolution (width, height)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="workspace/reloc3r_kitti_results")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    resolution = tuple(args.resolution)

    print(f"Scene: {args.scene}, Top-K: {args.topk}, Resolution: {resolution}")

    # Load data.
    db_entries = load_kitti_entries(args.data_root, args.scene, "train")
    query_entries = load_kitti_entries(args.data_root, args.scene, "test")
    print(f"Database: {len(db_entries)} frames, Queries: {len(query_entries)} frames")

    # Step 1: NetVLAD retrieval (same as reloc3r eval_visloc.py).
    print("Loading NetVLAD...")
    netvlad_model = NetVLAD(NetVLAD.default_conf).eval().to(device)
    db_descs = build_netvlad_descriptors(db_entries, netvlad_model, device)
    print("Retrieving top-K for each query...")
    query_topk = []
    for q_entry in tqdm(query_entries, desc="Retrieval"):
        q_data = load_image_for_netvlad(q_entry["image_path"], device)
        q_desc = netvlad_model(q_data)["global_descriptor"].cpu().squeeze()
        topk_idx = retrieve_topk(q_desc, db_descs, args.topk, device)
        query_topk.append(topk_idx)

    # Step 2: Reloc3r pairwise relpose.
    print("Loading Reloc3r model...")
    reloc3r_model = setup_reloc3r_relpose_model(args.model, device)

    print("Computing pairwise relative poses...")
    all_relposes = []  # [n_queries][topk] of 4x4
    all_db_poses = []  # [n_queries][topk] of 4x4
    for qi, q_entry in enumerate(tqdm(query_entries, desc="Relpose")):
        relposes_for_query = []
        db_poses_for_query = []
        for db_idx in query_topk[qi]:
            db_entry = db_entries[db_idx]
            views = prepare_reloc3r_pair(q_entry, db_entry, resolution)
            batch = [
                {k: v.unsqueeze(0) if torch.is_tensor(v) else v for k, v in views[0].items()},
                {k: v.unsqueeze(0) if torch.is_tensor(v) else v for k, v in views[1].items()},
            ]
            with torch.no_grad():
                pose_q2d = inference_relpose(batch, reloc3r_model, device)  # [1, 4, 4]
            Rt = np.eye(4)
            Rt[:3, :3] = pose_q2d[0, :3, :3].cpu().numpy()
            Rt[:3, 3] = pose_q2d[0, :3, 3].cpu().numpy()
            relposes_for_query.append(Rt)
            db_poses_for_query.append(db_entry["pose"].astype(np.float64))
        all_relposes.append(relposes_for_query)
        all_db_poses.append(db_poses_for_query)

    # Step 3: Motion averaging (same solver as eval_visloc.py).
    print("Running motion averaging...")
    solver = Reloc3rVisloc()
    rerrs, terrs = [], []
    for qi, q_entry in enumerate(query_entries):
        gt_q = q_entry["pose"].astype(np.float64)
        Rt = solver.motion_averaging(all_db_poses[qi], all_relposes[qi])
        rerr = get_rot_err(Rt[:3, :3], gt_q[:3, :3])
        terr = np.linalg.norm(Rt[:3, 3] - gt_q[:3, 3])
        rerrs.append(rerr)
        terrs.append(terr)

    med_terr = np.median(terrs)
    med_rerr = np.median(rerrs)
    print(f"\n[Reloc3r-512] Scene {args.scene} median pose error: {med_terr:.2f} m  {med_rerr:.2f} deg")

    # Save results.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / f"reloc3r_kitti_{args.scene}.npz",
        rotation_errors=np.array(rerrs, dtype=np.float32),
        translation_errors=np.array(terrs, dtype=np.float32),
    )
    print(f"Saved to {output_dir}/reloc3r_kitti_{args.scene}.npz")


if __name__ == "__main__":
    main()
