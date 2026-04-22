#!/usr/bin/env python3
"""Evaluate Reloc3r-512 on any visloc dataset (KITTI/EuRoC/ETH3D).

Uses the exact same preprocessing as reloc3r/eval_visloc.py:
- NetVLAD retrieval with BGR min-max normalized images
- Reloc3r relpose with reloc3r-style crop + ImgNorm
- Motion averaging with Reloc3rVisloc

Usage:
    python scripts/eval_reloc3r_visloc.py --data-root data/kitti_visloc --scene 00
    python scripts/eval_reloc3r_visloc.py --data-root data/euroc_visloc --scene MH_01_easy
    python scripts/eval_reloc3r_visloc.py --data-root data/eth3d_visloc --scene cables_1
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
    parser.add_argument("--save-failure-cases", type=int, default=0, metavar="N",
                        help="Save top-N worst failure cases with images and diagnostics.")
    parser.add_argument("--oracle-retrieval", type=str, default=None,
                        choices=["position", "combined"],
                        help="Bypass NetVLAD and use GT pose for top-K selection. "
                             "position: rank by translation distance. "
                             "combined: rank by translation + rotation angular distance.")
    return parser.parse_args()


def compute_oracle_topk(db_entries, query_entries, top_k, mode="combined"):
    """Top-K DB indices for each query using GT poses (no visual features)."""
    db_pos = np.array([e["pose"][:3, 3] for e in db_entries], dtype=np.float64)
    db_rot = np.array([e["pose"][:3, :3] for e in db_entries], dtype=np.float64)
    out = np.zeros((len(query_entries), top_k), dtype=np.int64)
    for qi, q in enumerate(query_entries):
        q_pos = q["pose"][:3, 3].astype(np.float64)
        q_rot = q["pose"][:3, :3].astype(np.float64)
        pos_d = np.linalg.norm(db_pos - q_pos, axis=1)
        if mode == "position":
            scores = pos_d
        else:  # combined
            R_rel = np.einsum("ij,njk->nik", q_rot.T, db_rot)
            trace = np.einsum("nii->n", R_rel)
            rot_d = np.arccos(np.clip((trace - 1) / 2, -1.0 + 1e-8, 1.0 - 1e-8))
            scores = pos_d + rot_d
        idx = np.argsort(scores)[: min(top_k, len(scores))]
        out[qi, : len(idx)] = idx
    return out


def main():
    args = parse_args()
    device = torch.device(args.device)
    resolution = tuple(args.resolution)

    print(f"Scene: {args.scene}, Top-K: {args.topk}, Resolution: {resolution}")

    # Load data.
    db_entries = load_kitti_entries(args.data_root, args.scene, "train")
    query_entries = load_kitti_entries(args.data_root, args.scene, "test")
    print(f"Database: {len(db_entries)} frames, Queries: {len(query_entries)} frames")

    # Step 1: retrieval — either NetVLAD or GT-based oracle.
    if args.oracle_retrieval is not None:
        print(f"[INFO] Oracle retrieval ({args.oracle_retrieval}): bypassing NetVLAD.")
        oracle = compute_oracle_topk(db_entries, query_entries, args.topk, mode=args.oracle_retrieval)
        query_topk = [oracle[qi] for qi in range(len(query_entries))]
    else:
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
    rerrs, terrs, pred_poses = [], [], []
    for qi, q_entry in enumerate(query_entries):
        gt_q = q_entry["pose"].astype(np.float64)
        Rt = solver.motion_averaging(all_db_poses[qi], all_relposes[qi])
        rerr = get_rot_err(Rt[:3, :3], gt_q[:3, :3])
        terr = np.linalg.norm(Rt[:3, 3] - gt_q[:3, 3])
        rerrs.append(rerr)
        terrs.append(terr)
        pred_poses.append(Rt)

    med_terr = np.median(terrs)
    med_rerr = np.median(rerrs)
    retrieval_tag = f"oracle-{args.oracle_retrieval}" if args.oracle_retrieval else "netvlad"
    print(f"\n[Reloc3r-512][{retrieval_tag}] Scene {args.scene} median pose error: "
          f"{med_terr:.2f} m  {med_rerr:.2f} deg")

    # Save results.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"reloc3r_{retrieval_tag}_{args.scene}.npz"
    np.savez(
        out_file,
        rotation_errors=np.array(rerrs, dtype=np.float32),
        translation_errors=np.array(terrs, dtype=np.float32),
    )
    print(f"Saved to {out_file}")

    # Save failure cases.
    if args.save_failure_cases > 0:
        save_failure_cases(
            db_entries=db_entries,
            query_entries=query_entries,
            query_topk=query_topk,
            pred_poses=pred_poses,
            rerrs=rerrs,
            terrs=terrs,
            output_dir=str(output_dir),
            top_n=args.save_failure_cases,
        )


def save_failure_cases(
    db_entries: list[dict],
    query_entries: list[dict],
    query_topk: list[np.ndarray],
    pred_poses: list[np.ndarray],
    rerrs: list[float],
    terrs: list[float],
    output_dir: str,
    top_n: int = 10,
) -> None:
    """Save the top-N worst failure cases with images and diagnostics."""
    import shutil
    import json

    rot_errs = np.array(rerrs, dtype=np.float32)
    trans_errs = np.array(terrs, dtype=np.float32)
    combined = rot_errs + trans_errs * 10.0
    worst_indices = np.argsort(combined)[::-1][:top_n]

    fc_dir = Path(output_dir) / "failure_cases"
    fc_dir.mkdir(parents=True, exist_ok=True)

    db_positions = np.array([e["pose"][:3, 3] for e in db_entries])

    summary_rows = []
    for rank, qi in enumerate(worst_indices):
        case_dir = fc_dir / f"rank{rank:02d}_query{qi:04d}"
        case_dir.mkdir(parents=True, exist_ok=True)

        q_entry = query_entries[qi]
        gt_pose = q_entry["pose"].astype(np.float64)
        q_pos = gt_pose[:3, 3]

        # Copy query image.
        q_img_src = q_entry["image_path"]
        shutil.copy2(q_img_src, case_dir / f"query_{Path(q_img_src).name}")

        # Copy retrieved candidates and compute spatial distances.
        tk = query_topk[qi]
        retrieval_info = []
        for k, db_idx in enumerate(tk):
            db_entry = db_entries[db_idx]
            db_img_src = db_entry["image_path"]
            shutil.copy2(db_img_src, case_dir / f"candidate_top{k}_{Path(db_img_src).name}")

            db_pos = db_entry["pose"].astype(np.float64)[:3, 3]
            spatial_dist = float(np.linalg.norm(q_pos - db_pos))
            gt_rel = np.linalg.inv(db_entry["pose"].astype(np.float64)) @ gt_pose
            gt_t_norm = float(np.linalg.norm(gt_rel[:3, 3]))

            retrieval_info.append({
                "rank": k,
                "db_index": int(db_idx),
                "db_image": Path(db_img_src).name,
                "spatial_distance_m": round(spatial_dist, 3),
                "gt_relative_t_norm_m": round(gt_t_norm, 3),
            })

        nn_dist = float(np.linalg.norm(db_positions - q_pos, axis=1).min())
        nn_idx = int(np.linalg.norm(db_positions - q_pos, axis=1).argmin())

        pred_pose = pred_poses[qi]

        case_info = {
            "rank": rank,
            "query_index": int(qi),
            "query_image": Path(q_img_src).name,
            "rotation_error_deg": round(float(rot_errs[qi]), 3),
            "translation_error_m": round(float(trans_errs[qi]), 3),
            "combined_score": round(float(combined[qi]), 3),
            "gt_position": q_pos.tolist(),
            "pred_position": pred_pose[:3, 3].tolist(),
            "position_error_vector": (pred_pose[:3, 3] - gt_pose[:3, 3]).tolist(),
            "nearest_db_distance_m": round(nn_dist, 3),
            "nearest_db_index": nn_idx,
            "retrieved_candidates": retrieval_info,
        }

        np.savetxt(case_dir / "gt_pose.txt", gt_pose, fmt="%.8f")
        np.savetxt(case_dir / "pred_pose.txt", pred_pose, fmt="%.8f")

        with open(case_dir / "info.json", "w") as f:
            json.dump(case_info, f, indent=2)
        summary_rows.append(case_info)

    with open(fc_dir / "summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)

    print(f"[Failure Cases] Saved {len(worst_indices)} cases to {fc_dir}/")
    for row in summary_rows[:5]:
        print(f"  rank={row['rank']} q={row['query_index']}: "
              f"t_err={row['translation_error_m']:.2f}m, r_err={row['rotation_error_deg']:.1f}°, "
              f"nn_dist={row['nearest_db_distance_m']:.1f}m, "
              f"top1_spatial={row['retrieved_candidates'][0]['spatial_distance_m']:.1f}m")


if __name__ == "__main__":
    main()
