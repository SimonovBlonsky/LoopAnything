#!/usr/bin/env python3
"""Convert KITTI Odometry sequences into a visual localization dataset.

For each sequence with loop closures, splits frames into database (first
traversal) and query (revisiting frames), so that the evaluation measures
relocalization in previously visited areas — matching the 7Scenes/Cambridge
protocol used by eval_training_free_visloc.py.

Usage:
    python scripts/prepare_kitti_visloc.py \
        --kitti-root /data/datasets/kitti_odometry/dataset \
        --output-dir data/kitti_visloc \
        --sequences 00 05 06 07 \
        --revisit-radius 25.0 \
        --db-subsample 5

Output per sequence:
    data/kitti_visloc/{seq}/
        TrainSplit.txt      (database frame indices)
        TestSplit.txt       (query frame indices)
        frames/
            {frame_id:06d}.png  (symlink to original image)
            {frame_id:06d}.pose.txt  (4x4 c2w matrix)
        intrinsics.txt      (3x3 camera intrinsic matrix)

Then evaluate with:
    python ablation/eval_training_free_visloc.py \
        --dataset kitti --scene 00 \
        --data-root data/kitti_visloc ...
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare KITTI Odometry for visual localization")
    parser.add_argument("--kitti-root", type=str, required=True,
                        help="Path to kitti_odometry/dataset/")
    parser.add_argument("--output-dir", type=str, default="data/kitti_visloc",
                        help="Output directory for converted dataset")
    parser.add_argument("--sequences", type=str, nargs="+", default=["00", "05", "06", "07"],
                        help="Sequences to convert (must have GT poses)")
    parser.add_argument("--revisit-radius", type=float, default=25.0,
                        help="Distance threshold (m) to consider a frame as revisiting")
    parser.add_argument("--db-subsample", type=int, default=5,
                        help="Subsample database frames (take every N-th frame)")
    parser.add_argument("--query-subsample", type=int, default=1,
                        help="Subsample query frames")
    parser.add_argument("--camera", type=str, default="image_2",
                        choices=["image_0", "image_1", "image_2", "image_3"],
                        help="Which camera to use")
    return parser.parse_args()


def load_poses(pose_file: str) -> np.ndarray:
    """Load KITTI poses [N, 4, 4] as c2w matrices."""
    raw = np.loadtxt(pose_file).reshape(-1, 3, 4)
    n = raw.shape[0]
    poses = np.zeros((n, 4, 4), dtype=np.float64)
    poses[:, :3, :] = raw
    poses[:, 3, 3] = 1.0
    # KITTI poses are already c2w (camera 0 to world).
    return poses


def load_intrinsics(calib_file: str, camera: str = "image_2") -> np.ndarray:
    """Extract 3x3 intrinsics from KITTI calib.txt."""
    cam_idx = {"image_0": 0, "image_1": 1, "image_2": 2, "image_3": 3}[camera]
    with open(calib_file) as f:
        for line in f:
            if line.startswith(f"P{cam_idx}:"):
                vals = list(map(float, line.strip().split()[1:]))
                P = np.array(vals).reshape(3, 4)
                K = P[:3, :3].copy()
                return K.astype(np.float32)
    raise ValueError(f"P{cam_idx} not found in {calib_file}")


def split_db_query(
    positions: np.ndarray,
    revisit_radius: float,
    db_subsample: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Split frames into database (first traversal) and query (revisiting).

    Strategy:
    - Database = first-half frames, subsampled by db_subsample.
    - Query = second-half frames within revisit_radius of some ACTUAL DB frame.
    - This ensures every query has at least one spatially close DB frame.
    """
    n = len(positions)
    half = n // 2

    db_indices = np.array(list(range(0, half, db_subsample)))
    db_positions = positions[db_indices]

    query_indices = []
    for i in range(half, n):
        dists = np.linalg.norm(db_positions - positions[i], axis=1)
        if dists.min() < revisit_radius:
            query_indices.append(i)

    return db_indices, np.array(query_indices)


def convert_sequence(
    kitti_root: str,
    output_dir: str,
    seq: str,
    camera: str,
    revisit_radius: float,
    db_subsample: int,
    query_subsample: int,
):
    seq_dir = Path(kitti_root) / "sequences" / seq
    pose_file = Path(kitti_root) / "poses" / f"{seq}.txt"
    calib_file = seq_dir / "calib.txt"
    img_dir = seq_dir / camera / "data" if (seq_dir / camera / "data").is_dir() else seq_dir / camera

    if not pose_file.exists():
        print(f"  [SKIP] No GT poses for sequence {seq}")
        return

    poses = load_poses(str(pose_file))
    K = load_intrinsics(str(calib_file), camera)
    positions = poses[:, :3, 3]
    n = len(poses)

    n_images = len(list(img_dir.glob("*.png")))
    assert n_images == n, f"Frame count mismatch: {n_images} images vs {n} poses"

    # Split into db and query (db_subsample applied BEFORE revisit check).
    db_indices, query_indices = split_db_query(positions, revisit_radius, db_subsample)

    if len(query_indices) == 0:
        print(f"  [SKIP] No revisiting frames in sequence {seq}")
        return

    # Subsample queries only.
    query_indices = query_indices[::query_subsample]

    print(f"  Database: {len(db_indices)} frames, Query: {len(query_indices)} frames")

    # Write output.
    out_dir = Path(output_dir) / seq
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Write intrinsics.
    np.savetxt(out_dir / "intrinsics.txt", K, fmt="%.6f")

    # Write pose files and symlink images.
    all_indices = set(db_indices.tolist()) | set(query_indices.tolist())
    for idx in sorted(all_indices):
        frame_name = f"{idx:06d}"
        # Symlink image.
        src_img = img_dir / f"{idx:06d}.png"
        dst_img = frames_dir / f"{frame_name}.color.png"
        if not dst_img.exists():
            os.symlink(src_img.resolve(), dst_img)
        # Write c2w pose (same format as 7Scenes .pose.txt).
        np.savetxt(frames_dir / f"{frame_name}.pose.txt", poses[idx], fmt="%.12e")

    # Write split files (same format as 7Scenes TrainSplit/TestSplit).
    with open(out_dir / "TrainSplit.txt", "w") as f:
        for idx in db_indices:
            f.write(f"{idx:06d}\n")
    with open(out_dir / "TestSplit.txt", "w") as f:
        for idx in query_indices:
            f.write(f"{idx:06d}\n")

    # Write metadata.
    with open(out_dir / "info.txt", "w") as f:
        f.write(f"sequence: {seq}\n")
        f.write(f"camera: {camera}\n")
        f.write(f"total_frames: {n}\n")
        f.write(f"db_frames: {len(db_indices)}\n")
        f.write(f"query_frames: {len(query_indices)}\n")
        f.write(f"revisit_radius: {revisit_radius}\n")
        f.write(f"db_subsample: {db_subsample}\n")
        f.write(f"query_subsample: {query_subsample}\n")


def main():
    args = parse_args()

    print(f"KITTI root: {args.kitti_root}")
    print(f"Output dir: {args.output_dir}")
    print(f"Sequences: {args.sequences}")
    print(f"Revisit radius: {args.revisit_radius}m")
    print(f"DB subsample: every {args.db_subsample} frames")
    print()

    for seq in args.sequences:
        print(f"Processing sequence {seq}...")
        convert_sequence(
            kitti_root=args.kitti_root,
            output_dir=args.output_dir,
            seq=seq,
            camera=args.camera,
            revisit_radius=args.revisit_radius,
            db_subsample=args.db_subsample,
            query_subsample=args.query_subsample,
        )

    print(f"\nDone. Dataset written to {args.output_dir}/")
    print("Evaluate with:")
    print(f"  python ablation/eval_training_free_visloc.py \\")
    print(f"    --dataset kitti --scene 00 \\")
    print(f"    --data-root {args.output_dir} ...")


if __name__ == "__main__":
    main()
