#!/usr/bin/env python3
"""Convert ETH3D SLAM sequences into visual localization dataset.

ETH3D format:
- rgb/{timestamp}.png
- groundtruth.txt: timestamp tx ty tz qx qy qz qw
- calibration.txt: fx fy cx cy (single line)

Split: first half = database, second half revisiting first half = queries.

Usage:
    python scripts/prepare_eth3d_visloc.py \
        --eth3d-root /data/datasets/ETH3DSLAM \
        --output-dir data/eth3d_visloc
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eth3d-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="data/eth3d_visloc")
    parser.add_argument("--split", type=str, default="training",
                        choices=["training", "test"])
    parser.add_argument("--sequences", type=str, nargs="+", default=None)
    parser.add_argument("--revisit-radius", type=float, default=0.5,
                        help="Revisit radius in meters (ETH3D is small-scale indoor)")
    parser.add_argument("--db-subsample", type=int, default=5)
    parser.add_argument("--query-subsample", type=int, default=5)
    return parser.parse_args()


def load_eth3d_gt(gt_path: str):
    """Load ETH3D groundtruth.txt: timestamp tx ty tz qx qy qz qw."""
    poses = {}
    with open(gt_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            ts = float(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = [tx, ty, tz]
            poses[ts] = T  # c2w
    return poses


def load_eth3d_rgb_list(rgb_txt_path: str):
    """Load rgb.txt: timestamp rgb/filename."""
    entries = []
    with open(rgb_txt_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            ts = float(parts[0])
            filename = parts[1]
            entries.append((ts, filename))
    return entries


def load_eth3d_intrinsics(calib_path: str):
    """Load calibration.txt: fx fy cx cy."""
    with open(calib_path) as f:
        line = f.readline().strip()
    vals = [float(x) for x in line.split()]
    fx, fy, cx, cy = vals[0], vals[1], vals[2], vals[3]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


def nearest_pose(gt_poses: dict, query_ts: float, max_dt: float = 0.05):
    """Find nearest GT pose within max_dt seconds."""
    best_ts, best_dt = None, float("inf")
    for ts in gt_poses:
        dt = abs(ts - query_ts)
        if dt < best_dt:
            best_dt = dt
            best_ts = ts
    if best_ts is not None and best_dt <= max_dt:
        return gt_poses[best_ts]
    return None


def convert_sequence(eth3d_root, output_dir, split, seq_name,
                     revisit_radius, db_subsample, query_subsample):
    seq_dir = Path(eth3d_root) / split / seq_name
    gt_path = seq_dir / "groundtruth.txt"
    rgb_txt = seq_dir / "rgb.txt"
    calib_path = seq_dir / "calibration.txt"

    if not gt_path.exists() or not rgb_txt.exists() or not calib_path.exists():
        print(f"  [SKIP] Missing files for {seq_name}")
        return

    K = load_eth3d_intrinsics(str(calib_path))
    gt_poses = load_eth3d_gt(str(gt_path))
    rgb_entries = load_eth3d_rgb_list(str(rgb_txt))

    frames = []
    for ts, filename in rgb_entries:
        pose = nearest_pose(gt_poses, ts)
        if pose is None:
            continue
        img_path = str(seq_dir / filename)
        if os.path.exists(img_path):
            frames.append({"ts": ts, "image_path": img_path, "pose": pose})

    if len(frames) < 20:
        print(f"  [SKIP] Too few frames ({len(frames)}) for {seq_name}")
        return

    positions = np.array([f["pose"][:3, 3] for f in frames])
    n = len(frames)
    half = n // 2

    db_indices = list(range(0, half, db_subsample))
    db_positions = positions[db_indices]

    query_indices = []
    for i in range(half, n, query_subsample):
        dists = np.linalg.norm(db_positions - positions[i], axis=1)
        if dists.min() < revisit_radius:
            query_indices.append(i)

    if len(query_indices) == 0:
        print(f"  [SKIP] No revisiting frames in {seq_name}")
        return

    print(f"  {seq_name}: DB={len(db_indices)}, Query={len(query_indices)}")

    out_dir = Path(output_dir) / seq_name
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(out_dir / "intrinsics.txt", K, fmt="%.6f")

    all_indices = set(db_indices) | set(query_indices)
    for idx in sorted(all_indices):
        f = frames[idx]
        fid = f"{idx:06d}"
        src = Path(f["image_path"])
        dst = frames_dir / f"{fid}.color.png"
        if not dst.exists():
            os.symlink(src.resolve(), dst)
        np.savetxt(frames_dir / f"{fid}.pose.txt", f["pose"], fmt="%.12e")

    with open(out_dir / "TrainSplit.txt", "w") as fp:
        for idx in db_indices:
            fp.write(f"{idx:06d}\n")
    with open(out_dir / "TestSplit.txt", "w") as fp:
        for idx in query_indices:
            fp.write(f"{idx:06d}\n")

    with open(out_dir / "info.txt", "w") as fp:
        fp.write(f"sequence: {seq_name}\ntotal_frames: {n}\n")
        fp.write(f"db_frames: {len(db_indices)}\nquery_frames: {len(query_indices)}\n")
        fp.write(f"revisit_radius: {revisit_radius}\n")


def main():
    args = parse_args()
    eth3d_root = Path(args.eth3d_root)
    split_dir = eth3d_root / args.split

    if args.sequences:
        sequences = args.sequences
    else:
        sequences = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])

    print(f"ETH3D root: {eth3d_root}, split: {args.split}")
    print(f"Sequences: {len(sequences)}\n")

    for seq in sequences:
        convert_sequence(str(eth3d_root), args.output_dir, args.split, seq,
                        args.revisit_radius, args.db_subsample, args.query_subsample)

    print(f"\nDone. Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
