#!/usr/bin/env python3
"""Convert CMU Seasons dataset into visual localization format.

CMU Seasons format:
- images/slice{N}/database/*.jpg  (reference images, sunny+no foliage)
- images/slice{N}/query/*.jpg     (query images, various conditions, NO GT pose)
- 3D-models/nvm_models/slice{N}.nvm  (NVM with DB poses)
- intrinsics/c0_calib.txt, c1_calib.txt
- query_lists/slice{N}.queries_with_intrinsics.txt

Two evaluation modes:
1. Self-eval: split DB into train/test to evaluate locally
2. Benchmark: DB as train, query images as test (no GT, generate submission file)

Usage:
    # Self-eval (local metrics)
    python scripts/prepare_cmu_seasons_visloc.py \
        --cmu-root /data/datasets/CMUSeasons \
        --output-dir data/cmu_visloc \
        --mode self_eval --slices 2 3

    # Benchmark submission
    python scripts/prepare_cmu_seasons_visloc.py \
        --cmu-root /data/datasets/CMUSeasons \
        --output-dir data/cmu_visloc_benchmark \
        --mode benchmark --slices 2 3
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmu-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="data/cmu_visloc")
    parser.add_argument("--slices", type=str, nargs="+", default=None,
                        help="Slices to convert (default: all available)")
    parser.add_argument("--mode", type=str, default="self_eval",
                        choices=["self_eval", "benchmark"],
                        help="self_eval: split DB for local eval. benchmark: DB=train, query=test (no GT)")
    parser.add_argument("--db-subsample", type=int, default=3,
                        help="Subsample DB frames (self_eval mode)")
    parser.add_argument("--camera", type=str, default="c0", choices=["c0", "c1", "both"],
                        help="Which camera to use")
    return parser.parse_args()


def rotation_from_quaternion(qvec):
    """Convert [w, x, y, z] quaternion to 3x3 rotation matrix."""
    norm = np.linalg.norm(qvec)
    if norm < 1e-10:
        raise ValueError(f"Degenerate quaternion with norm {norm}")
    qvec = qvec / norm
    qr, qi, qj, qk = qvec[0], qvec[1], qvec[2], qvec[3]
    R = np.zeros((3, 3))
    R[0, 0] = 1 - 2 * (qj ** 2 + qk ** 2)
    R[0, 1] = 2 * (qi * qj - qk * qr)
    R[0, 2] = 2 * (qi * qk + qj * qr)
    R[1, 0] = 2 * (qi * qj + qk * qr)
    R[1, 1] = 1 - 2 * (qi ** 2 + qk ** 2)
    R[1, 2] = 2 * (qj * qk - qi * qr)
    R[2, 0] = 2 * (qi * qk - qj * qr)
    R[2, 1] = 2 * (qj * qk + qi * qr)
    R[2, 2] = 1 - 2 * (qi ** 2 + qj ** 2)
    return R


def read_nvm(nvm_path: str, images_dir: str):
    """Read NVM file, return list of {image_path, pose_c2w, intrinsics, camera}."""
    with open(nvm_path) as f:
        lines = f.readlines()

    # Line 0: "NVM_V3" or "NVM_V3 FixedK ..."
    # Line 1: number of cameras (but sometimes directly the camera entries)
    # Find the line with the number of cameras
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if line and not line.startswith("NVM"):
            try:
                n_cameras = int(line)
                idx += 1
                break
            except ValueError:
                pass
        idx += 1
    else:
        # No count line found, try to parse directly after NVM header
        idx = 1
        n_cameras = 0
        # Count non-empty lines until we hit a blank or a non-camera line
        while idx < len(lines) and lines[idx].strip():
            n_cameras += 1
            idx += 1
        idx = 1  # Reset to start parsing

    entries = []
    for i in range(n_cameras):
        if idx + i >= len(lines):
            break
        parts = lines[idx + i].strip().split()
        if len(parts) < 10:
            continue

        rel_path = parts[0]  # e.g. "database/img_00119_c0_..."
        focal = float(parts[1])
        qw, qx, qy, qz = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        cx, cy, cz = float(parts[6]), float(parts[7]), float(parts[8])

        # Determine camera from filename
        cam = "c0" if "_c0_" in rel_path else "c1"

        # Construct full image path
        img_path = os.path.join(images_dir, rel_path)
        if not os.path.exists(img_path):
            # Try .jpg extension
            img_path_jpg = img_path.replace(".png", ".jpg")
            if os.path.exists(img_path_jpg):
                img_path = img_path_jpg

        # NVM convention: qvec is w2c rotation, center is camera center in world
        R = rotation_from_quaternion(np.array([qw, qx, qy, qz]))
        T = -R @ np.array([cx, cy, cz])
        Rt = np.eye(4)
        Rt[:3, :3] = R
        Rt[:3, 3] = T
        pose_c2w = np.linalg.inv(Rt)

        # Per-image intrinsics from NVM focal + image center assumption
        # CMU Seasons images are 1024x768
        K = np.array([[focal, 0, 512.0], [0, focal, 384.0], [0, 0, 1]], dtype=np.float32)

        entries.append({
            "image_path": img_path,
            "rel_path": rel_path,
            "pose": pose_c2w.astype(np.float32),
            "intrinsics": K,
            "camera": cam,
        })

    return entries


def load_intrinsics(calib_path: str):
    """Load 3x3 intrinsics from CMU calib file."""
    K = np.loadtxt(calib_path).astype(np.float32)
    return K


def load_query_list(query_list_path: str, images_dir: str):
    """Load query image list with per-query intrinsics."""
    entries = []
    with open(query_list_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            rel_path = parts[0]  # query/img_...
            # PINHOLE w h fx fy cx cy
            fx, fy = float(parts[4]), float(parts[5])
            cx, cy = float(parts[6]), float(parts[7])
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

            cam = "c0" if "_c0_" in rel_path else "c1"
            img_path = os.path.join(images_dir, rel_path)

            entries.append({
                "image_path": img_path,
                "rel_path": rel_path,
                "intrinsics": K,
                "camera": cam,
            })
    return entries


def write_visloc_format(entries_train, entries_test, output_dir, has_gt_test=True):
    """Write entries in our standard visloc format."""
    out_dir = Path(output_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    all_entries = entries_train + entries_test
    for i, entry in enumerate(all_entries):
        fid = f"{i:06d}"
        src = Path(entry["image_path"])
        dst = frames_dir / f"{fid}.color.png"
        if not dst.exists() and src.exists():
            # Symlink (works for both .jpg and .png)
            os.symlink(src.resolve(), dst)
        if "pose" in entry:
            np.savetxt(frames_dir / f"{fid}.pose.txt", entry["pose"], fmt="%.12e")
        entry["_fid"] = fid

    # Use first entry's intrinsics as representative
    if entries_train:
        np.savetxt(out_dir / "intrinsics.txt", entries_train[0]["intrinsics"], fmt="%.6f")

    n_train = len(entries_train)
    with open(out_dir / "TrainSplit.txt", "w") as f:
        for entry in entries_train:
            f.write(f"{entry['_fid']}\n")
    with open(out_dir / "TestSplit.txt", "w") as f:
        for entry in entries_test:
            f.write(f"{entry['_fid']}\n")

    # Save mapping for benchmark submission reconstruction
    mapping = {}
    for entry in all_entries:
        mapping[entry["_fid"]] = entry.get("rel_path", "")
    import json
    with open(out_dir / "fid_to_relpath.json", "w") as f:
        json.dump(mapping, f, indent=2)

    with open(out_dir / "info.txt", "w") as f:
        f.write(f"train_frames: {len(entries_train)}\n")
        f.write(f"test_frames: {len(entries_test)}\n")
        f.write(f"test_has_gt: {has_gt_test}\n")


def convert_slice(cmu_root, output_dir, slice_name, mode, db_subsample, camera):
    images_dir = Path(cmu_root) / "images" / slice_name
    nvm_path = Path(cmu_root) / "3D-models" / "nvm_models" / f"{slice_name}.nvm"

    if not nvm_path.exists():
        print(f"  [SKIP] No NVM for {slice_name}")
        return

    # Load DB entries from NVM
    db_entries = read_nvm(str(nvm_path), str(images_dir))

    # Filter by camera
    if camera != "both":
        db_entries = [e for e in db_entries if e["camera"] == camera]

    if len(db_entries) < 10:
        print(f"  [SKIP] Too few DB entries ({len(db_entries)}) for {slice_name}")
        return

    if mode == "self_eval":
        # Split DB into train/test
        n = len(db_entries)
        half = n // 2
        train_entries = db_entries[:half:db_subsample]
        # Test: second half entries that are within 25m of train
        train_positions = np.array([e["pose"][:3, 3] for e in train_entries])
        test_entries = []
        for e in db_entries[half:]:
            pos = e["pose"][:3, 3]
            dist = np.linalg.norm(train_positions - pos, axis=1).min()
            if dist < 25.0:
                test_entries.append(e)

        if len(test_entries) == 0:
            print(f"  [SKIP] No revisiting frames in {slice_name}")
            return

        print(f"  {slice_name} (self_eval): DB={len(train_entries)}, Query={len(test_entries)}")
        out = Path(output_dir) / slice_name
        write_visloc_format(train_entries, test_entries, str(out), has_gt_test=True)

    elif mode == "benchmark":
        # DB = all database images (subsampled), Query = query images (no GT)
        train_entries = db_entries[::db_subsample]
        query_list_path = Path(cmu_root) / "query_lists" / f"{slice_name}.queries_with_intrinsics.txt"
        if not query_list_path.exists():
            print(f"  [SKIP] No query list for {slice_name}")
            return
        query_entries = load_query_list(str(query_list_path), str(images_dir))
        if camera != "both":
            query_entries = [e for e in query_entries if e["camera"] == camera]

        print(f"  {slice_name} (benchmark): DB={len(train_entries)}, Query={len(query_entries)}")
        out = Path(output_dir) / slice_name
        write_visloc_format(train_entries, query_entries, str(out), has_gt_test=False)


def main():
    args = parse_args()
    cmu_root = Path(args.cmu_root)

    if args.slices:
        slices = [f"slice{s}" if not s.startswith("slice") else s for s in args.slices]
    else:
        nvm_dir = cmu_root / "3D-models" / "nvm_models"
        slices = sorted([p.stem for p in nvm_dir.glob("*.nvm")])

    print(f"CMU Seasons root: {cmu_root}")
    print(f"Mode: {args.mode}, Camera: {args.camera}")
    print(f"Slices: {slices}\n")

    for sl in slices:
        print(f"Processing {sl}...")
        convert_slice(str(cmu_root), args.output_dir, sl, args.mode,
                     args.db_subsample, args.camera)

    print(f"\nDone. Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
