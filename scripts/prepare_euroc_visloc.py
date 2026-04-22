#!/usr/bin/env python3
"""Convert EuRoC MAV sequences into visual localization dataset.

EuRoC format:
- cam0/data/{timestamp}.png (grayscale 752x480)
- state_groundtruth_estimate0/data.csv (timestamp, tx,ty,tz, qw,qx,qy,qz, ...)
- cam0/sensor.yaml (T_BS body-to-sensor, intrinsics)

NOTE on evaluation protocol:
EuRoC is poorly suited to the single-sequence train/query split used by
7Scenes/Cambridge. MAV trajectories cover a 6D (position+orientation) space
densely in pose but sparsely in any single traversal, so queries routinely
lack same-orientation DB neighbors. Use cross-sequence mode:
    V1: db=V1_01+V1_02, query=V1_03
    V2: db=V2_01+V2_02, query=V2_03
    MH: db=MH_01+MH_02+MH_03, query=MH_04+MH_05

Modes:
- self_eval (legacy): split one sequence (first half=db, second half=query).
- cross_sequence (recommended): combine multiple sequences per scene (V1/V2/MH).

Usage:
    # Cross-sequence (default scene splits)
    python scripts/prepare_euroc_visloc.py \
        --euroc-root /data/datasets/EuRoC_mav \
        --output-dir data/euroc_visloc \
        --mode cross_sequence

    # Self-eval (legacy, single-sequence split)
    python scripts/prepare_euroc_visloc.py \
        --euroc-root /data/datasets/EuRoC_mav \
        --output-dir data/euroc_visloc_selfeval \
        --mode self_eval
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


DEFAULT_SCENE_SPLITS = {
    # scene_name: (db_sequences, query_sequences)
    "V1": (["V1_01_easy", "V1_02_medium"], ["V1_03_difficult"]),
    "V2": (["V2_01_easy", "V2_02_medium"], ["V2_03_difficult"]),
    "MH": (["MH_01_easy", "MH_02_easy", "MH_03_medium"],
           ["MH_04_difficult", "MH_05_difficult"]),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--euroc-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="data/euroc_visloc")
    parser.add_argument("--mode", type=str, default="cross_sequence",
                        choices=["self_eval", "cross_sequence"],
                        help="self_eval: split one sequence. cross_sequence: combine multiple sequences per scene.")
    parser.add_argument("--sequences", type=str, nargs="+", default=None,
                        help="(self_eval only) Sequences to convert. Default: all with GT.")
    # Cross-sequence split overrides.
    parser.add_argument("--scenes", type=str, nargs="+", default=None,
                        help="(cross_sequence only) Scenes to build. Default: V1 V2 MH.")
    parser.add_argument("--v1-db", type=str, nargs="+",
                        default=DEFAULT_SCENE_SPLITS["V1"][0])
    parser.add_argument("--v1-query", type=str, nargs="+",
                        default=DEFAULT_SCENE_SPLITS["V1"][1])
    parser.add_argument("--v2-db", type=str, nargs="+",
                        default=DEFAULT_SCENE_SPLITS["V2"][0])
    parser.add_argument("--v2-query", type=str, nargs="+",
                        default=DEFAULT_SCENE_SPLITS["V2"][1])
    parser.add_argument("--mh-db", type=str, nargs="+",
                        default=DEFAULT_SCENE_SPLITS["MH"][0])
    parser.add_argument("--mh-query", type=str, nargs="+",
                        default=DEFAULT_SCENE_SPLITS["MH"][1])
    parser.add_argument("--camera", type=str, default="cam0")
    parser.add_argument("--revisit-radius", type=float, default=2.0,
                        help="Revisit radius in meters (self_eval mode)")
    parser.add_argument("--db-subsample", type=int, default=5)
    parser.add_argument("--query-subsample", type=int, default=5)
    return parser.parse_args()


def load_euroc_sensor_yaml(yaml_path: str):
    """Parse EuRoC sensor.yaml for T_BS and intrinsics."""
    with open(yaml_path) as f:
        text = f.read()

    # Parse T_BS
    match = re.search(r"T_BS:.*?data:\s*\[(.*?)\]", text, re.DOTALL)
    T_BS = np.array([float(x) for x in match.group(1).split(",")]).reshape(4, 4)

    # Parse intrinsics [fu, fv, cu, cv]
    match = re.search(r"intrinsics:\s*\[(.*?)\]", text)
    fu, fv, cu, cv = [float(x) for x in match.group(1).split(",")]
    K = np.array([[fu, 0, cu], [0, fv, cv], [0, 0, 1]], dtype=np.float32)

    # Parse resolution
    match = re.search(r"resolution:\s*\[(\d+),\s*(\d+)\]", text)
    W, H = int(match.group(1)), int(match.group(2))

    return T_BS, K, (W, H)


def load_euroc_gt(gt_csv_path: str):
    """Load EuRoC groundtruth: returns dict timestamp_ns -> c2w 4x4."""
    poses = {}
    with open(gt_csv_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0].startswith("#"):
                continue
            ts = int(row[0])
            tx, ty, tz = float(row[1]), float(row[2]), float(row[3])
            qw, qx, qy, qz = float(row[4]), float(row[5]), float(row[6]), float(row[7])
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = [tx, ty, tz]
            poses[ts] = T  # body-to-world
    return poses


def load_euroc_image_timestamps(data_csv_path: str):
    """Load cam timestamps from data.csv."""
    timestamps = []
    with open(data_csv_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0].startswith("#"):
                continue
            timestamps.append(int(row[0]))
    return timestamps


def interpolate_pose(gt_poses: dict, query_ts: int):
    """Find nearest GT pose for a camera timestamp."""
    ts_list = sorted(gt_poses.keys())
    idx = np.searchsorted(ts_list, query_ts)
    if idx == 0:
        return gt_poses[ts_list[0]]
    if idx >= len(ts_list):
        return gt_poses[ts_list[-1]]
    # Nearest neighbor
    if abs(ts_list[idx] - query_ts) < abs(ts_list[idx - 1] - query_ts):
        return gt_poses[ts_list[idx]]
    return gt_poses[ts_list[idx - 1]]


def load_sequence_frames(euroc_root, seq_name, camera):
    """Load all (image_path, c2w pose) frames for a single EuRoC sequence.

    Returns (frames, K) where frames is a list of dicts with keys
    ts/image_path/pose, and K is the camera intrinsic matrix.
    """
    seq_dir = Path(euroc_root) / seq_name / "mav0"
    cam_dir = seq_dir / camera
    gt_path = seq_dir / "state_groundtruth_estimate0" / "data.csv"

    if not gt_path.exists():
        return [], None

    T_BS, K, _ = load_euroc_sensor_yaml(str(cam_dir / "sensor.yaml"))
    gt_body = load_euroc_gt(str(gt_path))
    cam_timestamps = load_euroc_image_timestamps(str(cam_dir / "data.csv"))
    T_BS_inv = np.linalg.inv(T_BS)

    frames = []
    for ts in cam_timestamps:
        T_WB = interpolate_pose(gt_body, ts)
        c2w = T_WB @ T_BS_inv
        img_path = str(cam_dir / "data" / f"{ts}.png")
        if os.path.exists(img_path):
            frames.append({
                "ts": ts,
                "image_path": img_path,
                "pose": c2w,
                "source_seq": seq_name,
            })
    return frames, K


def convert_cross_sequence(euroc_root, output_dir, scene_name,
                           db_seqs, query_seqs, camera,
                           db_subsample, query_subsample):
    """Build a scene by combining frames from multiple EuRoC sequences.

    - db_seqs: sequences contributing database frames
    - query_seqs: sequences contributing query frames
    """
    # Load all frames from db sequences.
    db_frames_all = []
    intrinsics = None
    for seq in db_seqs:
        frames, K = load_sequence_frames(euroc_root, seq, camera)
        if not frames:
            print(f"  [SKIP] No frames for db sequence {seq}")
            continue
        db_frames_all.extend(frames)
        if intrinsics is None:
            intrinsics = K
    db_frames = db_frames_all[::db_subsample]

    # Load all frames from query sequences.
    query_frames_all = []
    for seq in query_seqs:
        frames, K = load_sequence_frames(euroc_root, seq, camera)
        if not frames:
            print(f"  [SKIP] No frames for query sequence {seq}")
            continue
        query_frames_all.extend(frames)
    query_frames = query_frames_all[::query_subsample]

    if not db_frames or not query_frames:
        print(f"  [SKIP] Empty db or query for scene {scene_name}")
        return

    print(f"  Scene {scene_name}: DB={len(db_frames)} (from {db_seqs}), "
          f"Query={len(query_frames)} (from {query_seqs})")

    # Write output with combined frame ids.
    out_dir = Path(output_dir) / scene_name
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(out_dir / "intrinsics.txt", intrinsics, fmt="%.6f")

    db_ids = []
    for i, f in enumerate(db_frames):
        fid = f"db_{i:06d}"
        db_ids.append(fid)
        src = Path(f["image_path"])
        dst = frames_dir / f"{fid}.color.png"
        if not dst.exists():
            os.symlink(src.resolve(), dst)
        np.savetxt(frames_dir / f"{fid}.pose.txt", f["pose"], fmt="%.12e")

    q_ids = []
    for i, f in enumerate(query_frames):
        fid = f"q_{i:06d}"
        q_ids.append(fid)
        src = Path(f["image_path"])
        dst = frames_dir / f"{fid}.color.png"
        if not dst.exists():
            os.symlink(src.resolve(), dst)
        np.savetxt(frames_dir / f"{fid}.pose.txt", f["pose"], fmt="%.12e")

    with open(out_dir / "TrainSplit.txt", "w") as fp:
        for fid in db_ids:
            fp.write(f"{fid}\n")
    with open(out_dir / "TestSplit.txt", "w") as fp:
        for fid in q_ids:
            fp.write(f"{fid}\n")

    with open(out_dir / "info.txt", "w") as fp:
        fp.write(f"scene: {scene_name}\ncamera: {camera}\n")
        fp.write(f"db_sequences: {db_seqs}\n")
        fp.write(f"query_sequences: {query_seqs}\n")
        fp.write(f"db_frames: {len(db_frames)}\n")
        fp.write(f"query_frames: {len(query_frames)}\n")
        fp.write(f"db_subsample: {db_subsample}\n")
        fp.write(f"query_subsample: {query_subsample}\n")


def convert_sequence(euroc_root, output_dir, seq_name, camera, revisit_radius,
                     db_subsample, query_subsample):
    seq_dir = Path(euroc_root) / seq_name / "mav0"
    cam_dir = seq_dir / camera
    gt_path = seq_dir / "state_groundtruth_estimate0" / "data.csv"

    if not gt_path.exists():
        print(f"  [SKIP] No GT for {seq_name}")
        return

    T_BS, K, (W, H) = load_euroc_sensor_yaml(str(cam_dir / "sensor.yaml"))
    gt_body = load_euroc_gt(str(gt_path))
    cam_timestamps = load_euroc_image_timestamps(str(cam_dir / "data.csv"))

    # Compute cam-to-world poses: c2w = T_WB @ T_BS^{-1}... wait
    # T_BS is body-to-sensor (cam). GT is body-to-world (T_WB).
    # c2w = T_WB @ inv(T_BS) would give sensor-to-world... no.
    # T_BS maps body frame to sensor frame: p_sensor = T_BS @ p_body
    # T_WB maps body frame to world: p_world = T_WB @ p_body
    # Camera-to-world: p_world = T_WB @ inv(T_BS) @ p_sensor
    # So c2w = T_WB @ inv(T_BS)
    T_BS_inv = np.linalg.inv(T_BS)

    frames = []
    for ts in cam_timestamps:
        T_WB = interpolate_pose(gt_body, ts)
        c2w = T_WB @ T_BS_inv
        img_path = str(cam_dir / "data" / f"{ts}.png")
        if os.path.exists(img_path):
            frames.append({"ts": ts, "image_path": img_path, "pose": c2w})

    if len(frames) < 20:
        print(f"  [SKIP] Too few frames ({len(frames)}) for {seq_name}")
        return

    positions = np.array([f["pose"][:3, 3] for f in frames])
    n = len(frames)
    half = n // 2

    # DB = first half, subsampled
    db_indices = list(range(0, half, db_subsample))
    db_positions = positions[db_indices]

    # Query = second half frames that revisit DB
    query_indices = []
    for i in range(half, n, query_subsample):
        dists = np.linalg.norm(db_positions - positions[i], axis=1)
        if dists.min() < revisit_radius:
            query_indices.append(i)

    if len(query_indices) == 0:
        print(f"  [SKIP] No revisiting frames in {seq_name}")
        return

    print(f"  Database: {len(db_indices)}, Query: {len(query_indices)}")

    # Write output
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
        fp.write(f"sequence: {seq_name}\ncamera: {camera}\n")
        fp.write(f"total_frames: {n}\ndb_frames: {len(db_indices)}\nquery_frames: {len(query_indices)}\n")
        fp.write(f"revisit_radius: {revisit_radius}\n")


def main():
    args = parse_args()
    euroc_root = Path(args.euroc_root)

    print(f"EuRoC root: {euroc_root}")
    print(f"Mode: {args.mode}\n")

    if args.mode == "self_eval":
        if args.sequences:
            sequences = args.sequences
        else:
            sequences = sorted([d.name for d in euroc_root.iterdir()
                               if d.is_dir() and (d / "mav0").is_dir()])

        print(f"Sequences: {sequences}\n")
        for seq in sequences:
            print(f"Processing {seq}...")
            convert_sequence(str(euroc_root), args.output_dir, seq, args.camera,
                            args.revisit_radius, args.db_subsample, args.query_subsample)

    elif args.mode == "cross_sequence":
        # Build scene → (db, query) from CLI-overridable defaults.
        scene_splits = {
            "V1": (args.v1_db, args.v1_query),
            "V2": (args.v2_db, args.v2_query),
            "MH": (args.mh_db, args.mh_query),
        }
        selected_scenes = args.scenes or list(scene_splits.keys())
        print(f"Scenes: {selected_scenes}\n")
        for scene in selected_scenes:
            if scene not in scene_splits:
                print(f"  [SKIP] Unknown scene {scene} (expected V1/V2/MH)")
                continue
            db_seqs, q_seqs = scene_splits[scene]
            print(f"Processing scene {scene} (db={db_seqs}, query={q_seqs})...")
            convert_cross_sequence(
                str(euroc_root), args.output_dir, scene,
                db_seqs, q_seqs, args.camera,
                args.db_subsample, args.query_subsample,
            )

    print(f"\nDone. Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
