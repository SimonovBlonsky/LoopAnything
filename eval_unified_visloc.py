from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Add paths for external dependencies
PROJECT_ROOT = Path(__file__).resolve().parent
# Resolve NeurIPS26 repo root: walk up until we find the reloc3r/ sibling.
_candidate = PROJECT_ROOT
for _p in [PROJECT_ROOT] + list(PROJECT_ROOT.parents):
    if (_p / "reloc3r").is_dir():
        _candidate = _p
        break
REPO_ROOT = _candidate
SRC_ROOT = PROJECT_ROOT / "src"
for path in (SRC_ROOT, str(REPO_ROOT)):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from depth_anything_3.model.unified_pipeline_helper import build_unified_pipeline, load_config
from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri
from depth_anything_3.utils.geometry import affine_inverse
from utils.image import imread_cv2


# Import metrics from reloc3r
RELOC3R_ROOT = REPO_ROOT / "reloc3r"
if str(RELOC3R_ROOT) not in sys.path:
    sys.path.insert(0, str(RELOC3R_ROOT))
from reloc3r.utils.metric import get_rot_err


# Dataset configs
SEVEN_SCENES = {
    "root": str(REPO_ROOT / "reloc3r" / "data" / "7scenes"),
    "scenes": ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"],
    "intrinsics": np.array([[525.0, 0.0, 320.0], [0.0, 525.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32),
}
CAMBRIDGE = {
    "root": str(REPO_ROOT / "reloc3r" / "data" / "cambridge"),
    "scenes": ["GreatCourt", "KingsCollege", "OldHospital", "ShopFacade", "StMarysChurch"],
    "intrinsics": np.array([[1671.31, 0.0, 960.0], [0.0, 1671.31, 540.0], [0.0, 0.0, 1.0]], dtype=np.float32),
}
DATASET_CONFIGS = {"7scenes": SEVEN_SCENES, "cambridge": CAMBRIDGE}


def load_scene_images_and_poses(dataset_name, scene, split, data_root=None):
    """Load all images and GT poses for a scene split.

    Args:
        dataset_name: "7scenes" or "cambridge"
        scene: scene name
        split: "train" or "test"
        data_root: override default dataset root path

    Returns:
        list of dicts with keys: image_path, pose (4x4 np.array)
    """
    config = DATASET_CONFIGS[dataset_name]
    root = data_root if data_root else config["root"]

    if dataset_name == "7scenes":
        return _load_7scenes_split(root, scene, split)
    elif dataset_name == "cambridge":
        return _load_cambridge_split(root, scene, split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _load_7scenes_split(root, scene, split):
    """Load 7Scenes image paths and poses for a split.

    Uses TrainSplit.txt / TestSplit.txt to filter sequences.
    """
    scene_dir = Path(root) / scene

    # Read split file to get allowed sequences
    split_map = {"train": "TrainSplit.txt", "test": "TestSplit.txt"}
    split_file = scene_dir / split_map[split]
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    allowed_seqs = set()
    with open(split_file) as f:
        for line in f:
            line = line.strip()
            if line:
                # "sequence1" -> "seq-01", "sequence12" -> "seq-12"
                seq_num = int(line.replace("sequence", ""))
                allowed_seqs.add(f"seq-{seq_num:02d}")

    seq_dirs = sorted([d for d in scene_dir.iterdir() if d.is_dir() and d.name.startswith("seq-")])
    entries = []
    for seq_dir in seq_dirs:
        if seq_dir.name not in allowed_seqs:
            continue
        frames = sorted(seq_dir.glob("*.color.png"))
        for frame_path in frames:
            pose_path = str(frame_path).replace(".color.png", ".pose.txt")
            if Path(pose_path).exists():
                pose = np.loadtxt(pose_path).astype(np.float32)
                entries.append({
                    "image_path": str(frame_path),
                    "pose": pose,
                    "intrinsics": SEVEN_SCENES["intrinsics"].copy(),
                })
    return entries


def _load_cambridge_split(root, scene, split):
    """Load Cambridge image paths and poses for a split.

    Uses dataset_{split}.txt for the image list, and reconstruction.nvm
    (VisualSfM) for ground-truth poses. This matches reloc3r's protocol
    (see reloc3r/datasets/cambridge.py ReadModelVisualSfM).
    """
    scene_dir = Path(root) / scene

    # 1. Read SfM poses from reconstruction.nvm (same as reloc3r).
    params_dict = _read_cambridge_nvm(str(scene_dir))

    # 2. Read split image list from dataset_{split}.txt.
    split_file = scene_dir / f"dataset_{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    entries = []
    with open(split_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("Visual") or line.startswith("Image"):
                continue
            parts = line.split()
            if not parts:
                continue
            rel_path = parts[0]
            img_path = str(scene_dir / rel_path)
            if img_path not in params_dict:
                continue
            pose_c2w = params_dict[img_path]["pose_c2w"]
            intrinsics = params_dict[img_path]["intrinsics"]
            entries.append({
                "image_path": img_path,
                "pose": pose_c2w.astype(np.float32),
                "intrinsics": intrinsics,
            })
    return entries


def _rotation_from_quaternion(quad):
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix.

    Ported from reloc3r/datasets/cambridge.py.
    """
    norm = np.linalg.norm(quad)
    if norm < 1e-10:
        raise ValueError(f"Degenerate quaternion with norm {norm}")
    quad = quad / norm
    qr, qi, qj, qk = quad[0], quad[1], quad[2], quad[3]
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


def _read_cambridge_nvm(scene_dir, nvm_file="reconstruction.nvm"):
    """Read camera params from VisualSfM NVM file.

    Returns dict mapping absolute image path -> {intrinsics, pose_c2w}.
    Ported from reloc3r/datasets/cambridge.py ReadModelVisualSfM.
    """
    import os
    nvm_path = os.path.join(scene_dir, nvm_file)
    if not os.path.exists(nvm_path):
        raise FileNotFoundError(f"NVM file not found: {nvm_path}")
    with open(nvm_path) as f:
        lines = f.readlines()

    counter = 2
    n_images = int(lines[counter].strip())
    counter += 1

    params_dict = {}
    for _ in range(n_images):
        parts = lines[counter].strip().split()
        counter += 1

        imname = os.path.join(scene_dir, parts[0]).replace(".jpg", ".png")
        focal = float(parts[1])
        qvec = np.array([float(parts[k]) for k in range(2, 6)])
        center = np.array([float(parts[k]) for k in range(6, 9)])

        # Per-image intrinsics from SfM (same as reloc3r cambridge.py).
        import imagesize
        width, height = imagesize.get(imname)
        cx, cy = width / 2.0, height / 2.0
        intrinsics = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

        # NVM convention: qvec is w2c rotation, center is camera center in world.
        R = _rotation_from_quaternion(qvec)
        T = -R @ center
        Rt = np.eye(4)
        Rt[:3, :3] = R
        Rt[:3, 3] = T
        pose_c2w = np.linalg.inv(Rt)

        params_dict[imname] = {"pose_c2w": pose_c2w, "intrinsics": intrinsics}

    return params_dict


def preprocess_image(image_path, target_size=(504, 504)):
    """Load and preprocess image for DA3 backbone."""
    import cv2
    img = imread_cv2(image_path)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)  # [3, H, W]
    return img


@torch.no_grad()
def build_database(pipeline, db_entries, device, batch_size=16, target_size=(504, 504),
                   cache_dir=None):
    """Build database features and save all to disk as numpy memmap files.

    Args:
        cache_dir: directory for memmap files. If None, uses a temp directory.

    Returns:
        desc_mmap: np.memmap [N, D] on disk
        patch_mmap: np.memmap [N, P, C] on disk
        cam_mmap: np.memmap [N, C] on disk
    """
    import tempfile

    if cache_dir is None:
        cache_dir = Path(tempfile.mkdtemp(prefix="unified_db_"))
    else:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    N = len(db_entries)

    # First batch to determine shapes
    first_batch = torch.stack([preprocess_image(db_entries[0]["image_path"], target_size)])
    first_batch = first_batch.unsqueeze(1).to(device)
    desc0, patch0, cam0 = pipeline.extract_database_features(first_batch)
    D = desc0.shape[1]
    P, C_patch = patch0.shape[1], patch0.shape[2]
    C_cam = cam0.shape[1]

    # Create memmap files for all three
    desc_mmap = np.memmap(cache_dir / "db_descriptors.mmap", dtype=np.float32, mode="w+", shape=(N, D))
    patch_mmap = np.memmap(cache_dir / "db_patch_tokens.mmap", dtype=np.float32, mode="w+", shape=(N, P, C_patch))
    cam_mmap = np.memmap(cache_dir / "db_camera_tokens.mmap", dtype=np.float32, mode="w+", shape=(N, C_cam))

    print(f"Database cache dir: {cache_dir}")
    print(f"  descriptors:  [{N}, {D}] = {N * D * 4 / 1e6:.0f} MB")
    print(f"  patch_tokens: [{N}, {P}, {C_patch}] = {N * P * C_patch * 4 / 1e9:.1f} GB")
    print(f"  camera_tokens: [{N}, {C_cam}] = {N * C_cam * 4 / 1e6:.0f} MB")

    for i in tqdm(range(0, N, batch_size), desc="Building database"):
        batch_entries = db_entries[i:i+batch_size]
        images = torch.stack([preprocess_image(e["image_path"], target_size) for e in batch_entries])
        images = images.unsqueeze(1).to(device)

        descriptors, patch_tokens, camera_tokens = pipeline.extract_database_features(images)

        bs = descriptors.shape[0]
        desc_mmap[i:i+bs] = descriptors.cpu().numpy()
        patch_mmap[i:i+bs] = patch_tokens.cpu().numpy()
        cam_mmap[i:i+bs] = camera_tokens.cpu().numpy()

    desc_mmap.flush()
    patch_mmap.flush()
    cam_mmap.flush()

    return desc_mmap, patch_mmap, cam_mmap


@torch.no_grad()
def evaluate_scene(pipeline, db_entries, query_entries, db_descriptors, db_patch_tokens,
                   db_camera_tokens, device, top_k=10, target_size=(504, 504)):
    """Evaluate unified pipeline on a single scene.

    All db_* args are np.memmap on disk; only top-K slices are loaded per query.
    """
    rotation_errors = []
    translation_errors = []

    # Load descriptors to device once (small enough: N*D*4 bytes)
    db_desc_device = torch.from_numpy(np.array(db_descriptors)).to(device)

    for q_entry in tqdm(query_entries, desc="Evaluating queries"):
        query_img = preprocess_image(q_entry["image_path"], target_size)
        query_img = query_img.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 3, H, W]

        # Retrieval
        query_desc = pipeline.retrieval_only(query_img)  # [1, D]
        sims = torch.nn.functional.cosine_similarity(query_desc[0].unsqueeze(0), db_desc_device, dim=1)
        k = min(top_k, sims.shape[0])
        topk_indices = sims.topk(k).indices.cpu().numpy()

        # Load only top-K features from memmap (disk → GPU, tiny amount)
        cand_patch = torch.from_numpy(np.array(db_patch_tokens[topk_indices])).unsqueeze(0).to(device)
        cand_cam = torch.from_numpy(np.array(db_camera_tokens[topk_indices])).unsqueeze(0).to(device)
        cand_desc = torch.from_numpy(np.array(db_descriptors[topk_indices])).to(device)

        # Run full pipeline
        output = pipeline(query_img, cand_patch, cand_cam, cand_desc)

        # Decode pose_enc to camera-to-world matrix
        pose_enc = output.pose_enc  # [B, 1, 9]
        c2w, _ixt = pose_encoding_to_extri_intri(pose_enc, target_size)
        pred_pose = c2w[0, 0].cpu().numpy()  # [4, 4] camera-to-world

        if pred_pose.shape[0] == 3:
            full_pose = np.eye(4, dtype=np.float32)
            full_pose[:3, :] = pred_pose
            pred_pose = full_pose

        gt_pose = q_entry["pose"]

        # Compute errors (same as reloc3r)
        rerr = get_rot_err(pred_pose[:3, :3], gt_pose[:3, :3])
        terr = np.linalg.norm(pred_pose[:3, 3] - gt_pose[:3, 3])

        rotation_errors.append(rerr)
        translation_errors.append(terr)

    return rotation_errors, translation_errors


def main():
    parser = argparse.ArgumentParser(description="Evaluate unified visual localization pipeline")
    parser.add_argument("--model-config", type=str, required=True, help="Path to unified_pipeline.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint")
    parser.add_argument("--dataset", type=str, required=True, choices=["7scenes", "cambridge"])
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default="workspace/eval_results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, nargs=2, default=[504, 504])
    parser.add_argument("--data-root", type=str, default=None, help="Override default dataset root path")
    parser.add_argument("--cache-dir", type=str, default=None, help="Directory for database feature cache (memmap)")
    args = parser.parse_args()

    config = load_config(args.model_config)
    pipeline = build_unified_pipeline(config, device=args.device)

    if args.checkpoint and Path(args.checkpoint).is_file():
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        state_dict = ckpt.get("state_dict", ckpt)
        pipeline.load_state_dict(state_dict, strict=False)

    pipeline.eval()

    target_size = tuple(args.image_size)
    db_entries = load_scene_images_and_poses(args.dataset, args.scene, "train", data_root=args.data_root)
    query_entries = load_scene_images_and_poses(args.dataset, args.scene, "test", data_root=args.data_root)

    print(f"Database: {len(db_entries)} images, Queries: {len(query_entries)} images")

    db_descriptors, db_patch_tokens, db_camera_tokens = build_database(
        pipeline, db_entries, args.device, batch_size=args.batch_size, target_size=target_size,
        cache_dir=args.cache_dir,
    )

    rerrs, terrs = evaluate_scene(
        pipeline, db_entries, query_entries,
        db_descriptors, db_patch_tokens, db_camera_tokens,
        args.device, top_k=args.top_k, target_size=target_size,
    )

    med_rerr = np.median(rerrs)
    med_terr = np.median(terrs)
    print(f"Scene {args.scene} median pose error: {med_terr:.2f} m  {med_rerr:.2f} deg")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / f"{args.dataset}_{args.scene}_results.npz",
        rotation_errors=np.array(rerrs),
        translation_errors=np.array(terrs),
    )


if __name__ == "__main__":
    main()
