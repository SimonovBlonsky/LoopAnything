from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Add paths for external dependencies
PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parents[2]  # ~/code/NeurIPS26
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
    "root": str(REPO_ROOT / "data" / "7scenes"),
    "scenes": ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"],
    "intrinsics": np.array([[525.0, 0.0, 320.0], [0.0, 525.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32),
}
CAMBRIDGE = {
    "root": str(REPO_ROOT / "data" / "cambridge"),
    "scenes": ["GreatCourt", "KingsCollege", "OldHospital", "ShopFacade", "StMarysChurch"],
    "intrinsics": np.array([[1671.31, 0.0, 960.0], [0.0, 1671.31, 540.0], [0.0, 0.0, 1.0]], dtype=np.float32),
}
DATASET_CONFIGS = {"7scenes": SEVEN_SCENES, "cambridge": CAMBRIDGE}


def load_scene_images_and_poses(dataset_name, scene, split):
    """Load all images and GT poses for a scene split.

    Args:
        dataset_name: "7scenes" or "cambridge"
        scene: scene name
        split: "train" or "test"

    Returns:
        list of dicts with keys: image_path, pose (4x4 np.array)
    """
    config = DATASET_CONFIGS[dataset_name]
    root = config["root"]

    if dataset_name == "7scenes":
        return _load_7scenes_split(root, scene, split)
    elif dataset_name == "cambridge":
        return _load_cambridge_split(root, scene, split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _load_7scenes_split(root, scene, split):
    """Load 7Scenes image paths and poses for a split."""
    scene_dir = Path(root) / scene
    seq_dirs = sorted([d for d in scene_dir.iterdir() if d.is_dir() and d.name.startswith("seq-")])
    entries = []
    for seq_dir in seq_dirs:
        frames = sorted(seq_dir.glob("*.color.png"))
        for frame_path in frames:
            pose_path = str(frame_path).replace(".color.png", ".pose.txt")
            if Path(pose_path).exists():
                pose = np.loadtxt(pose_path).astype(np.float32)
                entries.append({"image_path": str(frame_path), "pose": pose})
    return entries


def _load_cambridge_split(root, scene, split):
    """Load Cambridge image paths and poses for a split."""
    scene_dir = Path(root) / scene
    split_file = scene_dir / f"dataset_{split}.txt"
    entries = []
    if split_file.exists():
        with open(split_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 7:
                    img_path = scene_dir / parts[0]
                    # Cambridge format: img_path qw qx qy qz tx ty tz
                    qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
                    pose = _quat_trans_to_pose(qw, qx, qy, qz, tx, ty, tz)
                    entries.append({"image_path": str(img_path), "pose": pose})
    return entries


def _quat_trans_to_pose(qw, qx, qy, qz, tx, ty, tz):
    """Convert quaternion + translation to 4x4 pose matrix."""
    from scipy.spatial.transform import Rotation
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R
    pose[:3, 3] = [tx, ty, tz]
    return pose


def preprocess_image(image_path, target_size=(504, 504)):
    """Load and preprocess image for DA3 backbone."""
    import cv2
    img = imread_cv2(image_path)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)  # [3, H, W]
    return img


@torch.no_grad()
def build_database(pipeline, db_entries, device, batch_size=16, target_size=(504, 504)):
    """Build database of descriptors and final features."""
    all_descriptors = []
    all_patch_tokens = []
    all_camera_tokens = []

    for i in tqdm(range(0, len(db_entries), batch_size), desc="Building database"):
        batch_entries = db_entries[i:i+batch_size]
        images = torch.stack([preprocess_image(e["image_path"], target_size) for e in batch_entries])
        images = images.unsqueeze(1).to(device)  # [B, 1, 3, H, W]

        descriptors, patch_tokens, camera_tokens = pipeline.extract_database_features(images)
        all_descriptors.append(descriptors.cpu())
        all_patch_tokens.append(patch_tokens.cpu())
        all_camera_tokens.append(camera_tokens.cpu())

    return (
        torch.cat(all_descriptors, dim=0),
        torch.cat(all_patch_tokens, dim=0),
        torch.cat(all_camera_tokens, dim=0),
    )


@torch.no_grad()
def evaluate_scene(pipeline, db_entries, query_entries, db_descriptors, db_patch_tokens,
                   db_camera_tokens, device, top_k=10, target_size=(504, 504)):
    """Evaluate unified pipeline on a single scene."""
    rotation_errors = []
    translation_errors = []

    for q_entry in tqdm(query_entries, desc="Evaluating queries"):
        query_img = preprocess_image(q_entry["image_path"], target_size)
        query_img = query_img.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 3, H, W]

        # First get top-K indices via retrieval
        query_desc = pipeline.retrieval_only(query_img)  # [1, D]
        sims = torch.nn.functional.cosine_similarity(query_desc[0].unsqueeze(0), db_descriptors.to(device), dim=1)
        k = min(top_k, sims.shape[0])
        topk_indices = sims.topk(k).indices.cpu()

        # Get candidate features
        cand_patch = db_patch_tokens[topk_indices].unsqueeze(0).to(device)  # [1, K, P, C]
        cand_cam = db_camera_tokens[topk_indices].unsqueeze(0).to(device)  # [1, K, C]
        cand_desc = db_descriptors[topk_indices].to(device)  # [K, D]

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
    args = parser.parse_args()

    config = load_config(args.model_config)
    pipeline = build_unified_pipeline(config, device=args.device)

    if args.checkpoint and Path(args.checkpoint).is_file():
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        state_dict = ckpt.get("state_dict", ckpt)
        pipeline.load_state_dict(state_dict, strict=False)

    pipeline.eval()

    target_size = tuple(args.image_size)
    db_entries = load_scene_images_and_poses(args.dataset, args.scene, "train")
    query_entries = load_scene_images_and_poses(args.dataset, args.scene, "test")

    print(f"Database: {len(db_entries)} images, Queries: {len(query_entries)} images")

    db_descriptors, db_patch_tokens, db_camera_tokens = build_database(
        pipeline, db_entries, args.device, batch_size=args.batch_size, target_size=target_size
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
