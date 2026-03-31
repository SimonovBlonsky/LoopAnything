from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
REPO_ROOT = PROJECT_ROOT.parents[2]
SALAD_ROOT = PROJECT_ROOT / "da3_streaming" / "loop_utils" / "salad"
for path in (str(SRC_ROOT), str(REPO_ROOT), str(SALAD_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

RELOC3R_ROOT = REPO_ROOT / "reloc3r"
if str(RELOC3R_ROOT) not in sys.path:
    sys.path.insert(0, str(RELOC3R_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from depth_anything_3.model.unified_pipeline_helper import build_unified_pipeline, load_config
from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri
from eval_unified_visloc import load_scene_images_and_poses, preprocess_image
from reloc3r.utils.metric import get_rot_err


def load_dino_salad(salad_checkpoint, device):
    """Load standalone DINO SALAD model."""
    from vpr_model import VPRModel
    import torchvision.transforms as T
    from torch import nn

    # Load from the SALAD codebase
    from models.helper import get_model
    model = get_model("dinov2_vitb14", num_channels=768, num_clusters=64, cluster_dim=128, token_dim=256)
    checkpoint = torch.load(salad_checkpoint, map_location="cpu")
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint, strict=False)
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_salad_descriptors(salad_model, entries, device, batch_size=16, target_size=(504, 504)):
    """Extract DINO SALAD descriptors for all images."""
    all_descriptors = []
    for i in tqdm(range(0, len(entries), batch_size), desc="Extracting DINO SALAD descriptors"):
        batch_entries = entries[i:i+batch_size]
        images = torch.stack([preprocess_image(e["image_path"], target_size) for e in batch_entries])
        images = images.to(device)
        descriptors = salad_model(images)
        all_descriptors.append(descriptors.cpu())
    return torch.cat(all_descriptors, dim=0)


@torch.no_grad()
def evaluate_ablation(
    pipeline, db_entries, query_entries,
    salad_db_descriptors, salad_query_descriptors,
    device, top_k=10, target_size=(504, 504),
):
    rotation_errors = []
    translation_errors = []

    for q_idx, q_entry in enumerate(tqdm(query_entries, desc="Evaluating (DINO SALAD + our pose)")):
        # DINO SALAD retrieval
        q_desc = salad_query_descriptors[q_idx]
        sims = F.cosine_similarity(q_desc.unsqueeze(0), salad_db_descriptors, dim=1)
        k = min(top_k, sims.shape[0])
        topk_indices = sims.topk(k).indices.cpu().tolist()

        # Our pose estimation
        query_img = preprocess_image(q_entry["image_path"], target_size)
        query_input = query_img.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 3, H, W]

        cand_images = []
        for db_idx in topk_indices:
            cand_img = preprocess_image(db_entries[db_idx]["image_path"], target_size)
            cand_images.append(cand_img)
        cand_images = torch.stack(cand_images).unsqueeze(0).to(device)  # [1, K, 3, H, W]

        output = pipeline.pose_only(query_input, cand_images)

        # Decode pose_enc to c2w matrix
        pose_enc = output.pose_enc  # [B, 1, 9]
        c2w, _ixt = pose_encoding_to_extri_intri(pose_enc, target_size)
        pred_pose = c2w[0, 0].cpu().numpy()  # [4, 4] camera-to-world

        if pred_pose.shape[0] == 3:
            full_pose = np.eye(4, dtype=np.float32)
            full_pose[:3, :] = pred_pose
            pred_pose = full_pose

        gt_pose = q_entry["pose"]
        rerr = get_rot_err(pred_pose[:3, :3], gt_pose[:3, :3])
        terr = np.linalg.norm(pred_pose[:3, 3] - gt_pose[:3, 3])
        rotation_errors.append(rerr)
        translation_errors.append(terr)

    return rotation_errors, translation_errors


def main():
    parser = argparse.ArgumentParser(description="Ablation: DINO SALAD retrieval + our pose estimation")
    parser.add_argument("--unified-config", type=str, required=True)
    parser.add_argument("--unified-checkpoint", type=str, default=None)
    parser.add_argument("--salad-checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["7scenes", "cambridge"])
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default="workspace/ablation_results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, nargs=2, default=[504, 504])
    parser.add_argument("--data-root", type=str, default=None, help="Override default dataset root path")
    args = parser.parse_args()

    config = load_config(args.unified_config)
    pipeline = build_unified_pipeline(config, device=args.device)
    if args.unified_checkpoint and Path(args.unified_checkpoint).is_file():
        ckpt = torch.load(args.unified_checkpoint, map_location=args.device)
        pipeline.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    pipeline.eval()

    salad_model = load_dino_salad(args.salad_checkpoint, args.device)

    target_size = tuple(args.image_size)
    db_entries = load_scene_images_and_poses(args.dataset, args.scene, "train", data_root=args.data_root)
    query_entries = load_scene_images_and_poses(args.dataset, args.scene, "test", data_root=args.data_root)

    salad_db_descs = extract_salad_descriptors(salad_model, db_entries, args.device, args.batch_size, target_size)
    salad_query_descs = extract_salad_descriptors(salad_model, query_entries, args.device, args.batch_size, target_size)

    rerrs, terrs = evaluate_ablation(
        pipeline, db_entries, query_entries,
        salad_db_descs, salad_query_descs,
        args.device, args.top_k, target_size,
    )

    print(f"[Ablation: DINO SALAD + our pose] Scene {args.scene} median pose error: {np.median(terrs):.2f} m  {np.median(rerrs):.2f} deg")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / f"ablation2_{args.dataset}_{args.scene}.npz",
             rotation_errors=np.array(rerrs), translation_errors=np.array(terrs))


if __name__ == "__main__":
    main()
