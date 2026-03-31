from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
REPO_ROOT = PROJECT_ROOT.parents[2]
for path in (str(SRC_ROOT), str(REPO_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

RELOC3R_ROOT = REPO_ROOT / "reloc3r"
if str(RELOC3R_ROOT) not in sys.path:
    sys.path.insert(0, str(RELOC3R_ROOT))

import numpy as np
import torch
from tqdm import tqdm

from depth_anything_3.model.unified_pipeline_helper import build_unified_pipeline, load_config
from eval_unified_visloc import (
    build_database,
    load_scene_images_and_poses,
    preprocess_image,
)
from reloc3r.reloc3r_relpose import Reloc3rRelpose
from reloc3r.reloc3r_visloc import Reloc3rVisloc
from reloc3r.utils.metric import get_rot_err


def setup_reloc3r(model_str, device):
    """Load reloc3r model."""
    model = eval(model_str)
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate_ablation(
    pipeline, reloc3r_model, db_entries, query_entries,
    db_descriptors, device, top_k=10, target_size=(504, 504),
):
    reloc3r_visloc = Reloc3rVisloc()
    rotation_errors = []
    translation_errors = []

    for q_entry in tqdm(query_entries, desc="Evaluating (our VPR + reloc3r pose)"):
        query_img = preprocess_image(q_entry["image_path"], target_size)
        query_input = query_img.unsqueeze(0).unsqueeze(0).to(device)

        # Our retrieval
        query_desc = pipeline.retrieval_only(query_input)
        sims = torch.nn.functional.cosine_similarity(
            query_desc[0].unsqueeze(0), db_descriptors.to(device), dim=1
        )
        k = min(top_k, sims.shape[0])
        topk_indices = sims.topk(k).indices.cpu().tolist()

        # reloc3r pose regression for each pair
        pose_db_list = []
        pose_q2d_list = []
        for db_idx in topk_indices:
            db_entry = db_entries[db_idx]
            # Load images for reloc3r (it uses its own preprocessing)
            batch = {
                "view1": {"img": preprocess_image(db_entry["image_path"], target_size).unsqueeze(0).to(device)},
                "view2": {"img": query_img.unsqueeze(0).to(device)},
            }
            pred = reloc3r_model(batch)
            pred_pose = pred["pose"][0].cpu().numpy()  # [4, 4] relative pose

            pose_db_list.append(db_entry["pose"])
            pose_q2d_list.append(pred_pose)

        # Motion averaging
        Rt = reloc3r_visloc.motion_averaging(pose_db_list, pose_q2d_list)

        gt_pose = q_entry["pose"]
        rerr = get_rot_err(Rt[:3, :3], gt_pose[:3, :3])
        terr = np.linalg.norm(Rt[:3, 3] - gt_pose[:3, 3])
        rotation_errors.append(rerr)
        translation_errors.append(terr)

    return rotation_errors, translation_errors


def main():
    parser = argparse.ArgumentParser(description="Ablation: our VPR + reloc3r pose regression")
    parser.add_argument("--unified-config", type=str, required=True)
    parser.add_argument("--unified-checkpoint", type=str, default=None)
    parser.add_argument("--reloc3r-model", type=str, default="Reloc3rRelpose(img_size=512)")
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

    reloc3r_model = setup_reloc3r(args.reloc3r_model, args.device)

    target_size = tuple(args.image_size)
    db_entries = load_scene_images_and_poses(args.dataset, args.scene, "train", data_root=args.data_root)
    query_entries = load_scene_images_and_poses(args.dataset, args.scene, "test", data_root=args.data_root)

    db_descriptors, _, _ = build_database(pipeline, db_entries, args.device, args.batch_size, target_size)

    rerrs, terrs = evaluate_ablation(
        pipeline, reloc3r_model, db_entries, query_entries,
        db_descriptors, args.device, args.top_k, target_size,
    )

    print(f"[Ablation: our VPR + reloc3r] Scene {args.scene} median pose error: {np.median(terrs):.2f} m  {np.median(rerrs):.2f} deg")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / f"ablation1_{args.dataset}_{args.scene}.npz",
             rotation_errors=np.array(rerrs), translation_errors=np.array(terrs))


if __name__ == "__main__":
    main()
