#!/usr/bin/env python3
"""Pairwise DA3 reconstruction and pose evaluation for failure-case debugging.

Given two images (candidate + query) and their GT c2w poses, runs DA3's native
API to:
1. Predict relative pose (query expressed in candidate frame)
2. Compare with GT relative pose
3. Export point cloud + camera frustums as PLY for visual inspection

Usage:
    python scripts/da3_pairwise_debug.py \
        --candidate-image workspace/DA3/.../candidate_top0_000410.color.png \
        --query-image     workspace/DA3/.../query_001474.color.png \
        --candidate-pose  data/euroc_visloc/V1_03_difficult/frames/000410.pose.txt \
        --query-pose      data/euroc_visloc/V1_03_difficult/frames/001474.pose.txt \
        --output-dir      workspace/DA3_pairwise_debug/V1_03_q0080_top0

Shortcut: give a failure_case directory and the script auto-picks the files.
    python scripts/da3_pairwise_debug.py \
        --failure-case-dir workspace/DA3/.../rank00_query0080 \
        --scene-frames-dir data/euroc_visloc/V1_03_difficult/frames \
        --candidate-rank 0 \
        --output-dir workspace/DA3_pairwise_debug/rank00
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
# Walk up to find NeurIPS26 repo root (contains `reloc3r`).
REPO_ROOT = PROJECT_ROOT
for _p in [PROJECT_ROOT] + list(PROJECT_ROOT.parents):
    if (_p / "reloc3r").is_dir():
        REPO_ROOT = _p
        break
for _p in (str(SRC_ROOT), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from depth_anything_3.api import DepthAnything3  # noqa: E402
from depth_anything_3.utils.geometry import as_homogeneous, unproject_depth  # noqa: E402


# ---------------------------------------------------------------------------
# Pose helpers
# ---------------------------------------------------------------------------
def relative_pose_in_ref(c2w_ref: np.ndarray, c2w_target: np.ndarray) -> np.ndarray:
    """Return target's pose expressed in ref's coordinate frame.

    T = inv(c2w_ref) @ c2w_target. This maps points in target camera frame
    to points in ref camera frame.
    """
    return np.linalg.inv(c2w_ref) @ c2w_target


def pose_errors(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Compute rotation, translation direction, and translation magnitude errors."""
    R_pred, t_pred = pred[:3, :3], pred[:3, 3]
    R_gt, t_gt = gt[:3, :3], gt[:3, 3]

    # Rotation (geodesic angle).
    R_rel = R_pred.T @ R_gt
    rot_err = float(np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1.0, 1.0))))

    # Translation direction (angle between pred and gt translation vectors).
    tp_n = t_pred / (np.linalg.norm(t_pred) + 1e-12)
    tg_n = t_gt / (np.linalg.norm(t_gt) + 1e-12)
    trans_ang = float(np.degrees(np.arccos(np.clip(np.dot(tp_n, tg_n), -1.0, 1.0))))

    # Translation magnitude.
    mag_pred = float(np.linalg.norm(t_pred))
    mag_gt = float(np.linalg.norm(t_gt))

    # Translation scale ratio.
    scale_ratio = mag_pred / (mag_gt + 1e-12)

    # Translation vector L2.
    trans_l2 = float(np.linalg.norm(t_pred - t_gt))

    return {
        "rotation_deg": rot_err,
        "translation_direction_deg": trans_ang,
        "pred_translation_norm_m": mag_pred,
        "gt_translation_norm_m": mag_gt,
        "scale_ratio": scale_ratio,
        "translation_l2_m": trans_l2,
    }


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------
def load_image_rgb_uint8(path: str) -> np.ndarray:
    from PIL import Image
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# PLY export
# ---------------------------------------------------------------------------
def _camera_frustum_lines(c2w: np.ndarray, scale: float = 0.1):
    """Return (points, edges) representing a camera frustum in world coords."""
    # Points in camera frame: origin + 4 far-plane corners.
    pts_cam = np.array([
        [0, 0, 0],
        [-scale, -scale, 2 * scale],  # top-left
        [+scale, -scale, 2 * scale],  # top-right
        [+scale, +scale, 2 * scale],  # bottom-right
        [-scale, +scale, 2 * scale],  # bottom-left
    ])
    # Apply c2w: p_world = R @ p_cam + t
    R, t = c2w[:3, :3], c2w[:3, 3]
    pts_world = pts_cam @ R.T + t
    edges = [
        (0, 1), (0, 2), (0, 3), (0, 4),  # from origin to corners
        (1, 2), (2, 3), (3, 4), (4, 1),  # far-plane rectangle
    ]
    return pts_world, edges


def save_ply_scene(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray | None,
    cameras: list[tuple[str, np.ndarray, tuple[int, int, int]]],
) -> None:
    """Save point cloud + camera frustums as a PLY file.

    Args:
        points: [N, 3] world-space points.
        colors: [N, 3] uint8 colors or None.
        cameras: list of (name, c2w, rgb_color) for frustum rendering.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Build vertices (points + camera frustum vertices).
    all_vertices = [points]
    all_colors = [colors if colors is not None else np.full_like(points, 200, dtype=np.uint8)]

    edge_list = []  # list of (v_start, v_end, rgb)
    offset = len(points)
    for name, c2w, rgb in cameras:
        pts, edges = _camera_frustum_lines(c2w, scale=0.15)
        all_vertices.append(pts)
        all_colors.append(np.tile(np.asarray(rgb, dtype=np.uint8)[None], (len(pts), 1)))
        for a, b in edges:
            edge_list.append((offset + a, offset + b, rgb))
        offset += len(pts)

    vertices = np.vstack(all_vertices)
    vcolors = np.vstack(all_colors).astype(np.uint8)

    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element edge {len(edge_list)}\n")
        f.write("property int vertex1\nproperty int vertex2\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for v, c in zip(vertices, vcolors):
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]} {c[1]} {c[2]}\n")
        for a, b, rgb in edge_list:
            f.write(f"{a} {b} {rgb[0]} {rgb[1]} {rgb[2]}\n")


def save_depth_png(depth: np.ndarray, path: Path) -> None:
    """Save a colorized depth map as PNG."""
    import matplotlib.cm as cm
    from PIL import Image
    d = np.asarray(depth, dtype=np.float32)
    d = np.squeeze(d)
    valid = d > 0
    if valid.any():
        lo, hi = np.percentile(d[valid], [5, 95])
    else:
        lo, hi = 0, 1
    d_norm = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    cmap = cm.get_cmap("turbo")
    rgba = (cmap(d_norm) * 255).astype(np.uint8)
    Image.fromarray(rgba[..., :3]).save(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-image", type=str, default=None)
    parser.add_argument("--query-image", type=str, default=None)
    parser.add_argument("--candidate-pose", type=str, default=None,
                        help="4x4 c2w txt file (row-major).")
    parser.add_argument("--query-pose", type=str, default=None)
    # Shortcut: pass a failure-case dir + scene-frames dir
    parser.add_argument("--failure-case-dir", type=str, default=None,
                        help="Path to rankXX_queryYYYY/ with info.json")
    parser.add_argument("--scene-frames-dir", type=str, default=None,
                        help="Path to visloc frames/ dir containing fid.pose.txt files")
    parser.add_argument("--candidate-rank", type=int, default=0,
                        help="Which retrieved candidate to use (0-indexed)")
    parser.add_argument("--model-path", type=str, default="depth-anything/DA3-LARGE-1.1")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--process-res-method", type=str, default="upper_bound_resize")
    parser.add_argument("--ref-view-strategy", type=str, default="first",
                        help="'first' forces candidate (view 0) as reference.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-points", type=int, default=50000,
                        help="Subsample point cloud to this many points for smaller PLY.")
    return parser.parse_args()


def resolve_inputs(args):
    """Resolve (cand_img, query_img, cand_pose, query_pose) from either direct
    args or --failure-case-dir shortcut."""
    if args.candidate_image and args.query_image and args.candidate_pose and args.query_pose:
        return args.candidate_image, args.query_image, args.candidate_pose, args.query_pose

    if not args.failure_case_dir:
        raise ValueError("Must provide either all four --*-image/--*-pose args "
                         "or --failure-case-dir + --scene-frames-dir.")

    fc_dir = Path(args.failure_case_dir)
    info = json.load(open(fc_dir / "info.json"))

    # Find query image in fc_dir.
    query_name = info["query_image"]
    q_img = fc_dir / f"query_{query_name}"
    if not q_img.exists():
        # Fall back: any file starting with "query_"
        matches = sorted(fc_dir.glob("query_*"))
        q_img = matches[0] if matches else None
    if q_img is None or not q_img.exists():
        raise FileNotFoundError(f"Query image not found in {fc_dir}")

    # Find candidate image at requested rank.
    cand_info = info["retrieved_candidates"][args.candidate_rank]
    cand_name = cand_info["db_image"]
    c_img = fc_dir / f"candidate_top{args.candidate_rank}_{cand_name}"
    if not c_img.exists():
        matches = sorted(fc_dir.glob(f"candidate_top{args.candidate_rank}_*"))
        c_img = matches[0] if matches else None
    if c_img is None or not c_img.exists():
        raise FileNotFoundError(f"Candidate image rank {args.candidate_rank} not found in {fc_dir}")

    # Resolve pose files via scene-frames-dir.
    if not args.scene_frames_dir:
        raise ValueError("--scene-frames-dir is required when using --failure-case-dir")
    frames_dir = Path(args.scene_frames_dir)

    # Query pose: the visloc frame id comes from info["query_index"] in the original
    # eval loop. Failure-case info.json doesn't store the visloc fid directly,
    # so we use the TestSplit.txt line at query_index.
    scene_dir = frames_dir.parent
    q_fid = None
    with open(scene_dir / "TestSplit.txt") as f:
        lines = [l.strip() for l in f if l.strip()]
        q_fid = lines[info["query_index"]]
    q_pose = frames_dir / f"{q_fid}.pose.txt"

    # Candidate pose: index into TrainSplit.txt.
    c_fid = None
    with open(scene_dir / "TrainSplit.txt") as f:
        lines = [l.strip() for l in f if l.strip()]
        c_fid = lines[cand_info["db_index"]]
    c_pose = frames_dir / f"{c_fid}.pose.txt"

    return str(c_img), str(q_img), str(c_pose), str(q_pose)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cand_img_path, query_img_path, cand_pose_path, query_pose_path = resolve_inputs(args)

    print(f"Candidate image: {cand_img_path}")
    print(f"Query     image: {query_img_path}")
    print(f"Candidate pose : {cand_pose_path}")
    print(f"Query     pose : {query_pose_path}")

    # Load GT poses.
    c2w_cand_gt = np.loadtxt(cand_pose_path).astype(np.float64)
    c2w_query_gt = np.loadtxt(query_pose_path).astype(np.float64)
    # Accept 3x4 or 4x4.
    def to_4x4(p):
        if p.shape == (4, 4):
            return p
        m = np.eye(4)
        m[:3, :4] = p
        return m
    c2w_cand_gt = to_4x4(c2w_cand_gt)
    c2w_query_gt = to_4x4(c2w_query_gt)
    gt_rel = relative_pose_in_ref(c2w_cand_gt, c2w_query_gt)

    # Load images.
    cand_img = load_image_rgb_uint8(cand_img_path)
    query_img = load_image_rgb_uint8(query_img_path)

    # Load DA3 model.
    device = torch.device(args.device)
    print(f"Loading DA3 model: {args.model_path}")
    model = DepthAnything3.from_pretrained(args.model_path).to(device)
    model.eval()

    # Preprocess: candidate is view 0 (reference), query is view 1.
    print("Preprocessing images via DA3 InputProcessor...")
    imgs_cpu, _, _ = model.input_processor(
        [cand_img, query_img],
        process_res=args.process_res,
        process_res_method=args.process_res_method,
        num_workers=1,
        print_progress=False,
        sequential=True,
        desc=None,
    )
    imgs = imgs_cpu.view(1, 2, *imgs_cpu.shape[1:]).to(device).float()

    # Forward.
    print("Running DA3 forward pass...")
    with torch.inference_mode():
        out = model.forward(
            imgs,
            extrinsics=None, intrinsics=None,
            export_feat_layers=[],
            infer_gs=False,
            use_ray_pose=False,
            ref_view_strategy=args.ref_view_strategy,
        )

    pred_ext = as_homogeneous(out["extrinsics"])   # [1, 2, 4, 4] (w2c)
    pred_int = out["intrinsics"]                   # [1, 2, 3, 3]
    pred_depth = out.get("depth", None)            # [1, 2, H, W, 1] or None

    # Relative pose: query expressed in candidate frame.
    # pred_ext[0,0] = w2c_cand, pred_ext[0,1] = w2c_query (same DA3 internal world frame)
    # T_query_in_cand = w2c_cand @ c2w_query = pred_ext[0,0] @ inv(pred_ext[0,1])
    pred_rel_t = pred_ext[0, 0] @ torch.linalg.inv(pred_ext[0, 1])
    pred_rel = pred_rel_t.cpu().numpy().astype(np.float64)

    # Errors.
    errs = pose_errors(pred_rel, gt_rel)

    # ---------------- Print & save summary ----------------
    lines = []
    lines.append("=" * 60)
    lines.append("DA3 Pairwise Reconstruction & Pose Evaluation")
    lines.append("=" * 60)
    lines.append(f"Candidate: {cand_img_path}")
    lines.append(f"Query    : {query_img_path}")
    lines.append("")
    lines.append("GT relative pose (query in candidate frame):")
    lines.append(np.array_str(gt_rel, precision=6, suppress_small=True))
    lines.append(f"  |t_gt|     = {np.linalg.norm(gt_rel[:3, 3]):.4f} m")
    lines.append("")
    lines.append("Predicted relative pose (query in candidate frame):")
    lines.append(np.array_str(pred_rel, precision=6, suppress_small=True))
    lines.append(f"  |t_pred|   = {np.linalg.norm(pred_rel[:3, 3]):.4f} m")
    lines.append("")
    lines.append("Errors:")
    lines.append(f"  Rotation:               {errs['rotation_deg']:.2f} deg")
    lines.append(f"  Translation direction:  {errs['translation_direction_deg']:.2f} deg")
    lines.append(f"  Translation magnitude:  pred={errs['pred_translation_norm_m']:.3f}m  gt={errs['gt_translation_norm_m']:.3f}m")
    lines.append(f"  Scale ratio (pred/gt):  {errs['scale_ratio']:.3f}")
    lines.append(f"  Translation L2:         {errs['translation_l2_m']:.3f} m")
    lines.append("=" * 60)
    summary = "\n".join(lines)
    print(summary)
    (out_dir / "summary.txt").write_text(summary + "\n")

    # Save poses.
    np.savetxt(out_dir / "pred_rel_pose_query_in_cand.txt", pred_rel, fmt="%.8f")
    np.savetxt(out_dir / "gt_rel_pose_query_in_cand.txt", gt_rel, fmt="%.8f")

    np.savez(
        out_dir / "result.npz",
        pred_rel=pred_rel,
        gt_rel=gt_rel,
        pred_extrinsics_w2c=pred_ext[0].cpu().numpy(),
        pred_intrinsics=pred_int[0].cpu().numpy(),
        c2w_cand_gt=c2w_cand_gt,
        c2w_query_gt=c2w_query_gt,
        **errs,
    )

    # ---------------- Visualization ----------------
    if pred_depth is None:
        print("[WARN] DA3 output has no 'depth' field — skipping point cloud export.")
    else:
        print("Unprojecting depth to point cloud...")
        # Normalize depth shape to [B, V, H, W, 1] expected by unproject_depth.
        # DA3 may return [B, V, H, W], [B, V, 1, H, W], or [B, V, H, W, 1].
        d = pred_depth
        if d.ndim == 4:                          # [B, V, H, W]
            d = d.unsqueeze(-1)
        elif d.ndim == 5 and d.shape[2] == 1:    # [B, V, 1, H, W] → [B, V, H, W, 1]
            d = d.permute(0, 1, 3, 4, 2).contiguous()
        # At this point d should be [B, V, H, W, 1].
        assert d.ndim == 5 and d.shape[-1] == 1, f"Unexpected depth shape: {pred_depth.shape}"

        c2w_pred = torch.linalg.inv(pred_ext)  # [1, 2, 4, 4]
        points_world = unproject_depth(d, pred_int, c2w=c2w_pred)  # [1, 2, H, W, 3]
        points_world = points_world[0].cpu().numpy()  # [2, H, W, 3]
        depth_np = d[0, ..., 0].cpu().numpy()  # [2, H, W]
        imgs_np = imgs[0].cpu().numpy().transpose(0, 2, 3, 1)  # [2, H, W, 3]

        # Undo ImageNet norm for visualization colors.
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        imgs_vis = np.clip(imgs_np * std + mean, 0, 1)
        colors_uint8 = (imgs_vis * 255).astype(np.uint8)

        # Flatten and concatenate both views.
        pts = points_world.reshape(-1, 3)
        cols = colors_uint8.reshape(-1, 3)
        valid = (depth_np > 0).reshape(-1)
        pts = pts[valid]
        cols = cols[valid]

        # Subsample.
        if len(pts) > args.max_points:
            idx = np.random.RandomState(0).choice(len(pts), args.max_points, replace=False)
            pts = pts[idx]
            cols = cols[idx]

        # Camera frustums in DA3 internal world frame.
        c2w_pred_np = c2w_pred[0].cpu().numpy()  # [2, 4, 4]
        # Also show the GT query frustum in candidate's frame (red),
        # by placing candidate at c2w_pred_cand and putting query at c2w_cand_pred @ gt_rel.
        c2w_query_gt_in_pred_world = c2w_pred_np[0] @ gt_rel  # candidate's predicted c2w @ GT rel

        cameras = [
            ("pred_cand", c2w_pred_np[0], (0, 255, 0)),     # green
            ("pred_query", c2w_pred_np[1], (0, 128, 255)),  # blue
            ("gt_query", c2w_query_gt_in_pred_world, (255, 0, 0)),  # red
        ]

        ply_path = out_dir / "pointcloud.ply"
        save_ply_scene(ply_path, pts, cols, cameras)
        print(f"Saved point cloud: {ply_path}")
        print("  Camera frustum colors:")
        print("    GREEN = predicted candidate (reference)")
        print("    BLUE  = predicted query")
        print("    RED   = GT query (placed relative to predicted candidate)")

        # Save depth visualizations.
        save_depth_png(depth_np[0], out_dir / "depth_candidate.png")
        save_depth_png(depth_np[1], out_dir / "depth_query.png")
        np.save(out_dir / "depth_candidate.npy", depth_np[0])
        np.save(out_dir / "depth_query.npy", depth_np[1])

    print(f"\nAll outputs saved to: {out_dir}/")


if __name__ == "__main__":
    main()
