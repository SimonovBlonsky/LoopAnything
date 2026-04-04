from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


SUPPORTED_BACKENDS = ("dino_salad", "da3_salad")
SUPPORTED_POSE_PATHS = ("cam_dec", "ray", "both")
SUPPORTED_ANCHOR_MODES = ("multi_ref_alignment", "top1_anchor")


def load_config(config_path: str) -> dict[str, Any]:
    from depth_anything_3.model.unified_pipeline_helper import load_config as _load_config

    return _load_config(config_path)


def build_unified_pipeline(config: dict[str, Any], device: str = "cpu"):
    from depth_anything_3.model.unified_pipeline_helper import build_unified_pipeline as _builder

    return _builder(config, device=device)


def load_scene_images_and_poses(
    dataset_name: str,
    scene: str,
    split: str,
    data_root: str | None = None,
):
    from eval_unified_visloc import load_scene_images_and_poses as _loader

    return _loader(dataset_name, scene, split, data_root=data_root)


def preprocess_image(image_path: str, target_size: tuple[int, int] = (504, 504)) -> torch.Tensor:
    from eval_unified_visloc import preprocess_image as _preprocess

    return _preprocess(image_path, target_size=target_size)


def get_rot_err(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    from reloc3r.utils.metric import get_rot_err as _get_rot_err

    return float(_get_rot_err(rot_a, rot_b))


def pose_encoding_to_extri_intri(pose_encoding: torch.Tensor, image_size_hw: tuple[int, int]):
    from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri as _decoder

    return _decoder(pose_encoding, image_size_hw)


def validate_retrieval_backend(backend: str) -> str:
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported retrieval backend: {backend}. "
            f"Expected one of {SUPPORTED_BACKENDS}."
        )
    return backend


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training-free visual localization baseline")
    parser.add_argument("--unified-config", type=str, default="configs/unified_pipeline.yaml")
    parser.add_argument("--unified-checkpoint", type=str, default=None)
    parser.add_argument("--salad-checkpoint", type=str, default=None)
    parser.add_argument("--dataset", type=str, required=True, choices=["7scenes", "cambridge"])
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument(
        "--retriever-backend",
        "--backend",
        dest="retriever_backend",
        type=str,
        default="dino_salad",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-m", type=int, default=3)
    parser.add_argument("--pose-path", type=str, default="cam_dec", choices=list(SUPPORTED_POSE_PATHS))
    parser.add_argument(
        "--anchor-mode",
        type=str,
        default="multi_ref_alignment",
        choices=list(SUPPORTED_ANCHOR_MODES),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, nargs=2, default=[504, 504])
    parser.add_argument("--output-dir", type=str, default="workspace/ablation_results")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    args.retriever_backend = validate_retrieval_backend(args.retriever_backend)
    args.backend = args.retriever_backend  # backward compatibility for tests/scripts
    return args


def select_topk_topm(
    sims: torch.Tensor,
    top_k: int,
    top_m: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sims.ndim != 1:
        raise ValueError("sims must be a 1D tensor")
    if top_m > top_k:
        raise ValueError("top_m must be <= top_k")
    if top_k <= 0 or top_m <= 0:
        raise ValueError("top_k and top_m must be positive")

    k = min(top_k, int(sims.numel()))
    topk_indices = sims.topk(k).indices
    m = min(top_m, k)
    topm_indices = topk_indices[:m]
    return topk_indices, topm_indices


def _get_field(output: Any, field: str) -> Any:
    if isinstance(output, dict):
        return output.get(field)
    return getattr(output, field, None)


def _resolve_cam_dec(output: Any) -> dict[str, Any]:
    extrinsics = _get_field(output, "extrinsics")
    intrinsics = _get_field(output, "intrinsics")
    if extrinsics is None or intrinsics is None:
        raise ValueError("cam_dec output requires `extrinsics` and `intrinsics`.")
    return {
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "pose_enc": _get_field(output, "pose_enc"),
    }


def _resolve_ray(output: Any) -> dict[str, Any]:
    extrinsics = _get_field(output, "ray_extrinsics")
    intrinsics = _get_field(output, "ray_intrinsics")
    if extrinsics is None or intrinsics is None:
        extrinsics = _get_field(output, "extrinsics")
        intrinsics = _get_field(output, "intrinsics")
    if extrinsics is None or intrinsics is None:
        raise ValueError("ray output requires ray-specific or generic extrinsics/intrinsics fields.")
    return {
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
    }


def resolve_pose_output(output: Any, pose_path: str) -> dict[str, dict[str, Any]]:
    if pose_path not in SUPPORTED_POSE_PATHS:
        raise ValueError(f"Unsupported pose_path: {pose_path}")
    if pose_path == "cam_dec":
        return {"cam_dec": _resolve_cam_dec(output)}
    if pose_path == "ray":
        return {"ray": _resolve_ray(output)}
    return {"cam_dec": _resolve_cam_dec(output), "ray": _resolve_ray(output)}


def validate_runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    validate_retrieval_backend(args.retriever_backend)
    if args.top_k <= 0 or args.top_m <= 0:
        raise ValueError("top_k and top_m must be positive")
    if args.top_m > args.top_k:
        raise ValueError("top_m must be <= top_k")
    if args.anchor_mode not in SUPPORTED_ANCHOR_MODES:
        raise ValueError(f"Unsupported anchor-mode: {args.anchor_mode}")
    if args.retriever_backend == "dino_salad" and not args.salad_checkpoint:
        raise ValueError("dino_salad backend requires --salad-checkpoint")
    return args


def _load_checkpoint_if_exists(pipeline: Any, checkpoint_path: str | None, device: str):
    if checkpoint_path and Path(checkpoint_path).is_file():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        pipeline.load_state_dict(state_dict, strict=False)


def build_pose_pipeline(unified_config: str, unified_checkpoint: str | None, device: str):
    config = load_config(unified_config)
    pipeline = build_unified_pipeline(config, device=device)
    _load_checkpoint_if_exists(pipeline, unified_checkpoint, device)
    pipeline.eval()
    return pipeline, config


def build_da3_salad_retriever(unified_config: str, unified_checkpoint: str | None, device: str):
    pipeline, _config = build_pose_pipeline(unified_config, unified_checkpoint, device)

    @torch.no_grad()
    def retriever(images: torch.Tensor) -> torch.Tensor:
        return pipeline.retrieval_only(images.to(device))

    return pipeline, retriever


def _ensure_salad_path() -> None:
    import sys

    project_root = Path(__file__).resolve().parents[1]
    salad_root = project_root / "da3_streaming" / "loop_utils" / "salad"
    salad_path = str(salad_root)
    if salad_path not in sys.path:
        sys.path.insert(0, salad_path)


def load_dino_salad_retriever(salad_checkpoint: str, device: str):
    _ensure_salad_path()
    from models.helper import get_model

    checkpoint_path = Path(salad_checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SALAD checkpoint not found: {salad_checkpoint}")

    model = get_model(
        "dinov2_vitb14",
        num_channels=768,
        num_clusters=64,
        cluster_dim=128,
        token_dim=256,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    @torch.no_grad()
    def retriever(images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 5:
            images = images[:, 0]
        return model(images.to(device))

    return retriever


@torch.no_grad()
def extract_descriptors(
    entries: list[dict[str, Any]],
    retriever,
    device: str,
    batch_size: int = 16,
    target_size: tuple[int, int] = (504, 504),
) -> torch.Tensor:
    descriptors = []
    for i in tqdm(range(0, len(entries), batch_size), desc="Extracting descriptors"):
        batch_entries = entries[i : i + batch_size]
        images = torch.stack([preprocess_image(e["image_path"], target_size) for e in batch_entries])
        images = images.unsqueeze(1).to(device)
        desc = retriever(images)
        descriptors.append(desc.detach().cpu())
    return torch.cat(descriptors, dim=0)


def _as_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_4x4_batch(extrinsics: np.ndarray | torch.Tensor) -> np.ndarray:
    ext = _as_numpy(extrinsics).astype(np.float64)
    if ext.ndim == 3:
        ext = ext[None]
    if ext.ndim != 4:
        raise ValueError(f"Expected [B, S, 3/4, 4], got {ext.shape}")
    if ext.shape[-2:] == (4, 4):
        return ext
    if ext.shape[-2:] == (3, 4):
        out = np.zeros((*ext.shape[:-2], 4, 4), dtype=np.float64)
        out[..., :3, :4] = ext
        out[..., 3, 3] = 1.0
        return out
    raise ValueError(f"Unsupported extrinsics shape: {ext.shape}")


def _extract_group_c2w(
    resolved_output: dict[str, dict[str, Any]],
    branch: str,
    image_size_hw: tuple[int, int],
) -> np.ndarray:
    branch_out = resolved_output[branch]
    pose_enc = branch_out.get("pose_enc")
    if branch == "cam_dec" and pose_enc is not None:
        c2w, _ = pose_encoding_to_extri_intri(pose_enc, image_size_hw)
        return _to_4x4_batch(c2w)

    ext = _to_4x4_batch(branch_out["extrinsics"])
    if branch == "cam_dec":
        # cam_dec path stores w2c in output.extrinsics; invert to get c2w.
        return np.linalg.inv(ext)
    return ext


def _estimate_query_pose_from_group(
    pred_group_c2w: np.ndarray,
    ref_gt_c2w: np.ndarray,
    anchor_mode: str,
) -> tuple[np.ndarray, str]:
    mode = anchor_mode
    if mode == "multi_ref_alignment" and ref_gt_c2w.shape[0] < 2:
        mode = "top1_anchor"
    if mode == "multi_ref_alignment":
        try:
            return align_query_pose_multi_ref(pred_group_c2w, ref_gt_c2w), mode
        except ValueError:
            # Degenerate geometry can happen for some branches (e.g., ray in tiny scenes).
            mode = "top1_anchor"
    if mode == "top1_anchor":
        return align_query_pose_top1_anchor(pred_group_c2w, ref_gt_c2w), mode
    raise ValueError(f"Unsupported anchor mode: {anchor_mode}")


@torch.no_grad()
def evaluate_scene_training_free(
    pose_pipeline: Any,
    retriever,
    db_entries: list[dict[str, Any]],
    query_entries: list[dict[str, Any]],
    db_descriptors: torch.Tensor | np.ndarray,
    device: str,
    top_k: int,
    top_m: int,
    pose_path: str,
    anchor_mode: str,
    target_size: tuple[int, int],
    config: dict[str, Any],
    retriever_backend: str,
) -> dict[str, Any]:
    db_desc_device = db_descriptors
    if isinstance(db_desc_device, np.ndarray):
        db_desc_device = torch.from_numpy(db_desc_device)
    db_desc_device = db_desc_device.to(device)

    rotation_errors: list[float] = []
    translation_errors: list[float] = []
    topk_all: list[np.ndarray] = []
    topm_all: list[np.ndarray] = []
    effective_modes: list[str] = []
    query_poses_by_branch: dict[str, list[np.ndarray]] = {"cam_dec": [], "ray": []}

    for q_entry in tqdm(query_entries, desc="Evaluating queries"):
        query_img = preprocess_image(q_entry["image_path"], target_size).to(device)
        query_input = query_img.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, H, W]
        query_desc = retriever(query_input)  # [1, D]

        sims = F.cosine_similarity(query_desc[0].unsqueeze(0), db_desc_device, dim=1)
        topk_indices_t, topm_indices_t = select_topk_topm(sims, top_k=top_k, top_m=top_m)
        topk_indices = topk_indices_t.detach().cpu().numpy().astype(np.int64)
        topm_indices = topm_indices_t.detach().cpu().numpy().astype(np.int64)
        topk_all.append(topk_indices)
        topm_all.append(topm_indices)

        candidate_images = [
            preprocess_image(db_entries[idx]["image_path"], target_size) for idx in topm_indices.tolist()
        ]
        candidate_input = torch.stack(candidate_images, dim=0).unsqueeze(0).to(device)

        output = pose_pipeline.pose_only(query_input, candidate_input, pose_path=pose_path)
        resolved = resolve_pose_output(output, pose_path)

        ref_gt = np.stack([db_entries[idx]["pose"] for idx in topm_indices.tolist()], axis=0).astype(np.float64)

        branch_modes: dict[str, str] = {}
        for branch in resolved.keys():
            pred_group = _extract_group_c2w(resolved, branch, target_size)[0]
            query_pose, mode_used = _estimate_query_pose_from_group(pred_group, ref_gt, anchor_mode)
            query_poses_by_branch[branch].append(query_pose)
            branch_modes[branch] = mode_used

        primary_branch = "cam_dec" if "cam_dec" in resolved else "ray"
        pred_pose = query_poses_by_branch[primary_branch][-1]
        gt_pose = np.asarray(q_entry["pose"], dtype=np.float64)
        rotation_errors.append(get_rot_err(pred_pose[:3, :3], gt_pose[:3, :3]))
        translation_errors.append(float(np.linalg.norm(pred_pose[:3, 3] - gt_pose[:3, 3])))
        effective_modes.append(branch_modes[primary_branch])

    payload: dict[str, Any] = {
        "rotation_errors": np.asarray(rotation_errors, dtype=np.float32),
        "translation_errors": np.asarray(translation_errors, dtype=np.float32),
        "topk_indices": np.asarray(topk_all, dtype=object),
        "topm_indices": np.asarray(topm_all, dtype=object),
        "effective_anchor_modes": np.asarray(effective_modes, dtype=object),
        "retriever_backend": retriever_backend,
        "pose_path": pose_path,
        "primary_pose_branch": "cam_dec" if pose_path in ("cam_dec", "both") else "ray",
        "config": config,
    }
    if query_poses_by_branch["cam_dec"]:
        payload["query_poses_cam_dec"] = np.stack(query_poses_by_branch["cam_dec"], axis=0)
    if query_poses_by_branch["ray"]:
        payload["query_poses_ray"] = np.stack(query_poses_by_branch["ray"], axis=0)
    return payload


def save_result_payload(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = dict(payload)
    save_dict["config"] = np.array(payload["config"], dtype=object)
    np.savez(output_path, **save_dict)


def _as_pose_array(poses: np.ndarray | list[np.ndarray], name: str) -> np.ndarray:
    arr = np.asarray(poses, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1:] != (4, 4):
        raise ValueError(f"{name} must have shape [N, 4, 4], got {arr.shape}")
    return arr


def _estimate_sim3(src_pts: np.ndarray, dst_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Estimate Sim(3) from src to dst: dst = s * R * src + t."""
    if src_pts.shape != dst_pts.shape:
        raise ValueError("src_pts and dst_pts must share shape [N, 3]")
    if src_pts.ndim != 2 or src_pts.shape[1] != 3:
        raise ValueError("src_pts and dst_pts must have shape [N, 3]")
    if src_pts.shape[0] < 2:
        raise ValueError("At least 2 points are required for Sim(3) estimation.")

    n = src_pts.shape[0]
    src_mean = src_pts.mean(axis=0)
    dst_mean = dst_pts.mean(axis=0)
    src_centered = src_pts - src_mean
    dst_centered = dst_pts - dst_mean

    cov = (dst_centered.T @ src_centered) / n
    u, svals, vt = np.linalg.svd(cov)
    d = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        d[2, 2] = -1.0
    rot = u @ d @ vt

    src_var = np.sum(src_centered**2) / n
    if src_var <= 1e-12:
        raise ValueError("Degenerate source geometry for Sim(3) estimation.")
    scale = float(np.sum(svals * np.diag(d)) / src_var)
    trans = dst_mean - scale * (rot @ src_mean)
    return rot, trans, scale


def _apply_sim3_to_pose(pose: np.ndarray, rot: np.ndarray, trans: np.ndarray, scale: float) -> np.ndarray:
    aligned = np.eye(4, dtype=np.float64)
    aligned[:3, :3] = rot @ pose[:3, :3]
    aligned[:3, 3] = scale * (rot @ pose[:3, 3]) + trans
    return aligned


def align_query_pose_multi_ref(pred_group: np.ndarray, ref_gt: np.ndarray) -> np.ndarray:
    pred_group = _as_pose_array(pred_group, "pred_group")
    ref_gt = _as_pose_array(ref_gt, "ref_gt")
    if pred_group.shape[0] < 2:
        raise ValueError("pred_group must contain query plus at least one reference pose.")
    if pred_group.shape[0] - 1 != ref_gt.shape[0]:
        raise ValueError("ref_gt count must match number of reference poses in pred_group.")

    pred_refs_centers = pred_group[1:, :3, 3]
    gt_refs_centers = ref_gt[:, :3, 3]
    rot, trans, scale = _estimate_sim3(pred_refs_centers, gt_refs_centers)
    return _apply_sim3_to_pose(pred_group[0], rot, trans, scale)


def align_query_pose_top1_anchor(pred_group: np.ndarray, ref_gt: np.ndarray) -> np.ndarray:
    pred_group = _as_pose_array(pred_group, "pred_group")
    ref_gt = _as_pose_array(ref_gt, "ref_gt")
    if pred_group.shape[0] < 2:
        raise ValueError("pred_group must contain query plus at least one reference pose.")
    if ref_gt.shape[0] < 1:
        raise ValueError("ref_gt must contain at least one reference pose.")

    ref_transform = ref_gt[0] @ np.linalg.inv(pred_group[1])
    return ref_transform @ pred_group[0]


def main() -> None:
    args = parse_args()
    validate_runtime_args(args)
    target_size = (args.image_size[0], args.image_size[1])

    if args.retriever_backend == "da3_salad":
        pose_pipeline, retriever = build_da3_salad_retriever(
            args.unified_config, args.unified_checkpoint, args.device,
        )
        config = load_config(args.unified_config)
    else:
        pose_pipeline, config = build_pose_pipeline(args.unified_config, args.unified_checkpoint, args.device)
        retriever = load_dino_salad_retriever(args.salad_checkpoint, args.device)

    db_entries = load_scene_images_and_poses(args.dataset, args.scene, "train", data_root=args.data_root)
    query_entries = load_scene_images_and_poses(args.dataset, args.scene, "test", data_root=args.data_root)

    db_descriptors = extract_descriptors(
        db_entries,
        retriever,
        device=args.device,
        batch_size=args.batch_size,
        target_size=target_size,
    )

    result = evaluate_scene_training_free(
        pose_pipeline=pose_pipeline,
        retriever=retriever,
        db_entries=db_entries,
        query_entries=query_entries,
        db_descriptors=db_descriptors,
        device=args.device,
        top_k=args.top_k,
        top_m=args.top_m,
        pose_path=args.pose_path,
        anchor_mode=args.anchor_mode,
        target_size=target_size,
        config=config,
        retriever_backend=args.retriever_backend,
    )

    med_t = float(np.median(result["translation_errors"]))
    med_r = float(np.median(result["rotation_errors"]))
    print(
        f"[Training-Free][{args.retriever_backend}] Scene {args.scene} "
        f"median pose error: {med_t:.2f} m  {med_r:.2f} deg"
    )

    output_path = Path(args.output_dir) / (
        f"training_free_{args.retriever_backend}_{args.dataset}_{args.scene}.npz"
    )
    save_result_payload(result, output_path)
    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
