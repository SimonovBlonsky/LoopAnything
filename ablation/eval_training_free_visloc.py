from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


SUPPORTED_BACKENDS = ("dino_salad", "da3_salad", "netvlad")
SUPPORTED_POSE_PATHS = ("cam_dec", "ray", "both", "relpose_head")
SUPPORTED_ANCHOR_MODES = ("reloc3r_motion_averaging", "multi_ref_alignment", "top1_anchor")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
REPO_ROOT = PROJECT_ROOT.parents[2]
RELOC3R_ROOT = REPO_ROOT / "reloc3r"
SALAD_ROOT = PROJECT_ROOT / "da3_streaming" / "loop_utils" / "salad"
NETVLAD_ROOT = REPO_ROOT / "netvlad_image_retrieval"


def _bootstrap_import_paths(sys_path: list[str] | None = None) -> list[str]:
    target = sys.path if sys_path is None else sys_path
    ordered_paths = [
        str(PROJECT_ROOT),
        str(SRC_ROOT),
        str(REPO_ROOT),
        str(RELOC3R_ROOT),
    ]
    for path in reversed(ordered_paths):
        if path not in target:
            target.insert(0, path)
    return target


_bootstrap_import_paths()


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


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def preprocess_image(image_path: str, target_size: tuple[int, int] = (504, 504)) -> torch.Tensor:
    """Load and resize an image to [0, 1] range.

    This function intentionally does NOT apply model-specific normalization.
    Each consumer (retriever, pose pipeline) is responsible for its own
    normalization requirements.

    Args:
        image_path: path to the image file.
        target_size: (H, W) target dimensions.

    Returns:
        [3, H, W] float32 tensor in [0, 1] range.
    """
    import cv2
    from utils.image import imread_cv2

    img = imread_cv2(image_path)
    # cv2.resize expects (width, height); target_size is (H, W).
    img = cv2.resize(img, (target_size[1], target_size[0]))
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)  # [3, H, W]
    return img


def _apply_imagenet_norm(images: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet normalization to images in [0, 1] range.

    Works with any shape that has a channel dimension at position -3
    (e.g. [3, H, W], [B, 3, H, W], [B, V, 3, H, W]).
    """
    mean = IMAGENET_MEAN.to(images.device)
    std = IMAGENET_STD.to(images.device)
    return (images - mean) / std


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
    parser.add_argument("--relpose-checkpoint", type=str, default=None,
                        help="Path to trained RelPoseHead checkpoint (required for --pose-path relpose_head)")
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
        default="reloc3r_motion_averaging",
        choices=list(SUPPORTED_ANCHOR_MODES),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, nargs=2, default=[504, 504])
    parser.add_argument("--output-dir", type=str, default="workspace/ablation_results")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--cpu-fallback-max-queries",
        type=int,
        default=0,
        help="Cap query count for bounded smoke runtime; especially useful when CUDA falls back to CPU.",
    )
    parser.add_argument(
        "--cpu-fallback-max-db-entries",
        type=int,
        default=0,
        help="Cap database size for bounded smoke runtime; especially useful when CUDA falls back to CPU.",
    )
    parser.add_argument(
        "--scale-diagnostics",
        action="store_true",
        default=False,
        help="Enable per-query scale diagnostics comparing pred vs GT translation norms.",
    )
    args = parser.parse_args(argv)
    args.retriever_backend = validate_retrieval_backend(args.retriever_backend)
    args.backend = args.retriever_backend  # backward compatibility for tests/scripts
    return args


def default_data_root(dataset: str) -> str:
    """Resolve reloc3r dataset roots relative to the NeurIPS26 workspace."""
    dataset_dir = "7scenes" if dataset == "7scenes" else "cambridge"
    return str(REPO_ROOT / "reloc3r" / "data" / dataset_dir)


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
    if args.retriever_backend == "netvlad" and args.salad_checkpoint:
        print("[INFO] --salad-checkpoint is ignored for the netvlad backend.")
    if args.pose_path == "relpose_head" and not args.relpose_checkpoint:
        raise ValueError("relpose_head pose path requires --relpose-checkpoint")
    if args.pose_path == "relpose_head" and args.anchor_mode != "reloc3r_motion_averaging":
        raise ValueError("relpose_head only supports reloc3r_motion_averaging anchor mode")
    return args


def resolve_runtime_device(requested_device: str) -> str:
    """Resolve a usable runtime device, with a safe CUDA fallback."""
    if requested_device != "cuda":
        return requested_device
    if not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    try:
        # Force CUDA initialization early so we can recover before model loading.
        _ = torch.zeros(1, device="cuda")
        return "cuda"
    except Exception as exc:
        print(f"[WARN] CUDA init failed ({exc}). Falling back to CPU.")
        return "cpu"


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
        return pipeline.retrieval_only(_apply_imagenet_norm(images).to(device))

    return pipeline, retriever


def _ensure_salad_path(sys_path: list[str] | None = None) -> list[str]:
    target = _bootstrap_import_paths(sys_path)
    salad_path = str(SALAD_ROOT)
    if salad_path not in target:
        # Keep SALAD ahead of repo root so `models.helper` resolves from the SALAD package.
        target.insert(0, salad_path)
    return target


def _purge_salad_modules(modules: dict[str, Any] | None = None) -> None:
    """Drop top-level SALAD modules that would shadow repo-local packages."""
    registry = sys.modules if modules is None else modules
    for name, module in list(registry.items()):
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        try:
            module_path = Path(module_file).resolve()
        except OSError:
            continue
        if not str(module_path).startswith(str(SALAD_ROOT.resolve())):
            continue
        if name == "utils" or name.startswith("utils.") or name == "models" or name.startswith("models."):
            registry.pop(name, None)


def load_dino_salad_retriever(salad_checkpoint: str, device: str):
    salad_path = str(SALAD_ROOT)
    added_path = salad_path not in sys.path
    if added_path:
        _ensure_salad_path()
    try:
        if device == "cpu":
            # Local DINOv2 falls back to PyTorch attention when xFormers is disabled.
            os.environ["XFORMERS_DISABLED"] = "1"
        from vpr_model import VPRModel
    finally:
        if added_path and salad_path in sys.path:
            sys.path.remove(salad_path)

    checkpoint_path = Path(salad_checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SALAD checkpoint not found: {salad_checkpoint}")

    # Match the local SALAD checkpoint recipe used by the bundled eval script.
    model = VPRModel(
        backbone_arch="dinov2_vitb14",
        backbone_config={
            "num_trainable_blocks": 4,
            "return_token": True,
            "norm_layer": True,
        },
        agg_arch="SALAD",
        agg_config={
            "num_channels": 768,
            "num_clusters": 16,
            "cluster_dim": 32,
            "token_dim": 32,
        },
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    _purge_salad_modules()
    model = model.to(device)
    model.eval()

    @torch.no_grad()
    def retriever(images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 5:
            images = images[:, 0]
        return model(_apply_imagenet_norm(images).to(device))

    return retriever


def load_netvlad_retriever(device: str):
    """Load the VGG16-NetVLAD-Pitts30K retriever (same model as reloc3r).

    NetVLAD expects images in [0, 1] range and applies its own MATLAB-based
    mean subtraction internally. Since ``preprocess_image`` already produces
    [0, 1] tensors, no additional normalization is needed here.
    """
    netvlad_path = str(NETVLAD_ROOT)
    added_path = netvlad_path not in sys.path
    if added_path:
        sys.path.insert(0, netvlad_path)
    try:
        from netvlad import NetVLAD
    finally:
        if added_path and netvlad_path in sys.path:
            sys.path.remove(netvlad_path)

    model = NetVLAD(NetVLAD.default_conf).eval().to(device)

    @torch.no_grad()
    def retriever(images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 5:
            images = images[:, 0]  # [B, 3, H, W]
        return model({"image": images.to(device)})["global_descriptor"]

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
        # cam_dec branch is most reliable when decoded directly from pose_enc.
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
    if anchor_mode == "reloc3r_motion_averaging":
        raise ValueError("reloc3r_motion_averaging requires pairwise relative-pose evaluation.")

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


# ---------------------------------------------------------------------------
# Scale diagnostics (enabled via --scale-diagnostics flag)
# ---------------------------------------------------------------------------
_SCALE_DIAG_ENABLED = False
_SCALE_DIAG_LOG: list[dict] = []


def _log_scale_diagnostics(
    relposes_q2d: np.ndarray,
    ref_gt_c2w: np.ndarray,
    gt_query_c2w: np.ndarray,
    branch: str,
    query_idx: int,
):
    """Compare predicted vs GT relative translation norms for one query."""
    K = relposes_q2d.shape[0]
    pred_norms = []
    gt_norms = []
    scale_ratios = []

    for i in range(K):
        # Predicted q2d translation norm.
        pred_t_norm = float(np.linalg.norm(relposes_q2d[i, :3, 3]))
        pred_norms.append(pred_t_norm)

        # GT q2d = inv(c2w_db) @ c2w_query.
        gt_q2d = np.linalg.inv(ref_gt_c2w[i]) @ gt_query_c2w
        gt_t_norm = float(np.linalg.norm(gt_q2d[:3, 3]))
        gt_norms.append(gt_t_norm)

        # Scale ratio (pred / gt). Avoid division by zero.
        if gt_t_norm > 1e-8:
            scale_ratios.append(pred_t_norm / gt_t_norm)

    entry = {
        "query_idx": query_idx,
        "branch": branch,
        "pred_t_norms": pred_norms,
        "gt_t_norms": gt_norms,
        "scale_ratios": scale_ratios,
    }
    _SCALE_DIAG_LOG.append(entry)

    # Print per-query summary every 50 queries.
    if (query_idx + 1) % 50 == 0 and scale_ratios:
        ratios = np.array(scale_ratios)
        print(
            f"  [ScaleDiag q={query_idx} {branch}] "
            f"pred_t_norm: median={np.median(pred_norms):.4f} std={np.std(pred_norms):.4f} | "
            f"gt_t_norm: median={np.median(gt_norms):.4f} | "
            f"scale_ratio: median={np.median(ratios):.4f} std={np.std(ratios):.4f} "
            f"min={ratios.min():.4f} max={ratios.max():.4f}"
        )


def summarize_scale_diagnostics() -> dict[str, Any] | None:
    """Print and return aggregate scale statistics after all queries."""
    if not _SCALE_DIAG_LOG:
        return None

    all_ratios = []
    all_pred_norms = []
    all_gt_norms = []
    for entry in _SCALE_DIAG_LOG:
        all_ratios.extend(entry["scale_ratios"])
        all_pred_norms.extend(entry["pred_t_norms"])
        all_gt_norms.extend(entry["gt_t_norms"])

    ratios = np.array(all_ratios)
    pred_norms = np.array(all_pred_norms)
    gt_norms = np.array(all_gt_norms)

    summary = {
        "n_pairs": len(ratios),
        "scale_ratio_median": float(np.median(ratios)),
        "scale_ratio_mean": float(np.mean(ratios)),
        "scale_ratio_std": float(np.std(ratios)),
        "scale_ratio_min": float(ratios.min()),
        "scale_ratio_max": float(ratios.max()),
        "scale_ratio_iqr": float(np.percentile(ratios, 75) - np.percentile(ratios, 25)),
        "pred_t_norm_median": float(np.median(pred_norms)),
        "pred_t_norm_std": float(np.std(pred_norms)),
        "gt_t_norm_median": float(np.median(gt_norms)),
        "gt_t_norm_std": float(np.std(gt_norms)),
    }

    print("\n===== Scale Diagnostics Summary =====")
    print(f"  Total pairs analyzed: {summary['n_pairs']}")
    print(f"  Predicted t_norm:  median={summary['pred_t_norm_median']:.4f}  std={summary['pred_t_norm_std']:.4f}")
    print(f"  GT t_norm:         median={summary['gt_t_norm_median']:.4f}  std={summary['gt_t_norm_std']:.4f}")
    print(f"  Scale ratio (pred/gt):")
    print(f"    median={summary['scale_ratio_median']:.4f}  mean={summary['scale_ratio_mean']:.4f}  std={summary['scale_ratio_std']:.4f}")
    print(f"    min={summary['scale_ratio_min']:.4f}  max={summary['scale_ratio_max']:.4f}  IQR={summary['scale_ratio_iqr']:.4f}")
    print("=====================================\n")
    return summary


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

    topk_all: list[np.ndarray] = []
    topm_all: list[np.ndarray] = []
    branch_rotation_errors: dict[str, list[float]] = {"cam_dec": [], "ray": []}
    branch_translation_errors: dict[str, list[float]] = {"cam_dec": [], "ray": []}
    branch_effective_modes: dict[str, list[str]] = {"cam_dec": [], "ray": []}
    query_poses_by_branch: dict[str, list[np.ndarray]] = {"cam_dec": [], "ray": []}

    for q_entry in tqdm(query_entries, desc="Evaluating queries"):
        query_img = preprocess_image(q_entry["image_path"], target_size).to(device)
        query_input = query_img.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, H, W]
        query_desc = retriever(query_input)  # [1, D]
        gt_pose = np.asarray(q_entry["pose"], dtype=np.float64)

        sims = F.cosine_similarity(query_desc[0].unsqueeze(0), db_desc_device, dim=1)
        topk_indices_t, topm_indices_t = select_topk_topm(sims, top_k=top_k, top_m=top_m)
        topk_indices = topk_indices_t.detach().cpu().numpy().astype(np.int64)
        topm_indices = topm_indices_t.detach().cpu().numpy().astype(np.int64)
        topk_all.append(topk_indices)
        if anchor_mode == "reloc3r_motion_averaging":
            # Use the full retrieved top-K set for pairwise relpose + motion averaging.
            pose_indices = topk_indices
            topm_all.append(pose_indices)
            ref_gt = np.stack(
                [db_entries[idx]["pose"] for idx in pose_indices.tolist()],
                axis=0,
            ).astype(np.float64)

            if pose_path == "relpose_head":
                # RelPoseHead: directly predict pairwise q2d relative pose.
                relposes_q2d_list: list[np.ndarray] = []
                for db_idx in pose_indices.tolist():
                    candidate_image = preprocess_image(db_entries[db_idx]["image_path"], target_size)
                    candidate_input = candidate_image.unsqueeze(0).unsqueeze(0).to(device)
                    rel_pose = pose_pipeline.pairwise_relpose(
                        _apply_imagenet_norm(query_input),
                        _apply_imagenet_norm(candidate_input),
                    )
                    relposes_q2d_list.append(rel_pose[0].detach().cpu().numpy().astype(np.float64))

                relposes_q2d = np.stack(relposes_q2d_list, axis=0)
                query_pose = estimate_query_pose_motion_averaging(relposes_q2d, ref_gt)
                branch = "relpose_head"
                query_poses_by_branch.setdefault(branch, []).append(query_pose)
                branch_rotation_errors.setdefault(branch, []).append(
                    get_rot_err(query_pose[:3, :3], gt_pose[:3, :3])
                )
                branch_translation_errors.setdefault(branch, []).append(
                    float(np.linalg.norm(query_pose[:3, 3] - gt_pose[:3, 3]))
                )
                branch_effective_modes.setdefault(branch, []).append(anchor_mode)
                continue

            # cam_dec / ray / both: derive relative pose from predicted c2w group.
            relposes_by_branch: dict[str, list[np.ndarray]] = {"cam_dec": [], "ray": []}

            for db_idx in pose_indices.tolist():
                candidate_image = preprocess_image(db_entries[db_idx]["image_path"], target_size)
                candidate_input = candidate_image.unsqueeze(0).unsqueeze(0).to(device)

                # DA3 backbone requires ImageNet normalization.
                output = pose_pipeline.pose_only(
                    _apply_imagenet_norm(query_input), _apply_imagenet_norm(candidate_input),
                    pose_path=pose_path,
                )
                resolved = resolve_pose_output(output, pose_path)

                for branch in resolved.keys():
                    pred_group = _extract_group_c2w(resolved, branch, target_size)[0]
                    relposes_by_branch[branch].append(group_to_query_to_db_relative_pose(pred_group))

            for branch in resolved.keys():
                relposes_q2d = np.stack(relposes_by_branch[branch], axis=0)

                # --- Scale diagnostic ---
                if _SCALE_DIAG_ENABLED:
                    _log_scale_diagnostics(
                        relposes_q2d, ref_gt, gt_pose, branch,
                        query_idx=len(topk_all) - 1,
                    )

                query_pose = estimate_query_pose_motion_averaging(relposes_q2d, ref_gt)
                query_poses_by_branch[branch].append(query_pose)
                branch_rotation_errors[branch].append(get_rot_err(query_pose[:3, :3], gt_pose[:3, :3]))
                branch_translation_errors[branch].append(
                    float(np.linalg.norm(query_pose[:3, 3] - gt_pose[:3, 3]))
                )
                branch_effective_modes[branch].append(anchor_mode)
            continue

        topm_all.append(topm_indices)
        candidate_images = [
            preprocess_image(db_entries[idx]["image_path"], target_size) for idx in topm_indices.tolist()
        ]
        candidate_input = torch.stack(candidate_images, dim=0).unsqueeze(0).to(device)

        # DA3 backbone requires ImageNet normalization.
        output = pose_pipeline.pose_only(
            _apply_imagenet_norm(query_input), _apply_imagenet_norm(candidate_input),
            pose_path=pose_path,
        )
        resolved = resolve_pose_output(output, pose_path)

        ref_gt = np.stack([db_entries[idx]["pose"] for idx in topm_indices.tolist()], axis=0).astype(np.float64)

        for branch in resolved.keys():
            pred_group = _extract_group_c2w(resolved, branch, target_size)[0]
            query_pose, mode_used = _estimate_query_pose_from_group(pred_group, ref_gt, anchor_mode)
            query_poses_by_branch[branch].append(query_pose)
            branch_rotation_errors[branch].append(get_rot_err(query_pose[:3, :3], gt_pose[:3, :3]))
            branch_translation_errors[branch].append(
                float(np.linalg.norm(query_pose[:3, 3] - gt_pose[:3, 3]))
            )
            branch_effective_modes[branch].append(mode_used)

    if pose_path == "relpose_head":
        primary_branch = "relpose_head"
    elif pose_path in ("cam_dec", "both"):
        primary_branch = "cam_dec"
    else:
        primary_branch = "ray"
    payload: dict[str, Any] = {
        "topk_indices": np.asarray(topk_all, dtype=object),
        "topm_indices": np.asarray(topm_all, dtype=object),
        "retriever_backend": retriever_backend,
        "pose_path": pose_path,
        "primary_pose_branch": primary_branch,
        "config": config,
    }
    for branch in ("cam_dec", "ray", "relpose_head"):
        errs = branch_rotation_errors.get(branch, [])
        if errs:
            payload[f"rotation_errors_{branch}"] = np.asarray(errs, dtype=np.float32)
            payload[f"translation_errors_{branch}"] = np.asarray(
                branch_translation_errors[branch], dtype=np.float32
            )
            payload[f"effective_anchor_modes_{branch}"] = np.asarray(
                branch_effective_modes[branch], dtype=object
            )
        poses = query_poses_by_branch.get(branch, [])
        if poses:
            payload[f"query_poses_{branch}"] = np.stack(poses, axis=0)
    payload["rotation_errors"] = payload[f"rotation_errors_{primary_branch}"]
    payload["translation_errors"] = payload[f"translation_errors_{primary_branch}"]
    payload["effective_anchor_modes"] = payload[f"effective_anchor_modes_{primary_branch}"]
    return payload


def save_result_payload(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = dict(payload)
    save_dict["config"] = np.array(payload["config"], dtype=object)
    np.savez(output_path, **save_dict)


def summarize_result_medians(payload: dict[str, Any]) -> list[dict[str, float | str]]:
    summaries: list[dict[str, float | str]] = []
    for branch in ("cam_dec", "ray"):
        rotation_key = f"rotation_errors_{branch}"
        translation_key = f"translation_errors_{branch}"
        if rotation_key not in payload or translation_key not in payload:
            continue
        summaries.append(
            {
                "branch": branch,
                "median_translation": float(np.median(payload[translation_key])),
                "median_rotation": float(np.median(payload[rotation_key])),
            }
        )
    return summaries


def build_output_path(
    output_dir: str | Path,
    retriever_backend: str,
    dataset: str,
    scene: str,
    pose_path: str,
    anchor_mode: str,
) -> Path:
    return Path(output_dir) / (
        f"training_free_{retriever_backend}_{dataset}_{scene}_{pose_path}_{anchor_mode}.npz"
    )


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


def group_to_query_to_db_relative_pose(pred_group: np.ndarray) -> np.ndarray:
    """Convert a 2-view [query, db] group into reloc3r's query-to-db relative pose."""
    pred_group = _as_pose_array(pred_group, "pred_group")
    if pred_group.shape[0] != 2:
        raise ValueError("pred_group must contain exactly [query, db] poses for pairwise relpose.")
    return np.linalg.inv(pred_group[1]) @ pred_group[0]


def estimate_query_pose_motion_averaging(
    relposes_q2d: np.ndarray | list[np.ndarray],
    ref_gt: np.ndarray | list[np.ndarray],
) -> np.ndarray:
    """Fuse pairwise q->db predictions with reloc3r's motion averaging solver."""
    relposes_q2d = _as_pose_array(relposes_q2d, "relposes_q2d")
    ref_gt = _as_pose_array(ref_gt, "ref_gt")
    if relposes_q2d.shape[0] != ref_gt.shape[0]:
        raise ValueError("relposes_q2d count must match ref_gt count.")

    from reloc3r.reloc3r_visloc import Reloc3rVisloc

    solver = Reloc3rVisloc()
    return solver.motion_averaging(list(ref_gt), list(relposes_q2d))


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
    global _SCALE_DIAG_ENABLED, _SCALE_DIAG_LOG
    args = parse_args()
    validate_runtime_args(args)
    _SCALE_DIAG_ENABLED = args.scale_diagnostics
    _SCALE_DIAG_LOG = []
    runtime_device = resolve_runtime_device(args.device)
    target_size = (args.image_size[0], args.image_size[1])
    data_root = args.data_root or default_data_root(args.dataset)

    if args.retriever_backend == "da3_salad":
        pose_pipeline, retriever = build_da3_salad_retriever(
            args.unified_config, args.unified_checkpoint, runtime_device,
        )
        config = load_config(args.unified_config)
    else:
        pose_pipeline, config = build_pose_pipeline(
            args.unified_config, args.unified_checkpoint, runtime_device,
        )
        if args.retriever_backend == "netvlad":
            retriever = load_netvlad_retriever(runtime_device)
        else:
            retriever = load_dino_salad_retriever(args.salad_checkpoint, runtime_device)

    # Load RelPoseHead if requested.
    if args.pose_path == "relpose_head" and args.relpose_checkpoint:
        from depth_anything_3.model.rel_pose_head import RelPoseHead
        from depth_anything_3.model.unified_pipeline_helper import _unwrap_checkpoint_state_dict

        model_cfg = config.get("model", config)
        token_dim = model_cfg.get("rel_pose_head", {}).get("token_dim", 1536)
        head = RelPoseHead(token_dim=token_dim)
        ckpt = torch.load(args.relpose_checkpoint, map_location="cpu")
        sd = _unwrap_checkpoint_state_dict(ckpt)
        # Try prefixed keys first (Lightning wraps as "rel_pose_head.xxx").
        from depth_anything_3.model.unified_pipeline_helper import extract_prefixed_state_dict
        head_sd = extract_prefixed_state_dict(sd, ("rel_pose_head.",))
        if head_sd:
            head.load_state_dict(head_sd, strict=True)
        else:
            head.load_state_dict(sd, strict=False)
        head.to(runtime_device).eval()
        pose_pipeline.rel_pose_head = head
        print(f"[INFO] Loaded RelPoseHead from {args.relpose_checkpoint}")

    db_entries = load_scene_images_and_poses(args.dataset, args.scene, "train", data_root=data_root)
    query_entries = load_scene_images_and_poses(args.dataset, args.scene, "test", data_root=data_root)
    if args.cpu_fallback_max_db_entries > 0:
        # A bounded database keeps smoke runs practical during runtime verification.
        db_entries = db_entries[: args.cpu_fallback_max_db_entries]
        print(
            f"[WARN] Using first {len(db_entries)} database images for bounded smoke "
            "(set --cpu-fallback-max-db-entries <= 0 to disable)."
        )
    if args.cpu_fallback_max_queries > 0:
        query_entries = query_entries[: args.cpu_fallback_max_queries]
        print(
            f"[WARN] Evaluating only first {len(query_entries)} queries for bounded smoke "
            "(set --cpu-fallback-max-queries <= 0 to disable)."
        )

    db_descriptors = extract_descriptors(
        db_entries,
        retriever,
        device=runtime_device,
        batch_size=args.batch_size,
        target_size=target_size,
    )

    result = evaluate_scene_training_free(
        pose_pipeline=pose_pipeline,
        retriever=retriever,
        db_entries=db_entries,
        query_entries=query_entries,
        db_descriptors=db_descriptors,
        device=runtime_device,
        top_k=args.top_k,
        top_m=args.top_m,
        pose_path=args.pose_path,
        anchor_mode=args.anchor_mode,
        target_size=target_size,
        config=config,
        retriever_backend=args.retriever_backend,
    )

    for summary in summarize_result_medians(result):
        print(
            f"[Training-Free][{args.retriever_backend}][{summary['branch']}] Scene {args.scene} "
            f"median pose error: {summary['median_translation']:.2f} m  "
            f"{summary['median_rotation']:.2f} deg"
        )

    output_path = build_output_path(
        output_dir=args.output_dir,
        retriever_backend=args.retriever_backend,
        dataset=args.dataset,
        scene=args.scene,
        pose_path=args.pose_path,
        anchor_mode=args.anchor_mode,
    )
    save_result_payload(result, output_path)
    print(f"Saved results to: {output_path}")

    if _SCALE_DIAG_ENABLED:
        summarize_scale_diagnostics()


if __name__ == "__main__":
    main()
