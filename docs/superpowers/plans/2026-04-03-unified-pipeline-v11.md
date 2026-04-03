# Unified Pipeline v1.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace external CrossViewFusion with DA3's native multi-view alternate attention for pose estimation.

**Architecture:** Two-stage pipeline sharing one DA3 backbone. Stage 1: query single-view backbone pass for VPR retrieval. Stage 2: query + top-M candidates multi-view backbone pass using DA3's alternate attention, then cam_dec for pose.

**Tech Stack:** PyTorch, PyTorch Lightning, DA3 (DepthAnything3), SALAD aggregator, addict Dict

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/depth_anything_3/model/unified_pipeline.py` | Rewrite | Pipeline with two-stage forward: retrieval + multi-view pose |
| `src/depth_anything_3/model/unified_pipeline_helper.py` | Modify | Remove cross_view_fusion; add `backbone_from_layer` freeze; keep DA3Net reference |
| `src/depth_anything_3/model/cross_view_fusion.py` | Delete | Replaced by backbone alternate attention |
| `configs/eval_unified_v11.yaml` | Create | Eval / training-free baseline config |
| `eval_unified_visloc_v11.py` | Create | Eval script with cam_dec + ray dual pose paths |
| `configs/train_unified_stage1_v11.yaml` | Create | v1.1 training config |
| `train/train_unified_pipeline.py` | Rewrite | Two-stage training forward, no fusion |
| `ablation/eval_dinosalad_plus_ourpose.py` | Modify | Use multi-view backbone for pose path |
| `ablation/eval_ourvpr_plus_reloc3r.py` | Modify | Use new descriptor-only database |

---

### Task 1: Rewrite UnifiedPipeline — remove CrossViewFusion, add multi-view pose

**Files:**
- Modify: `src/depth_anything_3/model/unified_pipeline.py`

- [ ] **Step 1: Rewrite `unified_pipeline.py`**

Replace the entire file. Key changes: remove `cross_view_fusion`, `_run_pose_branch` now concatenates query+candidates and runs backbone multi-view, `extract_database_features` returns only descriptors, add `da3_net` (full DA3Net) for stage 2 ray path.

```python
from __future__ import annotations

import torch
import torch.nn as nn
from addict import Dict

from depth_anything_3.model.retrieval_strategy import BaseRetrievalStrategy
from depth_anything_3.model.vpr_feature_utils import (
    extract_aux_patch_tokens,
    patch_tokens_to_feature_dict,
)


class UnifiedPipeline(nn.Module):
    """Unified image retrieval + pose regression pipeline v1.1.

    Two-stage architecture:
        Stage 1 (retrieval): query single-view backbone → aux features → VPR descriptor
        Stage 2 (pose): [query, top-M candidates] multi-view backbone → alternate attention → cam_dec → pose

    Three calling modes:
        forward():        full pipeline (retrieval + pose)
        retrieval_only(): extract global descriptors only
        pose_only():      given pre-selected candidates, multi-view backbone → pose
    """

    PATCH_SIZE = 14

    def __init__(
        self,
        da3_backbone: nn.Module,
        feature_adapter: nn.Module,
        aggregator: nn.Module,
        retrieval_strategy: BaseRetrievalStrategy,
        da3_head: nn.Module,
        cam_dec: nn.Module,
        aux_layer: int = 5,
        pose_top_m: int = 3,
    ):
        super().__init__()
        self.da3_backbone = da3_backbone
        self.feature_adapter = feature_adapter
        self.aggregator = aggregator
        self.retrieval_strategy = retrieval_strategy
        self.da3_head = da3_head
        self.cam_dec = cam_dec
        self.aux_layer = aux_layer
        self.pose_top_m = pose_top_m

    # ----- Stage 1: Retrieval -----

    def _run_backbone_single(self, x: torch.Tensor):
        """Run backbone in single-view mode, exporting aux layer features.

        Args:
            x: [B, 1, 3, H, W] single-view input

        Returns:
            feats: list of (patch_tokens, camera_tokens) tuples
            aux_feats: list of aux feature tensors
            image_h, image_w: image dimensions
        """
        image_h, image_w = x.shape[-2], x.shape[-1]
        feats, aux_feats = self.da3_backbone(
            x, cam_token=None, export_feat_layers=[self.aux_layer],
            ref_view_strategy="saddle_balanced",
        )
        return feats, aux_feats, image_h, image_w

    def _run_vpr_branch(self, aux_feats, image_h, image_w):
        """VPR side branch: aux features -> adapter -> SALAD -> descriptor.

        Args:
            aux_feats: aux features from backbone
            image_h, image_w: image dimensions for spatial reshape

        Returns:
            descriptor: [B, D] global descriptor
        """
        patch_tokens = extract_aux_patch_tokens(aux_feats)
        feat_dict = patch_tokens_to_feature_dict(
            patch_tokens, image_h, image_w, self.PATCH_SIZE,
        )
        feat_dict = self.feature_adapter(feat_dict)
        descriptor = self.aggregator(
            (feat_dict["feature_map"], feat_dict["global_token"]),
        )
        return descriptor

    # ----- Stage 2: Multi-view Pose -----

    def _run_backbone_multiview(self, multi_view_input: torch.Tensor):
        """Run backbone in multi-view mode with alternate attention.

        Args:
            multi_view_input: [B, 1+M, 3, H, W] query (view 0) + M candidates

        Returns:
            feats: list of (patch_tokens [B, 1+M, P, C], camera_tokens [B, 1+M, C])
            image_h, image_w: image dimensions
        """
        image_h, image_w = multi_view_input.shape[-2], multi_view_input.shape[-1]
        feats, _ = self.da3_backbone(
            multi_view_input, cam_token=None, export_feat_layers=[],
            ref_view_strategy="saddle_balanced",
        )
        return feats, image_h, image_w

    def _run_pose_cam_dec(self, feats, image_h, image_w):
        """Pose via cam_dec path (differentiable, for training).

        Args:
            feats: backbone multi-view feats
            image_h, image_w: image dimensions

        Returns:
            Dict with pose_enc [B, 1+M, 9] and decoded extrinsics/intrinsics
        """
        from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri
        from depth_anything_3.utils.geometry import affine_inverse

        # cam_dec expects camera tokens [B, S, C]
        camera_tokens = feats[-1][1]  # [B, 1+M, C]
        pose_enc = self.cam_dec(camera_tokens)  # [B, 1+M, 9]

        c2w, ixt = pose_encoding_to_extri_intri(pose_enc, (image_h, image_w))

        output = Dict()
        output.pose_enc = pose_enc
        output.extrinsics = affine_inverse(c2w)
        output.intrinsics = ixt
        return output

    def _run_pose_ray(self, feats, image_h, image_w):
        """Pose via ray path (non-differentiable, for inference).

        Args:
            feats: backbone multi-view feats
            image_h, image_w: image dimensions

        Returns:
            Dict with extrinsics and intrinsics from ray map
        """
        from depth_anything_3.utils.geometry import affine_inverse
        from depth_anything_3.utils.ray_utils import get_extrinsic_from_camray

        head_out = self.da3_head(feats, image_h, image_w, patch_start_idx=0)

        output = Dict()
        if "ray" in head_out and "ray_conf" in head_out:
            pred_ext, pred_fl, pred_pp = get_extrinsic_from_camray(
                head_out.ray, head_out.ray_conf,
                head_out.ray.shape[-3], head_out.ray.shape[-2],
            )
            pred_ext = affine_inverse(pred_ext)  # w2c -> c2w
            pred_ext = pred_ext[:, :, :3, :]

            import torch
            pred_ixt = torch.eye(3, 3)[None, None].repeat(
                pred_ext.shape[0], pred_ext.shape[1], 1, 1,
            ).clone().to(pred_ext.device)
            pred_ixt[:, :, 0, 0] = pred_fl[:, :, 0] / 2 * image_w
            pred_ixt[:, :, 1, 1] = pred_fl[:, :, 1] / 2 * image_h
            pred_ixt[:, :, 0, 2] = pred_pp[:, :, 0] * image_w * 0.5
            pred_ixt[:, :, 1, 2] = pred_pp[:, :, 1] * image_h * 0.5

            output.extrinsics = pred_ext
            output.intrinsics = pred_ixt

        if "depth" in head_out:
            output.depth = head_out.depth

        return output

    # ----- Public API -----

    def forward(
        self,
        query_image: torch.Tensor,
        candidate_images: torch.Tensor,
    ) -> Dict:
        """Full unified forward: retrieval + multi-view pose.

        Args:
            query_image: [B, 1, 3, H, W]
            candidate_images: [B, K, 3, H, W] all K candidates (top-M selected internally)

        Returns:
            Dict with: pose_enc, extrinsics, intrinsics, query_descriptor, selected_indices
        """
        B, K = candidate_images.shape[:2]

        # Stage 1: Retrieval
        _, aux_feats, image_h, image_w = self._run_backbone_single(query_image)
        query_descriptor = self._run_vpr_branch(aux_feats, image_h, image_w)

        # Get candidate descriptors (no grad, frozen backbone)
        with torch.no_grad():
            cand_descs = []
            for k in range(K):
                cand_input = candidate_images[:, k:k+1]
                _, cand_aux, _, _ = self._run_backbone_single(cand_input)
                cand_desc = self._run_vpr_branch(cand_aux, image_h, image_w)
                cand_descs.append(cand_desc)
            cand_descs = torch.cat(cand_descs, dim=0)  # [B*K, D]

        # Select top-M
        M = min(self.pose_top_m, K)
        sims = torch.nn.functional.cosine_similarity(
            query_descriptor[0].unsqueeze(0), cand_descs[:K], dim=-1,
        )
        topm_indices = sims.topk(M).indices  # [M]

        # Gather top-M candidate images
        selected_cands = candidate_images[:, topm_indices]  # [B, M, 3, H, W]

        # Stage 2: Multi-view pose
        multi_view = torch.cat([query_image, selected_cands], dim=1)  # [B, 1+M, 3, H, W]
        feats, _, _ = self._run_backbone_multiview(multi_view)

        output = self._run_pose_cam_dec(feats, image_h, image_w)
        output.query_descriptor = query_descriptor
        output.selected_indices = topm_indices
        return output

    def retrieval_only(self, images: torch.Tensor) -> torch.Tensor:
        """Extract global descriptors only.

        Args:
            images: [B, 1, 3, H, W]

        Returns:
            descriptors: [B, D]
        """
        _, aux_feats, image_h, image_w = self._run_backbone_single(images)
        return self._run_vpr_branch(aux_feats, image_h, image_w)

    def pose_only(
        self,
        query_image: torch.Tensor,
        candidate_images: torch.Tensor,
        pose_path: str = "cam_dec",
    ) -> Dict:
        """Pose-only: given pre-selected candidates, run multi-view backbone.

        Args:
            query_image: [B, 1, 3, H, W]
            candidate_images: [B, M, 3, H, W] pre-selected candidates
            pose_path: "cam_dec", "ray", or "both"

        Returns:
            Dict with pose results
        """
        multi_view = torch.cat([query_image, candidate_images], dim=1)  # [B, 1+M, 3, H, W]
        feats, image_h, image_w = self._run_backbone_multiview(multi_view)

        output = Dict()

        if pose_path in ("cam_dec", "both"):
            cam_out = self._run_pose_cam_dec(feats, image_h, image_w)
            output.update(cam_out)

        if pose_path in ("ray", "both"):
            with torch.no_grad():
                ray_out = self._run_pose_ray(feats, image_h, image_w)
            if pose_path == "ray":
                output.update(ray_out)
            else:
                output.ray_extrinsics = ray_out.get("extrinsics")
                output.ray_intrinsics = ray_out.get("intrinsics")

        return output

    @torch.no_grad()
    def extract_database_features(self, images: torch.Tensor) -> torch.Tensor:
        """Offline: extract VPR descriptors for database images.

        Args:
            images: [B, 1, 3, H, W]

        Returns:
            descriptors: [B, D]
        """
        _, aux_feats, image_h, image_w = self._run_backbone_single(images)
        return self._run_vpr_branch(aux_feats, image_h, image_w)
```

- [ ] **Step 2: Verify no syntax errors**

Run: `cd /home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1 && python -c "from depth_anything_3.model.unified_pipeline import UnifiedPipeline; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/depth_anything_3/model/unified_pipeline.py
git commit -m "feat: rewrite UnifiedPipeline v1.1 with multi-view backbone pose"
```

---

### Task 2: Update unified_pipeline_helper — remove fusion, add backbone_from_layer freeze

**Files:**
- Modify: `src/depth_anything_3/model/unified_pipeline_helper.py`

- [ ] **Step 1: Rewrite `unified_pipeline_helper.py`**

Remove cross_view_fusion imports and construction. Update `_apply_freeze` to support `backbone_from_layer`. Keep `da3_net` reference in pipeline for ray path. Update `build_unified_pipeline` to pass `pose_top_m`.

```python
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import torch
import yaml

from depth_anything_3.api import DepthAnything3
from depth_anything_3.model import VPRaggregators
from depth_anything_3.model.retrieval_strategy import build_retrieval_strategy
from depth_anything_3.model.unified_pipeline import UnifiedPipeline
from depth_anything_3.model.vpr_feature_adapter import (
    DualBranchFeatureAdapter,
    IdentityFeatureAdapter,
    PatchOnlyFeatureAdapter,
)


# ---------------------------------------------------------------------------
# VPR component builders (migrated from vpr_helper.py)
# ---------------------------------------------------------------------------


def build_aggregator(agg_arch: str, agg_config: dict | None = None):
    """Build a VPR aggregator by name."""
    agg_config = {} if agg_config is None else dict(agg_config)
    name = agg_arch.lower()
    if name == "cosplace":
        return VPRaggregators.CosPlace(**agg_config)
    if name == "gem":
        agg_config.setdefault("p", 3)
        return VPRaggregators.GeMPool(**agg_config)
    if name == "convap":
        return VPRaggregators.ConvAP(**agg_config)
    if name == "mixvpr":
        return VPRaggregators.MixVPR(**agg_config)
    if name == "salad":
        return VPRaggregators.SALAD(**agg_config)
    raise ValueError(f"Unsupported aggregator: {agg_arch}")


def build_feature_adapter(adapter_arch: str | None = None, adapter_config: dict | None = None):
    """Build a feature adapter by name."""
    adapter_config = {} if adapter_config is None else dict(adapter_config)
    if adapter_arch is None:
        return IdentityFeatureAdapter()
    name = str(adapter_arch).lower()
    if name == "identity":
        return IdentityFeatureAdapter()
    if name == "patch_only":
        return PatchOnlyFeatureAdapter(**adapter_config)
    if name == "dual_branch":
        return DualBranchFeatureAdapter(**adapter_config)
    raise ValueError(f"Unsupported feature adapter: {adapter_arch}")


def extract_prefixed_state_dict(
    state_dict: Mapping[str, torch.Tensor], prefixes: Iterable[str],
) -> dict[str, torch.Tensor]:
    """Extract keys matching any prefix, stripping the prefix."""
    extracted = {}
    for key, value in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                extracted[key[len(prefix):]] = value
                break
    return extracted


def _unwrap_checkpoint_state_dict(checkpoint):
    """Unwrap a Lightning/PyTorch checkpoint to its raw state_dict."""
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Expected checkpoint to contain a mapping of parameters")
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], Mapping):
        return checkpoint["state_dict"]
    if "model" in checkpoint and isinstance(checkpoint["model"], Mapping):
        return checkpoint["model"]
    return checkpoint


VPR_ADAPTER_PREFIXES = (
    "feature_adapter.",
    "vpr_model.feature_adapter.",
    "module.vpr_model.feature_adapter.",
)
VPR_AGGREGATOR_PREFIXES = (
    "aggregator.",
    "vpr_model.aggregator.",
    "module.vpr_model.aggregator.",
    "model.aggregator.",
    "module.aggregator.",
)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _load_vpr_weights(feature_adapter, aggregator, vpr_checkpoint_path):
    """Load VPR checkpoint into feature_adapter and aggregator."""
    checkpoint = torch.load(Path(vpr_checkpoint_path), map_location="cpu")
    state_dict = _unwrap_checkpoint_state_dict(checkpoint)

    adapter_sd = extract_prefixed_state_dict(state_dict, VPR_ADAPTER_PREFIXES)
    if adapter_sd:
        feature_adapter.load_state_dict(adapter_sd, strict=True)

    agg_sd = extract_prefixed_state_dict(state_dict, VPR_AGGREGATOR_PREFIXES)
    if agg_sd:
        aggregator.load_state_dict(agg_sd, strict=True)


def _apply_freeze(pipeline: UnifiedPipeline, freeze_config: dict):
    """Freeze modules according to config.

    Supports backbone_from_layer: freeze layers [0, N), unfreeze [N, end].
    When backbone_from_layer == -1 (default), all backbone layers are frozen.
    """
    backbone_frozen = freeze_config.get("backbone", True)
    backbone_from_layer = freeze_config.get("backbone_from_layer", -1)

    if backbone_frozen:
        if backbone_from_layer == -1:
            # Freeze entire backbone
            pipeline.da3_backbone.requires_grad_(False)
            pipeline.da3_backbone.eval()
        else:
            # Freeze layers [0, backbone_from_layer), unfreeze [backbone_from_layer, end]
            pipeline.da3_backbone.requires_grad_(False)
            pipeline.da3_backbone.eval()
            # Unfreeze blocks from backbone_from_layer onward
            blocks = pipeline.da3_backbone.blocks
            for i in range(backbone_from_layer, len(blocks)):
                blocks[i].requires_grad_(True)
                blocks[i].train()
            # Also unfreeze the final LayerNorm
            if hasattr(pipeline.da3_backbone, "norm"):
                pipeline.da3_backbone.norm.requires_grad_(True)
                pipeline.da3_backbone.norm.train()

    if freeze_config.get("vpr", True):
        pipeline.feature_adapter.requires_grad_(False)
        pipeline.feature_adapter.eval()
        pipeline.aggregator.requires_grad_(False)
        pipeline.aggregator.eval()

    head_flag = freeze_config.get("head", False)
    if freeze_config.get("da3_head", head_flag):
        pipeline.da3_head.requires_grad_(False)
        pipeline.da3_head.eval()
    if freeze_config.get("cam_dec", head_flag):
        pipeline.cam_dec.requires_grad_(False)
        pipeline.cam_dec.eval()


def build_unified_pipeline(config: dict, device: str = "cpu") -> UnifiedPipeline:
    """Build UnifiedPipeline v1.1 from config dict.

    Args:
        config: dict with top-level "model" key, or flat model config.
    """
    model_config = config.get("model", config)

    # 1. Load DA3 model and extract backbone + head + cam_dec
    da3_model_name = model_config.get("da3_model_name_or_path", "depth-anything/DA3-BASE")
    da3_wrapper = DepthAnything3.from_pretrained(da3_model_name)
    da3_net = da3_wrapper.model  # DepthAnything3Net

    backbone = da3_net.backbone
    da3_head = da3_net.head
    cam_dec = da3_net.cam_dec

    # 2. Build VPR components
    agg_arch = model_config.get("agg_arch", "salad")
    agg_config = model_config.get("agg_config", {
        "num_channels": 768, "num_clusters": 16, "cluster_dim": 32, "token_dim": 32,
    })
    aggregator = build_aggregator(agg_arch, agg_config=agg_config)

    adapter_arch = model_config.get("feature_adapter_arch", "patch_only")
    adapter_config = model_config.get("feature_adapter_config", {"channels": 768})
    feature_adapter = build_feature_adapter(adapter_arch, adapter_config=adapter_config)

    # Load VPR checkpoint if provided
    vpr_ckpt = model_config.get("vpr_checkpoint")
    if vpr_ckpt and Path(vpr_ckpt).is_file():
        _load_vpr_weights(feature_adapter, aggregator, vpr_ckpt)

    # 3. Build retrieval strategy
    retrieval_config = model_config.get("retrieval", {
        "strategy": "soft_attention", "top_k": 10, "temperature": 1.0,
    })
    retrieval_strategy = build_retrieval_strategy(retrieval_config)

    aux_layer = model_config.get("aux_layer", 5)
    pose_top_m = model_config.get("pose_top_m", 3)

    # 4. Assemble pipeline
    pipeline = UnifiedPipeline(
        da3_backbone=backbone,
        feature_adapter=feature_adapter,
        aggregator=aggregator,
        retrieval_strategy=retrieval_strategy,
        da3_head=da3_head,
        cam_dec=cam_dec,
        aux_layer=aux_layer,
        pose_top_m=pose_top_m,
    )

    # 5. Apply freeze
    freeze_config = model_config.get("freeze", {
        "backbone": True, "vpr": True, "da3_head": True, "cam_dec": False,
    })
    _apply_freeze(pipeline, freeze_config)

    pipeline.to(device)
    return pipeline
```

- [ ] **Step 2: Verify no syntax errors**

Run: `cd /home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1 && python -c "from depth_anything_3.model.unified_pipeline_helper import build_unified_pipeline; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/depth_anything_3/model/unified_pipeline_helper.py
git commit -m "feat: update helper for v1.1 — remove fusion, add backbone_from_layer"
```

---

### Task 3: Delete cross_view_fusion.py

**Files:**
- Delete: `src/depth_anything_3/model/cross_view_fusion.py`

- [ ] **Step 1: Remove the file**

```bash
git rm src/depth_anything_3/model/cross_view_fusion.py
```

- [ ] **Step 2: Verify no remaining imports**

Run: `cd /home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1 && grep -r "cross_view_fusion" src/ --include="*.py"`
Expected: no output (no references remain)

- [ ] **Step 3: Commit**

```bash
git commit -m "refactor: delete cross_view_fusion.py — replaced by backbone alternate attention"
```

---

### Task 4: Create eval config and training-free baseline eval script

**Files:**
- Create: `configs/eval_unified_v11.yaml`
- Create: `eval_unified_visloc_v11.py`

- [ ] **Step 1: Create eval config**

```yaml
model:
  da3_model_name_or_path: "depth-anything/DA3-BASE"
  vpr_checkpoint: "checkpoints/image_retrieval/DA3_vprmodel_patchonlyadapter_aux5.ckpt"
  aux_layer: 5
  pose_top_m: 3

  feature_adapter_arch: "patch_only"
  feature_adapter_config:
    channels: 768
    bottleneck: 640

  agg_arch: "salad"
  agg_config:
    num_channels: 768
    num_clusters: 16
    cluster_dim: 32
    token_dim: 32

  freeze:
    backbone: true
    vpr: true
    da3_head: true
    cam_dec: true

  retrieval:
    strategy: "soft_attention"
    top_k: 10
    temperature: 1.0

eval:
  dataset: "7scenes"
  scenes:
    - heads
  retrieval_top_k: 10
  pose_top_m: 3
  pose_path: "both"
  image_size: [504, 504]
  batch_size: 16
```

- [ ] **Step 2: Create eval script `eval_unified_visloc_v11.py`**

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parents[0]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (SRC_ROOT, str(REPO_ROOT)):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from depth_anything_3.model.unified_pipeline_helper import build_unified_pipeline, load_config
from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri
from depth_anything_3.utils.geometry import affine_inverse

RELOC3R_ROOT = REPO_ROOT / "reloc3r"
if str(RELOC3R_ROOT) not in sys.path:
    sys.path.insert(0, str(RELOC3R_ROOT))
from reloc3r.utils.metric import get_rot_err

# Reuse dataset loaders from v1.0
from eval_unified_visloc import (
    load_scene_images_and_poses,
    preprocess_image,
)


@torch.no_grad()
def build_database(pipeline, db_entries, device, batch_size=16, target_size=(504, 504)):
    """Build database: extract VPR descriptors only (no patch/camera tokens needed in v1.1).

    Returns:
        db_descriptors: torch.Tensor [N, D] on device
        db_image_paths: list of str
    """
    all_descs = []
    for i in tqdm(range(0, len(db_entries), batch_size), desc="Building database"):
        batch_entries = db_entries[i:i + batch_size]
        images = torch.stack([preprocess_image(e["image_path"], target_size) for e in batch_entries])
        images = images.unsqueeze(1).to(device)  # [B, 1, 3, H, W]
        descs = pipeline.extract_database_features(images)  # [B, D]
        all_descs.append(descs.cpu())

    db_descriptors = torch.cat(all_descs, dim=0)  # [N, D]
    db_image_paths = [e["image_path"] for e in db_entries]
    return db_descriptors, db_image_paths


@torch.no_grad()
def evaluate_scene(
    pipeline, db_entries, query_entries, db_descriptors, db_image_paths,
    device, top_k=10, pose_top_m=3, pose_path="cam_dec", target_size=(504, 504),
):
    """Evaluate on a single scene.

    Args:
        pose_path: "cam_dec", "ray", or "both"
    """
    db_desc_device = db_descriptors.to(device)

    results = {"cam_dec": {"rerrs": [], "terrs": []}, "ray": {"rerrs": [], "terrs": []}}

    for q_entry in tqdm(query_entries, desc="Evaluating queries"):
        query_img = preprocess_image(q_entry["image_path"], target_size)
        query_input = query_img.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 3, H, W]

        # Stage 1: Retrieval
        query_desc = pipeline.retrieval_only(query_input)  # [1, D]
        sims = F.cosine_similarity(query_desc[0].unsqueeze(0), db_desc_device, dim=1)
        k = min(top_k, sims.shape[0])
        topk_indices = sims.topk(k).indices.cpu().tolist()

        # Select top-M from top-K
        M = min(pose_top_m, k)
        topm_indices = topk_indices[:M]

        # Load candidate images
        cand_images = torch.stack([
            preprocess_image(db_image_paths[idx], target_size) for idx in topm_indices
        ]).unsqueeze(0).to(device)  # [1, M, 3, H, W]

        # Stage 2: Multi-view pose
        output = pipeline.pose_only(query_input, cand_images, pose_path=pose_path)

        gt_pose = q_entry["pose"]

        # cam_dec results
        if pose_path in ("cam_dec", "both") and "pose_enc" in output:
            pose_enc = output.pose_enc  # [1, 1+M, 9]
            c2w, _ = pose_encoding_to_extri_intri(pose_enc, target_size)
            # Query is view 0
            pred_pose = c2w[0, 0].cpu().numpy()
            if pred_pose.shape[0] == 3:
                full = np.eye(4, dtype=np.float32)
                full[:3, :] = pred_pose
                pred_pose = full

            rerr = get_rot_err(pred_pose[:3, :3], gt_pose[:3, :3])
            terr = np.linalg.norm(pred_pose[:3, 3] - gt_pose[:3, 3])
            results["cam_dec"]["rerrs"].append(rerr)
            results["cam_dec"]["terrs"].append(terr)

        # ray results
        if pose_path in ("ray", "both"):
            ray_ext_key = "ray_extrinsics" if pose_path == "both" else "extrinsics"
            if ray_ext_key in output and output[ray_ext_key] is not None:
                ray_ext = output[ray_ext_key]
                # ray extrinsics are w2c, need c2w
                pred_pose_ray = affine_inverse(ray_ext)[0, 0].cpu().numpy()
                if pred_pose_ray.shape[0] == 3:
                    full = np.eye(4, dtype=np.float32)
                    full[:3, :] = pred_pose_ray
                    pred_pose_ray = full

                rerr = get_rot_err(pred_pose_ray[:3, :3], gt_pose[:3, :3])
                terr = np.linalg.norm(pred_pose_ray[:3, 3] - gt_pose[:3, 3])
                results["ray"]["rerrs"].append(rerr)
                results["ray"]["terrs"].append(terr)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate unified pipeline v1.1")
    parser.add_argument("--config", type=str, required=True, help="Path to eval config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint (omit for training-free baseline)")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--pose-path", type=str, default=None, choices=["cam_dec", "ray", "both"])
    parser.add_argument("--pose-top-m", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="workspace/eval_results_v11")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, nargs=2, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    eval_config = config.get("eval", {})

    # CLI args override config
    dataset = args.dataset or eval_config.get("dataset", "7scenes")
    scenes = [args.scene] if args.scene else eval_config.get("scenes", ["heads"])
    pose_path = args.pose_path or eval_config.get("pose_path", "cam_dec")
    pose_top_m = args.pose_top_m or eval_config.get("pose_top_m", 3)
    top_k = args.top_k or eval_config.get("retrieval_top_k", 10)
    batch_size = args.batch_size or eval_config.get("batch_size", 16)
    target_size = tuple(args.image_size) if args.image_size else tuple(eval_config.get("image_size", [504, 504]))

    pipeline = build_unified_pipeline(config, device=args.device)

    if args.checkpoint and Path(args.checkpoint).is_file():
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        state_dict = ckpt.get("state_dict", ckpt)
        # Strip "pipeline." prefix from Lightning checkpoint keys
        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith("pipeline."):
                cleaned[k[len("pipeline."):]] = v
            else:
                cleaned[k] = v
        pipeline.load_state_dict(cleaned, strict=False)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("Running training-free baseline (no checkpoint loaded)")

    pipeline.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for scene in scenes:
        print(f"\n{'='*60}")
        print(f"Scene: {scene}")
        print(f"{'='*60}")

        db_entries = load_scene_images_and_poses(dataset, scene, "train", data_root=args.data_root)
        query_entries = load_scene_images_and_poses(dataset, scene, "test", data_root=args.data_root)
        print(f"Database: {len(db_entries)} images, Queries: {len(query_entries)} images")

        db_descriptors, db_image_paths = build_database(
            pipeline, db_entries, args.device, batch_size=batch_size, target_size=target_size,
        )

        results = evaluate_scene(
            pipeline, db_entries, query_entries, db_descriptors, db_image_paths,
            args.device, top_k=top_k, pose_top_m=pose_top_m,
            pose_path=pose_path, target_size=target_size,
        )

        # Print results
        for path_name in ("cam_dec", "ray"):
            rerrs = results[path_name]["rerrs"]
            terrs = results[path_name]["terrs"]
            if rerrs:
                med_rerr = np.median(rerrs)
                med_terr = np.median(terrs)
                print(f"  [{path_name}] median: {med_terr:.4f} m  {med_rerr:.2f} deg  ({len(rerrs)} queries)")

        # Save results
        np.savez(
            output_dir / f"v11_{dataset}_{scene}_results.npz",
            cam_dec_rerrs=np.array(results["cam_dec"]["rerrs"]),
            cam_dec_terrs=np.array(results["cam_dec"]["terrs"]),
            ray_rerrs=np.array(results["ray"]["rerrs"]),
            ray_terrs=np.array(results["ray"]["terrs"]),
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify no syntax errors**

Run: `cd /home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1 && python -c "import ast; ast.parse(open('eval_unified_visloc_v11.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add configs/eval_unified_v11.yaml eval_unified_visloc_v11.py
git commit -m "feat: add v1.1 eval script with training-free baseline and dual pose paths"
```

---

### Task 5: Create v1.1 training config

**Files:**
- Create: `configs/train_unified_stage1_v11.yaml`

- [ ] **Step 1: Create training config**

```yaml
model:
  da3_model_name_or_path: "depth-anything/DA3-BASE"
  vpr_checkpoint: "checkpoints/image_retrieval/DA3_vprmodel_patchonlyadapter_aux5.ckpt"
  aux_layer: 5
  pose_top_m: 3

  feature_adapter_arch: "patch_only"
  feature_adapter_config:
    channels: 768
    bottleneck: 640

  agg_arch: "salad"
  agg_config:
    num_channels: 768
    num_clusters: 16
    cluster_dim: 32
    token_dim: 32

  freeze:
    backbone: true
    backbone_from_layer: -1
    vpr: false
    da3_head: true
    cam_dec: false

  retrieval:
    strategy: "soft_attention"
    top_k: 10
    temperature: 1.0

training:
  dataset: "7scenes"
  data_root: "/mnt/nas_9/group/chenguyuan/NeurIPS26/LoopAnything-dev/reloc3r/data/7scenes"
  scenes:
    - chess
    - fire
    - heads
    - office
    - pumpkin
    - redkitchen
    - stairs
  batch_size: 2
  num_workers: 8
  max_epochs: 50
  image_size: [504, 504]

  optimizer:
    name: "adamw"
    lr: 1.0e-4
    weight_decay: 1.0e-4

  scheduler:
    name: "cosine"
    T_max: 50

  loss:
    rotation_weight: 1.0
    translation_weight: 1.0

  candidate_sampling:
    pos_threshold: 1.0
    neg_threshold: 3.0
    pos_ratio: 0.7
    distance_alpha: 0.5
    scene_sampling: "round_robin"

  eval:
    eval_every_n_epoch: 1
    val_scenes:
      - heads
    top_k: 10
    pose_top_m: 3
```

- [ ] **Step 2: Commit**

```bash
git add configs/train_unified_stage1_v11.yaml
git commit -m "feat: add v1.1 training config"
```

---

### Task 6: Rewrite training script

**Files:**
- Modify: `train/train_unified_pipeline.py`

- [ ] **Step 1: Rewrite `train/train_unified_pipeline.py`**

Key changes: remove all fusion-related code, training forward uses two-stage (retrieval + multi-view backbone), loss only on query pose (view 0).

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
REPO_ROOT = PROJECT_ROOT.parents[2]
for path in (str(SRC_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from depth_anything_3.data.unified_visloc_dataset import UnifiedVislocDataset
from depth_anything_3.model.unified_pipeline_helper import build_unified_pipeline, load_config
from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri

RELOC3R_ROOT = REPO_ROOT / "reloc3r"
if str(RELOC3R_ROOT) not in sys.path:
    sys.path.insert(0, str(RELOC3R_ROOT))
from reloc3r.utils.metric import get_rot_err


def geodesic_rotation_loss(pred_R: torch.Tensor, gt_R: torch.Tensor) -> torch.Tensor:
    """Geodesic distance between rotation matrices.

    Args:
        pred_R: [B, 3, 3]
        gt_R: [B, 3, 3]

    Returns:
        Scalar loss (mean geodesic distance in radians)
    """
    R_rel = torch.bmm(pred_R.transpose(1, 2), gt_R)
    trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
    cos_angle = torch.clamp((trace - 1) / 2, -1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(cos_angle)
    return angle.mean()


def translation_loss(pred_t: torch.Tensor, gt_t: torch.Tensor) -> torch.Tensor:
    """L2 distance between predicted and GT translations."""
    return (pred_t - gt_t).norm(dim=1).mean()


class UnifiedPipelineLightningModule(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.model_config = config["model"]
        self.train_config = config["training"]

        self.pipeline = build_unified_pipeline(self.model_config)

        self.rotation_weight = self.train_config["loss"].get("rotation_weight", 1.0)
        self.translation_weight = self.train_config["loss"].get("translation_weight", 1.0)
        self.pose_top_m = self.model_config.get("pose_top_m", 3)

        self.save_hyperparameters(config)

    def training_step(self, batch, batch_idx):
        query_image = batch["query_image"].unsqueeze(1)  # [B, 1, 3, H, W]
        query_pose = batch["query_pose"]  # [B, 4, 4]
        candidate_images = batch["candidate_images"]  # [B, K, 3, H, W]

        B, K = candidate_images.shape[:2]
        pipeline = self.pipeline
        image_h, image_w = query_image.shape[-2], query_image.shape[-1]

        # Stage 1: Retrieval — get query descriptor
        _, aux_feats, _, _ = pipeline._run_backbone_single(query_image)
        query_descriptor = pipeline._run_vpr_branch(aux_feats, image_h, image_w)

        # Get candidate descriptors (no grad for frozen backbone)
        with torch.no_grad():
            cand_descs = []
            for k in range(K):
                cand_input = candidate_images[:, k:k+1]
                _, cand_aux, _, _ = pipeline._run_backbone_single(cand_input)
                cand_desc = pipeline._run_vpr_branch(cand_aux, image_h, image_w)
                cand_descs.append(cand_desc)
            all_cand_descs = torch.cat(cand_descs, dim=0)  # [B*K, D]

        # Select top-M candidates
        M = min(self.pose_top_m, K)
        sims = F.cosine_similarity(
            query_descriptor[0].unsqueeze(0), all_cand_descs[:K], dim=-1,
        )
        topm_indices = sims.topk(M).indices  # [M]
        selected_cands = candidate_images[:, topm_indices]  # [B, M, 3, H, W]

        # Stage 2: Multi-view pose
        multi_view = torch.cat([query_image, selected_cands], dim=1)  # [B, 1+M, 3, H, W]
        feats, _, _ = pipeline._run_backbone_multiview(multi_view)

        # cam_dec on all views, but loss only on query (view 0)
        camera_tokens = feats[-1][1]  # [B, 1+M, C]
        pose_enc = pipeline.cam_dec(camera_tokens)  # [B, 1+M, 9]

        # Decode query pose (view 0)
        query_pose_enc = pose_enc[:, :1, :]  # [B, 1, 9]
        c2w, _ixt = pose_encoding_to_extri_intri(query_pose_enc, (image_h, image_w))

        pred_R = c2w[:, 0, :3, :3]  # [B, 3, 3]
        pred_t = c2w[:, 0, :3, 3]   # [B, 3]

        gt_R = query_pose[:, :3, :3]
        gt_t = query_pose[:, :3, 3]

        rot_loss = geodesic_rotation_loss(pred_R, gt_R)
        trans_loss = translation_loss(pred_t, gt_t)
        total_loss = self.rotation_weight * rot_loss + self.translation_weight * trans_loss

        self.log("train/rot_loss", rot_loss, prog_bar=True)
        self.log("train/trans_loss", trans_loss, prog_bar=True)
        self.log("train/total_loss", total_loss, prog_bar=True)

        return total_loss

    def on_validation_epoch_start(self):
        self._val_rerrs = []
        self._val_terrs = []

    def validation_step(self, batch, batch_idx):
        query_image = batch["query_image"].unsqueeze(1)
        query_pose = batch["query_pose"]
        candidate_images = batch["candidate_images"]

        B, K = candidate_images.shape[:2]
        M = min(self.pose_top_m, K)

        # Use top-M candidates (by index order, no retrieval needed in val)
        selected_cands = candidate_images[:, :M]  # [B, M, 3, H, W]

        output = self.pipeline.pose_only(query_image, selected_cands, pose_path="cam_dec")

        pose_enc = output.pose_enc  # [B, 1+M, 9]
        image_size = (query_image.shape[-2], query_image.shape[-1])
        c2w, _ixt = pose_encoding_to_extri_intri(pose_enc[:, :1, :], image_size)

        pred_R = c2w[:, 0, :3, :3].cpu().numpy()
        pred_t = c2w[:, 0, :3, 3].cpu().numpy()
        gt_R = query_pose[:, :3, :3].cpu().numpy()
        gt_t = query_pose[:, :3, 3].cpu().numpy()

        for i in range(pred_R.shape[0]):
            rerr = get_rot_err(pred_R[i], gt_R[i])
            terr = np.linalg.norm(pred_t[i] - gt_t[i])
            self._val_rerrs.append(rerr)
            self._val_terrs.append(terr)

    def on_validation_epoch_end(self):
        if not self._val_rerrs:
            return
        med_rerr = np.median(self._val_rerrs)
        med_terr = np.median(self._val_terrs)
        self.log("val/rot_err_deg", med_rerr, prog_bar=True)
        self.log("val/trans_err_m", med_terr, prog_bar=True)
        print(
            f"\n[Epoch {self.current_epoch}] Val median pose error: "
            f"{med_terr:.4f} m  {med_rerr:.2f} deg  "
            f"({len(self._val_rerrs)} queries)",
            flush=True,
        )

    def configure_optimizers(self):
        opt_config = self.train_config["optimizer"]
        params = [p for p in self.pipeline.parameters() if p.requires_grad]
        if not params:
            raise ValueError("No trainable parameters found")

        if opt_config["name"] == "adamw":
            optimizer = torch.optim.AdamW(
                params, lr=opt_config["lr"],
                weight_decay=opt_config.get("weight_decay", 1e-4),
            )
        elif opt_config["name"] == "adam":
            optimizer = torch.optim.Adam(
                params, lr=opt_config["lr"],
                weight_decay=opt_config.get("weight_decay", 1e-4),
            )
        elif opt_config["name"] == "sgd":
            optimizer = torch.optim.SGD(
                params, lr=opt_config["lr"],
                weight_decay=opt_config.get("weight_decay", 1e-4),
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config['name']}")

        sched_config = self.train_config.get("scheduler", {})
        if sched_config.get("name") == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=sched_config.get("T_max", 50),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }

        return optimizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train unified pipeline v1.1")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", type=int, default=1)
    return parser.parse_args()


def _build_datasets(train_config):
    """Load scene entries and build UnifiedVislocDataset instances."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from eval_unified_visloc import load_scene_images_and_poses

    dataset_name = train_config["dataset"]
    sampling = train_config.get("candidate_sampling", {})
    image_size = tuple(train_config.get("image_size", [504, 504]))
    data_root = train_config.get("data_root", None)
    num_candidates = train_config.get("eval", {}).get("top_k", 10)

    def make_dataset(scene, split):
        entries = load_scene_images_and_poses(dataset_name, scene, split, data_root=data_root)
        return UnifiedVislocDataset(
            entries=entries,
            num_candidates=num_candidates,
            pos_threshold=sampling.get("pos_threshold", 1.0),
            neg_threshold=sampling.get("neg_threshold", 3.0),
            pos_ratio=sampling.get("pos_ratio", 0.7),
            distance_alpha=sampling.get("distance_alpha", 0.5),
            image_size=image_size,
        )

    return make_dataset


def build_scene_dataloaders(train_config):
    """Build train and val dataloaders."""
    make_dataset = _build_datasets(train_config)
    scenes = train_config["scenes"]
    batch_size = train_config["batch_size"]
    num_workers = train_config.get("num_workers", 8)

    train_datasets = [make_dataset(scene, "train") for scene in scenes]
    combined_train = torch.utils.data.ConcatDataset(train_datasets)
    train_loader = DataLoader(
        combined_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )

    eval_config = train_config.get("eval", {})
    val_scenes = eval_config.get("val_scenes", [scenes[0]])
    if isinstance(val_scenes, str):
        val_scenes = [val_scenes]

    val_datasets = [make_dataset(scene, "test") for scene in val_scenes]
    combined_val = torch.utils.data.ConcatDataset(val_datasets)
    val_loader = DataLoader(
        combined_val, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"Train: {len(combined_train)} samples from {len(scenes)} scenes")
    print(f"Val:   {len(combined_val)} samples from {val_scenes}")

    return train_loader, val_loader


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    config = load_config(args.config)
    train_config = config["training"]

    model = UnifiedPipelineLightningModule(config)
    train_loader, val_loader = build_scene_dataloaders(train_config)

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        monitor="val/trans_err_m",
        filename="unified_v11_{epoch:02d}_terr{val/trans_err_m:.4f}",
        auto_insert_metric_name=False,
        save_top_k=3,
        save_last=True,
        mode="min",
    )

    eval_config = train_config.get("eval", {})
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        max_epochs=train_config["max_epochs"],
        check_val_every_n_epoch=eval_config.get("eval_every_n_epoch", 1),
        callbacks=[checkpoint_cb],
        precision="16-mixed",
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        default_root_dir="./logs/unified_pipeline_v11/",
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume_from,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify no syntax errors**

Run: `cd /home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1 && python -c "import ast; ast.parse(open('train/train_unified_pipeline.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add train/train_unified_pipeline.py
git commit -m "feat: rewrite training script for v1.1 two-stage pipeline"
```

---

### Task 7: Update ablation scripts

**Files:**
- Modify: `ablation/eval_dinosalad_plus_ourpose.py`
- Modify: `ablation/eval_ourvpr_plus_reloc3r.py`

- [ ] **Step 1: Update `ablation/eval_dinosalad_plus_ourpose.py`**

The pose path now uses multi-view backbone via `pipeline.pose_only()`. The existing code at lines 78-91 already does this correctly — `pipeline.pose_only(query_input, cand_images)` concatenates views and runs backbone. Only change needed: the `pose_only` call now accepts `pose_path` parameter.

Replace lines 87-91:

Old:
```python
        output = pipeline.pose_only(query_input, cand_images)
```

New:
```python
        output = pipeline.pose_only(query_input, cand_images, pose_path="cam_dec")
```

- [ ] **Step 2: Update `ablation/eval_ourvpr_plus_reloc3r.py`**

The `build_database` function in v1.1 returns `(db_descriptors, db_image_paths)` instead of `(desc_mmap, patch_mmap, cam_mmap)`.

Replace line 118:

Old:
```python
    db_descriptors, _, _ = build_database(pipeline, db_entries, args.device, args.batch_size, target_size)
```

New:
```python
    from eval_unified_visloc_v11 import build_database as build_database_v11
    db_descriptors, _ = build_database_v11(pipeline, db_entries, args.device, args.batch_size, target_size)
```

Also update the import at line 23-27 to not import old `build_database`:

Old:
```python
from eval_unified_visloc import (
    build_database,
    load_scene_images_and_poses,
    preprocess_image,
)
```

New:
```python
from eval_unified_visloc import (
    load_scene_images_and_poses,
    preprocess_image,
)
from eval_unified_visloc_v11 import build_database
```

And update line 118:

Old:
```python
    db_descriptors, _, _ = build_database(pipeline, db_entries, args.device, args.batch_size, target_size)
```

New:
```python
    db_descriptors, _ = build_database(pipeline, db_entries, args.device, args.batch_size, target_size)
```

- [ ] **Step 3: Verify no syntax errors**

Run: `cd /home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1 && python -c "import ast; ast.parse(open('ablation/eval_dinosalad_plus_ourpose.py').read()); ast.parse(open('ablation/eval_ourvpr_plus_reloc3r.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add ablation/eval_dinosalad_plus_ourpose.py ablation/eval_ourvpr_plus_reloc3r.py
git commit -m "fix: update ablation scripts for v1.1 pipeline interface"
```

---

### Task 8: Run training-free baseline

**Files:**
- No file changes — this is a validation run

- [ ] **Step 1: Run training-free baseline on heads scene**

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1
python eval_unified_visloc_v11.py \
    --config configs/eval_unified_v11.yaml \
    --dataset 7scenes \
    --scene heads \
    --pose-path both \
    --pose-top-m 3 \
    --device cuda
```

Expected: script runs to completion, prints median rotation and translation errors for both cam_dec and ray paths. Check:
- No NaN or crashes
- Results are printed for both paths
- Note the baseline numbers for comparison with future training

- [ ] **Step 2: Record baseline results**

Save the output numbers. These are the reference for whether training improves things.
