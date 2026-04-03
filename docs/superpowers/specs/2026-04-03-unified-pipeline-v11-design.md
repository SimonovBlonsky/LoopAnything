# Unified Visual Localization Pipeline v1.1 Design Spec

## Problem Statement

v1.0 of the unified pipeline uses a post-backbone `CrossViewFusion` module (2-layer cross-attention) to fuse query and candidate features. This fundamentally cannot replicate what DA3's backbone does internally: **alternate attention** across views at every other transformer layer from `alt_start` onward. The result is that the camera token fed to `cam_dec` lacks multi-view geometric information, making absolute pose regression impossible regardless of training.

v1.1 replaces the external fusion module with DA3's native multi-view alternate attention by feeding query + candidates **together** into the backbone.

## Architecture Overview

Two-stage pipeline sharing a single DA3 backbone (same weights, two forward passes):

```
Stage 1 — Retrieval:
  Query [B,1,3,H,W] → DA3 Backbone (single-view)
    → aux_feats (layer aux_layer) → Adapter → SALAD → query_descriptor [B,D]
    → cosine similarity vs database → top-K → select top-M candidates

Stage 2 — Pose Estimation:
  [Query, top-M candidates] [B, 1+M, 3, H, W] → DA3 Backbone (multi-view)
    → alternate attention (local + global layers) → feats per view
    → cam_dec(query_camera_token) → pose_enc → c2w (training path, differentiable)
    → da3_head(feats) → ray → get_extrinsic_from_camray → c2w (inference path, optional)
```

### Key Changes from v1.0

| Aspect | v1.0 | v1.1 |
|--------|------|------|
| Multi-view fusion | External CrossViewFusion (2-layer cross-attn) | DA3 backbone alternate attention (native) |
| Pose input | `cam_dec(enhanced_cam)` — single-view camera token + shallow fusion | `cam_dec(camera_token)` — camera token after deep multi-view interaction in backbone |
| Stage 2 input | Pre-extracted patch_tokens + camera_tokens | Raw candidate images (must re-run backbone) |
| Database storage | descriptor + patch_tokens + camera_tokens (~8MB/image) | descriptor + image path (~KB/image) |
| Training-free baseline | Not possible | Possible (all components have pretrained weights) |

## Pipeline Modules

| Module | Source | Stage | Trainable (default) |
|--------|--------|-------|---------------------|
| `da3_backbone` | DA3 pretrained | 1 + 2 | Frozen |
| `feature_adapter` | VPR pretrained | 1 | Trainable |
| `aggregator` (SALAD) | VPR pretrained | 1 | Trainable |
| `retrieval_strategy` | Reused from v1.0 | 1 | N/A (no params) |
| `da3_head` | DA3 pretrained | 2 (inference only) | Frozen |
| `cam_dec` | DA3 pretrained | 2 | Trainable |

Removed: `cross_view_fusion`

## Calling Modes

```python
# Mode 1: Full pipeline — retrieval + pose
output = pipeline.forward(query_image, candidate_images)
# query_image: [B, 1, 3, H, W]
# candidate_images: [B, K, 3, H, W]
# Returns: output.pose_enc, output.query_descriptor, output.extrinsics, ...

# Mode 2: Retrieval only — extract VPR descriptors
descriptors = pipeline.retrieval_only(images)
# images: [B, 1, 3, H, W]
# Returns: descriptors [B, D]

# Mode 3: Pose only — given pre-selected candidates, multi-view backbone forward
output = pipeline.pose_only(query_image, candidate_images)
# Concatenates [query, candidates] → backbone multi-view → cam_dec → pose
# Returns: output.pose_enc, output.extrinsics, output.intrinsics
```

### VPR Branch Configuration

- `aux_layer: int` (configurable, default 5) — which backbone layer's features to use for VPR
- Stage 1 backbone forward passes `export_feat_layers=[aux_layer]`
- Exported aux_feats → `feature_adapter` → `aggregator` → global descriptor

### Retrieval Configuration

- `retrieval_top_k: int` (default 10) — number of candidates from database similarity search
- `pose_top_m: int` (default 3, M <= K) — number of candidates fed into Stage 2 backbone
- Retrieval strategy: `SoftAttentionRetrieval` (training), hard top-K mask (eval)

## Milestone 0: Training-Free Baseline

Before any training code, validate the architecture with pure pretrained weights.

### Procedure

1. Load pretrained VPR model (adapter + SALAD) and original DA3 model (backbone + head + cam_dec)
2. On 7scenes `heads` scene:
   a. Offline: all train images → `retrieval_only()` → VPR descriptors → build database
   b. Per test query: descriptor → cosine similarity → top-M candidates
   c. `[query, top-M candidates]` as `[1, 1+M, 3, H, W]` → DA3 forward (multi-view)
   d. Extract pose via both cam_dec path and ray path
   e. Compare with GT pose → rotation error (deg) + translation error (m)

### Success Criteria

- Runs without errors (validates multi-view backbone concatenation logic)
- Report median rotation/translation error for both cam_dec and ray paths
- Sub-meter accuracy → architecture direction confirmed
- Poor accuracy but no crashes → architecture viable, training needed
- NaN or crashes → concatenation logic needs debugging

### Configuration

```yaml
eval:
  pose_top_m: 3
  retrieval_top_k: 10
  scenes: [heads]
  pose_path: "both"   # cam_dec + ray
```

## Training

### Loss

- `geodesic_rotation_loss(pred_R, gt_R)` — geodesic distance between rotation matrices
- `translation_loss(pred_t, gt_t)` — L2 distance
- Total: `rotation_weight * rot_loss + translation_weight * trans_loss`
- Only query pose supervised (index 0 of multi-view output); candidate poses ignored

### Training Forward Flow

1. Query single-view backbone → aux_feats → VPR descriptor (gradient if VPR trainable)
2. Database descriptors (pre-extracted, no grad)
3. Retrieval → top-K → select top-M indices
4. Load top-M candidate original images
5. `[query, top-M candidates]` → backbone multi-view forward (gradient depends on freeze config)
6. `cam_dec(camera_tokens[:, 0])` → query pose_enc → decode → c2w
7. Pose loss vs GT

### Freeze Strategy

All configurable via YAML. Default values for initial experiments:

```yaml
freeze:
  backbone: true
  backbone_from_layer: -1    # -1 = all frozen; set to N to freeze layers [0,N) and unfreeze layers [N,end]
  vpr: false                 # adapter + SALAD trainable
  da3_head: true             # frozen, not used in training loss
  cam_dec: false             # trainable, pose output
```

Freeze strategy is explicitly left for ablation experiments. The config exposes full control.

### Optimizer / Scheduler

- AdamW (lr, weight_decay configurable)
- CosineAnnealing (T_max configurable)
- Gradient clipping: norm=1.0
- Mixed precision: 16-mixed

## Evaluation Pipeline

Shared logic with training-free baseline, differing only in loaded weights.

### Flow

1. Offline: all train images → `retrieval_only()` → VPR descriptors → store
2. Per test query:
   a. `retrieval_only()` → query descriptor
   b. Cosine similarity → top-K → select top-M candidate indices
   c. Load top-M candidate original images
   d. `pose_only(query, candidates)` → backbone multi-view → pose
   e. Report cam_dec result; optionally ray path result
3. Aggregate: median rotation error, median translation error

### Storage

Database stores only VPR descriptors and image paths. No patch_tokens or camera_tokens needed (candidates go through backbone at inference time).

## File Changes

### New Files

| File | Purpose |
|------|---------|
| `eval_unified_visloc_v11.py` | Training-free baseline + eval (unified script; `--checkpoint` flag distinguishes) |
| `configs/train_unified_stage1_v11.yaml` | v1.1 training config |
| `configs/eval_unified_v11.yaml` | v1.1 eval / baseline config |

### Modified Files

| File | Changes |
|------|---------|
| `src/depth_anything_3/model/unified_pipeline.py` | Rewrite `forward()`, `pose_only()`, `extract_database_features()`; remove cross_view_fusion dependency; add `pose_top_m` config |
| `src/depth_anything_3/model/unified_pipeline_helper.py` | Remove cross_view_fusion construction; update `_apply_freeze()` for `backbone_from_layer`; update `build_unified_pipeline()` |
| `train/train_unified_pipeline.py` | Two-stage training forward; remove fusion-related code and NaN debug for fusion |
| `ablation/eval_dinosalad_plus_ourpose.py` | Pose path: cross_view_fusion + cam_dec → multi-view backbone + cam_dec |
| `ablation/eval_ourvpr_plus_reloc3r.py` | Adapt to new `build_database` interface (descriptor + image paths only) |

### Deleted Files

| File | Reason |
|------|--------|
| `src/depth_anything_3/model/cross_view_fusion.py` | Replaced by backbone alternate attention |
