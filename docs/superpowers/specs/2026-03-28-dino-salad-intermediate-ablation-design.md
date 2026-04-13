# DINO SALAD Intermediate-Layer Ablation Design

> **SUPERSEDED (2026-04-06):** Earlier design iteration. Current work is tracked in [`2026-04-06-relpose-head-training-spec.md`](2026-04-06-relpose-head-training-spec.md).

## Goal
Build a general DINOv2 intermediate-layer ablation training framework for SALAD, with the first target experiment defined as:

- backbone: `dinov2_vitb14`
- feature source: `layer 5`
- backbone mode: frozen
- training scope: aggregator only
- aggregator config: `num_channels=768`, `num_clusters=16`, `cluster_dim=32`, `token_dim=32`
- data: `GSVCitiesDataModule`
- validation: `pitts30k_val`, `pitts30k_test`

The purpose is to isolate whether the remaining gap versus DA3 comes from the DA3 backbone itself or from the frozen mid-layer retrieval protocol.

## Scope
This work introduces a new ablation-only training path under:

- `da3_streaming/loop_utils/salad/ablations/train_ablations.py`
- `da3_streaming/loop_utils/salad/ablations/models/`

It may reuse existing SALAD dataloaders, aggregators, losses, miners, and validation utilities, but it should not replace or refactor the original SALAD training path in:

- `da3_streaming/loop_utils/salad/main.py`
- `da3_streaming/loop_utils/salad/vpr_model.py`
- `da3_streaming/loop_utils/salad/models/backbones/`

The ablation path should remain isolated so that the original SALAD baseline stays unchanged.

## Design Principles

### 1. Match the DA3 ablation contract
The DINO ablation path should use the same high-level retrieval contract as the DA3 path:

- encoder returns a feature dictionary
- retrieval aggregation consumes `feature_map` plus `global_token`
- the training wrapper decides which parameters are trainable

This keeps the DINO and DA3 experiments comparable. The main variable should be the encoder feature source, not the VPR plumbing.

### 2. Make the backbone interface general
The new DINO encoder should support both final-layer and intermediate-layer retrieval experiments through a single interface:

- `output_layer=-1` means final DINO output
- `output_layer in [0, num_blocks-1]` means a specific transformer block output

This allows one framework to cover:

- original-style final-output experiments
- layer sweeps
- the frozen layer-5 fair-control experiment

### 3. Keep ablation logic separate from production logic
The ablation framework may duplicate a small amount of code instead of adding conditionals to the original SALAD trainer. This is preferred because it reduces coupling and makes experiment protocol inspection easier.

## File Layout

### 1. `da3_streaming/loop_utils/salad/ablations/models/dinov2_intermediate.py`
Defines a reusable DINOv2 encoder module for retrieval ablations.

Responsibilities:

- load `facebookresearch/dinov2` backbones through `torch.hub`
- expose both final and intermediate transformer block outputs
- return a feature dictionary with:
  - `patch_tokens`
  - `feature_map`
  - `global_token`
  - `spatial_shape`
- support freezing the full backbone
- support training the last `N` blocks when desired
- keep the backbone in `eval` when fully frozen
- validate shapes and finiteness

### 2. `da3_streaming/loop_utils/salad/ablations/models/vpr_model.py`
Defines a lightweight VPR assembly module modeled after the DA3 retrieval stack.

Responsibilities:

- hold `encoder`, `aggregator`, and `agg_arch`
- call `encoder(x, **kwargs)`
- feed SALAD with `(feature_map, global_token)`
- feed non-SALAD aggregators with `feature_map`
- optionally return both descriptors and feature dictionaries
- fail fast on non-finite descriptors

This is intentionally a plain `nn.Module`, not a Lightning module.

### 3. `da3_streaming/loop_utils/salad/ablations/models/vpr_helper.py`
Defines helper functions to build the ablation model.

Responsibilities:

- build the intermediate-layer DINO encoder
- build the aggregator from existing SALAD aggregator classes
- load aggregator-only weights from `dino_salad_512_32.ckpt`
- return a fully assembled ablation `VPRModel`

This file should mirror the DA3 helper style rather than the original SALAD helper style.

### 4. `da3_streaming/loop_utils/salad/ablations/train_ablations.py`
Defines the ablation training entrypoint.

Responsibilities:

- parse experiment arguments
- build the datamodule
- build the ablation VPR model
- wrap it in a Lightning training module
- configure optimizer, scheduler, mining, and validation
- print startup diagnostics that confirm the exact training protocol

## Encoder Contract
The DINO encoder should accept images shaped `[B, 3, H, W]` where `H` and `W` are divisible by 14.

The returned feature dictionary must contain:

- `patch_tokens`: `[B, HW, C]`
- `feature_map`: `[B, C, H/14, W/14]`
- `global_token`: `[B, C]`
- `spatial_shape`: `(H/14, W/14)`

For intermediate layers:

- compute tokens by running DINO blocks up to `output_layer`
- optionally apply `model.norm` if `norm_layer=True`
- use `tokens[:, 0]` as `global_token`
- use `tokens[:, 1:]` as patch tokens

For final output:

- the same contract applies, using the normal final DINO token sequence

This keeps the retrieval descriptor interface identical across final-layer and intermediate-layer experiments.

## Training Wrapper
The ablation training path should define a dedicated Lightning module inside `train_ablations.py`.

Responsibilities:

- hold the ablation `VPRModel`
- reuse original SALAD utilities:
  - `MultiSimilarityLoss`
  - `MultiSimilarityMiner`
  - `LinearLR`
  - Pitts recall validation
- support two optimization modes:
  - `train_aggregator_only=True`
  - `train_aggregator_only=False`

When `train_aggregator_only=True`:

- only aggregator parameters are passed to the optimizer
- startup checks must verify that no encoder parameters remain trainable

When `freeze_backbone=True`:

- encoder parameters must all have `requires_grad=False`
- encoder must remain in `eval`

## First Experiment Defaults
The default CLI and construction path should match the intended fair-control experiment:

- `backbone_arch='dinov2_vitb14'`
- `output_layer=5`
- `freeze_backbone=True`
- `num_trainable_blocks=0`
- `train_aggregator_only=True`
- `agg_arch='SALAD'`
- `agg_config={'num_channels': 768, 'num_clusters': 16, 'cluster_dim': 32, 'token_dim': 32}`
- `agg_ckpt_path='da3_streaming/loop_utils/salad/weights/dino_salad_512_32.ckpt'`
- `lr=6e-5`
- `optimizer='adamw'`
- `weight_decay=9.5e-9`
- `lr_sched='linear'`
- `lr_sched_args={'start_factor': 1, 'end_factor': 0.2, 'total_iters': 4000}`

This makes the first run directly comparable to the DA3 layer-5 frozen-backbone aggregator-only experiment.

## CLI Surface
`train_ablations.py` should expose a compact but general CLI.

Required experiment controls:

- `--backbone-arch`
- `--backbone-layer`
- `--freeze-backbone`
- `--num-trainable-blocks`
- `--train-aggregator-only`
- `--agg-arch`
- `--agg-ckpt-path`
- `--agg-num-channels`
- `--agg-num-clusters`
- `--agg-cluster-dim`
- `--agg-token-dim`

The script should print startup diagnostics including:

- backbone arch
- selected output layer
- whether the backbone is frozen
- how many blocks are trainable
- whether training is aggregator-only
- total parameters
- trainable parameters
- optimizer parameter count

## Validation
Validation should match the existing SALAD protocol:

- use `GSVCitiesDataModule`
- validate on `pitts30k_val` and `pitts30k_test`
- compute descriptors in dataset order
- split reference/query sets using existing metadata
- compute recall through `utils.get_validation_recalls`
- checkpoint on `pitts30k_val/R1`

## Error Handling
The ablation path should fail early for:

- invalid `output_layer`
- non-divisible spatial image shapes
- malformed token counts that cannot reshape into a feature map
- non-finite encoder features
- non-finite descriptors
- missing aggregator checkpoint
- missing aggregator-prefixed keys in checkpoint
- `train_aggregator_only=True` while encoder parameters still require gradients

## Testing Plan
Implementation verification should include:

### Unit tests
- encoder returns correct feature-dict shapes for an intermediate layer
- encoder returns final-layer features when `output_layer=-1`
- frozen encoder parameters all have `requires_grad=False`
- frozen encoder remains in `eval`
- optimizer excludes encoder parameters in aggregator-only mode
- ablation `VPRModel` routes SALAD inputs through `(feature_map, global_token)`

### Smoke checks
- ablation training script constructs successfully
- one synthetic forward pass returns finite descriptors
- one optimizer construction check confirms correct parameter counts

Full training is not required for implementation completion, but the new ablation entrypoint should be runnable with the existing data and trainer stack.

## Rationale
The point of this framework is not just to run one layer-5 experiment. It is to create a controlled DINOv2 ablation path that mirrors the DA3 retrieval protocol closely enough to answer:

1. what performance ceiling comes from using frozen mid-layer retrieval features with a retrained SALAD head
2. whether DA3 underperforms because of its backbone representation, rather than because the retrieval head or training recipe is mismatched

By holding the retrieval head, optimizer family, dataset, and validation protocol constant while changing only the encoder feature source, this ablation framework provides a cleaner answer than reusing the original end-to-end SALAD trainer.
