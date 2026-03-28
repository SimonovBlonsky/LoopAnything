# DA3 Dual-Branch VPR Adapter Design

## Goal
Add a lightweight retrieval adapter on top of frozen `DA3-BASE` layer-5 retrieval features so that the VPR path becomes:

- `DA3 layer-5 aux feature`
- `Dual-Branch Adapter`
- `SALAD aggregator`

The training protocol for this experiment is:

- freeze the full DA3 backbone
- train only the adapter and SALAD aggregator
- initialize SALAD from `dino_salad_512_32.ckpt`
- train on `GSVCitiesDataModule`
- validate with the existing VPR recall pipeline

The purpose is to test whether a small retrieval-specific adapter can recover part of the remaining gap between:

- `DA3 layer-5 + frozen backbone + retrained SALAD`
- `DINO frozen retrieval baselines`
- `DINO partially fine-tuned SALAD`

without unfreezing the DA3 backbone.

## Motivation
The current results support three conclusions:

1. DA3 layer-5 features already carry useful retrieval signal.
2. The remaining gap is not primarily caused by the SALAD aggregator alone.
3. A likely bottleneck is that the retrieval branch is forced to consume frozen shared features with no retrieval-specific adaptation before SALAD.

This suggests the next minimal experiment should not unfreeze DA3 yet. Instead, it should add a small learnable adapter between the DA3 encoder and SALAD so that:

- the shared DA3 representation remains unchanged
- the retrieval branch gains a small amount of task-specific capacity
- the comparison to the frozen-backbone protocol stays clean

## Scope
This work targets the existing DA3 VPR path in:

- `src/depth_anything_3/model/vpr_model.py`
- `src/depth_anything_3/model/vpr_helper.py`
- `src/depth_anything_3/model/vpr_encoder_adapter.py`
- `train/train_salad_aggregator.py`

It may introduce new retrieval modules under `src/depth_anything_3/model/`, but it should not refactor the DA3 backbone itself.

This work should not:

- modify the DA3 backbone weights during training
- modify SALAD internals
- replace the existing retrieval contract used by the DA3 VPR stack

## Design Principles

### 1. Preserve the current retrieval contract
The DA3 encoder path already returns a stable feature dictionary:

- `patch_tokens`
- `feature_map`
- `global_token`
- `spatial_shape`

The adapter should accept and return the same structure. This keeps the rest of the VPR stack unchanged and makes ablations easy to compare.

### 2. Adapt local and global retrieval signals separately
SALAD consumes both:

- a local feature map
- a global token

The adapter should therefore not be a single local-only projection. It should provide separate adaptation capacity for:

- local spatial retrieval features
- the global token branch

This is the main reason to choose a dual-branch design instead of a single `1x1` projection.

### 3. Stay lightweight but not minimalistic
The user selected an indicator-priority setting, not a strict parameter-minimization setting. The adapter should therefore remain small relative to the backbone, but it may use:

- channel bottlenecks
- depthwise spatial mixing
- residual updates
- global-token conditioning on pooled local context

if they are likely to improve retrieval quality.

### 4. Keep the backbone fully frozen
This experiment is specifically meant to isolate the value of a retrieval-specific adapter under a frozen DA3 backbone.

The training wrapper must enforce:

- no DA3 backbone parameters trainable
- only adapter and aggregator parameters passed to the optimizer

## Architecture

### 1. Retrieval Data Flow
The updated DA3 retrieval path should be:

1. `DA3EncoderAdapter` extracts `layer-5` auxiliary features.
2. The adapter receives the DA3 feature dictionary.
3. The adapter outputs an adapted feature dictionary with the same keys.
4. `VPRModel` feeds the adapted features into SALAD.

This keeps the current encoder and aggregator boundaries intact while inserting one retrieval-specific transformation stage.

### 2. Adapter Placement
The adapter should sit between:

- `encoder`
- `aggregator`

inside the generic DA3 `VPRModel`.

`VPRModel` should become:

- `encoder -> optional feature_adapter -> aggregator`

If no adapter is configured, the default path should remain equivalent to the current implementation.

### 3. Dual-Branch Adapter
The adapter should operate on the DA3 feature dictionary and produce a new one with the same schema.

#### Local Branch
Input:

- `feature_map` with shape `[B, C, H, W]`

Recommended structure:

- channel-last layer normalization
- `1x1` reduction from `C=768` to a bottleneck width
- `GELU`
- depthwise `3x3` convolution
- `1x1` expansion back to `768`
- residual add with the original feature map

Recommended default bottleneck width:

- `192` or `256`

The local branch is responsible for learning retrieval-specific local corrections before SALAD clustering.

#### Global Branch
Input:

- original `global_token`
- global average pooled adapted local map

The branch should concatenate these two signals and learn a token correction.

Recommended structure:

- concatenate `[global_token, pooled_local]` to size `1536`
- `LayerNorm`
- `Linear(1536 -> 256)`
- `GELU`
- `Linear(256 -> 768)`
- residual-style update against the original global token

This branch is designed to give SALAD a stronger scene token than the raw DA3 layer-5 token alone.

### 4. Output Contract
The adapter output must remain:

- `patch_tokens`
- `feature_map`
- `global_token`
- `spatial_shape`

Suggested behavior:

- local branch produces the adapted `feature_map`
- `patch_tokens` are regenerated from the adapted `feature_map`
- global branch produces the adapted `global_token`
- `spatial_shape` is passed through unchanged

This keeps all downstream code compatible.

## Module Layout

### 1. `src/depth_anything_3/model/vpr_feature_adapter.py`
Create a new module containing:

- `IdentityFeatureAdapter`
- `DualBranchFeatureAdapter`

Responsibilities:

- operate on the DA3 retrieval feature dict
- preserve output schema
- validate finite outputs
- support residual initialization

### 2. `src/depth_anything_3/model/vpr_model.py`
Extend the VPR assembly module so it can optionally own:

- `encoder`
- `feature_adapter`
- `aggregator`

If no adapter is passed, it should default to identity behavior.

### 3. `src/depth_anything_3/model/vpr_helper.py`
Add helper assembly support for:

- `build_feature_adapter(adapter_arch, adapter_config=None)`
- `build_vpr_model(..., feature_adapter_arch=None, feature_adapter_config=None, ...)`

This keeps model construction centralized and makes training/evaluation code consistent.

### 4. `train/train_da3_adapter_salad.py`
Create a new training entrypoint rather than overloading `train_salad_aggregator.py`.

Responsibilities:

- build the frozen DA3 layer-5 encoder path
- build the dual-branch adapter
- build SALAD
- load SALAD checkpoint weights
- enforce adapter-only plus aggregator-only optimization
- reuse the current loss, miner, scheduler, datamodule, and validation pipeline

The existing `train_salad_aggregator.py` should remain as the adapter-free baseline.

## Training Protocol

### 1. Default Experiment
The first default run should be:

- DA3 model: `DA3-BASE`
- feature source: `aux`
- aux layer: `5`
- adapter: `dual_branch`
- aggregator: `SALAD`
- SALAD config:
  - `num_channels=768`
  - `num_clusters=16`
  - `cluster_dim=32`
  - `token_dim=32`
- SALAD init:
  - `da3_streaming/loop_utils/salad/weights/dino_salad_512_32.ckpt`
- trainable modules:
  - adapter
  - aggregator
- frozen modules:
  - full DA3 backbone

### 2. Optimization
Recommended defaults:

- optimizer: `AdamW`
- adapter learning rate: `1e-4`
- aggregator learning rate: `6e-5`
- weight decay: `9.5e-9`
- scheduler:
  - `LinearLR`
  - `start_factor=1.0`
  - `end_factor=0.2`
  - `total_iters=4000`

If per-module learning rates are inconvenient in the first implementation, a single shared learning rate may be used initially, but separate parameter groups are preferred.

### 3. Initialization
Recommended adapter initialization:

- zero-initialize the final projection in each branch
- keep the initial adapter behavior close to identity

This reduces startup instability and makes the adapter act like a small residual correction instead of a full feature rewrite from iteration 0.

## CLI Surface
The new training script should expose at least:

- `--da3-model-name-or-path`
- `--feature-source`
- `--aux-layer`
- `--feature-adapter-arch`
- `--adapter-local-bottleneck`
- `--adapter-global-hidden-dim`
- `--agg-arch`
- `--agg-num-clusters`
- `--agg-cluster-dim`
- `--agg-token-dim`
- `--aggregator-ckpt-path`
- `--train-aggregator-only`
- `--train-adapter-only`

The script should print startup diagnostics including:

- DA3 model name
- selected aux layer
- adapter architecture
- whether DA3 is frozen
- total parameters
- trainable parameters
- trainable parameter counts for:
  - adapter
  - aggregator
  - encoder

## Validation
Validation should reuse the current DA3/SALAD training protocol:

- `GSVCitiesDataModule`
- `pitts30k_val`
- `pitts30k_test`
- the existing recall computation utilities

The main model-selection metric should remain:

- `pitts30k_val/R1`

## Ablations
The following experiment table is recommended.

### Core Ablations
1. `DA3 layer-5 + SALAD`
   - no adapter

2. `DA3 layer-5 + Patch-Only Adapter + SALAD`
   - local branch only

3. `DA3 layer-5 + Dual-Branch Adapter + SALAD`
   - recommended main experiment

4. `DA3 layer-5 + Dual-Branch Adapter + SALAD (no SALAD warm start)`
   - tests how much gain comes specifically from the adapter versus pretrained SALAD initialization

### Optional Token Ablations
5. `Dual-Branch Adapter` with global branch using only original `global_token`
6. `Dual-Branch Adapter` with global branch using `concat(global_token, pooled_local)`

The second of these should be treated as the recommended default.

## Success Criteria
The experiment should be considered successful if it shows a meaningful improvement over:

- the current `DA3 layer-5 + SALAD` frozen-backbone baseline

without:

- unfreezing DA3
- changing the backbone feature source
- changing the SALAD architecture itself

The main empirical question is whether this adapter closes a substantial portion of the frozen-protocol gap while preserving the unified DA3 trunk.

## Error Handling
The new training path should fail early for:

- non-finite adapted features
- non-finite descriptors
- shape mismatches between patch tokens and feature maps
- trainable backbone parameters when the backbone is meant to be frozen
- empty or incompatible SALAD checkpoint loads

## Non-Goals
This spec does not cover:

- unfreezing any DA3 backbone blocks
- LoRA or other parameter-efficient backbone tuning
- multi-layer DA3 feature fusion
- joint retrieval plus pose multi-task training

Those should remain follow-up experiments after this adapter-only retrieval study.
