# DA3 SALAD Aggregator Retraining Design

## Goal
Train a new SALAD aggregator on top of a frozen DA3 backbone, using DA3-BASE layer-5 features for visual place recognition. The experiment must keep the SALAD aggregator architecture unchanged, reuse the existing `dino_salad_512_32.ckpt` as initialization, and modify only `LoopAnything/train/train_salad_aggregator.py`.

## Scope
This design only covers a training script change. It does not modify:
- `src/depth_anything_3/model/vpr_model.py`
- `src/depth_anything_3/model/vpr_helper.py`
- any SALAD aggregator implementation
- dataloaders or dataset paths
- DA3 model code

The objective is to answer a narrow question: can a retrained SALAD aggregator adapt to frozen DA3 layer-5 features well enough to recover a large portion of the gap versus original DINO SALAD?

## Fixed Experimental Definition
The training setup is fixed as follows:
- Backbone: `DA3-BASE` loaded via `DepthAnything3.from_pretrained("depth-anything/DA3-BASE")`
- Backbone weights: frozen for the entire run
- Retrieval feature source: DA3 transformer `aux layer 5`
- Local branch input: normalized layer-5 AUX patch tokens reshaped into a `768`-channel SALAD feature map
- Token branch input: normalized layer-5 DA3 `cam_token` from the same layer, used in place of DINO CLS token
- Aggregator initialization: `da3_streaming/loop_utils/salad/weights/dino_salad_512_32.ckpt`
- Aggregator config: `num_channels=768`, `num_clusters=16`, `cluster_dim=32`, `token_dim=32` to match `dino_salad_512_32.ckpt`
- Validation sets: `pitts30k_val`, `pitts30k_test`
- Data module: `GSVCitiesDataModule` with its existing hard-coded paths

This keeps the experiment aligned with prior analysis: DA3 layer 5 is the strongest single retrieval layer, and the token branch should use the DA3 camera token rather than a zero token or patch pooling.

## Architecture
The training script will define two local modules.

### 1. `DA3Layer5CamTokenEncoder`
A lightweight `nn.Module` used only inside the training script.

Responsibilities:
- load `DA3-BASE`
- freeze all DA3 parameters
- call the DA3 transformer path that exposes the full normalized token sequence for `aux layer 5` before special-token removal
- explicitly avoid the DA3 final exported feature path with `cat_token=True`, because that path produces `1536`-dim features and would not match the chosen SALAD checkpoint
- split the normalized layer-5 sequence into:
  - `global_token = token[:, 0]` which is DA3 `cam_token`
  - `patch_tokens = token[:, 1:]`
- reshape patch tokens into `[B, C, H/14, W/14]`
- return a feature dictionary compatible with `src/depth_anything_3/model/vpr_model.py`

The encoder must not use the current `DA3EncoderAdapter` because that adapter discards special tokens in AUX mode and replaces the token branch with patch-token mean pooling, which does not match this experiment. The encoder must consume the `768`-dim AUX representation, not the `1536`-dim concatenated final export.

### 2. `DA3SALADLightningModule`
A lightweight `pl.LightningModule` wrapper defined in the training script.

Responsibilities:
- construct `VPRModel(encoder, aggregator, agg_arch="SALAD")`
- initialize the SALAD aggregator from `dino_salad_512_32.ckpt`
- keep encoder frozen and train only aggregator parameters
- reuse original SALAD training behavior:
  - `MultiSimilarityLoss`
  - `MultiSimilarityMiner`
  - linear LR schedule
  - Pitts recall validation
- log `loss`, `b_acc`, `R1`, `R5`, `R10`

## Training Flow
For each train batch:
1. flatten place batches from `[BS, N, C, H, W]` to `[BS*N, C, H, W]`
2. run frozen DA3 layer-5 encoder
3. run SALAD aggregator through current `VPRModel`
4. compute metric-learning loss with miner
5. backpropagate through aggregator only

For each validation set:
1. compute descriptors in original dataset order
2. split reference/query descriptors using existing Pitts metadata
3. compute recall via SALAD `get_validation_recalls`
4. checkpoint on `pitts30k_val/R1`

## Optimizer and Scheduler
Use the same default optimizer setup as original SALAD unless later experiments justify changes:
- optimizer: `AdamW`
- learning rate: `6e-5`
- weight decay: `9.5e-9`
- scheduler: `LinearLR`
- scheduler args:
  - `start_factor=1`
  - `end_factor=0.2`
  - `total_iters=4000`

This keeps the comparison focused on backbone-feature replacement rather than hyperparameter search.

## Initialization Rules
Aggregator initialization will load only aggregator-prefixed weights from the existing SALAD checkpoint. Backbone weights from the SALAD checkpoint are ignored.

The DA3 backbone comes only from `DepthAnything3.from_pretrained("depth-anything/DA3-BASE")`. A local Hugging Face snapshot path is acceptable only if it resolves to the same pretrained model through the same API. There is no weight sharing or checkpoint merging between DA3 and original SALAD backbone weights.

## Freezing Rules
All parameters under the DA3 encoder are frozen.
Only parameters under the SALAD aggregator are passed to the optimizer.

The script must explicitly verify this split when it starts, for example by printing:
- total parameters
- trainable parameters
- trainable parameter names or module groups

## Validation and Success Criteria
Primary validation goal:
- training remains numerically stable
- no NaNs in descriptors or loss
- recall improves substantially over current training-free DA3 + SALAD baselines

The first acceptance threshold is not “match original SALAD”. The initial success criterion is:
- clear improvement over the current training-free DA3 best result
- stable validation behavior on `pitts30k_val`
- non-collapsed descriptors

If this succeeds but remains well below original SALAD, the next stage would be retrieval-specific fine-tuning of a small subset of DA3 blocks near layer 5. That is out of scope for this script-only change.

## Error Handling
The script should fail early for:
- missing DA3 checkpoint or config
- missing SALAD checkpoint
- aggregator weight prefix mismatch
- non-finite descriptors or losses
- attempts to optimize frozen encoder parameters
- unexpected DA3 layer-5 token dimensionality

## Testing Plan
Implementation verification should include:
- import / construction smoke test for the training script
- assertion that encoder parameters are frozen and aggregator parameters are trainable
- one forward pass on a synthetic batch to verify descriptor shape and finite outputs
- one validation-step smoke test to confirm recall path still runs

Full training is not required to validate the design, but the script should be runnable end to end with the existing data module and trainer settings.

## Rationale
Current experiments show that:
- training-free DA3 layer-5 features carry meaningful retrieval information
- simple scale calibration helps, but does not close the gap to original SALAD
- therefore the remaining gap is due to representation mismatch between DA3 features and an aggregator trained on DINOv2-B outputs

Retraining only the aggregator is the cheapest and cleanest next experiment. It isolates whether the mismatch is mostly in the aggregator adaptation layer, while avoiding the added complexity of backbone fine-tuning.
