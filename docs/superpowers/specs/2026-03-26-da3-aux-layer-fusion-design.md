# DA3 AUX Layer Fusion Design

## 1. Goal

Add a training-free retrieval ablation to the existing DA3 + SALAD evaluation path by allowing multiple DA3 auxiliary transformer layers to be combined into a single feature map before entering the unchanged SALAD aggregator.

The immediate goal is experimental, not architectural expansion:

- keep the current DA3 VPR path working exactly as it does today
- keep the existing SALAD checkpoint reusable without retraining
- support simple, non-learned multi-layer fusion of DA3 AUX features
- evaluate whether fused AUX features improve place-recognition recall on SPEDTEST

This is explicitly an ablation feature. It should be easy to remove if it does not help.

## 2. Scope and Constraints

### In scope

- extend `src/depth_anything_3/model/vpr_encoder_adapter.py`
- extend `eval_da3_vpr.py`
- add tests for adapter fusion behavior and CLI argument wiring
- support training-free fusion modes that preserve feature dimensionality

### Out of scope

- modifying the SALAD aggregator implementation
- modifying SALAD checkpoints
- modifying `vpr_helper.py`
- modifying `vpr_model.py` beyond its current behavior
- adding learned fusion weights
- adding concat-based fusion that changes channel dimensions
- changing the recall computation or dataset logic
- integrating fusion logic into `debug_vpr_feature_maps.py` in this step

### Hard constraints

- existing SALAD checkpoints must remain loadable
- existing single-layer DA3 AUX evaluation must remain intact
- the default DA3 VPR behavior must remain unchanged unless the new fusion arguments are explicitly used

## 3. Existing Technical Facts

### Current DA3 VPR path

`eval_da3_vpr.py` constructs a DA3-backed `VPRModel` and currently configures its encoder with:

- `feature_source = final | aux`
- `aux_layer = N`

When `feature_source=aux`, the current adapter requests exactly one exported layer from the DA3 backbone and reshapes the returned patch tokens into a single feature map.

### Current DA3 AUX feature contract

The DA3 backbone returns AUX outputs through `export_feat_layers`. For the single-view retrieval case, the relevant AUX tensor contract is:

- `[B, 1, N, C]`

The current adapter removes the single-view dimension and treats the result as:

- patch tokens: `[B, N, C]`
- feature map: `[B, C, Hp, Wp]`

### Why fusion must preserve channels

The existing SALAD checkpoint expects a fixed `num_channels` input. Therefore concat-based fusion is intentionally excluded from this step because it would alter the expected channel width and break checkpoint compatibility.

## 4. Selected Design

The selected design is adapter-level AUX fusion.

Instead of building fusion logic in the evaluation script, the adapter will support exporting multiple AUX layers and combining them before feature-map construction. The evaluation script will only parse and forward fusion parameters.

This keeps the experiment reusable and keeps the evaluation script thin.

## 5. Component Design

### 5.1 `vpr_encoder_adapter.py`

#### New responsibilities

In addition to the current single-layer AUX path, the adapter will support:

- selecting multiple AUX layers
- combining their patch-token tensors using a training-free rule
- producing one fused feature map from the combined patch tokens

#### New configuration fields

The adapter should support the following fields:

- `aux_layers: list[int] | None`
- `layer_combine: str = "single"`
- `layer_weights: list[float] | None`

The existing fields remain:

- `feature_source`
- `aux_layer`
- `feat_layer`
- `ref_view_strategy`
- `patch_size`

#### Fusion modes

Supported fusion modes:

- `single`
- `avg`
- `sum`
- `weighted_avg`

Fusion is applied to patch tokens with shape `[B, N, C]` before reshaping to `[B, C, Hp, Wp]`.

#### Priority rules

When `feature_source=aux`:

1. if `aux_layers` is provided, use it
2. otherwise fall back to `aux_layer`

When `feature_source=final`, AUX fusion arguments must not affect behavior.

#### Combination formulas

Given per-layer patch token tensors `T_1 ... T_k`:

- `single`: use the only requested layer tensor directly
- `avg`: `mean(T_i)`
- `sum`: `sum(T_i)`
- `weighted_avg`: `sum(w_i * T_i) / sum(w_i)`

The adapter will continue to derive `global_token` from the fused patch tokens via mean pooling, preserving the current ablation-friendly behavior.

#### Required helper functions

The implementation should isolate the new behavior behind small helpers, for example:

- `_resolve_requested_aux_layers(...)`
- `_validate_layer_combine(...)`
- `_combine_aux_patch_tokens(...)`
- `_extract_aux_features(...)`

The current final-feature path should remain separate and minimally touched.

### 5.2 `eval_da3_vpr.py`

#### New CLI arguments

Add:

- `--aux-layers 3 4 5`
- `--layer-combine single|avg|sum|weighted_avg`
- `--layer-weights 0.2 0.3 0.5`

#### Responsibility

The evaluation script should only:

- parse the new arguments
- validate obvious CLI-level incompatibilities
- pass the resulting values onto `model.encoder`

It should not implement fusion itself.

## 6. Validation and Error Rules

### Valid combinations

#### `feature_source=final`

Valid only when fusion is effectively disabled.

Recommended rule:

- `layer_combine` must remain `single`
- `aux_layers` must be absent
- `layer_weights` must be absent

#### `feature_source=aux`

If `aux_layers` is absent:

- use `aux_layer`
- only `single` is valid

If `aux_layers` is present:

- `single` requires exactly one layer
- `avg` and `sum` require at least two layers
- `weighted_avg` requires at least two layers and matching `layer_weights`

### Runtime validation

The adapter should fail clearly when:

- any AUX layer index is negative
- any requested AUX layer exceeds the backbone depth
- `weighted_avg` is selected without weights
- the number of weights does not match the number of AUX layers
- the weight sum is zero
- the backbone returns fewer AUX tensors than requested
- fused patch tokens cannot be reshaped into a valid feature map

No silent fallbacks should be added.

## 7. Backward Compatibility

Backward compatibility is a primary design requirement.

Expected preserved behaviors:

- `feature_source=final` behaves exactly as before
- `feature_source=aux` with only `aux_layer` behaves exactly as before
- existing commands that do not mention fusion arguments remain valid
- existing SALAD checkpoints remain usable because channel count is unchanged

This ensures the ablation can be added and later removed without disturbing the original DA3 VPR integration.

## 8. Testing Plan for the Future Implementation

### Adapter tests

Add or extend tests to cover:

1. existing single-layer AUX path still works
2. `aux_layers=[3,4,5]` with `avg`
3. `aux_layers=[3,4,5]` with `sum`
4. `aux_layers=[3,4,5]` with `weighted_avg`
5. `weighted_avg` fails on length mismatch
6. `single` fails when multiple AUX layers are provided
7. invalid fusion arguments do not affect `final` mode silently

### Eval-script tests

Add or extend tests to cover:

1. parsing `--aux-layers`
2. parsing `--layer-combine`
3. parsing `--layer-weights`
4. forwarding those values into `model.encoder`
5. rejecting incompatible argument combinations

## 9. Intended Experiments

The first intended SPEDTEST experiments after implementation are:

- baseline single layer: `aux_layer=5`
- `aux_layers=4 5`, `layer_combine=avg`
- `aux_layers=3 4 5`, `layer_combine=avg`
- `aux_layers=4 5`, `layer_combine=weighted_avg`, `layer_weights=0.3 0.7`
- `aux_layers=3 4 5`, `layer_combine=weighted_avg`, `layer_weights=0.2 0.3 0.5`

The comparison metric remains Recall@1/5/10/15/20/25 on SPEDTEST.

## 10. Success Criteria

This design is successful if:

- the old DA3 VPR commands still behave identically
- multi-layer fusion can be enabled only through explicit CLI flags
- the existing SALAD checkpoint remains reusable
- SPEDTEST recall can be compared cleanly between:
  - single AUX layer
  - multi-layer avg
  - multi-layer sum
  - multi-layer weighted avg

The design does not require fusion to outperform the single-layer baseline. It only requires the experiment to be clean, reproducible, and reversible.
