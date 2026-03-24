# DA3 VPR Integration Design

## 1. Goal

Integrate a retrieval-oriented VPR model into the DA3 codebase by reusing the existing Depth Anything 3 DINOv2 encoder and plugging its output features into a SALAD-style aggregator interface.

The immediate goal is inference-first:

- read DA3 encoder weights or reuse an already-constructed DA3 model
- extract final-layer patch tokens from the DA3 encoder
- convert those tokens into the input format expected by VPR aggregators
- load an aggregator from the migrated `VPRaggregators` package
- optionally load the aggregator weights from a SALAD checkpoint
- produce a global descriptor through a unified `VPRModel`

This design is intentionally not a training-focused port of SALAD. It is a clean inference-oriented integration that preserves future extensibility.

## 2. Scope and Constraints

### In scope

- Refactor `src/depth_anything_3/model/vpr_model.py` into an inference-oriented module.
- Refactor `src/depth_anything_3/model/vpr_helper.py` into an assembly / checkpoint-loading helper.
- Add a small DA3-specific encoder adapter layer.
- Support both:
  - passing in an existing `DepthAnything3` / `DepthAnything3Net` instance
  - constructing from DA3 config + DA3 weights
- Support loading aggregator weights from a SALAD checkpoint.
- Preserve the existing aggregator selection interface so `SALAD`, `GeM`, `ConvAP`, `MixVPR`, and similar modules can still be swapped.

### Out of scope

- Modifying DA3 core model files such as `da3.py`, `cam_dec.py`, or `vision_transformer.py`
- Reintroducing the old Lightning training flow from SALAD as a first-class DA3 training path
- Editing vendored SALAD code under `da3_streaming/loop_utils/salad`
- Adding retrieval training, pairwise gate training, or pose-head training in this step
- Changing `da3_streaming` integration in this step

### File ownership

Primary files to modify:

- `src/depth_anything_3/model/vpr_model.py`
- `src/depth_anything_3/model/vpr_helper.py`

Permitted new file:

- `src/depth_anything_3/model/vpr_encoder_adapter.py`

Small adjacent additions are acceptable only if they directly support these files and keep the architecture cleaner.

## 3. Existing Technical Facts

### DA3 encoder output contract

The DA3 backbone returns tuples of `(patch_tokens, camera_tokens)` for intermediate layers.

For the final layer in the single-view retrieval setting, the relevant shapes are effectively:

- `patch_tokens`: `[B, 1, N, C]`
- `camera_tokens`: `[B, 1, C]`

For VPR inference, the adapter will normalize this into:

- `patch_tokens`: `[B, N, C]`
- `feature_map`: `[B, C, Hp, Wp]`
- `global_token`: `[B, C]`

with `N == Hp * Wp`.

### Aggregator input contract

The migrated aggregators are not all identical.

- `SALAD` expects `(feature_map, global_token)`
- `GeM`, `ConvAP`, and similar modules expect `feature_map`

Therefore the VPR model should not expose raw DA3 tokens directly to aggregator modules. Instead it should normalize DA3 outputs into a common feature dictionary and let the VPR model route the right pieces into each aggregator.

### Weight loading fact

SALAD evaluation loads a single checkpoint into a full model, but in the DA3 integration we do not want to load the SALAD backbone weights. We only want to recover the aggregator weights from that checkpoint and combine them with DA3 encoder weights.

That means the new helper layer must extract only aggregator-prefixed keys from the SALAD checkpoint.

## 4. Selected Design

The selected design is:

- interface choice: support both explicit DA3 model injection and DA3 auto-construction (`C`)
- internal organization: helper assembles components, model only defines forward (`2`)

This yields three layers:

1. `vpr_encoder_adapter.py`
   - DA3-specific feature extraction
   - converts DA3 backbone outputs into a normalized retrieval feature dictionary

2. `vpr_model.py`
   - pure inference module
   - owns `encoder` and `aggregator`
   - defines the descriptor forward path only

3. `vpr_helper.py`
   - constructs DA3 encoder adapter
   - constructs aggregator
   - extracts aggregator weights from SALAD checkpoints
   - assembles a final `VPRModel`

This structure keeps DA3-specific logic out of aggregator code and keeps checkpoint assembly out of the model forward path.

## 5. Component Design

### 5.1 `vpr_encoder_adapter.py`

#### Responsibility

Wrap a DA3 model and expose retrieval-friendly features from its encoder.

#### Proposed class

```python
class DA3EncoderAdapter(nn.Module):
    def __init__(
        self,
        da3_model: nn.Module,
        feat_layer: int = -1,
        ref_view_strategy: str = "saddle_balanced",
        patch_size: int = 14,
    ):
        ...
```

#### Forward behavior

Input:

- `[B, 3, H, W]` or `[B, 1, 3, H, W]`

Internal behavior:

- normalize input to `[B, S, 3, H, W]` with `S=1` for retrieval
- call the DA3 backbone
- select one feature layer, defaulting to the last layer
- unpack `(patch_tokens, camera_tokens)`
- remove the single-view dimension
- reshape patch tokens into a 2D feature map

Output dictionary:

```python
{
    "patch_tokens": Tensor[B, N, C],
    "feature_map": Tensor[B, C, Hp, Wp],
    "global_token": Tensor[B, C],
    "spatial_shape": (Hp, Wp),
}
```

#### Design notes

- The adapter should be the only place that knows DA3’s token layout details.
- It should validate that `N == Hp * Wp` and raise a clear error if not.
- It should not own checkpoint loading.
- It should not know anything about aggregator choice.

### 5.2 `vpr_model.py`

#### Responsibility

Own an encoder and an aggregator, and define a stable descriptor forward path.

#### Proposed class

```python
class VPRModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        aggregator: nn.Module,
        agg_arch: str,
    ):
        ...
```

#### Proposed methods

```python
def extract_features(self, x, **kwargs) -> dict: ...
def aggregate(self, features: dict) -> torch.Tensor: ...
def forward(self, x, return_features: bool = False, **kwargs): ...
```

#### Aggregator routing rule

The model should use a small internal dispatch rule:

- if `agg_arch` is SALAD-like, pass `(feature_map, global_token)`
- otherwise pass `feature_map`

#### Why this design

This keeps the model generic while still respecting aggregator-specific interfaces. It also avoids encoding SALAD assumptions into every aggregator.

#### Important refactor

The current migrated `vpr_model.py` still contains training-only Lightning logic from SALAD. That should be removed from the inference-oriented DA3 VPR model.

This means the DA3 version of `VPRModel` should become a plain `nn.Module`, not a `LightningModule`.

### 5.3 `vpr_helper.py`

#### Responsibility

Build and wire the model from DA3 and SALAD sources.

#### Core helper functions

```python
def build_da3_model(...)
def build_da3_encoder_adapter(...)
def build_aggregator(agg_arch, agg_config)
def build_vpr_model(
    *,
    da3_model=None,
    da3_config_path=None,
    da3_weight_path=None,
    da3_model_name_or_path=None,
    agg_arch,
    agg_config=None,
    aggregator_ckpt_path=None,
    strict=True,
    feat_layer=-1,
    ref_view_strategy="saddle_balanced",
    patch_size=14,
)
```

#### Checkpoint functions

```python
def extract_prefixed_state_dict(state_dict, prefixes)
def load_aggregator_weights_from_salad_ckpt(aggregator, ckpt_path, strict=True)
```

#### Expected supported construction modes

1. Existing DA3 model instance

```python
build_vpr_model(
    da3_model=existing_model,
    agg_arch="SALAD",
    agg_config=..., 
    aggregator_ckpt_path=...
)
```

This is the highest-priority DA3 source. If `da3_model` is provided, helper code must use it directly and ignore DA3 construction arguments.

2. DA3 config + DA3 weights

```python
build_vpr_model(
    da3_config_path=...,
    da3_weight_path=...,
    agg_arch="SALAD",
    agg_config=...,
    aggregator_ckpt_path=...
)
```

This mode is valid only when `da3_model` is not provided. `da3_config_path` and `da3_weight_path` must be provided together.

3. DA3 HF/local model name or path if already supported by the local DA3 API wrapper

```python
build_vpr_model(
    da3_model_name_or_path=...,
    agg_arch="SALAD",
    agg_config=...,
    aggregator_ckpt_path=...
)
```

This is the fallback mode. It is valid only when `da3_model` is absent and `da3_config_path` / `da3_weight_path` are absent.

#### DA3 source precedence and exclusivity

The helper must resolve DA3 sources in this order:

1. `da3_model`
2. `da3_config_path` + `da3_weight_path`
3. `da3_model_name_or_path`

Valid calls must provide exactly one DA3 source mode. If arguments from multiple modes are mixed, helper code should raise a clear `ValueError` instead of silently choosing one.

#### Aggregator checkpoint extraction

The helper must support common prefixes such as:

- `aggregator.`
- `model.aggregator.`
- `module.aggregator.`

It should strip the prefix and load only that sub-state into the chosen aggregator.

Before prefix extraction, the helper should unwrap common trainer checkpoint containers by checking, in order:

1. a top-level `state_dict` field
2. a top-level `model` field if it already looks like a flat parameter dictionary
3. the checkpoint object itself if it is already a flat parameter dictionary

If no recognized aggregator-prefixed keys are found after unwrapping, the helper must raise a clear error rather than partially loading or silently continuing.

This is the core mechanism that allows DA3 encoder + SALAD aggregator composition.

## 6. Proposed Runtime Data Flow

### Case A: existing DA3 model object

```text
existing DA3 model
    -> DA3EncoderAdapter
    -> feature dict {patch_tokens, feature_map, global_token}
    -> VPRModel.aggregate(...)
    -> global descriptor
```

### Case B: DA3 config + weights + SALAD checkpoint

```text
DA3 config + DA3 weights
    -> build DA3 model
    -> wrap with DA3EncoderAdapter

SALAD ckpt
    -> build aggregator
    -> extract aggregator-prefixed weights
    -> load into aggregator

encoder + aggregator
    -> assemble VPRModel
    -> forward
    -> global descriptor
```

## 7. Error Handling and Validation Rules

The implementation should fail clearly in these situations:

- DA3 source is underspecified
  - neither an existing DA3 model nor a DA3 construction path is given
- aggregator arch is unsupported
- aggregator checkpoint does not contain recognizable aggregator-prefixed keys
- patch token count cannot be reshaped into a valid spatial map
- DA3 backbone output contract is not the expected `(patch_tokens, camera_tokens)` structure

Recommended validation checks:

- assert single-view retrieval path for the first implementation
- assert `feature_map.dtype` is finite and `float32` or a valid inference dtype
- assert descriptor output is finite

## 8. Testing Plan for the Future Implementation

The implementation plan should include at least these tests:

1. DA3 adapter shape test
- given stub DA3 outputs, verify `patch_tokens`, `feature_map`, and `global_token` shapes

2. Aggregator routing test
- SALAD receives `(feature_map, global_token)`
- GeM / ConvAP receive `feature_map`

3. Aggregator checkpoint extraction test
- given a fake full-model checkpoint, only aggregator-prefixed keys are loaded

4. Existing-model assembly test
- passing an existing DA3 model yields a working `VPRModel`

5. Config-and-weight assembly test
- helper constructs DA3, wraps adapter, builds aggregator, and loads weights

6. End-to-end descriptor smoke test
- DA3 adapter + SALAD aggregator produce a finite descriptor of expected shape

## 9. Non-Goals and Deliberate Omissions

The following are explicitly deferred:

- preserving Lightning training hooks in the new DA3-side `vpr_model.py`
- adding retrieval loss or validation hooks to this module
- integrating this VPR model into `da3_streaming` in the same step
- adding learned DA3-specific retrieval heads
- reusing SALAD backbone weights directly

These are deferred to keep the first implementation focused and low-risk.

## 10. Recommendation

Implement the DA3-side VPR stack as:

- a DA3 encoder adapter
- a pure inference `VPRModel`
- a helper-led assembly and checkpoint-loading layer

This is the cleanest way to combine:

- DA3 encoder weights
- SALAD aggregator modules
- SALAD aggregator checkpoint weights

without tangling training logic, DA3 internals, and aggregator routing into one file.
