# Training-Free Visual Localization Baseline Design

## Goal
Build a training-free visual localization baseline for 7Scenes under the reloc3r-style `TrainSplit/TestSplit` protocol:

- retrieval backend: standalone `DINO-SALAD` or pretrained `DA3-SALAD`
- pose backend: DA3 multi-view pose network
- no additional training, no finetuning, no calibration-on-val
- evaluation target: fair comparison against reloc3r under the same train/test scene split style

The baseline should isolate retrieval quality from pose estimation quality while keeping the protocol transparent and reproducible.

## Scope
This work introduces a dedicated evaluation path for training-free visual localization. It will:

- reuse the existing 7Scenes split loader used by the current visloc scripts
- support two retrieval backends:
  - `dino_salad`
  - `da3_salad`
- use the current unified pipeline v1.1 multi-view pose path
- evaluate on 7Scenes with train split as database and test split as queries
- report translation and rotation errors per query and per scene

This work does not:

- add new training code
- change the reloc3r evaluation protocol
- replace the existing DA3 benchmark dataset wrapper
- merge the old `eval_unified_visloc.py` entrypoint into the new baseline

## Existing Technical Facts

### 1. Current unified pipeline is already multi-view at pose time
The current `src/depth_anything_3/model/unified_pipeline.py` is no longer the earlier cross-view-fusion design. It now does:

- single-view retrieval descriptor extraction from DA3 aux features
- top-candidate selection
- multi-view DA3 backbone inference on `[query, references...]`
- pose prediction from the DA3 multi-view branch

This matches the intended training-free baseline structure well.

### 2. The current helper and eval script are partially stale
`src/depth_anything_3/model/unified_pipeline_helper.py` and `eval_unified_visloc.py` still assume the older pipeline contract that used:

- cross-view fusion modules
- database patch tokens
- database camera tokens
- a full forward signature based on pre-extracted candidate features

Those assumptions no longer match the current `UnifiedPipeline` public API. The new training-free baseline should not be built by extending the stale evaluation path.

### 3. DA3 pose output is multi-view group pose, not directly scene-absolute query pose
When DA3 runs on `[query, refs...]`, its predicted camera poses are group-consistent predictions for the multi-view set. To compare against 7Scenes query ground truth in the reloc3r-style protocol, the predicted group must be anchored to the scene frame.

The chosen anchoring rule is:

- use the retrieved reference images' ground-truth poses
- align the predicted reference-camera group to those ground-truth reference poses with Umeyama
- apply the same alignment to the predicted query camera
- then evaluate the aligned query pose against query ground truth

This avoids abusing a single reference as the only anchor and better reflects the multi-view model's intended usage.

## Design Principles

### 1. Fairness means same protocol, not identical internals
The baseline should keep the protocol fixed across retrieval backends:

- same dataset split
- same image sizing policy at pose time
- same `top-K` retrieval pool
- same `top-M` references sent to the pose network
- same pose backend
- same absolute-pose recovery rule
- same metrics

The retrieval encoders may keep their own required preprocessing if needed. Fairness comes from a fixed evaluation protocol, not from forcing mismatched backbone preprocessing.

### 2. Retrieval and pose must remain explicitly decoupled
The code should make it obvious which part of the result comes from:

- retrieval quality
- DA3 multi-view pose quality

The script should therefore separate:

- descriptor extraction
- retrieval ranking
- pose inference
- multi-reference alignment
- metric computation

### 3. The new baseline should be isolated from stale evaluation code
Instead of retrofitting the old `eval_unified_visloc.py` entrypoint, create a new dedicated script for the training-free baseline. Small duplication is acceptable if it keeps the logic auditable and avoids binding new experiments to the obsolete pipeline contract.

### 4. Multi-reference alignment is the primary absolute-pose protocol
Two anchoring modes are useful:

- `top1_anchor`
- `multi_ref_alignment`

But the primary reported protocol should be `multi_ref_alignment`, because it uses the full set of references actually consumed by the DA3 pose network.

## Selected Design
Introduce a new evaluation script:

- `ablation/eval_training_free_visloc.py`

This script will:

1. Load a 7Scenes scene using reloc3r-style train/test splits
2. Build database descriptors with the selected retrieval backend
3. Extract a query descriptor for each test image
4. Retrieve `top-K` nearest database images by cosine similarity
5. Select `top-M` references for pose
6. Run the DA3 multi-view pose path on `[query, top-M refs]`
7. Recover the query absolute pose through multi-reference Umeyama alignment
8. Report and save localization errors

The script will support:

- two retrieval backends
- three pose output modes: `cam_dec`, `ray`, `both`
- two anchoring modes: `multi_ref_alignment`, `top1_anchor`

Default reporting:

- retrieval backend: user-selected
- pose path: `cam_dec`
- anchor mode: `multi_ref_alignment`

## File Layout

### 1. `ablation/eval_training_free_visloc.py`
Defines the training-free visual localization evaluation entrypoint.

Responsibilities:

- parse CLI arguments
- load scene train/test splits
- load retrieval backend
- load DA3 pose backend
- build database descriptors
- perform query-time retrieval
- run pose inference with `top-M` references
- align predicted group poses to scene coordinates
- compute metrics and save outputs

This file is the main experiment protocol and should be readable end to end.

### 2. `src/depth_anything_3/model/unified_pipeline_helper.py`
Must be updated so the builder matches the current unified pipeline v1.1 constructor and public methods.

Responsibilities after update:

- assemble the current `UnifiedPipeline`
- build retrieval modules and VPR components
- load pretrained DA3-SALAD VPR weights
- expose a `pose_top_m` configuration field
- avoid assuming cross-view-fusion modules exist in the active pipeline contract

This change is required so the new evaluation script can reuse the standard pipeline builder instead of manually reassembling DA3 components.

### 3. `tests/test_training_free_visloc.py`
Defines focused tests for protocol-critical logic.

Responsibilities:

- validate `top-K` to `top-M` selection behavior
- validate multi-reference absolute-pose recovery
- validate pose-path routing for `cam_dec`, `ray`, and `both`
- validate retrieval backend interface compatibility using mocks or small fake tensors

The tests should not depend on real checkpoints or large external datasets.

## Runtime Data Flow

### Case A: `dino_salad` retrieval backend
1. Load standalone DINO-SALAD model from:
   - `da3_streaming/loop_utils/salad/weights/dino_salad_512_32.ckpt`
2. Encode all train-split images to database descriptors
3. Encode each test-split query image to a query descriptor
4. Retrieve `top-K` train images by cosine similarity
5. Take `top-M` references for pose
6. Run DA3 multi-view pose on `[query, refs]`
7. Align the predicted multi-view camera group to the reference GT poses
8. Evaluate the aligned query pose

### Case B: `da3_salad` retrieval backend
1. Build unified pipeline with pretrained retrieval weights from:
   - `checkpoints/image_retrieval/DA3_vprmodel_patchonlyadapter_aux5.ckpt`
2. Use `retrieval_only()` for database/query descriptor extraction
3. Perform the same `top-K` retrieval and `top-M` pose flow as in Case A
4. Use the same absolute-pose recovery and metrics as in Case A

This keeps the only intended variable equal to the retrieval backend.

## Absolute Pose Recovery

### Primary mode: `multi_ref_alignment`
Inputs:

- predicted poses for `[query, ref1, ref2, ..., refM]`
- GT poses for `[ref1, ref2, ..., refM]`

Procedure:

1. Extract the predicted reference poses from the multi-view output
2. Convert predicted and GT poses to a common convention for alignment
3. Estimate a similarity transform with Umeyama using the reference subset only
4. Apply the transform to the predicted query pose
5. Return the transformed query pose as the localization output

This is the primary reported protocol.

### Secondary mode: `top1_anchor`
Inputs:

- predicted query pose
- predicted pose of the first selected reference
- GT pose of the first selected reference

Procedure:

- compute a single-reference anchoring transform from the predicted top-1 reference to its GT pose
- apply it to the predicted query pose

This mode is useful as a diagnostic or ablation, but should not be the main result.

## Pose Paths

### 1. `cam_dec`
Use the DA3 camera decoder path as the default reported pose output.

Rationale:

- already used by the current unified pipeline flow
- closest to the existing ablation scripts
- most natural default for training-free comparison

### 2. `ray`
Use DA3's ray-based pose recovery path as an optional alternative.

Rationale:

- may behave differently from `cam_dec`
- useful for robustness comparisons

### 3. `both`
Run both paths and report both outputs when available.

Rationale:

- avoids committing the baseline conclusions to one pose decoder
- preserves future experiment flexibility

## CLI Surface
`ablation/eval_training_free_visloc.py` should expose:

- `--retriever-backend` with choices `dino_salad`, `da3_salad`
- `--unified-config`
- `--unified-checkpoint`
- `--salad-checkpoint`
- `--dataset` with at least `7scenes`
- `--scene`
- `--top-k`
- `--top-m`
- `--pose-path` with choices `cam_dec`, `ray`, `both`
- `--anchor-mode` with choices `multi_ref_alignment`, `top1_anchor`
- `--batch-size`
- `--device`
- `--image-size`
- `--data-root`
- `--output-dir`

Backend-specific argument rules:

- `dino_salad` requires `--salad-checkpoint`
- `da3_salad` uses the pretrained DA3-SALAD retrieval weights from config unless explicitly overridden

## Output Contract
Per-scene execution should:

- print median translation and rotation errors
- save a scene result file under `workspace/ablation_results/`

Recommended output filename:

- `training_free_<backend>_<pose_path>_<anchor_mode>_<dataset>_<scene>.npz`

The result file should contain at least:

- `rotation_errors`
- `translation_errors`
- `topk_indices`
- `topm_indices`
- `query_image_paths`
- `db_image_paths`
- `config`

This supports downstream auditing and fair comparison debugging.

## Validation and Error Handling
The implementation should fail early for:

- unsupported retrieval backend
- missing required checkpoint for `dino_salad`
- missing or incompatible DA3-SALAD checkpoint weights
- `top_m > top_k`
- empty database split
- empty query split
- failure to obtain enough valid references for multi-reference alignment
- missing pose output for the selected pose path

The script should also print startup diagnostics:

- retrieval backend
- pose path
- anchor mode
- dataset and scene
- `top-K`
- `top-M`
- image size
- database/query counts

## Testing Plan for the Future Implementation

### 1. Unit tests
Add focused tests for:

- `top-K` retrieval ranking and `top-M` truncation
- multi-reference alignment recovering the correct query pose from synthetic poses
- `top1_anchor` recovery on a synthetic example
- `cam_dec` / `ray` / `both` output routing

### 2. Builder contract tests
Add a test that verifies the pipeline helper builds the current v1.1 unified pipeline without assuming cross-view-fusion constructor arguments.

### 3. Smoke evaluation test
Add a small mocked smoke test that:

- creates a tiny fake database
- creates one fake query
- runs through retrieval, pose routing, alignment, and metric computation
- verifies the result file schema

## Recommendation
Implement the training-free baseline as a new dedicated ablation script, not as an extension of the stale `eval_unified_visloc.py` path.

The primary reported experiment should be:

- protocol: reloc3r-style `TrainSplit/TestSplit`
- retrieval: `dino_salad` or `da3_salad`
- pose path: `cam_dec`
- anchoring: `multi_ref_alignment`
- retrieval/pose split: `top-K` retrieval followed by `top-M` pose references

This is the cleanest and fairest path for comparison against reloc3r while staying aligned with the current unified pipeline v1.1 model contract.
