# Training-Free Visual Localization Pipeline Dev Log

## Scope

This document records the execution history for the training-free 7Scenes visual localization baseline built in the `unified_pipeline1.1` worktree. It summarizes the planning artifacts, the Task 1-4 implementation sequence, the associated git commits, the main code changes, and the verification evidence.

Worktree:

- `/home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1`

Final branch state when the work finished:

- `HEAD`: `99d87b58cbcea0afa840f2482cbd6e57b9c9b58b`
- worktree status: clean

## Goal

Build a training-free visual localization baseline on 7Scenes under the reloc3r-style `TrainSplit/TestSplit` protocol:

- retrieval backend: `dino_salad` or `da3_salad`
- pose backend: DA3 multi-view pose network
- no training, no finetuning, no validation-set calibration
- primary pose output: `cam_dec`
- supported pose outputs: `cam_dec`, `ray`, `both`
- primary anchoring mode: `multi_ref_alignment`
- secondary anchoring mode: `top1_anchor`
- retrieval protocol: `top-K` retrieval followed by `top-M` references sent into the DA3 pose branch

## Design And Plan Artifacts

### Spec

- file: [`docs/superpowers/specs/2026-04-04-training-free-visloc-baseline-design.md`](/home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1/docs/superpowers/specs/2026-04-04-training-free-visloc-baseline-design.md)
- commit: `88cf851`
- subject: `docs: add training-free visloc baseline spec`

The spec locked the following protocol decisions:

- use reloc3r-style `TrainSplit/TestSplit`
- compare retrieval backends fairly under one fixed pose/evaluation protocol
- create a dedicated evaluation script instead of extending stale `eval_unified_visloc.py`
- use `multi_ref_alignment` as the main absolute-pose recovery rule
- support both `cam_dec` and `ray`, with `cam_dec` as the default reported path

### Implementation Plan

- file: [`docs/superpowers/plans/2026-04-04-training-free-visloc-baseline.md`](/home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1/docs/superpowers/plans/2026-04-04-training-free-visloc-baseline.md)

The plan broke execution into four tasks:

1. Repair the unified pipeline builder contract
2. Add protocol tests for the training-free baseline
3. Implement retrieval backends and scene evaluation
4. Perform real runtime smoke verification and final polish

## Task 1: Repair The Unified Pipeline Builder Contract

### Purpose

Before adding the new baseline, align the helper and tests with the active `UnifiedPipeline v1.1` API instead of the older cross-view-fusion contract.

### Commits

- `0b6c929` `fix unified pipeline builder contract`
- `872f20d` `strengthen unified pipeline smoke coverage`
- `85e328c` `fix batched top-m selection in unified pipeline`

### Main Changes

- updated [`src/depth_anything_3/model/unified_pipeline_helper.py`](/home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1/src/depth_anything_3/model/unified_pipeline_helper.py) so the builder matches the current `UnifiedPipeline` constructor
- added `pose_top_m` support in [`configs/unified_pipeline.yaml`](/home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1/configs/unified_pipeline.yaml)
- refreshed [`tests/test_unified_pipeline.py`](/home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1/tests/test_unified_pipeline.py) to validate the active two-stage API
- fixed a real batch bug in [`src/depth_anything_3/model/unified_pipeline.py`](/home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1/src/depth_anything_3/model/unified_pipeline.py): `selected_indices` is now computed per sample instead of incorrectly using row 0 for the full batch

### Verification

Command:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1
PYTHONPATH=src conda run -n da3 python -m pytest tests/test_unified_pipeline.py -q
```

Observed result at the end of Task 1:

- `5 passed, 1 warning`

### Outcome

The helper, config, and tests were now aligned with the current `UnifiedPipeline v1.1`, which unblocked the baseline script.

## Task 2: Add Protocol Tests For The Training-Free Baseline

### Purpose

Create the pure helper layer and test the protocol-critical logic before wiring in real checkpoint loading and scene evaluation.

### Commit

- `61b56a2` `add training-free visloc protocol helpers`

### Main Changes

Created:

- [`ablation/eval_training_free_visloc.py`](/home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1/ablation/eval_training_free_visloc.py)
- [`tests/test_training_free_visloc.py`](/home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1/tests/test_training_free_visloc.py)

Added importable, mostly pure protocol helpers:

- `parse_args()`
- `select_topk_topm()`
- `resolve_pose_output()`
- `align_query_pose_multi_ref()`
- `align_query_pose_top1_anchor()`
- `validate_retrieval_backend()`

### Verification

Commands:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1
PYTHONPATH=src conda run -n da3 python -m pytest tests/test_training_free_visloc.py -q
```

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1
PYTHONPATH=src conda run -n da3 python -m pytest tests/test_unified_pipeline.py tests/test_training_free_visloc.py -q
```

Observed results at the end of Task 2:

- `7 passed in 1.04s`
- `12 passed, 1 warning in 2.99s`

### Outcome

The baseline protocol had a stable helper/test scaffold before any heavy model or dataset logic was connected.

## Task 3: Implement Retrieval Backends And Scene Evaluation

### Purpose

Turn the protocol scaffold into a complete training-free evaluation entrypoint that supports both retrieval backends, DA3 multi-view pose inference, absolute-pose recovery, metrics, and result saving.

### Commits

- `624a90f` `add training-free visloc baseline`
- `95578dd` `fix dual-branch visloc reporting`
- `ac7e6dd` `fix training-free visloc import bootstrap`

### Main Changes

#### New Evaluation Flow

[`ablation/eval_training_free_visloc.py`](/home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1/ablation/eval_training_free_visloc.py) was extended into the full baseline:

- load 7Scenes train split as the retrieval database
- load 7Scenes test split as the query set
- support two retrieval backends:
  - `dino_salad`
  - `da3_salad`
- run retrieval with cosine similarity
- select `top-K`, then route `top-M` references into `pipeline.pose_only()`
- recover absolute query pose with `multi_ref_alignment` by default
- compute translation and rotation errors
- save `.npz` results under `workspace/ablation_results/`

#### Pose Path Support

The script now supports:

- `--pose-path cam_dec`
- `--pose-path ray`
- `--pose-path both`

When `both` is selected, branch-specific metrics and arrays are saved instead of forcing a single merged result.

#### Import Bootstrap

The script gained path/bootstrap logic so it can be run directly from `ablation/` while still importing:

- `depth_anything_3`
- `eval_unified_visloc`
- `reloc3r`

This avoided fragile manual `PYTHONPATH`-dependent behavior at runtime.

### Verification

Command:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1
PYTHONPATH=src conda run -n da3 python -m pytest tests/test_training_free_visloc.py tests/test_unified_pipeline.py -q
```

Observed result at the end of Task 3:

- `19 passed, 1 warning in 3.21s`

Additional import verification:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1/ablation
PYTHONPATH=../src conda run -n da3 python -c "import eval_training_free_visloc; import eval_unified_visloc; from reloc3r.utils.metric import get_rot_err; print('ok')"
```

Observed result:

- `ok`

### Outcome

The baseline was functionally complete and import-safe, but still needed real runtime smoke validation on actual checkpoints and data.

## Task 4: Real Runtime Smoke Verification And Final Polish

### Purpose

Run the baseline through real runtime smoke checks, fix runtime-only issues, and add English comments to the code touched across the four tasks.

### Commits

- `3547212` `fix training-free visloc smoke runtime`
- `99d87b5` `fix bounded smoke defaults`

### Main Changes

#### Runtime Fixes In The Evaluation Script

[`ablation/eval_training_free_visloc.py`](/home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1/ablation/eval_training_free_visloc.py) received several runtime fixes:

- device resolution with explicit CUDA fallback behavior
- a default data-root helper
- bounded-smoke control flags:
  - `--cpu-fallback-max-queries`
  - `--cpu-fallback-max-db-entries`
- corrected `dino_salad` loading through the local SALAD `VPRModel` recipe
- cleanup for SALAD module shadowing so its top-level `utils` and `models` packages do not pollute the repository runtime
- CPU-safe xFormers disabling so DINOv2 can fall back to the PyTorch attention implementation when CUDA is unavailable or unstable

The last commit, `99d87b5`, set the bounded-smoke defaults back to disabled so normal full-scene evaluation is not silently truncated on real runs.

#### English Comments Added

Per the request to comment the code modified across the task series, English comments were added in the main production files touched by this work:

- [`ablation/eval_training_free_visloc.py`](/home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1/ablation/eval_training_free_visloc.py)
- [`src/depth_anything_3/model/unified_pipeline.py`](/home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1/src/depth_anything_3/model/unified_pipeline.py)
- [`src/depth_anything_3/model/unified_pipeline_helper.py`](/home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1/src/depth_anything_3/model/unified_pipeline_helper.py)

### Final Verification

Unit and integration-level verification:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1
PYTHONPATH=src conda run -n da3 python -m pytest tests/test_unified_pipeline.py tests/test_training_free_visloc.py -q
```

Observed result:

- `23 passed, 1 warning`

CLI smoke:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1
PYTHONPATH=src conda run -n da3 python ablation/eval_training_free_visloc.py --help
```

Observed result:

- help printed successfully

### Real Runtime Smoke Commands

Because CUDA initialization was unstable in the sandbox runtime, the final smoke runs were completed as bounded-smoke executions instead of full-scene full-GPU benchmarks.

`da3_salad` smoke:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1
PYTHONPATH=src conda run --no-capture-output -n da3 python -u ablation/eval_training_free_visloc.py \
  --retriever-backend da3_salad \
  --unified-config configs/unified_pipeline.yaml \
  --salad-checkpoint checkpoints/image_retrieval/DA3_vprmodel_patchonlyadapter_aux5.ckpt \
  --dataset 7scenes \
  --scene heads \
  --top-k 10 \
  --top-m 3 \
  --pose-path cam_dec \
  --anchor-mode multi_ref_alignment \
  --device cuda \
  --cpu-fallback-max-queries 4 \
  --cpu-fallback-max-db-entries 32 \
  --output-dir workspace/ablation_results
```

Observed smoke result:

- median translation error: `0.42 m`
- median rotation error: `137.78 deg`

`dino_salad` smoke:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1
PYTHONPATH=src conda run --no-capture-output -n da3 python -u ablation/eval_training_free_visloc.py \
  --retriever-backend dino_salad \
  --unified-config configs/unified_pipeline.yaml \
  --salad-checkpoint da3_streaming/loop_utils/salad/weights/dino_salad_512_32.ckpt \
  --dataset 7scenes \
  --scene heads \
  --top-k 10 \
  --top-m 3 \
  --pose-path cam_dec \
  --anchor-mode multi_ref_alignment \
  --device cuda \
  --cpu-fallback-max-queries 4 \
  --cpu-fallback-max-db-entries 32 \
  --output-dir workspace/ablation_results
```

Observed smoke result:

- median translation error: `0.34 m`
- median rotation error: `110.16 deg`

### Smoke Output Files

- [`workspace/ablation_results/training_free_da3_salad_7scenes_heads_cam_dec_multi_ref_alignment.npz`](/home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1/workspace/ablation_results/training_free_da3_salad_7scenes_heads_cam_dec_multi_ref_alignment.npz)
- [`workspace/ablation_results/training_free_dino_salad_7scenes_heads_cam_dec_multi_ref_alignment.npz`](/home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1/workspace/ablation_results/training_free_dino_salad_7scenes_heads_cam_dec_multi_ref_alignment.npz)

### Outcome

The baseline was verified end-to-end on real checkpoints and real 7Scenes data under bounded-smoke settings, and the default runtime behavior was left safe for later full-scene execution on a stable CUDA machine.

## Final Commit Chain

Relevant commits for this baseline, in chronological order:

1. `88cf851` `docs: add training-free visloc baseline spec`
2. `0b6c929` `fix unified pipeline builder contract`
3. `872f20d` `strengthen unified pipeline smoke coverage`
4. `85e328c` `fix batched top-m selection in unified pipeline`
5. `61b56a2` `add training-free visloc protocol helpers`
6. `624a90f` `add training-free visloc baseline`
7. `95578dd` `fix dual-branch visloc reporting`
8. `ac7e6dd` `fix training-free visloc import bootstrap`
9. `3547212` `fix training-free visloc smoke runtime`
10. `99d87b5` `fix bounded smoke defaults`

## Final Deliverables

### Main Script

- [`ablation/eval_training_free_visloc.py`](/home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1/ablation/eval_training_free_visloc.py)

### Core Supporting Files

- [`src/depth_anything_3/model/unified_pipeline.py`](/home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1/src/depth_anything_3/model/unified_pipeline.py)
- [`src/depth_anything_3/model/unified_pipeline_helper.py`](/home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1/src/depth_anything_3/model/unified_pipeline_helper.py)
- [`configs/unified_pipeline.yaml`](/home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1/configs/unified_pipeline.yaml)
- [`tests/test_unified_pipeline.py`](/home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1/tests/test_unified_pipeline.py)
- [`tests/test_training_free_visloc.py`](/home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1/tests/test_training_free_visloc.py)

## Notes For Future Reads

- The runtime smoke metrics above are not the final benchmark numbers. They came from bounded-smoke runs designed to prove the execution path with real data and checkpoints.
- The default bounded-smoke limits are now disabled, so normal future runs will evaluate the full selected scene unless the bounded-smoke flags are explicitly passed.
- The script supports both retrieval backends and all three pose-path modes, but the current primary comparison protocol remains:
  - retrieval backend: user-chosen
  - pose path: `cam_dec`
  - anchor mode: `reloc3r_motion_averaging`
  - retrieval flow: `top-K retrieval -> top-K pairwise relpose -> motion averaging`

## Post-Completion Update: 2026-04-05 Fairness Alignment With Reloc3r

### Motivation

After running full `heads` experiments, the training-free baseline showed a large gap to reloc3r:

- `da3_salad` + `cam_dec`: `0.16 m / 77.24 deg`
- `dino_salad` + `cam_dec`: `0.19 m / 76.88 deg`
- `dino_salad` + `ray`: `0.22 m / 165.55 deg`

This led to a follow-up audit of the pose semantics and localization protocol. The audit found that the earlier training-free script was not sufficiently aligned with reloc3r's pose logic:

- reloc3r predicts pairwise relative pose `query -> db`
- reloc3r then fuses all retrieved pairs with `motion_averaging()`
- the training-free DA3 script instead predicted a multi-view camera group and anchored it with `multi_ref_alignment`

That difference made the comparison less fair, especially on 7Scenes `heads`, where the retrieved top-3 references were frequently close to collinear and the center-only Sim(3) alignment could produce unstable rotation estimates.

### Protocol Change

The default localization path in [`ablation/eval_training_free_visloc.py`](/home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1/ablation/eval_training_free_visloc.py) was updated to match reloc3r more closely:

- new default anchor mode: `reloc3r_motion_averaging`
- retrieval still uses the selected backend (`dino_salad` or `da3_salad`)
- pose now uses the full retrieved `top-K` set, not `top-M`, when `reloc3r_motion_averaging` is active
- for each retrieved DB image, the script runs DA3 pose on a 2-view pair
- the predicted 2-view DA3 group is converted into a pairwise relative pose with reloc3r's convention:
  - `query -> db = inv(db_pose) @ query_pose`
- all pairwise relative poses are then fused through reloc3r's `Reloc3rVisloc.motion_averaging()`

The older anchoring methods remain available for ablation only:

- `multi_ref_alignment`
- `top1_anchor`

### Main Code Changes

Updated in [`ablation/eval_training_free_visloc.py`](/home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1/ablation/eval_training_free_visloc.py):

- `SUPPORTED_ANCHOR_MODES` now includes `reloc3r_motion_averaging`
- `parse_args()` now defaults `--anchor-mode` to `reloc3r_motion_averaging`
- added `group_to_query_to_db_relative_pose()`
- added `estimate_query_pose_motion_averaging()`
- changed `evaluate_scene_training_free()` so the default path runs:
  - `top-K retrieval`
  - `top-K` pairwise DA3 pose calls
  - pairwise `query -> db` conversion
  - reloc3r motion averaging

Updated tests in [`tests/test_training_free_visloc.py`](/home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1/tests/test_training_free_visloc.py):

- added a test for the reloc3r relative-pose convention
- added a test for motion averaging recovery
- added a test confirming that `reloc3r_motion_averaging` uses the full `top-K` set for pose instead of truncating to `top-M`
- updated the CLI expectation so the default anchor mode is now `reloc3r_motion_averaging`

### Verification

Unit and integration verification after the fairness update:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1
PYTHONPATH=src conda run -n da3 python -m pytest tests/test_unified_pipeline.py tests/test_training_free_visloc.py -q
```

Observed result:

- `26 passed, 1 warning in 3.44s`

CLI verification:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1
PYTHONPATH=src conda run -n da3 python ablation/eval_training_free_visloc.py --help
```

Observed result:

- help printed successfully
- `--anchor-mode` choices now include `reloc3r_motion_averaging`

Bounded real runtime smoke after the fairness change:

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything/.worktrees/unified_pipeline1.1
PYTHONPATH=src conda run --no-capture-output -n da3 python ablation/eval_training_free_visloc.py \
  --retriever-backend da3_salad \
  --dataset 7scenes \
  --scene heads \
  --top-k 3 \
  --pose-path cam_dec \
  --device cpu \
  --cpu-fallback-max-queries 1 \
  --cpu-fallback-max-db-entries 16 \
  --output-dir /tmp/training_free_visloc_smoke
```

Observed result:

- `[Training-Free][da3_salad][cam_dec] Scene heads median pose error: 0.23 m  4.57 deg`
- result file:
  [`/tmp/training_free_visloc_smoke/training_free_da3_salad_7scenes_heads_cam_dec_reloc3r_motion_averaging.npz`](/tmp/training_free_visloc_smoke/training_free_da3_salad_7scenes_heads_cam_dec_reloc3r_motion_averaging.npz)

### Status Of This Update

This devlog amendment documents a post-completion local code update made after the original Task 1-4 sequence.

- original baseline commit history still ends at `99d87b5`
- this fairness-alignment update was recorded in the worktree and verified locally
- no new git commit had been created yet at the time this devlog entry was added
