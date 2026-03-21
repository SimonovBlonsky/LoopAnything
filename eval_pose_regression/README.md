# DA3 Relative Pose Evaluation

This folder provides a quick relative-pose benchmark for Depth Anything 3 on the same pair lists used by `reloc3r`:

- `reloc3r/data/scannet1500`
- `reloc3r/data/megadepth1500`

## What is measured

The evaluator matches `reloc3r/eval_relpose.py` exactly for metrics:

- rotation angular error
- translation-direction angular error
- `auc@5`, `auc@10`, `auc@20`, computed from `max(R error, T error)`

## How pose is predicted

For each image pair `(I1, I2)`:

1. preprocess the two images with DA3's official `InputProcessor`
2. run DA3's pose path through `DepthAnything3.forward(...)`
3. read the predicted per-view extrinsics
4. form the relative pose as `T_2to1 = E1 @ inv(E2)`

Here `E1` and `E2` are the per-view extrinsics predicted by DA3.

## Files

- `eval_relpose.py`: main entry point for both datasets
- `datasets.py`: lightweight readers for ScanNet1500 and MegaDepth1500
- `metrics.py`: reloc3r-compatible metrics
- `eval_scannet1500.sh`: convenience wrapper for ScanNet1500
- `eval_megadepth1500.sh`: convenience wrapper for MegaDepth1500

## Example usage

```bash
cd /home/chenguyuan/code/NeurIPS26/LoopAnything/eval_pose_regression
bash eval_scannet1500.sh
bash eval_megadepth1500.sh
```

You can override the checkpoint or preprocessing settings, for example:

```bash
bash eval_megadepth1500.sh --model-path /path/to/local/da3-checkpoint --batch-size 4 --use-ray-pose
```

## Speed output

The script also prints per-pair timing statistics:

- `ms_per_pair_model_only`
- `ms_per_pair_end_to_end`
- `pairs_per_sec_model_only`
- `pairs_per_sec_end_to_end`

This keeps the DA3 folder aligned with the VGGT evaluator layout and makes side-by-side speed comparisons easier.
