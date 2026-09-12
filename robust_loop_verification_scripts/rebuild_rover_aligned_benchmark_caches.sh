#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/chenguyuan/anaconda3/envs/da3/bin/python}"

FUSION_ROOT="${FUSION_ROOT:-/data/datasets/FusionPortable}"
GEODE_ROOT="${GEODE_ROOT:-/data/datasets/GEODE}"
NTU_ROOT="${NTU_ROOT:-/data/datasets/NTU-VIRAL}"
MAX_GT_DELTA_SEC="${MAX_GT_DELTA_SEC:-0.1}"

dry_run=0
check_only=0

usage() {
  cat <<'EOF'
Usage: rebuild_rover_aligned_benchmark_caches.sh [--dry-run] [--check-only]

Rebuilds VPR caches for the 10 frozen benchmark sequences using camera-frame
poses. The script validates source loop datasets and rebuilt cache manifests.
It does not run DA3, retrieval, the verifier, or metric evaluation.

Options:
  --dry-run     Print validation stages and preprocess commands without executing.
  --check-only  Validate source loop datasets without rebuilding caches.
  -h, --help    Show this help.

Environment:
  PYTHON_BIN, FUSION_ROOT, GEODE_ROOT, NTU_ROOT, MAX_GT_DELTA_SEC
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      dry_run=1
      shift
      ;;
    --check-only)
      check_only=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

print_command() {
  printf '[preprocess]'
  printf ' %q' "$@"
  printf '\n'
}

run_preprocess() {
  print_command "$@"
  if [[ "$dry_run" -eq 0 ]]; then
    "$@"
  fi
}

validate_datasets() {
  local mode="$1"
  echo "[validate] ${mode}"
  if [[ "$dry_run" -eq 1 ]]; then
    return
  fi

  FUSION_ROOT="$FUSION_ROOT" \
  GEODE_ROOT="$GEODE_ROOT" \
  NTU_ROOT="$NTU_ROOT" \
  MAX_GT_DELTA_SEC="$MAX_GT_DELTA_SEC" \
  PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "$PYTHON_BIN" - "$mode" <<'PY'
import json
import os
import sys
from bisect import bisect_left
from pathlib import Path

import numpy as np
import yaml


fusion_root = Path(os.environ["FUSION_ROOT"])
geode_root = Path(os.environ["GEODE_ROOT"])
ntu_root = Path(os.environ["NTU_ROOT"])
max_gt_delta_sec = float(os.environ["MAX_GT_DELTA_SEC"])
mode = sys.argv[1]

targets = [
    {
        "name": "FusionPortableV2/handheld/handheld_escalator00",
        "raw": fusion_root
        / "fusionportable_loop_dataset/handheld/handheld_escalator00/raw",
        "gt": None,
        "cache": fusion_root
        / "robust_loop_verifier_cache/FusionPortableV2/handheld/handheld_escalator00",
    },
    {
        "name": "FusionPortableV2/handheld/handheld_room00",
        "raw": fusion_root / "fusionportable_loop_dataset/handheld/handheld_room00/raw",
        "gt": None,
        "cache": fusion_root
        / "robust_loop_verifier_cache/FusionPortableV2/handheld/handheld_room00",
    },
    {
        "name": "FusionPortableV2/handheld/handheld_room01",
        "raw": fusion_root / "fusionportable_loop_dataset/handheld/handheld_room01/raw",
        "gt": None,
        "cache": fusion_root
        / "robust_loop_verifier_cache/FusionPortableV2/handheld/handheld_room01",
    },
    {
        "name": "FusionPortableV2/ugv/ugv_campus01",
        "raw": fusion_root / "fusionportable_loop_dataset/ugv/ugv_campus01/raw",
        "gt": None,
        "cache": fusion_root
        / "robust_loop_verifier_cache/FusionPortableV2/ugv/ugv_campus01",
    },
    {
        "name": "FusionPortableV2/ugv/ugv_parking01",
        "raw": fusion_root / "fusionportable_loop_dataset/ugv/ugv_parking01/raw",
        "gt": None,
        "cache": fusion_root
        / "robust_loop_verifier_cache/FusionPortableV2/ugv/ugv_parking01",
    },
    {
        "name": "GEODE/Offroad/Offroad02_beta",
        "raw": geode_root / "geode_loop_dataset/Offroad/Offroad02_beta/raw",
        "gt": None,
        "cache": geode_root
        / "robust_loop_verifier_cache/GEODE/Offroad/Offroad02_beta",
    },
    {
        "name": "GEODE/Offroad/Offroad05_beta",
        "raw": geode_root / "geode_loop_dataset/Offroad/Offroad05_beta/raw",
        "gt": None,
        "cache": geode_root
        / "robust_loop_verifier_cache/GEODE/Offroad/Offroad05_beta",
    },
    {
        "name": "NTU-VIRAL/NTU-VIRAL/eee_01",
        "raw": ntu_root / "ntu_viral_loop_dataset/NTU-VIRAL/eee_01/raw",
        "gt": None,
        "cache": ntu_root
        / "robust_loop_verifier_cache/NTU-VIRAL/NTU-VIRAL/eee_01",
    },
    {
        "name": "NTU-VIRAL/NTU-VIRAL/eee_02",
        "raw": ntu_root / "ntu_viral_loop_dataset/NTU-VIRAL/eee_02/raw",
        "gt": None,
        "cache": ntu_root
        / "robust_loop_verifier_cache/NTU-VIRAL/NTU-VIRAL/eee_02",
    },
    {
        "name": "NTU-VIRAL/NTU-VIRAL/nya_02",
        "raw": ntu_root / "ntu_viral_loop_dataset/NTU-VIRAL/nya_02/raw",
        "gt": None,
        "cache": ntu_root
        / "robust_loop_verifier_cache/NTU-VIRAL/NTU-VIRAL/nya_02",
    },
]


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return rows


def nonempty_lines(path):
    return [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def nested_value(data, dotted_key):
    value = data
    for part in dotted_key.split("."):
        value = value[part]
    return value


def load_extrinsic(raw):
    meta = json.loads((raw / "sequence_meta.json").read_text(encoding="utf-8"))
    transform = np.asarray(meta["T_camera_lidar"], dtype=np.float64).reshape(4, 4)
    calibration_path = Path(meta["calibration_file"])
    if not calibration_path.is_file():
        raise ValueError(f"calibration file does not exist: {calibration_path}")
    with calibration_path.open("r", encoding="utf-8") as handle:
        calibration = yaml.safe_load(handle)
    configured = np.asarray(
        nested_value(calibration, meta["calibration_key"]),
        dtype=np.float64,
    ).reshape(4, 4)
    if not np.allclose(transform, configured, atol=1e-12):
        raise ValueError("sequence metadata T_camera_lidar differs from calibration")
    rotation = transform[:3, :3]
    u, _, vt = np.linalg.svd(rotation)
    projected_rotation = u @ vt
    if np.linalg.det(projected_rotation) < 0.0:
        u[:, -1] *= -1.0
        projected_rotation = u @ vt
    if np.max(np.abs(projected_rotation - rotation)) > 1e-3:
        raise ValueError("T_camera_lidar rotation is not sufficiently close to SO(3)")
    effective_transform = transform.copy()
    effective_transform[:3, :3] = projected_rotation
    return meta, effective_transform


def nearest_gt_deltas(rows, gt_path):
    gt_timestamps = np.asarray(
        [float(line.split()[0]) for line in nonempty_lines(gt_path)],
        dtype=np.float64,
    )
    if len(gt_timestamps) == 0:
        raise ValueError(f"empty GT trajectory: {gt_path}")
    gt_timestamps.sort()
    deltas = []
    for row in rows:
        timestamp = float(row["timestamp"])
        position = bisect_left(gt_timestamps, timestamp)
        candidates = []
        if position < len(gt_timestamps):
            candidates.append(abs(float(gt_timestamps[position] - timestamp)))
        if position > 0:
            candidates.append(abs(float(gt_timestamps[position - 1] - timestamp)))
        deltas.append(min(candidates))
    return np.asarray(deltas, dtype=np.float64)


def validate_source(target):
    raw = target["raw"]
    required = (
        "trajectory_keyframes.txt",
        "trajectory_keyframe_indices.txt",
        "keyframes.jsonl",
        "keyframes_with_images.jsonl",
        "sequence_meta.json",
    )
    for filename in required:
        path = raw / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing or empty source file: {path}")

    rows = read_jsonl(raw / "keyframes_with_images.jsonl")
    base_rows = read_jsonl(raw / "keyframes.jsonl")
    trajectory_rows = nonempty_lines(raw / "trajectory_keyframes.txt")
    index_rows = nonempty_lines(raw / "trajectory_keyframe_indices.txt")
    counts = (len(rows), len(base_rows), len(trajectory_rows), len(index_rows))
    if len(set(counts)) != 1:
        raise ValueError(f"source row count mismatch: {counts}")
    if [int(row["keyframe_idx"]) for row in rows] != list(range(len(rows))):
        raise ValueError("source keyframe indices are not contiguous from zero")
    timestamps = [float(row["timestamp"]) for row in rows]
    if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
        raise ValueError("source keyframe timestamps are not strictly increasing")
    missing_images = [
        int(row["keyframe_idx"])
        for row in rows
        if row.get("has_image")
        and (
            not row.get("image_path")
            or not (raw / str(row["image_path"])).is_file()
        )
    ]
    if missing_images:
        raise ValueError(f"missing source images: {missing_images[:10]}")
    meta, _ = load_extrinsic(raw)
    if meta.get("loop_closure_enabled") is not False:
        raise ValueError("source metadata does not record loop_closure_enabled=false")

    gt = target["gt"]
    association = ""
    if gt is not None:
        if not gt.is_file():
            raise ValueError(f"GT trajectory does not exist: {gt}")
        deltas = nearest_gt_deltas(rows, gt)
        skipped = int(np.sum(deltas > max_gt_delta_sec))
        association = (
            f" gt_delta_max={float(np.max(deltas)):.6f}s"
            f" expected_skipped={skipped}"
        )
    images = sum(bool(row.get("has_image")) for row in rows)
    print(
        f"PASS source {target['name']}: keyframes={len(rows)} images={images}"
        f"{association}"
    )


def validate_cache(target):
    cache = target["cache"]
    manifest_path = cache / "manifest.json"
    keyframes_path = cache / "keyframes.jsonl"
    positives_path = cache / "positives.jsonl"
    for path in (manifest_path, keyframes_path, positives_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing or empty rebuilt cache file: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = read_jsonl(keyframes_path)
    positives = read_jsonl(positives_path)
    _, source_extrinsic = load_extrinsic(target["raw"])
    cached_extrinsic = np.asarray(
        manifest["T_camera_lidar"],
        dtype=np.float64,
    ).reshape(4, 4)
    if manifest.get("pose_frame") != "camera":
        raise ValueError(f"rebuilt cache is not camera-frame: {cache}")
    if manifest.get("source_trajectory_pose_frame") != "lidar":
        raise ValueError(f"rebuilt cache source pose frame is not lidar: {cache}")
    if manifest.get("gt_label_source") != "aster_slam_trajectory_keyframes":
        raise ValueError(f"rebuilt cache does not use AsterSLAM label trajectory: {cache}")
    if not np.allclose(cached_extrinsic, source_extrinsic, atol=1e-12):
        raise ValueError(f"rebuilt cache extrinsic differs from source metadata: {cache}")
    if int(manifest["keyframe_count"]) != len(rows) or len(rows) != len(positives):
        raise ValueError(f"rebuilt cache count mismatch: {cache}")
    missing_images = [
        int(row["idx"])
        for row in rows
        if row.get("image_path")
        and not (cache / str(row["image_path"])).is_file()
    ]
    if missing_images:
        raise ValueError(f"rebuilt cache contains missing images: {missing_images[:10]}")
    print(
        f"PASS cache {target['name']}: keyframes={len(rows)}"
        f" skipped_gt={int(manifest.get('skipped_gt_association_count', 0))}"
    )


if mode == "source loop datasets":
    for target in targets:
        validate_source(target)
elif mode == "rebuilt camera-frame caches":
    for target in targets:
        validate_cache(target)
else:
    raise ValueError(f"unknown validation mode: {mode}")
PY
}

cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

validate_datasets "source loop datasets"
if [[ "$check_only" -eq 1 ]]; then
  echo "Source loop dataset validation complete."
  exit 0
fi

fusion_helper="${SCRIPT_DIR}/generate_fusionportable_dataset_cache.sh"
fusion_cache_root="${FUSION_ROOT}/robust_loop_verifier_cache"

for sequence in handheld_escalator00 handheld_room00 handheld_room01; do
  run_preprocess env \
    PLATFORM=handheld \
    DATA_ROOT="$FUSION_ROOT" \
    OUTPUT_ROOT="$fusion_cache_root" \
    MAX_GT_DELTA_SEC="$MAX_GT_DELTA_SEC" \
    PYTHON_BIN="$PYTHON_BIN" \
    bash "$fusion_helper" "$sequence"
done

for sequence in ugv_campus01 ugv_parking01; do
  run_preprocess env \
    PLATFORM=ugv \
    DATA_ROOT="$FUSION_ROOT" \
    OUTPUT_ROOT="$fusion_cache_root" \
    MAX_GT_DELTA_SEC="$MAX_GT_DELTA_SEC" \
    PYTHON_BIN="$PYTHON_BIN" \
    bash "$fusion_helper" "$sequence"
done

for sequence in Offroad02_beta Offroad05_beta; do
  run_preprocess "$PYTHON_BIN" -m robust_loop_verifier.cli preprocess-fusionportable \
    --config "${REPO_ROOT}/configs/robust_loop_verifier/geode_offroad.yaml" \
    --raw-dir "${GEODE_ROOT}/geode_loop_dataset/Offroad/${sequence}/raw" \
    --gt-trajectory-file \
    "${GEODE_ROOT}/geode_loop_dataset/Offroad/${sequence}/raw/trajectory_keyframes.txt" \
    --sequence-name "$sequence" \
    --max-gt-delta-sec "$MAX_GT_DELTA_SEC"
done

for sequence in eee_01 eee_02 nya_02; do
  raw_dir="${NTU_ROOT}/ntu_viral_loop_dataset/NTU-VIRAL/${sequence}/raw"
  run_preprocess "$PYTHON_BIN" -m robust_loop_verifier.cli preprocess-fusionportable \
    --config "${REPO_ROOT}/configs/robust_loop_verifier/ntu_viral.yaml" \
    --raw-dir "$raw_dir" \
    --gt-trajectory-file "${raw_dir}/trajectory_keyframes.txt" \
    --sequence-name "$sequence" \
    --max-gt-delta-sec "$MAX_GT_DELTA_SEC"
done

validate_datasets "rebuilt camera-frame caches"
echo "Rebuilt all 10 camera-frame VPR caches."
