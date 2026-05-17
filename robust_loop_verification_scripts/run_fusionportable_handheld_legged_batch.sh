#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_fusionportable_handheld_legged_batch.sh [extra batch_fusionportable_handheld_legged.py args]

Default behavior:
  1. Scans handheld and legged sequences under fusionportable_loop_dataset.
  2. Keeps sequences with raw/evo_ape_summary.txt rmse < 0.1 m.
  3. Regenerates each VPR cache.
  4. Runs the robust loop verifier pipeline for each selected sequence.
  5. Writes batch_summary.{json,csv,md} and batch_metrics.{csv,md}.

Common environment overrides:
  PYTHON_BIN=/home/chenguyuan/anaconda3/envs/da3/bin/python
  LOOP_DATASET_ROOT=/data/datasets/FusionPortable/fusionportable_loop_dataset
  CACHE_ROOT=/data/datasets/FusionPortable/robust_loop_verifier_cache
  OUTPUT_BASE=<LoopAnything>/workspace/robust_loop_verifier_runs
  RUN_ID=20260517_all
  QUERY_LIMIT=200
  BACKEND=real
  ATE_RMSE_THRESHOLD_M=0.1
  DRY_RUN=1
  KEEP_GOING=1
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/chenguyuan/anaconda3/envs/da3/bin/python}"

args=(
  "--repo-root" "${REPO_ROOT}"
  "--loop-dataset-root" "${LOOP_DATASET_ROOT:-/data/datasets/FusionPortable/fusionportable_loop_dataset}"
  "--cache-root" "${CACHE_ROOT:-/data/datasets/FusionPortable/robust_loop_verifier_cache}"
  "--output-base" "${OUTPUT_BASE:-${REPO_ROOT}/workspace/robust_loop_verifier_runs}"
  "--dataset-name" "${DATASET_NAME:-FusionPortableV2}"
  "--platforms" "${PLATFORMS:-handheld,legged}"
  "--ate-rmse-threshold-m" "${ATE_RMSE_THRESHOLD_M:-0.1}"
  "--backend" "${BACKEND:-real}"
  "--python-bin" "${PYTHON_BIN}"
)

if [[ -n "${RUN_ID:-}" ]]; then
  args+=("--run-id" "${RUN_ID}")
fi
if [[ -n "${QUERY_LIMIT:-}" ]]; then
  args+=("--query-limit" "${QUERY_LIMIT}")
fi
if [[ -n "${CONFIG:-}" ]]; then
  args+=("--config" "${CONFIG}")
fi
if [[ -n "${BATCH_OUTPUT_ROOT:-}" ]]; then
  args+=("--batch-output-root" "${BATCH_OUTPUT_ROOT}")
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  args+=("--dry-run")
fi
if [[ "${KEEP_GOING:-0}" == "1" ]]; then
  args+=("--keep-going")
fi

cd "${REPO_ROOT}"
PYTHONPATH=src "${PYTHON_BIN}" "${SCRIPT_DIR}/batch_fusionportable_handheld_legged.py" "${args[@]}" "$@"
