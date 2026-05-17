#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_fusionportable_handheld_legged_batch_sweep.sh [extra batch_score_sweep.py args]

Preferred:
  BATCH_SUMMARY=<batch_root>/batch_summary.json \
    run_fusionportable_handheld_legged_batch_sweep.sh

Alternative:
  RUN_ID=<same run id used by run_fusionportable_handheld_legged_batch.sh> \
    run_fusionportable_handheld_legged_batch_sweep.sh

Common environment overrides:
  PYTHON_BIN=/home/chenguyuan/anaconda3/envs/da3/bin/python
  RUNS_ROOT=<LoopAnything>/workspace/robust_loop_verifier_runs
  LOOP_DATASET_ROOT=/data/datasets/FusionPortable/fusionportable_loop_dataset
  DATASET_NAME=FusionPortableV2
  PLATFORMS=handheld,legged
  ATE_RMSE_THRESHOLD_M=0.1
  GRAPH_WEIGHTS=0,0.25,0.5,1,2,4
  FUSION_WEIGHTS=0,0.25,0.5,1,2,4
  OUTPUT_ROOT=<custom sweep output root>
  SKIP_MISSING=1
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
  "--loop-dataset-root" "${LOOP_DATASET_ROOT:-/data/datasets/FusionPortable/fusionportable_loop_dataset}"
  "--runs-root" "${RUNS_ROOT:-${REPO_ROOT}/workspace/robust_loop_verifier_runs}"
  "--dataset-name" "${DATASET_NAME:-FusionPortableV2}"
  "--platforms" "${PLATFORMS:-handheld,legged}"
  "--ate-rmse-threshold-m" "${ATE_RMSE_THRESHOLD_M:-0.1}"
  "--graph-weights" "${GRAPH_WEIGHTS:-0,0.25,0.5,1,2,4}"
  "--fusion-weights" "${FUSION_WEIGHTS:-0,0.25,0.5,1,2,4}"
)

if [[ -n "${BATCH_SUMMARY:-}" ]]; then
  args+=("--batch-summary" "${BATCH_SUMMARY}")
fi
if [[ -n "${RUN_ID:-}" ]]; then
  args+=("--run-id" "${RUN_ID}")
fi
if [[ -n "${OUTPUT_ROOT:-}" ]]; then
  args+=("--output-root" "${OUTPUT_ROOT}")
fi
if [[ "${SKIP_MISSING:-0}" == "1" ]]; then
  args+=("--skip-missing")
fi

cd "${REPO_ROOT}"
PYTHONPATH=src "${PYTHON_BIN}" "${SCRIPT_DIR}/batch_score_sweep.py" "${args[@]}" "$@"
