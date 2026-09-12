#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/chenguyuan/anaconda3/envs/da3/bin/python}"
REVIEW_ROOT="${REVIEW_ROOT:-workspace/rover_aligned_benchmark/benchmark_v1_da3_auto_labels_v4_asterslam_main8_full_manual_review_20260701_013214_positive_recheck_main8_20260701_115646}"
CANDIDATE_RECORDS="${CANDIDATE_RECORDS:-workspace/rover_aligned_benchmark/benchmark_v1/candidate_records.jsonl}"
MR_TOLERANCE="${MR_TOLERANCE:-0.01}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

if [[ "${REVIEW_ROOT}" != /* ]]; then
  REVIEW_ROOT="${REPO_ROOT}/${REVIEW_ROOT}"
fi
if [[ "${CANDIDATE_RECORDS}" != /* ]]; then
  CANDIDATE_RECORDS="${REPO_ROOT}/${CANDIDATE_RECORDS}"
fi

args=(
  "${REVIEW_ROOT}"
  --candidate-records "${CANDIDATE_RECORDS}"
  --mr-tolerance "${MR_TOLERANCE}"
)

if [[ -n "${OUTPUT_DIR}" ]]; then
  if [[ "${OUTPUT_DIR}" != /* ]]; then
    OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"
  fi
  args+=(--output-dir "${OUTPUT_DIR}")
fi

cd "${REPO_ROOT}"
env PYTHONPATH=src "${PYTHON_BIN}" \
  robust_loop_verification_scripts/check_8sequence_benchmark/run_platform_shared_sweep.py \
  "${args[@]}"
