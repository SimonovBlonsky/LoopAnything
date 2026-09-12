#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  generate_fusionportable_dataset_cache.sh [SEQUENCE_NAME]

Examples:
  generate_fusionportable_dataset_cache.sh ugv_parking00
  generate_fusionportable_dataset_cache.sh ugv_parking01
  PLATFORM=handheld generate_fusionportable_dataset_cache.sh handheld_escalator00

Environment overrides:
  PLATFORM, RAW_DIR, GT_TRAJECTORY_FILE, DATA_ROOT, LOOP_DATASET_ROOT,
  GT_DATA_ROOT, PROCESSED_GT_ROOT, USE_PROCESSED_GT, OUTPUT_ROOT,
  CONFIG, PYTHON_BIN, MAX_GT_DELTA_SEC

Note:
  handheld, legged, and ugv default to AsterSLAM raw/trajectory_keyframes.txt
  as the GT label source; external GT paths are used only for other platforms.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "$#" -gt 1 ]]; then
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/chenguyuan/anaconda3/envs/da3/bin/python}"
BASE_CONFIG="${CONFIG:-${REPO_ROOT}/configs/robust_loop_verifier/fusionportablev2_handheld.yaml}"
DATA_ROOT="${DATA_ROOT:-/data/datasets/FusionPortable}"
LOOP_DATASET_ROOT="${LOOP_DATASET_ROOT:-${DATA_ROOT}/fusionportable_loop_dataset}"
GT_DATA_ROOT="${GT_DATA_ROOT:-${DATA_ROOT}}"
PROCESSED_GT_ROOT="${PROCESSED_GT_ROOT:-${DATA_ROOT}/processed_handheld_legged_gt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/robust_loop_verifier_cache}"
SEQUENCE_NAME="${1:-${SEQUENCE_NAME:-handheld_escalator00}}"
MAX_GT_DELTA_SEC="${MAX_GT_DELTA_SEC:-3.2}"
USE_PROCESSED_GT="${USE_PROCESSED_GT:-1}"

infer_platform() {
  local sequence_name="$1"
  local matches=()
  local platform
  for platform in handheld legged ugv vehicle; do
    if [[ -d "${LOOP_DATASET_ROOT}/${platform}/${sequence_name}/raw" ]]; then
      matches+=("${platform}")
    fi
  done

  if [[ "${#matches[@]}" -eq 0 ]]; then
    echo "Could not infer platform for sequence '${sequence_name}' under ${LOOP_DATASET_ROOT}" >&2
    return 1
  fi
  if [[ "${#matches[@]}" -gt 1 ]]; then
    echo "Sequence '${sequence_name}' matches multiple platforms: ${matches[*]}; set PLATFORM" >&2
    return 1
  fi
  echo "${matches[0]}"
}

PLATFORM="${PLATFORM:-$(infer_platform "${SEQUENCE_NAME}")}"
RAW_DIR="${RAW_DIR:-${LOOP_DATASET_ROOT}/${PLATFORM}/${SEQUENCE_NAME}/raw}"
DEFAULT_GT_FILE="${GT_DATA_ROOT}/${PLATFORM}/${SEQUENCE_NAME}/${SEQUENCE_NAME}.txt"
PROCESSED_GT_FILE="${PROCESSED_GT_ROOT}/${PLATFORM}/${SEQUENCE_NAME}/${SEQUENCE_NAME}.txt"

if [[ "${PLATFORM}" == "handheld" || "${PLATFORM}" == "legged" || "${PLATFORM}" == "ugv" ]]; then
  GT_TRAJECTORY_FILE="${RAW_DIR}/trajectory_keyframes.txt"
elif [[ -z "${GT_TRAJECTORY_FILE:-}" ]]; then
  if [[ "${USE_PROCESSED_GT}" != "0" && -f "${PROCESSED_GT_FILE}" ]]; then
    GT_TRAJECTORY_FILE="${PROCESSED_GT_FILE}"
  else
    GT_TRAJECTORY_FILE="${DEFAULT_GT_FILE}"
  fi
fi

if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "Config does not exist: ${BASE_CONFIG}" >&2
  exit 1
fi
if [[ ! -d "${RAW_DIR}" ]]; then
  echo "Raw directory does not exist: ${RAW_DIR}" >&2
  exit 1
fi
if [[ ! -f "${GT_TRAJECTORY_FILE}" ]]; then
  echo "GT trajectory file does not exist: ${GT_TRAJECTORY_FILE}" >&2
  exit 1
fi

TMP_CONFIG="$(mktemp "${TMPDIR:-/tmp}/robust-loop-verifier-${SEQUENCE_NAME}.XXXXXX.yaml")"
trap 'rm -f "${TMP_CONFIG}"' EXIT

cd "${REPO_ROOT}"

PYTHONPATH=src "${PYTHON_BIN}" - "${BASE_CONFIG}" "${TMP_CONFIG}" "${PLATFORM}" "${LOOP_DATASET_ROOT}" "${GT_DATA_ROOT}" "${OUTPUT_ROOT}" <<'PY'
import sys
from pathlib import Path

from robust_loop_verifier.io import read_yaml, write_yaml

base_config, output_config, platform, input_root, gt_root, output_root = sys.argv[1:]
data = dict(read_yaml(Path(base_config)))
data["platform"] = platform
data["input_root"] = input_root
data["gt_root"] = gt_root
data["output_root"] = output_root
write_yaml(Path(output_config), data)
PY

echo "Regenerating robust loop verifier cache"
echo "  base_config: ${BASE_CONFIG}"
echo "  temp_config: ${TMP_CONFIG}"
echo "  platform: ${PLATFORM}"
echo "  sequence_name: ${SEQUENCE_NAME}"
echo "  raw_dir: ${RAW_DIR}"
echo "  gt_trajectory_file: ${GT_TRAJECTORY_FILE}"
if [[ "${PLATFORM}" == "handheld" || "${PLATFORM}" == "legged" || "${PLATFORM}" == "ugv" ]]; then
  echo "  gt_label_source: aster_slam_trajectory_keyframes"
else
  echo "  gt_label_source: external_trajectory_timestamp_association"
fi
echo "  output_root: ${OUTPUT_ROOT}"
echo "  max_gt_delta_sec: ${MAX_GT_DELTA_SEC}"

PYTHONPATH=src "${PYTHON_BIN}" -m robust_loop_verifier.cli preprocess-fusionportable \
  --config "${TMP_CONFIG}" \
  --raw-dir "${RAW_DIR}" \
  --gt-trajectory-file "${GT_TRAJECTORY_FILE}" \
  --sequence-name "${SEQUENCE_NAME}" \
  --max-gt-delta-sec "${MAX_GT_DELTA_SEC}"
