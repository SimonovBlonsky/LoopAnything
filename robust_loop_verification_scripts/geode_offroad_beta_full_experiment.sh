#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  geode_offroad_beta_full_experiment.sh [options]

Default workflow:
  1. Export GEODE Offroad01_beta and Offroad02_beta raw datasets with AsterSLAM.
  2. Build/overwrite GEODE VPR caches.
  3. Run the robust loop verifier pipeline with support ensemble enabled.
  4. Run aggregate score sweep over the generated candidate records.

Examples:
  robust_loop_verification_scripts/geode_offroad_beta_full_experiment.sh
  robust_loop_verification_scripts/geode_offroad_beta_full_experiment.sh --dry-run
  robust_loop_verification_scripts/geode_offroad_beta_full_experiment.sh --skip-export
  QUERY_LIMIT=80 robust_loop_verification_scripts/geode_offroad_beta_full_experiment.sh

Options:
  --sequences CSV             Default: Offroad01_beta,Offroad02_beta
  --skip-export               Skip AsterSLAM raw export and reuse existing raw data.
  --no-overwrite_dataset      Reuse existing VPR caches when available.
  --overwrite_dataset         Regenerate VPR caches. Default.
  --support-ensemble          Enable support ensemble. Default.
  --no-support-ensemble       Disable support ensemble.
  --dry-run                   Print planned commands without running export/pipeline.
  --keep-going                Continue after per-sequence pipeline errors.
  -h, --help                  Show this help.

Environment overrides:
  GEODE_DATA_ROOT             Default: /data/datasets/GEODE/data/offroad
  LOOP_DATASET_ROOT           Default: /data/datasets/GEODE/geode_loop_dataset
  CACHE_ROOT                  Default: /data/datasets/GEODE/robust_loop_verifier_cache
  OUTPUT_BASE                 Default: LoopAnything/workspace/robust_loop_verifier_runs
  BATCH_ROOT                  Default: OUTPUT_BASE/GEODE/geode_offroad_beta_full_<RUN_ID>
  RUN_ID                      Default: current timestamp
  PYTHON_BIN                  Default: /home/chenguyuan/anaconda3/envs/da3/bin/python
  BACKEND                     Default: real
  QUERY_LIMIT                 Optional robust-loop-verifier query limit.
  MAX_GT_DELTA_SEC            Default: 0.1
  GEODE_LIDAR                 Default: geode_beta_livo
  GEODE_CALIB_NAME            Default: beta
  GEODE_IMAGE_TOPIC           Default: /left_camera/image/compressed
  ROSBAG_PLAY_ARGS            Default passed through to AsterSLAM export script.
  ALLOW_MISSING_IMAGES        Default passed through to AsterSLAM export script.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOOPANYTHING_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${LOOPANYTHING_ROOT}/.." && pwd)"
EXPORT_SCRIPT="${EXPORT_SCRIPT:-${WORKSPACE_ROOT}/AsterSLAM_ws/src/AsterSLAM/aster_slam/scripts/rerun_geode_offroad01_alpha_kf05_export.sh}"
BATCH_SCRIPT="${SCRIPT_DIR}/geode_robust_loop_verifier_batch_run.sh"
SCORE_SWEEP_SCRIPT="${SCRIPT_DIR}/batch_score_sweep.py"

SEQUENCES="${SEQUENCES:-Offroad01_beta,Offroad02_beta}"
SKIP_EXPORT=0
OVERWRITE_DATASET=1
SUPPORT_ENSEMBLE=1
DRY_RUN=0
KEEP_GOING=0

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --sequences)
      SEQUENCES="$2"
      shift 2
      ;;
    --skip-export)
      SKIP_EXPORT=1
      shift
      ;;
    --overwrite_dataset)
      OVERWRITE_DATASET=1
      shift
      ;;
    --no-overwrite_dataset)
      OVERWRITE_DATASET=0
      shift
      ;;
    --support-ensemble)
      SUPPORT_ENSEMBLE=1
      shift
      ;;
    --no-support-ensemble)
      SUPPORT_ENSEMBLE=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --keep-going)
      KEEP_GOING=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

GEODE_DATA_ROOT="${GEODE_DATA_ROOT:-/data/datasets/GEODE/data/offroad}"
LOOP_DATASET_ROOT="${LOOP_DATASET_ROOT:-/data/datasets/GEODE/geode_loop_dataset}"
CACHE_ROOT="${CACHE_ROOT:-/data/datasets/GEODE/robust_loop_verifier_cache}"
OUTPUT_BASE="${OUTPUT_BASE:-${LOOPANYTHING_ROOT}/workspace/robust_loop_verifier_runs}"
DATASET_NAME="${DATASET_NAME:-GEODE}"
PLATFORM="${PLATFORM:-Offroad}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
BATCH_ROOT="${BATCH_ROOT:-${OUTPUT_BASE}/${DATASET_NAME}/geode_offroad_beta_full_${RUN_ID}}"
PYTHON_BIN="${PYTHON_BIN:-/home/chenguyuan/anaconda3/envs/da3/bin/python}"
BACKEND="${BACKEND:-real}"
MAX_GT_DELTA_SEC="${MAX_GT_DELTA_SEC:-0.1}"
GEODE_LIDAR="${GEODE_LIDAR:-geode_beta_livo}"
GEODE_CALIB_NAME="${GEODE_CALIB_NAME:-beta}"
GEODE_IMAGE_TOPIC="${GEODE_IMAGE_TOPIC:-/left_camera/image/compressed}"

sequence_number() {
  local sequence="$1"
  if [[ ! "${sequence}" =~ ^Offroad0*([0-9]+)_beta$ ]]; then
    echo "Unsupported GEODE beta sequence name: ${sequence}" >&2
    exit 2
  fi
  echo "$((10#${BASH_REMATCH[1]}))"
}

print_config() {
  cat <<EOF
GEODE beta full experiment:
  sequences: ${SEQUENCES}
  geode_data_root: ${GEODE_DATA_ROOT}
  loop_dataset_root: ${LOOP_DATASET_ROOT}
  cache_root: ${CACHE_ROOT}
  output_base: ${OUTPUT_BASE}
  batch_root: ${BATCH_ROOT}
  run_id: ${RUN_ID}
  backend: ${BACKEND}
  support_ensemble: ${SUPPORT_ENSEMBLE}
  overwrite_dataset: ${OVERWRITE_DATASET}
  image_topic: ${GEODE_IMAGE_TOPIC}
  skip_export: ${SKIP_EXPORT}
  dry_run: ${DRY_RUN}
EOF
}

export_raw_sequence() {
  local sequence="$1"
  local number
  number="$(sequence_number "${sequence}")"
  local sequence_dir="${GEODE_DATA_ROOT}/Offroad${number}"
  local bag_path="${sequence_dir}/Offroad${number}_beta.bag"
  local gt_path="${sequence_dir}/Offroad${number}.txt"
  local output_raw_dir="${LOOP_DATASET_ROOT}/${PLATFORM}/${sequence}/raw"

  echo "[export] ${sequence}"
  echo "  bag: ${bag_path}"
  echo "  gt: ${gt_path}"
  echo "  output_raw_dir: ${output_raw_dir}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "  DRY-RUN: would call ${EXPORT_SCRIPT}"
    return
  fi

  SEQUENCE_NAME="${sequence}" \
  GEODE_LIDAR="${GEODE_LIDAR}" \
  GEODE_CALIB_NAME="${GEODE_CALIB_NAME}" \
  IMAGE_TOPIC="${GEODE_IMAGE_TOPIC}" \
  BAG_PATH="${bag_path}" \
  GEODE_GT_PATH="${gt_path}" \
  OUTPUT_RAW_DIR="${output_raw_dir}" \
  "${EXPORT_SCRIPT}"
}

run_geode_batch() {
  local batch_args=(
    --sequences "${SEQUENCES}"
    --loop-dataset-root "${LOOP_DATASET_ROOT}"
    --gt-data-root "${GEODE_DATA_ROOT}"
    --cache-root "${CACHE_ROOT}"
    --output-base "${OUTPUT_BASE}"
    --batch-output-root "${BATCH_ROOT}"
    --dataset-name "${DATASET_NAME}"
    --platform "${PLATFORM}"
    --run-id "${RUN_ID}"
    --backend "${BACKEND}"
    --max-gt-delta-sec "${MAX_GT_DELTA_SEC}"
  )

  if [[ "${OVERWRITE_DATASET}" == "1" ]]; then
    batch_args+=(--overwrite_dataset)
  fi
  if [[ "${SUPPORT_ENSEMBLE}" == "1" ]]; then
    batch_args+=(--support-ensemble)
  else
    batch_args+=(--no-support-ensemble)
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    batch_args+=(--dry-run)
  fi
  if [[ "${KEEP_GOING}" == "1" ]]; then
    batch_args+=(--keep-going)
  fi
  if [[ -n "${QUERY_LIMIT:-}" ]]; then
    batch_args+=(--query-limit "${QUERY_LIMIT}")
  fi

  echo "[run] ${BATCH_SCRIPT} ${batch_args[*]}"
  PYTHON_BIN="${PYTHON_BIN}" "${BATCH_SCRIPT}" "${batch_args[@]}"
}

run_batch_score_sweep() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[score-sweep] DRY-RUN: would aggregate ${BATCH_ROOT}/batch_summary.json"
    return
  fi

  echo "[score-sweep] ${BATCH_ROOT}/batch_summary.json"
  cd "${LOOPANYTHING_ROOT}"
  PYTHONPATH=src "${PYTHON_BIN}" "${SCORE_SWEEP_SCRIPT}" \
    --batch-summary "${BATCH_ROOT}/batch_summary.json" \
    --dataset-name "${DATASET_NAME}" \
    --output-root "${BATCH_ROOT}/score_sweep" \
    --skip-missing
}

print_config

IFS=',' read -r -a sequence_items <<< "${SEQUENCES}"
if [[ "${SKIP_EXPORT}" == "0" ]]; then
  for sequence in "${sequence_items[@]}"; do
    export_raw_sequence "${sequence//[[:space:]]/}"
  done
fi

run_geode_batch
run_batch_score_sweep

cat <<EOF
GEODE beta full experiment outputs:
  batch_summary: ${BATCH_ROOT}/batch_summary.json
  batch_metrics: ${BATCH_ROOT}/batch_metrics.md
  batch_metrics_average: ${BATCH_ROOT}/batch_metrics_average.md
  score_sweep_average: ${BATCH_ROOT}/score_sweep/batch_score_sweep_average.md
EOF
