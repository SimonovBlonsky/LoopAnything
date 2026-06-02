#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ntu_viral_full_experiment.sh [options]

Default workflow:
  1. Convert NTU-VIRAL GT CSV files to TUM trajectories with heading rotations.
  2. Export AsterSLAM raw datasets for eee_01, eee_02, nya_01, nya_02, nya_03.
  3. Build/overwrite NTU-VIRAL VPR caches.
  4. Run the robust loop verifier pipeline without support ensemble.
  5. Run aggregate score sweep over the generated candidate records.

Examples:
  robust_loop_verification_scripts/ntu_viral_full_experiment.sh
  robust_loop_verification_scripts/ntu_viral_full_experiment.sh --dry-run
  robust_loop_verification_scripts/ntu_viral_full_experiment.sh --skip-export
  QUERY_LIMIT=80 robust_loop_verification_scripts/ntu_viral_full_experiment.sh

Options:
  --sequences CSV             Default: eee_01,eee_02,nya_01,nya_02,nya_03
  --skip-export               Skip AsterSLAM raw export and reuse existing raw data.
  --no-overwrite_dataset      Reuse existing VPR caches when available.
  --overwrite_dataset         Regenerate VPR caches. Default.
  --dry-run                   Print planned commands without running export/pipeline.
  --keep-going                Continue after per-sequence pipeline errors.
  -h, --help                  Show this help.

Environment overrides:
  NTU_DATA_ROOT               Default: /data/datasets/NTU-VIRAL/data
  NTU_GT_ROOT                 Default: /data/datasets/NTU-VIRAL/groundtruth
  NTU_GT_TUM_ROOT             Default: /data/datasets/NTU-VIRAL/processed_gt_tum
  LOOP_DATASET_ROOT           Default: /data/datasets/NTU-VIRAL/ntu_viral_loop_dataset
  CACHE_ROOT                  Default: /data/datasets/NTU-VIRAL/robust_loop_verifier_cache
  OUTPUT_BASE                 Default: LoopAnything/workspace/robust_loop_verifier_runs
  BATCH_ROOT                  Default: OUTPUT_BASE/NTU-VIRAL/ntu_viral_full_<RUN_ID>
  RUN_ID                      Default: current timestamp
  PYTHON_BIN                  Default: /home/chenguyuan/anaconda3/envs/da3/bin/python
  BACKEND                     Default: real
  QUERY_LIMIT                 Optional robust-loop-verifier query limit.
  MAX_GT_DELTA_SEC            Default: 0.1
  IMAGE_TOPIC                 Default: /left/image_raw
  ROSBAG_PLAY_ARGS            Default passed through to AsterSLAM export script.
  ALLOW_MISSING_IMAGES        Default passed through to AsterSLAM export script.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOOPANYTHING_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${LOOPANYTHING_ROOT}/.." && pwd)"
EXPORT_SCRIPT="${EXPORT_SCRIPT:-${WORKSPACE_ROOT}/AsterSLAM_ws/src/AsterSLAM/aster_slam/scripts/rerun_ntu_viral_kf05_export.sh}"
BATCH_SCRIPT="${SCRIPT_DIR}/ntu_viral_robust_loop_verifier_batch_run.sh"
SCORE_SWEEP_SCRIPT="${SCRIPT_DIR}/batch_score_sweep.py"
CONVERT_GT_SCRIPT="${SCRIPT_DIR}/convert_ntu_viral_gt_to_tum.py"

SEQUENCES="${SEQUENCES:-eee_01,eee_02,nya_01,nya_02,nya_03}"
SKIP_EXPORT=0
OVERWRITE_DATASET=1
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

NTU_DATA_ROOT="${NTU_DATA_ROOT:-/data/datasets/NTU-VIRAL/data}"
NTU_GT_ROOT="${NTU_GT_ROOT:-/data/datasets/NTU-VIRAL/groundtruth}"
NTU_GT_TUM_ROOT="${NTU_GT_TUM_ROOT:-/data/datasets/NTU-VIRAL/processed_gt_tum}"
LOOP_DATASET_ROOT="${LOOP_DATASET_ROOT:-/data/datasets/NTU-VIRAL/ntu_viral_loop_dataset}"
CACHE_ROOT="${CACHE_ROOT:-/data/datasets/NTU-VIRAL/robust_loop_verifier_cache}"
OUTPUT_BASE="${OUTPUT_BASE:-${LOOPANYTHING_ROOT}/workspace/robust_loop_verifier_runs}"
DATASET_NAME="${DATASET_NAME:-NTU-VIRAL}"
PLATFORM="${PLATFORM:-NTU-VIRAL}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
BATCH_ROOT="${BATCH_ROOT:-${OUTPUT_BASE}/${DATASET_NAME}/ntu_viral_full_${RUN_ID}}"
PYTHON_BIN="${PYTHON_BIN:-/home/chenguyuan/anaconda3/envs/da3/bin/python}"
BACKEND="${BACKEND:-real}"
MAX_GT_DELTA_SEC="${MAX_GT_DELTA_SEC:-0.1}"
IMAGE_TOPIC="${IMAGE_TOPIC:-/left/image_raw}"

print_config() {
  cat <<EOF
NTU-VIRAL full experiment:
  sequences: ${SEQUENCES}
  ntu_data_root: ${NTU_DATA_ROOT}
  ntu_gt_root: ${NTU_GT_ROOT}
  ntu_gt_tum_root: ${NTU_GT_TUM_ROOT}
  loop_dataset_root: ${LOOP_DATASET_ROOT}
  cache_root: ${CACHE_ROOT}
  output_base: ${OUTPUT_BASE}
  batch_root: ${BATCH_ROOT}
  run_id: ${RUN_ID}
  backend: ${BACKEND}
  support_ensemble: 0
  overwrite_dataset: ${OVERWRITE_DATASET}
  image_topic: ${IMAGE_TOPIC}
  skip_export: ${SKIP_EXPORT}
  dry_run: ${DRY_RUN}
EOF
}

convert_gt_sequence() {
  local sequence="$1"
  local csv_path="${NTU_GT_ROOT}/${sequence}/ground_truth.csv"
  local tum_path="${NTU_GT_TUM_ROOT}/${sequence}.txt"

  echo "[gt] ${sequence}"
  echo "  csv: ${csv_path}"
  echo "  tum: ${tum_path}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "  DRY-RUN: would convert NTU-VIRAL GT CSV to TUM"
    return
  fi
  "${PYTHON_BIN}" "${CONVERT_GT_SCRIPT}" \
    --input "${csv_path}" \
    --output "${tum_path}" \
    --rotation-source auto
}

export_raw_sequence() {
  local sequence="$1"
  local bag_path="${NTU_DATA_ROOT}/${sequence}/${sequence}.bag"
  local gt_tum_path="${NTU_GT_TUM_ROOT}/${sequence}.txt"
  local output_raw_dir="${LOOP_DATASET_ROOT}/${PLATFORM}/${sequence}/raw"

  echo "[export] ${sequence}"
  echo "  bag: ${bag_path}"
  echo "  gt_tum: ${gt_tum_path}"
  echo "  output_raw_dir: ${output_raw_dir}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "  DRY-RUN: would call ${EXPORT_SCRIPT}"
    return
  fi

  SEQUENCE_NAME="${sequence}" \
  NTU_DATA_ROOT="${NTU_DATA_ROOT}" \
  BAG_PATH="${bag_path}" \
  IMAGE_TOPIC="${IMAGE_TOPIC}" \
  NTU_GT_TUM_PATH="${gt_tum_path}" \
  OUTPUT_RAW_DIR="${output_raw_dir}" \
  "${EXPORT_SCRIPT}"
}

run_ntu_batch() {
  local batch_args=(
    --sequences "${SEQUENCES}"
    --loop-dataset-root "${LOOP_DATASET_ROOT}"
    --gt-data-root "${NTU_GT_TUM_ROOT}"
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
for sequence in "${sequence_items[@]}"; do
  sequence="${sequence//[[:space:]]/}"
  convert_gt_sequence "${sequence}"
done

if [[ "${SKIP_EXPORT}" == "0" ]]; then
  for sequence in "${sequence_items[@]}"; do
    sequence="${sequence//[[:space:]]/}"
    export_raw_sequence "${sequence}"
  done
fi

run_ntu_batch
run_batch_score_sweep

cat <<EOF
NTU-VIRAL full experiment outputs:
  batch_summary: ${BATCH_ROOT}/batch_summary.json
  batch_metrics: ${BATCH_ROOT}/batch_metrics.md
  batch_metrics_average: ${BATCH_ROOT}/batch_metrics_average.md
  score_sweep_average: ${BATCH_ROOT}/score_sweep/batch_score_sweep_average.md
EOF
