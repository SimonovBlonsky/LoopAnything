#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  robust_loop_verification_scripts/rerun_gt_pose_threshold_main_experiment.sh [options]

Rebuilds the legacy GT-pose-threshold VPR caches and reruns the full-sequence
robust loop verifier experiments used by LoopAnything/main_experiment.md.

Label rule:
  positive if translation <= 2.0 m and rotation <= 45 deg, after the configured
  recent-keyframe exclusion. These thresholds are read from the existing
  robust_loop_verifier configs.

Default sequence set:
  FusionPortableV2:
    handheld_escalator00, handheld_escalator01, handheld_grass00,
    handheld_room00, handheld_room01,
    ugv_campus01, ugv_parking00, ugv_parking01, ugv_parking02, ugv_parking03
  GEODE:
    Offroad02_beta, Offroad05_beta
  NTU-VIRAL:
    eee_01, eee_02, nya_01, nya_02, nya_03

Options:
  --run-id ID              Run id. Default: gt_pose_threshold_YYYYmmdd_HHMMSS
  --overwrite_dataset      Regenerate VPR caches before running.
  --keep-going             Continue if one dataset/sequence group fails.
  --dry-run                Print planned commands without running.
  --backend real|mock      Default: BACKEND env or real.
  --query-limit N          Optional query limit for debugging.
  --skip-fusionportable    Skip FusionPortableV2.
  --skip-geode             Skip GEODE.
  --skip-ntu               Skip NTU-VIRAL.
  --fusionportable-sequences CSV
                           Comma-separated platform/sequence entries, e.g.
                           handheld/handheld_room00,ugv/ugv_parking01.
  --geode-sequences CSV    Default: Offroad02_beta,Offroad05_beta.
  --ntu-sequences CSV      Default: eee_01,eee_02,nya_01,nya_02,nya_03.
  -h, --help               Show this help.

Environment:
  PYTHON_BIN               Default: /home/chenguyuan/anaconda3/envs/da3/bin/python
  BACKEND                  Default: real
  CUDA_VISIBLE_DEVICES     Passed through to DA3.

Example:
  RUN_ID="gt_pose_threshold_$(date +%Y%m%d_%H%M%S)"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  nohup robust_loop_verification_scripts/rerun_gt_pose_threshold_main_experiment.sh \
    --run-id "$RUN_ID" --overwrite_dataset --keep-going \
    > "workspace/${RUN_ID}.log" 2>&1 &
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/chenguyuan/anaconda3/envs/da3/bin/python}"
BACKEND_VALUE="${BACKEND:-real}"
RUN_ID="gt_pose_threshold_$(date +%Y%m%d_%H%M%S)"
QUERY_LIMIT_VALUE=""
OVERWRITE_DATASET=0
KEEP_GOING=0
DRY_RUN=0
RUN_FUSIONPORTABLE=1
RUN_GEODE=1
RUN_NTU=1
FUSIONPORTABLE_SEQUENCES=(
  "handheld/handheld_escalator00"
  "handheld/handheld_escalator01"
  "handheld/handheld_grass00"
  "handheld/handheld_room00"
  "handheld/handheld_room01"
  "ugv/ugv_campus01"
  "ugv/ugv_parking00"
  "ugv/ugv_parking01"
  "ugv/ugv_parking02"
  "ugv/ugv_parking03"
)
GEODE_SEQUENCES="Offroad02_beta,Offroad05_beta"
NTU_SEQUENCES="eee_01,eee_02,nya_01,nya_02,nya_03"

require_value() {
  if [[ $# -lt 2 || -z "$2" || "$2" == -* ]]; then
    echo "Error: $1 requires a value" >&2
    usage >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      require_value "$@"
      RUN_ID="$2"
      shift 2
      ;;
    --overwrite_dataset)
      OVERWRITE_DATASET=1
      shift
      ;;
    --keep-going)
      KEEP_GOING=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --backend)
      require_value "$@"
      BACKEND_VALUE="$2"
      shift 2
      ;;
    --query-limit)
      require_value "$@"
      QUERY_LIMIT_VALUE="$2"
      shift 2
      ;;
    --skip-fusionportable)
      RUN_FUSIONPORTABLE=0
      shift
      ;;
    --skip-geode)
      RUN_GEODE=0
      shift
      ;;
    --skip-ntu)
      RUN_NTU=0
      shift
      ;;
    --fusionportable-sequences)
      require_value "$@"
      IFS=',' read -r -a FUSIONPORTABLE_SEQUENCES <<< "$2"
      shift 2
      ;;
    --geode-sequences)
      require_value "$@"
      GEODE_SEQUENCES="$2"
      shift 2
      ;;
    --ntu-sequences)
      require_value "$@"
      NTU_SEQUENCES="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${BACKEND_VALUE}" in
  real|mock) ;;
  *)
    echo "Error: --backend must be real or mock" >&2
    exit 2
    ;;
esac

OUTPUT_BASE="${REPO_ROOT}/workspace/robust_loop_verifier_runs"
BATCH_ROOT="${OUTPUT_BASE}/gt_pose_threshold_main_${RUN_ID}"
mkdir -p "${BATCH_ROOT}/logs"

run_cmd() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '[dry-run]'
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  printf '[run]'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

append_common_args() {
  local -n out_ref=$1
  out_ref+=("--run-id" "${RUN_ID}")
  out_ref+=("--backend" "${BACKEND_VALUE}")
  out_ref+=("--output-base" "${OUTPUT_BASE}")
  if [[ -n "${QUERY_LIMIT_VALUE}" ]]; then
    out_ref+=("--query-limit" "${QUERY_LIMIT_VALUE}")
  fi
  if [[ "${OVERWRITE_DATASET}" -eq 1 ]]; then
    out_ref+=("--overwrite_dataset")
  fi
  if [[ "${KEEP_GOING}" -eq 1 ]]; then
    out_ref+=("--keep-going")
  fi
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    out_ref+=("--dry-run")
  fi
}

run_group() {
  local name="$1"
  shift
  local log_path="${BATCH_ROOT}/logs/${name}.log"
  echo "[group] ${name}"
  echo "  log: ${log_path}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    run_cmd "$@"
    return 0
  fi
  set +e
  "$@" 2>&1 | tee "${log_path}"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ "${status}" -ne 0 ]]; then
    echo "[error] ${name} failed with status ${status}" >&2
    if [[ "${KEEP_GOING}" -ne 1 ]]; then
      exit "${status}"
    fi
  fi
}

echo "GT-pose-threshold main experiment"
echo "  run_id: ${RUN_ID}"
echo "  output_base: ${OUTPUT_BASE}"
echo "  batch_root: ${BATCH_ROOT}"
echo "  backend: ${BACKEND_VALUE}"
echo "  overwrite_dataset: ${OVERWRITE_DATASET}"
echo "  label_rule: translation<=2.0m, rotation<=45deg"
echo "  support_ensemble: disabled"

cd "${REPO_ROOT}"

if [[ "${RUN_FUSIONPORTABLE}" -eq 1 ]]; then
  handheld_names=()
  ugv_names=()
  for item in "${FUSIONPORTABLE_SEQUENCES[@]}"; do
    platform="${item%%/*}"
    sequence="${item#*/}"
    case "${platform}" in
      handheld) handheld_names+=("${sequence}") ;;
      ugv) ugv_names+=("${sequence}") ;;
      *)
        echo "Error: FusionPortable sequence must be platform/sequence, got ${item}" >&2
        exit 2
        ;;
    esac
  done
  handheld_csv="$(IFS=,; echo "${handheld_names[*]}")"
  ugv_csv="$(IFS=,; echo "${ugv_names[*]}")"
  args=(
    "${SCRIPT_DIR}/fusionportablev2_robust_loop_verifier_batch_run.sh"
    "--batch-output-root" "${BATCH_ROOT}/FusionPortableV2"
    "--platforms" "handheld,ugv"
    "--handheld-sequences" "${handheld_csv}"
    "--ugv-sequences" "${ugv_csv}"
  )
  append_common_args args
  run_group "fusionportablev2" "${args[@]}"
fi

if [[ "${RUN_GEODE}" -eq 1 ]]; then
  args=(
    "${SCRIPT_DIR}/geode_robust_loop_verifier_batch_run.sh"
    "--batch-output-root" "${BATCH_ROOT}/GEODE"
    "--sequences" "${GEODE_SEQUENCES}"
    "--no-support-ensemble"
    "--max-gt-delta-sec" "0.1"
  )
  append_common_args args
  run_group "geode" "${args[@]}"
fi

if [[ "${RUN_NTU}" -eq 1 ]]; then
  args=(
    "${SCRIPT_DIR}/ntu_viral_robust_loop_verifier_batch_run.sh"
    "--batch-output-root" "${BATCH_ROOT}/NTU-VIRAL"
    "--sequences" "${NTU_SEQUENCES}"
    "--max-gt-delta-sec" "0.1"
  )
  append_common_args args
  run_group "ntu_viral" "${args[@]}"
fi

cat > "${BATCH_ROOT}/README.md" <<EOF
# GT-Pose-Threshold Main Experiment

- Run id: \`${RUN_ID}\`
- Label rule: translation <= 2.0 m and rotation <= 45 deg, using the existing
  robust loop verifier configs.
- Support ensemble: disabled.
- FusionPortableV2 output: \`FusionPortableV2/\`
- GEODE output: \`GEODE/\`
- NTU-VIRAL output: \`NTU-VIRAL/\`

Each dataset directory contains \`batch_summary.json\`,
\`batch_metrics.csv\`, and \`batch_metrics_average.csv\`.
EOF

echo "Outputs:"
echo "  ${BATCH_ROOT}/README.md"
echo "  ${BATCH_ROOT}/FusionPortableV2/batch_metrics.csv"
echo "  ${BATCH_ROOT}/GEODE/batch_metrics.csv"
echo "  ${BATCH_ROOT}/NTU-VIRAL/batch_metrics.csv"
