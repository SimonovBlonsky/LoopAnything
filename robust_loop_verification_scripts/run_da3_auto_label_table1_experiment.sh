#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SOURCE_BENCHMARK_ROOT="workspace/rover_aligned_benchmark/benchmark_v1"
GEOMETRY_ROOT="workspace/rover_aligned_benchmark/benchmark_v1/da3_geometry_annotation_camera_frame_v4_asterslam_labels"
AUTO_BENCHMARK_ROOT="workspace/rover_aligned_benchmark/benchmark_v1_da3_auto_labels_v4_asterslam"
RUN_ID="${RUN_ID:-da3_auto_label_table1_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-/home/chenguyuan/anaconda3/envs/da3/bin/python}"
DEVICE="cuda"
BACKEND="real"
RECOMPUTE_SCORES=0
OVERWRITE_AUTO_LABELS=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: run_da3_auto_label_table1_experiment.sh [options]

Converts DA3 geometry_predictions automatic labels to a sealed benchmark root,
then runs the Table 1 comparison against DBoW2 / NetVLAD / SALAD / ROVER-like / LoopAnything.

Options:
  --source-benchmark-root PATH   Original frozen benchmark root.
  --geometry-root PATH           DA3 geometry annotation output root.
  --auto-benchmark-root PATH     Output sealed auto-label benchmark root.
  --run-id ID                    Run id for metrics/log naming.
  --python PATH                  Python executable.
  --device DEVICE                cuda|cpu; passed to scoring.
  --backend real|mock            Scoring backend.
  --overwrite-auto-labels        Recreate auto-label benchmark root files.
  --recompute-scores             Recompute method scores instead of reusing copied scores.
  --dry-run                      Print commands only.
  -h, --help                     Show help.
EOF
}

require_value() {
  if [[ $# -lt 2 || -z "$2" || "$2" == -* ]]; then
    echo "Error: $1 requires a value" >&2
    usage >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-benchmark-root)
      require_value "$@"
      SOURCE_BENCHMARK_ROOT="$2"
      shift 2
      ;;
    --geometry-root)
      require_value "$@"
      GEOMETRY_ROOT="$2"
      shift 2
      ;;
    --auto-benchmark-root)
      require_value "$@"
      AUTO_BENCHMARK_ROOT="$2"
      shift 2
      ;;
    --run-id)
      require_value "$@"
      RUN_ID="$2"
      shift 2
      ;;
    --python)
      require_value "$@"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --device)
      require_value "$@"
      DEVICE="$2"
      shift 2
      ;;
    --backend)
      require_value "$@"
      BACKEND="$2"
      shift 2
      ;;
    --overwrite-auto-labels)
      OVERWRITE_AUTO_LABELS=1
      shift
      ;;
    --recompute-scores)
      RECOMPUTE_SCORES=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
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

case "${BACKEND}" in
  real|mock) ;;
  *)
    echo "Error: --backend must be real or mock" >&2
    exit 2
    ;;
esac

resolve_repo_path() {
  if [[ "$1" = /* ]]; then
    printf '%s\n' "$1"
  else
    printf '%s/%s\n' "$REPO_ROOT" "$1"
  fi
}

SOURCE_BENCHMARK_ROOT="$(resolve_repo_path "$SOURCE_BENCHMARK_ROOT")"
GEOMETRY_ROOT="$(resolve_repo_path "$GEOMETRY_ROOT")"
AUTO_BENCHMARK_ROOT="$(resolve_repo_path "$AUTO_BENCHMARK_ROOT")"
METRICS_DIR="${AUTO_BENCHMARK_ROOT}/metrics/${RUN_ID}"

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

cd "$REPO_ROOT"

echo "DA3 auto-label Table 1 experiment"
echo "  source_benchmark_root: ${SOURCE_BENCHMARK_ROOT}"
echo "  geometry_root: ${GEOMETRY_ROOT}"
echo "  auto_benchmark_root: ${AUTO_BENCHMARK_ROOT}"
echo "  metrics_dir: ${METRICS_DIR}"
echo "  run_id: ${RUN_ID}"
echo "  backend: ${BACKEND}"
echo "  device: ${DEVICE}"
echo "  recompute_scores: ${RECOMPUTE_SCORES}"
echo "  overwrite_auto_labels: ${OVERWRITE_AUTO_LABELS}"

if [[ "${OVERWRITE_AUTO_LABELS}" -eq 0 \
  && -f "${AUTO_BENCHMARK_ROOT}/annotations.jsonl" \
  && -f "${AUTO_BENCHMARK_ROOT}/annotation_seal.json" ]]; then
  echo "[skip] auto-label conversion already exists: ${AUTO_BENCHMARK_ROOT}"
else
  convert_args=(
    env PYTHONPATH=src "${PYTHON_BIN}"
    robust_loop_verification_scripts/convert_da3_geometry_predictions_to_rover_annotations.py
    "${SOURCE_BENCHMARK_ROOT}"
    --geometry-root "${GEOMETRY_ROOT}"
    --output-root "${AUTO_BENCHMARK_ROOT}"
    --copy-scores
  )
  if [[ "${OVERWRITE_AUTO_LABELS}" -eq 1 ]]; then
    convert_args+=(--overwrite)
  fi
  run_cmd "${convert_args[@]}"
fi

table_args=(
  robust_loop_verification_scripts/run_rover_aligned_table1_experiment.sh
  --benchmark-root "${AUTO_BENCHMARK_ROOT}"
  --output-dir "${METRICS_DIR}"
  --python "${PYTHON_BIN}"
  --device "${DEVICE}"
  --backend "${BACKEND}"
)
if [[ "${RECOMPUTE_SCORES}" -eq 0 ]]; then
  table_args+=(--skip-existing)
fi
run_cmd "${table_args[@]}"

echo "Outputs:"
echo "  ${METRICS_DIR}/table1.md"
echo "  ${METRICS_DIR}/metrics_per_sequence.csv"
echo "  ${METRICS_DIR}/metrics_summary.json"
