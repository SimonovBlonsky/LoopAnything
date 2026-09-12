#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/chenguyuan/anaconda3/envs/da3/bin/python}"

benchmark_root="${REPO_ROOT}/workspace/rover_aligned_benchmark/benchmark_v1"
output_root=""
device="cuda"
backend="real"
pair_limit=""
port="8765"
min_translation_error_m="1.0"
max_translation_error_m="5.0"
translation_error_scale_ratio="0.2"
skip_build=0
summary_only=0

usage() {
  cat >&2 <<'EOF'
Usage: run_da3_geometry_annotation_prototype.sh [options]
  --benchmark-root PATH
  --output-root PATH
  --device DEVICE
  --backend real|mock
  --pair-limit POSITIVE_INT
  --min-translation-error-m FLOAT
  --max-translation-error-m FLOAT
  --translation-error-scale-ratio FLOAT
  --port POSITIVE_INT
  --skip-build
  --summary-only
EOF
}

require_value() {
  if [[ $# -lt 2 || -z "$2" || "$2" == -* ]]; then
    echo "error: $1 requires a value" >&2
    usage
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmark-root)
      require_value "$@"
      benchmark_root="$2"
      shift 2
      ;;
    --output-root)
      require_value "$@"
      output_root="$2"
      shift 2
      ;;
    --device)
      require_value "$@"
      device="$2"
      shift 2
      ;;
    --backend)
      require_value "$@"
      backend="$2"
      shift 2
      ;;
    --pair-limit)
      require_value "$@"
      pair_limit="$2"
      shift 2
      ;;
    --min-translation-error-m)
      require_value "$@"
      min_translation_error_m="$2"
      shift 2
      ;;
    --max-translation-error-m)
      require_value "$@"
      max_translation_error_m="$2"
      shift 2
      ;;
    --translation-error-scale-ratio)
      require_value "$@"
      translation_error_scale_ratio="$2"
      shift 2
      ;;
    --port)
      require_value "$@"
      port="$2"
      shift 2
      ;;
    --skip-build)
      skip_build=1
      shift
      ;;
    --summary-only)
      summary_only=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ "$backend" != "real" && "$backend" != "mock" ]]; then
  echo "error: --backend must be real or mock" >&2
  exit 2
fi
if [[ -n "$pair_limit" && ! "$pair_limit" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: --pair-limit must be a positive integer" >&2
  exit 2
fi
if [[ ! "$port" =~ ^[1-9][0-9]*$ || "$port" -gt 65535 ]]; then
  echo "error: --port must be an integer in [1, 65535]" >&2
  exit 2
fi
if [[ -z "$output_root" ]]; then
  output_root="${benchmark_root%/}/da3_geometry_annotation_v1"
fi

resolve_repo_path() {
  if [[ "$1" = /* ]]; then
    printf '%s\n' "$1"
  else
    printf '%s/%s\n' "$REPO_ROOT" "$1"
  fi
}

benchmark_root="$(resolve_repo_path "$benchmark_root")"
output_root="$(resolve_repo_path "$output_root")"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "$REPO_ROOT"

summary_script="${SCRIPT_DIR}/summarize_da3_geometry_annotations.py"

if [[ "$summary_only" -eq 1 ]]; then
  exec "$PYTHON_BIN" "$summary_script" \
    "$benchmark_root" \
    --geometry-root "$output_root"
fi

if [[ "$skip_build" -eq 0 ]]; then
  build_args=(
    "${SCRIPT_DIR}/build_da3_geometry_annotation.py"
    "$benchmark_root"
    --output-root "$output_root"
    --device "$device"
    --backend "$backend"
    --min-translation-error-m "$min_translation_error_m"
    --max-translation-error-m "$max_translation_error_m"
    --translation-error-scale-ratio "$translation_error_scale_ratio"
  )
  if [[ -n "$pair_limit" ]]; then
    build_args+=(--pair-limit "$pair_limit")
  fi
  "$PYTHON_BIN" "${build_args[@]}"
fi

if [[ -n "$pair_limit" ]]; then
  "$PYTHON_BIN" "$summary_script" \
    "$benchmark_root" \
    --geometry-root "$output_root" \
    --validate-predictions-only \
    --allow-partial-predictions
  echo "Smoke bundle ready: ${output_root}"
  exit 0
fi

"$PYTHON_BIN" "$summary_script" \
  "$benchmark_root" \
  --geometry-root "$output_root" \
  --validate-predictions-only

exec "$PYTHON_BIN" "${SCRIPT_DIR}/annotate_rover_aligned_benchmark.py" \
  "$output_root" \
  --port "$port" \
  --open
