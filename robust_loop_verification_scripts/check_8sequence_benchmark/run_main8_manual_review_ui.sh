#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/chenguyuan/anaconda3/envs/da3/bin/python}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"
OPEN_BROWSER="${OPEN_BROWSER:-1}"

usage() {
  cat <<'EOF'
Usage: run_main8_manual_review_ui.sh REVIEW_ROOT [options]

Start the benchmark annotation UI for a review root created by
prepare_main8_full_manual_review.sh.

Options:
  --python PATH       Python executable. Default: da3 env Python.
  --host HOST         Bind host. Default: 127.0.0.1.
  --port PORT         Bind port. Default: 8765.
  --no-open           Do not auto-open browser.
  -h, --help          Show this help.

Environment overrides:
  PYTHON_BIN
  HOST
  PORT
  OPEN_BROWSER
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

REVIEW_ROOT="$1"
shift

require_value() {
  if [[ $# -lt 2 || -z "$2" || "$2" == -* ]]; then
    echo "Error: $1 requires a value" >&2
    usage >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      require_value "$@"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --host)
      require_value "$@"
      HOST="$2"
      shift 2
      ;;
    --port)
      require_value "$@"
      PORT="$2"
      shift 2
      ;;
    --no-open)
      OPEN_BROWSER=0
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

if [[ "$REVIEW_ROOT" != /* ]]; then
  REVIEW_ROOT="${REPO_ROOT}/${REVIEW_ROOT}"
fi
if [[ ! -f "${REVIEW_ROOT}/benchmark_pairs.jsonl" ]]; then
  echo "Error: missing benchmark_pairs.jsonl under review root: ${REVIEW_ROOT}" >&2
  exit 1
fi

cd "$REPO_ROOT"

args=(
  env PYTHONPATH=src "$PYTHON_BIN"
  robust_loop_verification_scripts/annotate_rover_aligned_benchmark.py
  "$REVIEW_ROOT"
  --host "$HOST"
  --port "$PORT"
)
if [[ "$OPEN_BROWSER" -eq 1 ]]; then
  args+=(--open)
fi

echo "Starting annotation UI"
echo "  review_root: ${REVIEW_ROOT}"
echo "  url: http://${HOST}:${PORT}/"
echo "  finalize in browser after all pairs are labeled"

"${args[@]}"
