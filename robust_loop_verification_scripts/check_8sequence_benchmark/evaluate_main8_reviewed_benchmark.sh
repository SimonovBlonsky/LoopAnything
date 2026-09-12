#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/chenguyuan/anaconda3/envs/da3/bin/python}"
RUN_ID="${RUN_ID:-main8_manual_review_eval_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

usage() {
  cat <<'EOF'
Usage: evaluate_main8_reviewed_benchmark.sh REVIEW_ROOT [options]

Evaluate existing method score files against a finalized manual-review benchmark.
This does not recompute expensive model scores. It only reuses scores/*.jsonl and
the new annotations.jsonl from REVIEW_ROOT.

Options:
  --python PATH        Python executable. Default: da3 env Python.
  --run-id ID          Metrics run id.
  --output-dir PATH    Output metrics dir. Default: REVIEW_ROOT/metrics/RUN_ID.
  -h, --help           Show this help.

Environment overrides:
  PYTHON_BIN
  RUN_ID
  OUTPUT_DIR
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
    --run-id)
      require_value "$@"
      RUN_ID="$2"
      shift 2
      ;;
    --output-dir)
      require_value "$@"
      OUTPUT_DIR="$2"
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

if [[ "$REVIEW_ROOT" != /* ]]; then
  REVIEW_ROOT="${REPO_ROOT}/${REVIEW_ROOT}"
fi
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="${REVIEW_ROOT}/metrics/${RUN_ID}"
elif [[ "$OUTPUT_DIR" != /* ]]; then
  OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"
fi

if [[ ! -f "${REVIEW_ROOT}/annotations.jsonl" ]]; then
  echo "Error: missing finalized annotations.jsonl: ${REVIEW_ROOT}" >&2
  echo "Open the annotation UI and click Finalize after all pairs are labeled." >&2
  exit 1
fi
if [[ ! -f "${REVIEW_ROOT}/annotation_seal.json" ]]; then
  echo "Error: missing annotation_seal.json: ${REVIEW_ROOT}" >&2
  echo "Open the annotation UI and click Finalize after all pairs are labeled." >&2
  exit 1
fi

required_scores=(
  scores/dbow2.jsonl
  scores/netvlad.jsonl
  scores/salad.jsonl
  scores/boq_dinov2.jsonl
  scores/loftr.jsonl
  scores/dust3r.jsonl
  scores/mast3r.jsonl
  scores/vggt.jsonl
  scores/rover_like.jsonl
  scores/loopanything.jsonl
)
for score_file in "${required_scores[@]}"; do
  if [[ ! -f "${REVIEW_ROOT}/${score_file}" ]]; then
    echo "Error: missing score file: ${REVIEW_ROOT}/${score_file}" >&2
    exit 1
  fi
done

cd "$REPO_ROOT"

env PYTHONPATH=src "$PYTHON_BIN" \
  robust_loop_verification_scripts/evaluate_rover_aligned_benchmark.py \
  "$REVIEW_ROOT" \
  --method "DBoW2=scores/dbow2.jsonl" \
  --method "NetVLAD=scores/netvlad.jsonl" \
  --method "SALAD=scores/salad.jsonl" \
  --method "BoQ-dinov2=scores/boq_dinov2.jsonl" \
  --method "LoFTR=scores/loftr.jsonl" \
  --method "DUSt3R=scores/dust3r.jsonl" \
  --method "MAST3R=scores/mast3r.jsonl" \
  --method "VGGT-track=scores/vggt.jsonl" \
  --method "ROVER-like=scores/rover_like.jsonl" \
  --method "LoopAnything=scores/loopanything.jsonl" \
  --output-dir "$OUTPUT_DIR"

cat <<EOF
Evaluation outputs:
  ${OUTPUT_DIR}/table1.md
  ${OUTPUT_DIR}/metrics_per_sequence.csv
  ${OUTPUT_DIR}/metrics_summary.json

Note:
  Current paper Ours/Tuned LoopAnything is not scores/loopanything.jsonl.
  After the manual labels are finalized, rerun the calibrated candidate-record
  sweep separately to update Ours in table1.tex/table2.tex.
EOF
