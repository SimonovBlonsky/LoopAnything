#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BENCHMARK_ROOT="workspace/rover_aligned_benchmark/benchmark_v1"
PYTHON_BIN="${PYTHON_BIN:-/home/chenguyuan/anaconda3/envs/da3/bin/python}"
LOFTR_PYTHON="${LOFTR_PYTHON:-/home/chenguyuan/anaconda3/envs/loftr/bin/python}"
LOFTR_ROOT="${LOFTR_ROOT:-/home/chenguyuan/code/NeurIPS26/LoFTR}"
LOFTR_CKPT="${LOFTR_CKPT:-/home/chenguyuan/code/NeurIPS26/LoFTR/data/weights/indoor_ds_new.ckpt}"
DUST3R_PYTHON="${DUST3R_PYTHON:-/home/chenguyuan/anaconda3/envs/da3/bin/python}"
DUST3R_ROOT="${DUST3R_ROOT:-/home/chenguyuan/code/NeurIPS26/dust3r}"
DUST3R_CKPT="${DUST3R_CKPT:-/home/chenguyuan/code/NeurIPS26/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth}"
MAST3R_PYTHON="${MAST3R_PYTHON:-/home/chenguyuan/anaconda3/envs/mast3r/bin/python}"
MAST3R_ROOT="${MAST3R_ROOT:-/home/chenguyuan/code/NeurIPS26/mast3r}"
MAST3R_CKPT="${MAST3R_CKPT:-/home/chenguyuan/code/NeurIPS26/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"
DEVICE="cuda"
BACKEND="real"
OUTPUT_DIR=""
DRY_RUN=0
SKIP_EXISTING=0
INCLUDE_LOFTR=0
INCLUDE_DUST3R=0
INCLUDE_MAST3R=0

usage() {
  cat <<'EOF'
Usage: run_rover_aligned_table1_experiment.sh [options]

Runs the sealed ROVER-aligned benchmark Table 1 experiment:
  1. NetVLAD pair scoring
  2. SALAD pair scoring
  3. Loop verifier scoring, producing ROVER-like and LoopAnything scores
  4. Unified AP/MR evaluation table

Options:
  --benchmark-root PATH   Benchmark root (default: workspace/rover_aligned_benchmark/benchmark_v1)
  --output-dir PATH       Metrics output dir (default: BENCHMARK_ROOT/metrics)
  --python PATH           Python executable (default: PYTHON_BIN env or da3 env python)
  --device DEVICE         Scoring device passed to scorers (default: cuda)
  --backend real|mock     Scoring backend (default: real)
  --include-loftr         Also score/evaluate LoFTR geometric verification baseline
  --loftr-python PATH     LoFTR env Python (default: LOFTR_PYTHON env or loftr env python)
  --loftr-root PATH       LoFTR repository path (default: LOFTR_ROOT env or ../LoFTR)
  --loftr-ckpt PATH       LoFTR checkpoint path (default: LOFTR_CKPT env or indoor_ds_new.ckpt)
  --include-dust3r        Also score/evaluate DUSt3R geometric verification baseline
  --dust3r-python PATH    DUSt3R env Python (default: DUST3R_PYTHON env or da3 env python)
  --dust3r-root PATH      DUSt3R repository path (default: DUST3R_ROOT env or ../dust3r)
  --dust3r-ckpt PATH      DUSt3R checkpoint path (default: DUST3R_CKPT env or 512_dpt checkpoint)
  --include-mast3r        Also score/evaluate MAST3R geometric verification baseline
  --mast3r-python PATH    MAST3R env Python (default: MAST3R_PYTHON env or mast3r env python)
  --mast3r-root PATH      MAST3R repository path (default: MAST3R_ROOT env or ../mast3r)
  --mast3r-ckpt PATH      MAST3R checkpoint path (default: MAST3R_CKPT env or metric checkpoint)
  --skip-existing         Skip scorer steps whose expected score files already exist
  --dry-run               Print commands without executing them
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmark-root)
      BENCHMARK_ROOT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --backend)
      BACKEND="$2"
      shift 2
      ;;
    --include-loftr)
      INCLUDE_LOFTR=1
      shift
      ;;
    --loftr-python)
      LOFTR_PYTHON="$2"
      shift 2
      ;;
    --loftr-root)
      LOFTR_ROOT="$2"
      shift 2
      ;;
    --loftr-ckpt)
      LOFTR_CKPT="$2"
      shift 2
      ;;
    --include-dust3r)
      INCLUDE_DUST3R=1
      shift
      ;;
    --dust3r-python)
      DUST3R_PYTHON="$2"
      shift 2
      ;;
    --dust3r-root)
      DUST3R_ROOT="$2"
      shift 2
      ;;
    --dust3r-ckpt)
      DUST3R_CKPT="$2"
      shift 2
      ;;
    --include-mast3r)
      INCLUDE_MAST3R=1
      shift
      ;;
    --mast3r-python)
      MAST3R_PYTHON="$2"
      shift 2
      ;;
    --mast3r-root)
      MAST3R_ROOT="$2"
      shift 2
      ;;
    --mast3r-ckpt)
      MAST3R_CKPT="$2"
      shift 2
      ;;
    --skip-existing)
      SKIP_EXISTING=1
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

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${BENCHMARK_ROOT}/metrics"
fi

DBOW2_SCORE="${BENCHMARK_ROOT}/scores/dbow2.jsonl"
NETVLAD_SCORE="${BENCHMARK_ROOT}/scores/netvlad.jsonl"
SALAD_SCORE="${BENCHMARK_ROOT}/scores/salad.jsonl"
LOFTR_SCORE="${BENCHMARK_ROOT}/scores/loftr.jsonl"
DUST3R_SCORE="${BENCHMARK_ROOT}/scores/dust3r.jsonl"
MAST3R_SCORE="${BENCHMARK_ROOT}/scores/mast3r.jsonl"
ROVER_SCORE="${BENCHMARK_ROOT}/scores/rover_like.jsonl"
LOOPANYTHING_SCORE="${BENCHMARK_ROOT}/scores/loopanything.jsonl"
DBOW2_EVAL_SCORE="scores/dbow2.jsonl"
NETVLAD_EVAL_SCORE="scores/netvlad.jsonl"
SALAD_EVAL_SCORE="scores/salad.jsonl"
LOFTR_EVAL_SCORE="scores/loftr.jsonl"
DUST3R_EVAL_SCORE="scores/dust3r.jsonl"
MAST3R_EVAL_SCORE="scores/mast3r.jsonl"
ROVER_EVAL_SCORE="scores/rover_like.jsonl"
LOOPANYTHING_EVAL_SCORE="scores/loopanything.jsonl"

if [[ ! -f "${DBOW2_SCORE}" ]]; then
  echo "Error: missing frozen DBoW2 score file: ${DBOW2_SCORE}" >&2
  echo "Build the benchmark first; DBoW2 candidates are frozen by benchmark construction." >&2
  exit 1
fi

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

run_score_if_needed() {
  local method="$1"
  local expected_score="$2"
  if [[ "${SKIP_EXISTING}" -eq 1 && -f "${expected_score}" ]]; then
    echo "[skip] ${method}: ${expected_score} already exists"
    return 0
  fi
  run_cmd env PYTHONPATH=src "${PYTHON_BIN}" \
    robust_loop_verification_scripts/score_rover_aligned_benchmark.py \
    "${BENCHMARK_ROOT}" \
    --method "${method}" \
    --backend "${BACKEND}" \
    --device "${DEVICE}"
}

cd "${REPO_ROOT}"

echo "ROVER-aligned Table 1 experiment"
echo "  benchmark_root: ${BENCHMARK_ROOT}"
echo "  output_dir: ${OUTPUT_DIR}"
echo "  python: ${PYTHON_BIN}"
if [[ "${INCLUDE_LOFTR}" -eq 1 ]]; then
  echo "  loftr_python: ${LOFTR_PYTHON}"
  echo "  loftr_root: ${LOFTR_ROOT}"
  echo "  loftr_ckpt: ${LOFTR_CKPT}"
fi
if [[ "${INCLUDE_DUST3R}" -eq 1 ]]; then
  echo "  dust3r_python: ${DUST3R_PYTHON}"
  echo "  dust3r_root: ${DUST3R_ROOT}"
  echo "  dust3r_ckpt: ${DUST3R_CKPT}"
fi
if [[ "${INCLUDE_MAST3R}" -eq 1 ]]; then
  echo "  mast3r_python: ${MAST3R_PYTHON}"
  echo "  mast3r_root: ${MAST3R_ROOT}"
  echo "  mast3r_ckpt: ${MAST3R_CKPT}"
fi
echo "  backend: ${BACKEND}"
echo "  device: ${DEVICE}"
echo "  skip_existing: ${SKIP_EXISTING}"
echo "  dry_run: ${DRY_RUN}"

run_score_if_needed netvlad "${NETVLAD_SCORE}"
run_score_if_needed salad "${SALAD_SCORE}"
if [[ "${INCLUDE_LOFTR}" -eq 1 ]]; then
  if [[ "${SKIP_EXISTING}" -eq 1 && -f "${LOFTR_SCORE}" ]]; then
    echo "[skip] loftr: ${LOFTR_SCORE} already exists"
  else
    LOFTR_CMD=(
      env PYTHONPATH="${LOFTR_ROOT}" "${LOFTR_PYTHON}"
      baseline_scripts/score_loftr_rover_aligned_benchmark.py \
      "${BENCHMARK_ROOT}" \
      --loftr-root "${LOFTR_ROOT}" \
      --ckpt-path "${LOFTR_CKPT}" \
      --device "${DEVICE}"
    )
    run_cmd "${LOFTR_CMD[@]}"
  fi
fi
if [[ "${INCLUDE_DUST3R}" -eq 1 ]]; then
  if [[ "${SKIP_EXISTING}" -eq 1 && -f "${DUST3R_SCORE}" ]]; then
    echo "[skip] dust3r: ${DUST3R_SCORE} already exists"
  else
    run_cmd env PYTHONPATH="${DUST3R_ROOT}" "${DUST3R_PYTHON}" \
      baseline_scripts/score_dust3r_rover_aligned_benchmark.py \
      "${BENCHMARK_ROOT}" \
      --dust3r-root "${DUST3R_ROOT}" \
      --ckpt-path "${DUST3R_CKPT}" \
      --device "${DEVICE}"
  fi
fi
if [[ "${INCLUDE_MAST3R}" -eq 1 ]]; then
  if [[ "${SKIP_EXISTING}" -eq 1 && -f "${MAST3R_SCORE}" ]]; then
    echo "[skip] mast3r: ${MAST3R_SCORE} already exists"
  else
    run_cmd env PYTHONPATH="${MAST3R_ROOT}:${MAST3R_ROOT}/dust3r" "${MAST3R_PYTHON}" \
      baseline_scripts/score_mast3r_rover_aligned_benchmark.py \
      "${BENCHMARK_ROOT}" \
      --mast3r-root "${MAST3R_ROOT}" \
      --ckpt-path "${MAST3R_CKPT}" \
      --device "${DEVICE}"
  fi
fi
run_score_if_needed verifier "${LOOPANYTHING_SCORE}"

EVAL_CMD=(
  env PYTHONPATH=src "${PYTHON_BIN}"
  robust_loop_verification_scripts/evaluate_rover_aligned_benchmark.py
  "${BENCHMARK_ROOT}"
  --method "DBoW2=${DBOW2_EVAL_SCORE}"
  --method "NetVLAD=${NETVLAD_EVAL_SCORE}"
  --method "SALAD=${SALAD_EVAL_SCORE}"
)
if [[ "${INCLUDE_LOFTR}" -eq 1 ]]; then
  EVAL_CMD+=(--method "LoFTR=${LOFTR_EVAL_SCORE}")
fi
if [[ "${INCLUDE_DUST3R}" -eq 1 ]]; then
  EVAL_CMD+=(--method "DUSt3R=${DUST3R_EVAL_SCORE}")
fi
if [[ "${INCLUDE_MAST3R}" -eq 1 ]]; then
  EVAL_CMD+=(--method "MAST3R=${MAST3R_EVAL_SCORE}")
fi
EVAL_CMD+=(
  --method "ROVER-like=${ROVER_EVAL_SCORE}"
  --method "LoopAnything=${LOOPANYTHING_EVAL_SCORE}"
  --output-dir "${OUTPUT_DIR}"
)
run_cmd "${EVAL_CMD[@]}"

echo "Outputs:"
echo "  ${OUTPUT_DIR}/table1.md"
echo "  ${OUTPUT_DIR}/metrics_per_sequence.csv"
echo "  ${OUTPUT_DIR}/metrics_summary.json"
