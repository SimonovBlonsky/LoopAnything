#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_support_ensemble_score_sweep.sh [--help]

Runs a score sweep for support ensemble candidate records.

Environment overrides:
  RUN_ROOT            Required unless CANDIDATE_RECORDS is set.
  CANDIDATE_RECORDS   Candidate records path. Default: RUN_ROOT/candidate_records.jsonl
  OUTPUT_ROOT         Sweep output directory. Default: RUN_ROOT/support_ensemble_score_sweep
  GRAPH_WEIGHTS       Comma-separated graph weights. Default: 0,0.25,0.5,1,2,4
  FUSION_WEIGHTS      Comma-separated fusion weights. Default: 0,0.25,0.5,1,2,4
  PYTHON_BIN          Python executable. Default: /home/chenguyuan/anaconda3/envs/da3/bin/python

Artifacts:
  score_sweep_json    OUTPUT_ROOT/score_sweep.json
  score_sweep_md      OUTPUT_ROOT/score_sweep.md
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "$#" -gt 0 ]]; then
  usage >&2
  exit 2
fi

CALLER_PWD="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/chenguyuan/anaconda3/envs/da3/bin/python}"
GRAPH_WEIGHTS="${GRAPH_WEIGHTS:-0,0.25,0.5,1,2,4}"
FUSION_WEIGHTS="${FUSION_WEIGHTS:-0,0.25,0.5,1,2,4}"

canonical_path() {
  local path="$1"
  local dir
  local base
  dir="$(dirname "${path}")"
  base="$(basename "${path}")"
  if [[ "${path}" != /* ]]; then
    dir="${CALLER_PWD}/${dir}"
  fi
  printf '%s/%s\n' "$(cd "${dir}" && pwd)" "${base}"
}

if [[ -z "${CANDIDATE_RECORDS:-}" && -z "${RUN_ROOT:-}" ]]; then
  echo "RUN_ROOT is required unless CANDIDATE_RECORDS is set." >&2
  usage >&2
  exit 2
fi

if [[ -n "${RUN_ROOT:-}" ]]; then
  RUN_ROOT="$(canonical_path "${RUN_ROOT}")"
fi
if [[ -z "${CANDIDATE_RECORDS:-}" ]]; then
  CANDIDATE_RECORDS="${RUN_ROOT}/candidate_records.jsonl"
else
  CANDIDATE_RECORDS="$(canonical_path "${CANDIDATE_RECORDS}")"
fi
if [[ -z "${RUN_ROOT:-}" ]]; then
  RUN_ROOT="$(cd "$(dirname "${CANDIDATE_RECORDS}")" && pwd)"
fi
if [[ -z "${OUTPUT_ROOT:-}" ]]; then
  OUTPUT_ROOT="${RUN_ROOT}/support_ensemble_score_sweep"
else
  OUTPUT_ROOT="$(canonical_path "${OUTPUT_ROOT}")"
fi

cd "${REPO_ROOT}"

PYTHONPATH=src "${PYTHON_BIN}" -m robust_loop_verifier.cli sweep-scores \
  --candidate-records "${CANDIDATE_RECORDS}" \
  --output-root "${OUTPUT_ROOT}" \
  --graph-weights "${GRAPH_WEIGHTS}" \
  --fusion-weights "${FUSION_WEIGHTS}"

echo "score_sweep_json=${OUTPUT_ROOT}/score_sweep.json"
echo "score_sweep_md=${OUTPUT_ROOT}/score_sweep.md"
