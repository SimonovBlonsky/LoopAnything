#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_support_ensemble_verifier.sh [--help]

Runs the robust loop verifier with the FusionPortableV2 handheld support ensemble config.

Environment overrides:
  SEQUENCE_NAME       Sequence name. Default: handheld_escalator00
  PLATFORM            Platform name. Default: handheld
  DATASET_NAME        Dataset name. Default: FusionPortableV2
  CACHE_ROOT          Cache root. Default: /data/datasets/FusionPortable/robust_loop_verifier_cache
  CONFIG              Verifier config path. Default:
                      configs/robust_loop_verifier/fusionportablev2_handheld_support_ensemble.yaml
  PYTHON_BIN          Python executable. Default: /home/chenguyuan/anaconda3/envs/da3/bin/python
  BACKEND             Verifier backend. Default: real
  RUN_ID              Run identifier. Default: current timestamp
  OUTPUT_ROOT         Output directory. Default:
                      workspace/robust_loop_verifier_runs/.../<RUN_ID>_support_ensemble
  SEQUENCE_CACHE      Sequence cache directory. Default:
                      CACHE_ROOT/DATASET_NAME/PLATFORM/SEQUENCE_NAME
  QUERY_LIMIT         Optional query limit. If unset, the CLI default is used.

Artifacts:
  candidate_records   OUTPUT_ROOT/candidate_records.jsonl
  metrics_json        OUTPUT_ROOT/metrics.json
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SEQUENCE_NAME="${SEQUENCE_NAME:-handheld_escalator00}"
PLATFORM="${PLATFORM:-handheld}"
DATASET_NAME="${DATASET_NAME:-FusionPortableV2}"
CACHE_ROOT="${CACHE_ROOT:-/data/datasets/FusionPortable/robust_loop_verifier_cache}"
DEFAULT_CONFIG="${REPO_ROOT}/configs/robust_loop_verifier"
DEFAULT_CONFIG="${DEFAULT_CONFIG}/fusionportablev2_handheld_support_ensemble.yaml"
CONFIG="${CONFIG:-${DEFAULT_CONFIG}}"
PYTHON_BIN="${PYTHON_BIN:-/home/chenguyuan/anaconda3/envs/da3/bin/python}"
BACKEND="${BACKEND:-real}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
DEFAULT_OUTPUT_ROOT="${REPO_ROOT}/workspace/robust_loop_verifier_runs"
DEFAULT_OUTPUT_ROOT="${DEFAULT_OUTPUT_ROOT}/${DATASET_NAME}/${PLATFORM}/${SEQUENCE_NAME}"
DEFAULT_OUTPUT_ROOT="${DEFAULT_OUTPUT_ROOT}/${RUN_ID}_support_ensemble"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DEFAULT_OUTPUT_ROOT}}"
SEQUENCE_CACHE="${SEQUENCE_CACHE:-${CACHE_ROOT}/${DATASET_NAME}/${PLATFORM}/${SEQUENCE_NAME}}"

query_limit_args=()
if [[ -n "${QUERY_LIMIT:-}" ]]; then
  query_limit_args=(--query-limit "${QUERY_LIMIT}")
fi

cd "${REPO_ROOT}"

PYTHONPATH=src "${PYTHON_BIN}" -m robust_loop_verifier.cli run-cache \
  --sequence-cache "${SEQUENCE_CACHE}" \
  --config "${CONFIG}" \
  --output-root "${OUTPUT_ROOT}" \
  --backend "${BACKEND}" \
  "${query_limit_args[@]}"

echo "candidate_records=${OUTPUT_ROOT}/candidate_records.jsonl"
echo "metrics_json=${OUTPUT_ROOT}/metrics.json"
