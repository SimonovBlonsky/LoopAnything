#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/chenguyuan/anaconda3/envs/da3/bin/python}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/workspace/robust_loop_verifier_runs/FusionPortableV2/handheld/handheld_escalator00/20260517_211913}"
CANDIDATE_RECORDS="${CANDIDATE_RECORDS:-${RUN_ROOT}/candidate_records.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUN_ROOT}/score_sweep}"
GRAPH_WEIGHTS="${GRAPH_WEIGHTS:-0,0.25,0.5,1,2,4}"
FUSION_WEIGHTS="${FUSION_WEIGHTS:-0,0.25,0.5,1,2,4}"

cd "${REPO_ROOT}"

PYTHONPATH=src "${PYTHON_BIN}" -m robust_loop_verifier.cli sweep-scores \
  --candidate-records "${CANDIDATE_RECORDS}" \
  --output-root "${OUTPUT_ROOT}" \
  --graph-weights "${GRAPH_WEIGHTS}" \
  --fusion-weights "${FUSION_WEIGHTS}"

echo "score_sweep_json=${OUTPUT_ROOT}/score_sweep.json"
echo "score_sweep_md=${OUTPUT_ROOT}/score_sweep.md"
