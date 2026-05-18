#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  fusionportablev2_robust_loop_verifier_batch_run.sh [batch args]

Runs FusionPortableV2 robust loop verifier experiments:
  1. Select handheld sequences with AsterSLAM ATE RMSE < 0.1m by default.
  2. Select UGV campus00/01 and all parking sequences by default.
  3. Create missing VPR dataset caches unless --overwrite_dataset is set.
  4. Run the default robust loop verifier over the full cached sequence.
  5. Write batch_summary and main AP/MR tables.

Common args:
  --overwrite_dataset      Regenerate every selected VPR cache before running.
  --dry-run                Print planned work and write dry-run summary only.
  --keep-going             Continue after per-sequence failures.
  --platforms handheld,ugv Select platforms.
  --handheld-sequences A,B Override the hardcoded/auto handheld selection.
  --ugv-sequences A,B      Override the hardcoded UGV selection.
  --support-ensemble       Explicitly run the support-ensemble ablation config.

Common environment overrides:
  PYTHON_BIN=/home/chenguyuan/anaconda3/envs/da3/bin/python
  BACKEND=real
  CONFIG=<custom verifier config>
  RUN_ID=<shared run id>
  QUERY_LIMIT=<override full-sequence query limit>
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  echo
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
  PYTHON_BIN="${PYTHON_BIN:-/home/chenguyuan/anaconda3/envs/da3/bin/python}"
  PYTHONPATH=src "${PYTHON_BIN}" \
    "${SCRIPT_DIR}/fusionportablev2_robust_loop_verifier_batch_run.py" --help
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/chenguyuan/anaconda3/envs/da3/bin/python}"

args=(
  "--repo-root" "${REPO_ROOT}"
  "--python-bin" "${PYTHON_BIN}"
)

if [[ -n "${RUN_ID:-}" ]]; then
  args+=("--run-id" "${RUN_ID}")
fi
if [[ -n "${QUERY_LIMIT:-}" ]]; then
  args+=("--query-limit" "${QUERY_LIMIT}")
fi
if [[ -n "${BACKEND:-}" ]]; then
  args+=("--backend" "${BACKEND}")
fi
if [[ -n "${CONFIG:-}" ]]; then
  args+=("--config" "${CONFIG}")
fi

cd "${REPO_ROOT}"
PYTHONPATH=src "${PYTHON_BIN}" \
  "${SCRIPT_DIR}/fusionportablev2_robust_loop_verifier_batch_run.py" \
  "${args[@]}" "$@"
