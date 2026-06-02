#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ntu_viral_robust_loop_verifier_batch_run.sh [batch args]

Default:
  Builds/uses NTU-VIRAL VPR caches and runs the robust loop verifier without
  support ensemble.

Examples:
  ntu_viral_robust_loop_verifier_batch_run.sh --overwrite_dataset
  BACKEND=mock ntu_viral_robust_loop_verifier_batch_run.sh --query-limit 40

Common arguments:
  --overwrite_dataset       Regenerate VPR caches before running.
  --sequences CSV           Sequence list.
  --dry-run                 Print planned work and write batch summary only.

Environment overrides:
  PYTHON_BIN                Default: /home/chenguyuan/anaconda3/envs/da3/bin/python
  BACKEND                   Default: real
  QUERY_LIMIT               Optional full-pipeline query limit
  RUN_ID                    Optional run id
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  echo
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
  PYTHON_BIN="${PYTHON_BIN:-/home/chenguyuan/anaconda3/envs/da3/bin/python}"
  PYTHONPATH=src "${PYTHON_BIN}" \
    "${SCRIPT_DIR}/ntu_viral_robust_loop_verifier_batch_run.py" --help
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
  "${SCRIPT_DIR}/ntu_viral_robust_loop_verifier_batch_run.py" \
  "${args[@]}" "$@"
