#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_robust_loop_verifier_pipeline.sh [SEQUENCE_NAME_OR_CACHE_DIR]

Default:
  Runs FusionPortableV2/handheld/handheld_escalator00 from the preprocessed cache.

Examples:
  run_robust_loop_verifier_pipeline.sh
  run_robust_loop_verifier_pipeline.sh handheld_escalator00
  run_robust_loop_verifier_pipeline.sh ugv_parking01
  run_robust_loop_verifier_pipeline.sh /path/to/sequence/cache
  BACKEND=mock QUERY_LIMIT=40 run_robust_loop_verifier_pipeline.sh handheld_escalator00

Environment overrides:
  PYTHON_BIN, CONFIG, CACHE_ROOT, DATASET_NAME, PLATFORM, OUTPUT_BASE, OUTPUT_ROOT,
  BACKEND, QUERY_LIMIT, RUN_ID
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "$#" -gt 1 ]]; then
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/chenguyuan/anaconda3/envs/da3/bin/python}"
BASE_CONFIG="${CONFIG:-${REPO_ROOT}/configs/robust_loop_verifier/fusionportablev2_handheld.yaml}"
CACHE_ROOT="${CACHE_ROOT:-/data/datasets/FusionPortable/robust_loop_verifier_cache}"
DATASET_NAME="${DATASET_NAME:-FusionPortableV2}"
BACKEND="${BACKEND:-real}"
SEQUENCE_ARG="${1:-handheld_escalator00}"

resolve_sequence_cache() {
  local arg="$1"
  if [[ -d "${arg}" ]]; then
    readlink -f "${arg}"
    return 0
  fi

  if [[ -n "${PLATFORM:-}" ]]; then
    local explicit="${CACHE_ROOT}/${DATASET_NAME}/${PLATFORM}/${arg}"
    if [[ -d "${explicit}" ]]; then
      readlink -f "${explicit}"
      return 0
    fi
    echo "Sequence cache does not exist: ${explicit}" >&2
    return 1
  fi

  local matches=()
  local candidate
  for candidate in "${CACHE_ROOT}/${DATASET_NAME}"/*/"${arg}"; do
    if [[ -d "${candidate}" ]]; then
      matches+=("${candidate}")
    fi
  done

  if [[ "${#matches[@]}" -eq 0 ]]; then
    echo "Could not find sequence cache for '${arg}' under ${CACHE_ROOT}/${DATASET_NAME}" >&2
    return 1
  fi
  if [[ "${#matches[@]}" -gt 1 ]]; then
    echo "Sequence '${arg}' matches multiple caches; set PLATFORM or pass a cache path:" >&2
    printf '  %s\n' "${matches[@]}" >&2
    return 1
  fi
  readlink -f "${matches[0]}"
}

if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "Config does not exist: ${BASE_CONFIG}" >&2
  exit 1
fi

SEQUENCE_CACHE="$(resolve_sequence_cache "${SEQUENCE_ARG}")"
MANIFEST="${SEQUENCE_CACHE}/manifest.json"
if [[ ! -f "${MANIFEST}" ]]; then
  echo "Sequence cache is missing manifest.json: ${SEQUENCE_CACHE}" >&2
  exit 1
fi
if [[ ! -f "${SEQUENCE_CACHE}/keyframes.jsonl" || ! -f "${SEQUENCE_CACHE}/positives.jsonl" ]]; then
  echo "Sequence cache must contain keyframes.jsonl and positives.jsonl: ${SEQUENCE_CACHE}" >&2
  exit 1
fi

eval "$(
  PYTHONPATH=src "${PYTHON_BIN}" - "${MANIFEST}" <<'PY'
import json
import shlex
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
fields = {
    "CACHE_DATASET_NAME": manifest.get("dataset_name", "FusionPortableV2"),
    "CACHE_PLATFORM": manifest.get("platform", "unknown"),
    "CACHE_SEQUENCE_NAME": manifest.get("sequence_name", Path(sys.argv[1]).parent.name),
    "CACHE_KEYFRAME_COUNT": int(manifest.get("keyframe_count", 1000000000)),
}
for key, value in fields.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"

QUERY_LIMIT="${QUERY_LIMIT:-${CACHE_KEYFRAME_COUNT}}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/workspace/robust_loop_verifier_runs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${OUTPUT_BASE}/${CACHE_DATASET_NAME}/${CACHE_PLATFORM}/${CACHE_SEQUENCE_NAME}/${RUN_ID}}"

TMP_CONFIG="$(mktemp "${TMPDIR:-/tmp}/robust-loop-verifier-run-${CACHE_SEQUENCE_NAME}.XXXXXX.yaml")"
trap 'rm -f "${TMP_CONFIG}"' EXIT

cd "${REPO_ROOT}"

PYTHONPATH=src "${PYTHON_BIN}" - "${BASE_CONFIG}" "${TMP_CONFIG}" \
  "${CACHE_DATASET_NAME}" "${CACHE_PLATFORM}" "${OUTPUT_BASE}" <<'PY'
import sys
from pathlib import Path

from robust_loop_verifier.io import read_yaml, write_yaml

base_config, output_config, dataset_name, platform, output_root = sys.argv[1:]
data = dict(read_yaml(Path(base_config)))
data["dataset_name"] = dataset_name
data["platform"] = platform
data["output_root"] = output_root
write_yaml(Path(output_config), data)
PY

echo "Running robust loop verifier pipeline"
echo "  sequence_cache: ${SEQUENCE_CACHE}"
echo "  config: ${TMP_CONFIG}"
echo "  backend: ${BACKEND}"
echo "  query_limit: ${QUERY_LIMIT}"
echo "  output_root: ${OUTPUT_ROOT}"

PYTHONPATH=src "${PYTHON_BIN}" -m robust_loop_verifier.cli run-cache \
  --config "${TMP_CONFIG}" \
  --sequence-cache "${SEQUENCE_CACHE}" \
  --output-root "${OUTPUT_ROOT}" \
  --query-limit "${QUERY_LIMIT}" \
  --backend "${BACKEND}"

echo
echo "Metrics:"
cat "${OUTPUT_ROOT}/metrics.md"
echo
echo "Artifacts:"
echo "  candidate_records: ${OUTPUT_ROOT}/candidate_records.jsonl"
echo "  metrics_json: ${OUTPUT_ROOT}/metrics.json"
echo "  metrics_md: ${OUTPUT_ROOT}/metrics.md"
