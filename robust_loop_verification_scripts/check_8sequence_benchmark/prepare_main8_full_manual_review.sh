#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SOURCE_BENCHMARK_ROOT="${SOURCE_BENCHMARK_ROOT:-workspace/rover_aligned_benchmark/benchmark_v1_da3_auto_labels_v4_asterslam}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"

usage() {
  cat <<'EOF'
Usage: prepare_main8_full_manual_review.sh [options]

Create an unsealed review copy of the current benchmark. The copy keeps the full
10-sequence benchmark structure for compatibility, but pre-fills labels only for
the two excluded sequences:

  FusionPortableV2/handheld/handheld_room01
  NTU-VIRAL/NTU-VIRAL/eee_02

The annotation UI will therefore require manual review for all 3200 pairs from
the paper-facing eight sequences.

Options:
  --source-benchmark-root PATH   Source sealed benchmark root.
  --output-root PATH             Output review root. Defaults to timestamped root.
  -h, --help                     Show this help.

Environment overrides:
  SOURCE_BENCHMARK_ROOT
  OUTPUT_ROOT
EOF
}

require_value() {
  if [[ $# -lt 2 || -z "$2" || "$2" == -* ]]; then
    echo "Error: $1 requires a value" >&2
    usage >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-benchmark-root)
      require_value "$@"
      SOURCE_BENCHMARK_ROOT="$2"
      shift 2
      ;;
    --output-root)
      require_value "$@"
      OUTPUT_ROOT="$2"
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

resolve_repo_path() {
  if [[ "$1" = /* ]]; then
    printf '%s\n' "$1"
  else
    printf '%s/%s\n' "$REPO_ROOT" "$1"
  fi
}

SOURCE_BENCHMARK_ROOT="$(resolve_repo_path "$SOURCE_BENCHMARK_ROOT")"
if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="${REPO_ROOT}/workspace/rover_aligned_benchmark/benchmark_v1_da3_auto_labels_v4_asterslam_main8_full_manual_review_$(date +%Y%m%d_%H%M%S)"
else
  OUTPUT_ROOT="$(resolve_repo_path "$OUTPUT_ROOT")"
fi

if [[ ! -f "${SOURCE_BENCHMARK_ROOT}/annotations.jsonl" ]]; then
  echo "Error: missing source annotations.jsonl: ${SOURCE_BENCHMARK_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${SOURCE_BENCHMARK_ROOT}/annotation_seal.json" ]]; then
  echo "Error: missing source annotation_seal.json: ${SOURCE_BENCHMARK_ROOT}" >&2
  exit 1
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Error: output root already exists: ${OUTPUT_ROOT}" >&2
  exit 1
fi

cp -a "$SOURCE_BENCHMARK_ROOT" "$OUTPUT_ROOT"

SOURCE_BENCHMARK_ROOT="$SOURCE_BENCHMARK_ROOT" OUTPUT_ROOT="$OUTPUT_ROOT" python3 <<'PY'
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

source = Path(os.environ["SOURCE_BENCHMARK_ROOT"])
output = Path(os.environ["OUTPUT_ROOT"])

review_sequences = {
    "FusionPortableV2/handheld/handheld_escalator00",
    "FusionPortableV2/handheld/handheld_room00",
    "FusionPortableV2/ugv/ugv_campus01",
    "FusionPortableV2/ugv/ugv_parking01",
    "GEODE/Offroad/Offroad02_beta",
    "GEODE/Offroad/Offroad05_beta",
    "NTU-VIRAL/NTU-VIRAL/eee_01",
    "NTU-VIRAL/NTU-VIRAL/nya_02",
}

pairs = {}
sequence_counts = {}
for line in (output / "benchmark_pairs.jsonl").open(encoding="utf-8"):
    row = json.loads(line)
    pair_id = row["pair_id"]
    sequence_key = f'{row["dataset"]}/{row["platform"]}/{row["sequence"]}'
    pairs[pair_id] = sequence_key
    sequence_counts[sequence_key] = sequence_counts.get(sequence_key, 0) + 1

for filename in ("annotations.jsonl", "annotation_seal.json", "annotation_events.jsonl"):
    path = output / filename
    if path.exists():
        path.unlink()

prefilled = 0
left_for_review = 0
now = datetime.now(timezone.utc).isoformat()
with (output / "annotation_events.jsonl").open("w", encoding="utf-8") as handle:
    for line in (source / "annotations.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        pair_id = row["pair_id"]
        sequence_key = pairs[pair_id]
        if sequence_key in review_sequences:
            left_for_review += 1
            continue
        event = {
            "event_id": uuid.uuid4().hex,
            "action": "label",
            "pair_id": pair_id,
            "label": row["label"],
            "target_event_id": None,
            "annotated_at": now,
        }
        handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
        prefilled += 1

unexpected = sorted(set(review_sequences) - set(sequence_counts))
if unexpected:
    raise SystemExit(f"Missing expected review sequences: {unexpected}")

print(f"review_root={output}")
print(f"prefilled_excluded_pairs={prefilled}")
print(f"manual_review_pairs={left_for_review}")
print("manual_review_sequences:")
for key in sorted(review_sequences):
    print(f"  {key}: {sequence_counts[key]} pairs")
PY

cat <<EOF

Next command:
  bash ${SCRIPT_DIR}/run_main8_manual_review_ui.sh "${OUTPUT_ROOT}"
EOF
