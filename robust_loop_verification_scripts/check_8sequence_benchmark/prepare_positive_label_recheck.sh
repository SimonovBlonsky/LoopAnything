#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SOURCE_BENCHMARK_ROOT="${SOURCE_BENCHMARK_ROOT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
SCOPE="${SCOPE:-main8}"

usage() {
  cat <<'EOF'
Usage: prepare_positive_label_recheck.sh [SOURCE_ROOT] [options]

Create a stricter manual-review copy for positive-label rechecking. The output
benchmark keeps the same full benchmark structure, pre-fills labels that are not
selected for recheck, and leaves selected positive pairs for the annotation UI.

By default, the script auto-selects the newest finalized
benchmark_v1_da3_auto_labels_v4_asterslam_main8_full_manual_review_* root and
rechecks only positive pairs in the paper-facing eight sequences.

Options:
  --source-root PATH   Finalized benchmark root to recheck.
  --output-root PATH   Output review root. Defaults to timestamped root.
  --scope main8|all    main8 rechecks positives in the paper eight sequences;
                       all rechecks positives in all benchmark sequences.
  -h, --help           Show this help.

Environment overrides:
  SOURCE_BENCHMARK_ROOT
  OUTPUT_ROOT
  SCOPE

Typical usage:
  bash robust_loop_verification_scripts/check_8sequence_benchmark/prepare_positive_label_recheck.sh
  bash robust_loop_verification_scripts/check_8sequence_benchmark/run_main8_manual_review_ui.sh <printed_review_root>
EOF
}

require_value() {
  if [[ $# -lt 2 || -z "$2" || "$2" == -* ]]; then
    echo "Error: $1 requires a value" >&2
    usage >&2
    exit 2
  fi
}

if [[ $# -gt 0 && "$1" != -* ]]; then
  SOURCE_BENCHMARK_ROOT="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-root)
      require_value "$@"
      SOURCE_BENCHMARK_ROOT="$2"
      shift 2
      ;;
    --output-root)
      require_value "$@"
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --scope)
      require_value "$@"
      SCOPE="$2"
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

case "$SCOPE" in
  main8|all) ;;
  *)
    echo "Error: --scope must be main8 or all" >&2
    exit 2
    ;;
esac

REPO_ROOT="$REPO_ROOT" \
SOURCE_BENCHMARK_ROOT="$SOURCE_BENCHMARK_ROOT" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
SCOPE="$SCOPE" \
SCRIPT_DIR="$SCRIPT_DIR" \
python3 <<'PY'
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

repo_root = Path(os.environ["REPO_ROOT"])
source_arg = os.environ["SOURCE_BENCHMARK_ROOT"]
output_arg = os.environ["OUTPUT_ROOT"]
scope = os.environ["SCOPE"]
script_dir = Path(os.environ["SCRIPT_DIR"])

main8_sequences = {
    "FusionPortableV2/handheld/handheld_escalator00",
    "FusionPortableV2/handheld/handheld_room00",
    "FusionPortableV2/ugv/ugv_campus01",
    "FusionPortableV2/ugv/ugv_parking01",
    "GEODE/Offroad/Offroad02_beta",
    "GEODE/Offroad/Offroad05_beta",
    "NTU-VIRAL/NTU-VIRAL/eee_01",
    "NTU-VIRAL/NTU-VIRAL/nya_02",
}


def resolve_repo_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else repo_root / path


def latest_finalized_main8_root() -> Path:
    search_root = repo_root / "workspace" / "rover_aligned_benchmark"
    candidates = []
    for path in search_root.glob("benchmark_v1_da3_auto_labels_v4_asterslam_main8_full_manual_review_*"):
        if (path / "annotations.jsonl").is_file() and (path / "annotation_seal.json").is_file():
            candidates.append(path)
    if not candidates:
        raise SystemExit(
            "No finalized main8 manual-review benchmark found. "
            "Pass --source-root explicitly."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


source = resolve_repo_path(source_arg) if source_arg else latest_finalized_main8_root()
if output_arg:
    output = resolve_repo_path(output_arg)
else:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = source.parent / f"{source.name}_positive_recheck_{scope}_{timestamp}"

required = [
    "benchmark_pairs.jsonl",
    "manifest.json",
    "annotations.jsonl",
    "annotation_seal.json",
]
for filename in required:
    if not (source / filename).is_file():
        raise SystemExit(f"Missing source file: {source / filename}")
if output.exists():
    raise SystemExit(f"Output root already exists: {output}")

shutil.copytree(source, output, symlinks=True)
for filename in ("annotations.jsonl", "annotation_seal.json", "annotation_events.jsonl"):
    path = output / filename
    if path.exists():
        path.unlink()

pairs = {}
sequence_counts = {}
for line in (output / "benchmark_pairs.jsonl").open(encoding="utf-8"):
    row = json.loads(line)
    pair_id = row["pair_id"]
    sequence_key = f'{row["dataset"]}/{row["platform"]}/{row["sequence"]}'
    pairs[pair_id] = sequence_key
    sequence_counts[sequence_key] = sequence_counts.get(sequence_key, 0) + 1

annotations = []
for line in (source / "annotations.jsonl").open(encoding="utf-8"):
    row = json.loads(line)
    if row["pair_id"] not in pairs:
        raise SystemExit(f"Unknown annotation pair_id: {row['pair_id']}")
    annotations.append(row)
if len(annotations) != len(pairs):
    raise SystemExit(
        f"Annotation coverage mismatch: annotations={len(annotations)}, pairs={len(pairs)}"
    )

prefilled = 0
left_for_review = 0
left_by_sequence = {}
now = datetime.now(timezone.utc).isoformat()
with (output / "annotation_events.jsonl").open("w", encoding="utf-8") as handle:
    for row in annotations:
        pair_id = row["pair_id"]
        label = row["label"]
        sequence_key = pairs[pair_id]
        if scope == "main8":
            needs_review = label == 1 and sequence_key in main8_sequences
        else:
            needs_review = label == 1
        if needs_review:
            left_for_review += 1
            left_by_sequence[sequence_key] = left_by_sequence.get(sequence_key, 0) + 1
            continue
        event = {
            "event_id": uuid.uuid4().hex,
            "action": "label",
            "pair_id": pair_id,
            "label": label,
            "target_event_id": None,
            "annotated_at": now,
        }
        handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
        prefilled += 1

if scope == "main8":
    missing = sorted(main8_sequences - set(sequence_counts))
    if missing:
        raise SystemExit(f"Missing expected main8 sequences: {missing}")

print(f"source_root={source}")
print(f"review_root={output}")
print(f"scope={scope}")
print(f"prefilled_pairs={prefilled}")
print(f"positive_pairs_left_for_recheck={left_for_review}")
print("positive_pairs_left_by_sequence:")
for key in sorted(left_by_sequence):
    print(f"  {key}: {left_by_sequence[key]}")
print()
print("Next command:")
print(f'  bash {script_dir / "run_main8_manual_review_ui.sh"} "{output}"')
PY
