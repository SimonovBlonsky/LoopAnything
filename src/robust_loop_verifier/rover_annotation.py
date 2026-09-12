from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Literal, Mapping, Sequence, TextIO

from .io import read_json, read_jsonl

ANNOTATION_VERSION = 1
ANNOTATION_SEAL_VERSION = "rover_annotation_seal_v1"
FINALIZED_ANNOTATION_FIELDS = frozenset(
    {"pair_id", "label", "annotated_at", "annotation_version"}
)
EVENT_FIELDS = frozenset(
    {"event_id", "action", "pair_id", "label", "target_event_id", "annotated_at"}
)
CONTEXT_KEYS = (
    "query_prev",
    "query",
    "query_next",
    "candidate_prev",
    "candidate",
    "candidate_next",
)


@dataclass(frozen=True)
class AnnotationEvent:
    event_id: str
    action: Literal["label", "undo"]
    pair_id: str
    label: int | None
    target_event_id: str | None
    annotated_at: str

    def __post_init__(self) -> None:
        _validate_event(self)


def append_event(path: Path, event: AnnotationEvent) -> None:
    path = Path(path)
    seal_path = path.parent / "annotation_seal.json"
    if seal_path.exists():
        raise FileExistsError(f"annotation log is sealed: {seal_path}")
    _validate_event(event)
    existing_events = _read_events(path)
    if event.event_id in {existing.event_id for existing in existing_events}:
        raise ValueError(f"duplicate event_id: {event.event_id}")

    path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_appendable_jsonl(path)
    row = json.dumps(asdict(event), sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(row)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_parent(path)


def replay_events(path: Path) -> dict[str, int]:
    return {
        pair_id: event.label
        for pair_id, event in replay_resolved_label_events(path).items()
        if type(event.label) is int
    }


def replay_resolved_label_events(path: Path) -> dict[str, AnnotationEvent]:
    label_events, undone = _replay_label_events(path)
    resolved: dict[str, AnnotationEvent] = {}
    for event_id, event in label_events.items():
        if event_id not in undone:
            resolved.pop(event.pair_id, None)
            resolved[event.pair_id] = event
    return resolved


def deterministic_display_order(pair_ids: Iterable[str], seed: int) -> list[str]:
    return sorted(
        [str(pair_id) for pair_id in pair_ids],
        key=lambda pair_id: (
            hashlib.sha256(f"{int(seed)}\0{pair_id}".encode("utf-8")).hexdigest(),
            pair_id,
        ),
    )


def valid_image_context(pair: Mapping[str, object]) -> dict[str, str | None]:
    query_context = _validate_context_tuple(pair, "query_context")
    candidate_context = _validate_context_tuple(pair, "candidate_context")
    return {
        "query_prev": query_context[0],
        "query": query_context[1],
        "query_next": query_context[2],
        "candidate_prev": candidate_context[0],
        "candidate": candidate_context[1],
        "candidate_next": candidate_context[2],
    }


def opaque_image_tokens(
    pair: Mapping[str, object],
    token_secret: bytes | str,
) -> dict[str, str | None]:
    secret = _secret_bytes(token_secret)
    pair_id = str(pair["pair_id"])
    tokens: dict[str, str | None] = {}
    for slot, image_path in valid_image_context(pair).items():
        if image_path is None:
            tokens[slot] = None
            continue
        message = f"{pair_id}\0{slot}\0{image_path}".encode("utf-8")
        tokens[slot] = hmac.new(secret, message, hashlib.sha256).hexdigest()
    return tokens


def annotation_progress(pair_ids: Iterable[str], resolved: Mapping[str, object]) -> dict[str, int]:
    pair_id_list = [str(pair_id) for pair_id in pair_ids]
    resolved_ids = {str(pair_id) for pair_id in resolved}
    return {
        "completed": sum(1 for pair_id in pair_id_list if pair_id in resolved_ids),
        "total": len(pair_id_list),
    }


def finalize_annotations(
    pairs: Sequence[Mapping[str, object]],
    event_path: Path,
    output_path: Path,
    manifest_path: Path,
    seal_path: Path,
) -> None:
    manifest_path = Path(manifest_path)
    output_path = Path(output_path)
    seal_path = Path(seal_path)
    if output_path.exists():
        raise FileExistsError(f"annotations output already exists: {output_path}")
    if seal_path.exists():
        raise FileExistsError(f"annotation seal already exists: {seal_path}")
    pair_manifest_hash, benchmark_version = _verified_manifest_metadata(manifest_path)

    disk_pair_rows = list(read_jsonl(manifest_path.parent / "benchmark_pairs.jsonl"))
    caller_pair_rows = [_normalize_json_mapping(_as_mapping(pair)) for pair in pairs]
    if caller_pair_rows != disk_pair_rows:
        raise ValueError("caller pairs do not match on-disk benchmark_pairs.jsonl")
    pair_rows = disk_pair_rows
    pair_ids = _validate_unique_pair_ids(pair_rows)
    pair_id_set = set(pair_ids)
    resolved_events = replay_resolved_label_events(event_path)
    unknown_pair_ids = sorted(set(resolved_events) - pair_id_set)
    if unknown_pair_ids:
        raise ValueError(f"unknown pair_id in annotation log: {unknown_pair_ids[0]}")
    unlabeled = [pair_id for pair_id in pair_ids if pair_id not in resolved_events]
    if unlabeled:
        raise ValueError(f"unlabeled pairs remain: {len(unlabeled)}")

    rows = [
        {
            "pair_id": pair_id,
            "label": resolved_events[pair_id].label,
            "annotated_at": resolved_events[pair_id].annotated_at,
            "annotation_version": ANNOTATION_VERSION,
        }
        for pair_id in pair_ids
    ]
    _validate_finalized_annotations(rows, pair_rows)
    output_temp: Path | None = None
    seal_temp: Path | None = None
    output_published = False
    seal_published = False
    try:
        output_temp = _prepare_jsonl_temp(output_path, rows)
        annotation_hash = sha256_file(output_temp)
        seal_temp = _prepare_json_temp(
            seal_path,
            {
                "version": ANNOTATION_SEAL_VERSION,
                "annotation_version": ANNOTATION_VERSION,
                "benchmark_version": benchmark_version,
                "pair_manifest_sha256": pair_manifest_hash,
                "annotation_sha256": annotation_hash,
                "count": len(rows),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        os.replace(output_temp, output_path)
        output_temp = None
        output_published = True
        _fsync_parent(output_path)
        os.replace(seal_temp, seal_path)
        seal_temp = None
        seal_published = True
        _fsync_parent(seal_path)
    except Exception:
        if output_temp is not None:
            _unlink_and_fsync(output_temp)
        if seal_temp is not None:
            _unlink_and_fsync(seal_temp)
        if output_published and not seal_published:
            _unlink_and_fsync(output_path)
        raise


def verify_annotation_seal(
    manifest_path: Path,
    pairs_path: Path,
    annotations_path: Path,
    seal_path: Path,
) -> None:
    manifest_path = Path(manifest_path)
    pairs_path = Path(pairs_path)
    annotations_path = Path(annotations_path)
    seal_path = Path(seal_path)
    expected_pair_hash, benchmark_version = _verified_manifest_metadata(
        manifest_path,
        pairs_path,
    )
    seal = read_json(Path(seal_path))
    if seal.get("pair_manifest_sha256") != expected_pair_hash:
        raise ValueError("pair-manifest hash does not match annotation seal")
    if seal.get("version") != ANNOTATION_SEAL_VERSION:
        raise ValueError("annotation seal version mismatch")
    if type(seal.get("annotation_version")) is not int:
        raise ValueError("annotation seal annotation_version must be an integer")
    if seal["annotation_version"] != ANNOTATION_VERSION:
        raise ValueError("annotation seal annotation_version mismatch")
    if seal.get("benchmark_version") != benchmark_version:
        raise ValueError("annotation seal benchmark_version mismatch")
    count = seal.get("count")
    if type(count) is not int or count < 0:
        raise ValueError("annotation seal count must be a non-negative integer")
    _validate_completion_timestamp(seal.get("completed_at"))

    actual_annotation_hash = sha256_file(annotations_path)
    if seal.get("annotation_sha256") != actual_annotation_hash:
        raise ValueError("annotation hash does not match annotation seal")

    pair_rows = list(read_jsonl(pairs_path))
    annotation_rows = list(read_jsonl(annotations_path))
    _validate_finalized_annotations(annotation_rows, pair_rows)
    if count != len(annotation_rows):
        raise ValueError("annotation count does not match annotation seal")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_event(event: AnnotationEvent) -> None:
    if not isinstance(event.event_id, str) or not event.event_id:
        raise ValueError("event_id must be a non-empty string")
    if not isinstance(event.action, str):
        raise ValueError("action must be a string")
    if not isinstance(event.pair_id, str) or not event.pair_id:
        raise ValueError("pair_id must be a non-empty string")
    if not isinstance(event.annotated_at, str) or not event.annotated_at:
        raise ValueError("annotated_at must be a non-empty string")
    if event.action not in ("label", "undo"):
        raise ValueError(f"unsupported annotation action: {event.action}")
    if event.action == "label":
        if type(event.label) is not int or event.label not in (0, 1):
            raise ValueError("label events require label 0 or 1")
        if event.target_event_id is not None:
            raise ValueError("label events cannot target another event")
    if event.action == "undo":
        if (
            event.label is not None
            or not isinstance(event.target_event_id, str)
            or not event.target_event_id
        ):
            raise ValueError("undo events require label=None and target_event_id")


def _event_from_row(row: Mapping[str, object]) -> AnnotationEvent:
    missing_fields = EVENT_FIELDS - set(row)
    if missing_fields:
        raise ValueError(f"annotation event missing required field: {sorted(missing_fields)[0]}")
    return AnnotationEvent(
        event_id=row["event_id"],  # type: ignore[arg-type]
        action=row["action"],  # type: ignore[arg-type]
        pair_id=row["pair_id"],  # type: ignore[arg-type]
        label=row.get("label"),  # type: ignore[arg-type]
        target_event_id=row.get("target_event_id"),  # type: ignore[arg-type]
        annotated_at=row["annotated_at"],  # type: ignore[arg-type]
    )


def _read_events(path: Path) -> list[AnnotationEvent]:
    events: list[AnnotationEvent] = []
    seen_event_ids: set[str] = set()
    if not Path(path).exists():
        return events
    for row in read_jsonl(Path(path)):
        event = _event_from_row(row)
        if event.event_id in seen_event_ids:
            raise ValueError(f"duplicate event_id: {event.event_id}")
        seen_event_ids.add(event.event_id)
        events.append(event)
    return events


def _ensure_appendable_jsonl(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb") as handle:
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) != b"\n":
            raise ValueError("annotation event log must end with a newline before append")


def _replay_label_events(path: Path) -> tuple[dict[str, AnnotationEvent], set[str]]:
    label_events: dict[str, AnnotationEvent] = {}
    undone: set[str] = set()
    effective_by_pair: dict[str, str] = {}

    for event in _read_events(path):
        if event.action == "label":
            label_events[event.event_id] = event
            effective_by_pair[event.pair_id] = event.event_id
            continue

        target = event.target_event_id
        if target not in label_events:
            raise ValueError(f"invalid undo target: {target}")
        if target in undone:
            raise ValueError(f"invalid undo target already undone: {target}")
        if event.pair_id != label_events[target].pair_id:
            raise ValueError("undo pair_id does not match target event")
        if effective_by_pair.get(event.pair_id) != target:
            raise ValueError("undo target is not the currently effective label event")
        effective_event_ids = set(effective_by_pair.values())
        latest_effective_event_id = next(
            event_id
            for event_id in reversed(label_events)
            if event_id in effective_event_ids
        )
        if target != latest_effective_event_id:
            raise ValueError("undo target is not the globally latest effective label event")
        undone.add(target)
        replacement = next(
            (
                event_id
                for event_id, label_event in reversed(label_events.items())
                if label_event.pair_id == event.pair_id and event_id not in undone
            ),
            None,
        )
        if replacement is None:
            effective_by_pair.pop(event.pair_id, None)
        else:
            effective_by_pair[event.pair_id] = replacement
    return label_events, undone


def _prepare_jsonl_temp(
    path: Path,
    rows: Iterable[AnnotationEvent | Mapping[str, object]],
) -> Path:
    def write_rows(handle: TextIO) -> None:
        for row in rows:
            data = asdict(row) if isinstance(row, AnnotationEvent) else row
            if not isinstance(data, Mapping):
                raise ValueError("JSONL rows must be mappings")
            handle.write(json.dumps(data, sort_keys=True, separators=(",", ":")))
            handle.write("\n")

    return _prepare_text_temp(path, write_rows)


def _prepare_json_temp(path: Path, data: Mapping[str, object]) -> Path:
    def write_data(handle: TextIO) -> None:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return _prepare_text_temp(path, write_data)


def _prepare_text_temp(path: Path, writer: Callable[[TextIO], None]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temp_path = Path(temp_name)
    try:
        handle = os.fdopen(descriptor, "w", encoding="utf-8")
    except Exception:
        os.close(descriptor)
        _unlink_and_fsync(temp_path)
        raise
    try:
        with handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        _unlink_and_fsync(temp_path)
        raise
    return temp_path


def _fsync_parent(path: Path) -> None:
    parent = Path(path).parent
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(parent, flags)
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            pass
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass


def _unlink_and_fsync(path: Path) -> None:
    path = Path(path)
    if not path.exists():
        return
    path.unlink()
    _fsync_parent(path)


def _validate_context_tuple(
    pair: Mapping[str, object],
    field: str,
) -> tuple[str | None, str, str | None]:
    value = pair.get(field)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field} must contain previous/current/next image paths")
    converted = tuple(None if item is None else str(item) for item in value)
    if converted[1] is None:
        raise ValueError(f"{field} current image must not be None")
    return converted  # type: ignore[return-value]


def _secret_bytes(token_secret: bytes | str) -> bytes:
    return token_secret if isinstance(token_secret, bytes) else str(token_secret).encode("utf-8")


def _as_mapping(pair: Mapping[str, object] | object) -> Mapping[str, object]:
    if isinstance(pair, Mapping):
        return pair
    return asdict(pair)  # type: ignore[arg-type]


def _normalize_json_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    normalized = json.loads(json.dumps(value, sort_keys=True))
    if not isinstance(normalized, Mapping):
        raise ValueError("benchmark pair must normalize to a mapping")
    return normalized


def _validate_unique_pair_ids(pairs: Sequence[Mapping[str, object]]) -> list[str]:
    pair_ids: list[str] = []
    seen: set[str] = set()
    for pair in pairs:
        if "pair_id" not in pair:
            raise ValueError("benchmark pair missing pair_id")
        pair_id = pair["pair_id"]
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError("benchmark pair_id must be a non-empty string")
        if pair_id in seen:
            raise ValueError(f"duplicate pair_id in manifest: {pair_id}")
        seen.add(pair_id)
        pair_ids.append(pair_id)
    return pair_ids


def _validate_finalized_annotations(
    rows: Sequence[Mapping[str, object]],
    benchmark_pairs: Sequence[Mapping[str, object]],
) -> None:
    expected_pair_ids = _validate_unique_pair_ids(benchmark_pairs)
    expected_pair_id_set = set(expected_pair_ids)
    actual_pair_ids: list[str] = []
    seen_pair_ids: set[str] = set()
    for index, row in enumerate(rows):
        if set(row) != FINALIZED_ANNOTATION_FIELDS:
            raise ValueError(f"annotation row {index} has invalid fields")
        pair_id = row["pair_id"]
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError(f"annotation row {index} pair_id must be a non-empty string")
        if pair_id in seen_pair_ids:
            raise ValueError(f"duplicate annotation pair_id: {pair_id}")
        if pair_id not in expected_pair_id_set:
            raise ValueError(f"unknown annotation pair_id: {pair_id}")
        seen_pair_ids.add(pair_id)
        actual_pair_ids.append(pair_id)

        label = row["label"]
        if type(label) is not int or label not in (0, 1):
            raise ValueError(f"annotation row {index} label must be 0 or 1")
        annotated_at = row["annotated_at"]
        if not isinstance(annotated_at, str) or not annotated_at:
            raise ValueError(f"annotation row {index} annotated_at must be a non-empty string")
        if type(row["annotation_version"]) is not int:
            raise ValueError(f"annotation row {index} annotation_version must be an integer")
        if row["annotation_version"] != ANNOTATION_VERSION:
            raise ValueError(f"annotation row {index} annotation_version mismatch")

    if len(actual_pair_ids) != len(expected_pair_ids):
        raise ValueError("annotation coverage does not match benchmark pairs")
    if actual_pair_ids != expected_pair_ids:
        raise ValueError("annotation pair_id order does not match benchmark pairs")


def _validate_completion_timestamp(value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError("annotation seal completed_at must be a non-empty timestamp")
    try:
        completed_at = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("annotation seal completed_at must be a valid ISO timestamp") from exc
    if completed_at.tzinfo is None or completed_at.utcoffset() is None:
        raise ValueError("annotation seal completed_at must include a timezone")


def _verified_manifest_metadata(
    manifest_path: Path,
    pairs_path: Path | None = None,
) -> tuple[str, str]:
    manifest_path = Path(manifest_path)
    manifest = read_json(manifest_path)
    pair_hash = manifest.get("pair_manifest_sha256")
    if not isinstance(pair_hash, str) or not pair_hash:
        raise ValueError("manifest missing pair_manifest_sha256")
    pairs_path = (
        manifest_path.parent / "benchmark_pairs.jsonl"
        if pairs_path is None
        else Path(pairs_path)
    )
    actual_hash = sha256_file(pairs_path)
    if actual_hash != pair_hash:
        raise ValueError("pair-manifest hash does not match manifest")
    benchmark_version = manifest.get("benchmark_version")
    if not isinstance(benchmark_version, str) or not benchmark_version:
        raise ValueError("manifest benchmark_version must be a non-empty string")
    return pair_hash, benchmark_version
