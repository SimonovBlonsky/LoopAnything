from __future__ import annotations

import hashlib
import io
import json
import stat
import sys
from pathlib import Path

import pytest

LOOPANYTHING_ROOT = Path(__file__).resolve().parents[2]
if str(LOOPANYTHING_ROOT) not in sys.path:
    sys.path.insert(0, str(LOOPANYTHING_ROOT))

import robust_loop_verifier.rover_annotation as rover_annotation
from robust_loop_verifier.rover_annotation import (
    AnnotationEvent,
    annotation_progress,
    append_event,
    deterministic_display_order,
    finalize_annotations,
    opaque_image_tokens,
    replay_events,
    replay_resolved_label_events,
    valid_image_context,
    verify_annotation_seal,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_pairs(path: Path, pairs: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(pair, sort_keys=True) + "\n" for pair in pairs), encoding="utf-8"
    )


def _write_benchmark_files(
    root: Path, pairs: list[dict], manifest_extra: dict | None = None
) -> dict:
    pairs_path = root / "benchmark_pairs.jsonl"
    manifest_path = root / "manifest.json"
    _write_pairs(pairs_path, pairs)
    manifest = {
        "benchmark_version": "benchmark_v1",
        "pair_manifest_sha256": hashlib.sha256(pairs_path.read_bytes()).hexdigest(),
    }
    if manifest_extra:
        manifest.update(manifest_extra)
    _write_json(manifest_path, manifest)
    return manifest


def _pair(pair_id: str = "pair-a") -> dict:
    return {
        "pair_id": pair_id,
        "dataset": "d",
        "platform": "p",
        "sequence": "s",
        "query_context": (None, "images/query.png", "images/query-next.png"),
        "candidate_context": ("images/candidate-prev.png", "images/candidate.png", None),
        "rank": 1,
        "dbow2_score": 0.4,
        "query_idx": 10,
        "candidate_idx": 2,
    }


def _geometry_prediction(pair_id: str, automatic_label: int = 1, **updates) -> dict:
    prediction = {
        "pair_id": pair_id,
        "automatic_label": automatic_label,
        "support_idx": 28,
        "factor_status": "ok",
        "gt_rotation_angle_deg": 34.1,
        "gt_rotation_axis": [0.0, 0.0, 1.0],
        "gt_translation": [1.2, -0.1, 0.0],
        "gt_translation_norm_m": 1.204,
        "estimated_rotation_angle_deg": 31.2,
        "estimated_rotation_axis": [0.0, 0.0, 1.0],
        "estimated_translation": [1.1, -0.2, 0.0],
        "estimated_translation_norm_m": 1.118,
        "translation_error_m": 0.14,
        "effective_translation_error_threshold_m": 1.0,
        "rotation_error_deg": 3.2,
        "translation_direction_error_deg": 4.8,
        "automatic_label_reason": "accepted",
    }
    prediction.update(updates)
    return prediction


def _cache_state(tmp_path, pair_count: int = 1):
    annotator = pytest.importorskip(
        "robust_loop_verification_scripts.annotate_rover_aligned_benchmark"
    )
    benchmark_root = tmp_path / "benchmark"
    cache_root = tmp_path / "cache"
    (cache_root / "images").mkdir(parents=True)
    for image_name in [
        "query.png",
        "query-next.png",
        "candidate-prev.png",
        "candidate.png",
    ]:
        (cache_root / "images" / image_name).write_bytes(f"image:{image_name}".encode("utf-8"))
    benchmark_root.mkdir()
    pairs = [_pair(f"pair-{idx}") for idx in range(pair_count)]
    manifest = {
        "annotation_shuffle_seed": 1,
        "sequences": [
            {
                "dataset": "d",
                "platform": "p",
                "sequence": "s",
                "cache": str(cache_root),
            }
        ],
    }
    manifest = _write_benchmark_files(benchmark_root, pairs, manifest)
    return annotator.AnnotationServerState(
        benchmark_root=benchmark_root,
        pairs=pairs,
        manifest=manifest,
        token_secret=b"secret",
    )


class _FakeSocket:
    def __init__(self, request_bytes: bytes):
        self._reader = io.BytesIO(request_bytes)
        self._writer = io.BytesIO()

    def makefile(self, mode: str, buffering: int | None = None):
        if "r" in mode:
            return self._reader
        return self._writer

    def sendall(self, data: bytes) -> None:
        self._writer.write(data)


class _HandlerClient:
    def __init__(self, state):
        annotator = pytest.importorskip(
            "robust_loop_verification_scripts.annotate_rover_aligned_benchmark"
        )
        self.handler_cls = annotator.build_handler(state)

    def get_json(self, path: str) -> tuple[int, dict]:
        return self._request_json("GET", path, None)

    def post_json(self, path: str, payload: dict) -> tuple[int, dict]:
        return self._request_json("POST", path, payload)

    def post_raw(
        self,
        path: str,
        body: bytes,
        content_length: str | None = None,
    ) -> tuple[int, dict]:
        status, _, response_body = self._request_raw(
            "POST",
            path,
            body,
            content_length if content_length is not None else str(len(body)),
        )
        return status, json.loads(response_body.decode("utf-8"))

    def get_bytes(self, path: str) -> tuple[int, bytes]:
        status, _, body = self._request("GET", path, None)
        return status, body

    def _request_json(self, method: str, path: str, payload: dict | None) -> tuple[int, dict]:
        status, _, body = self._request(method, path, payload)
        return status, json.loads(body.decode("utf-8"))

    def _request(
        self,
        method: str,
        path: str,
        payload: dict | None,
    ) -> tuple[int, dict[str, str], bytes]:
        body = b"" if payload is None else json.dumps(payload).encode("utf-8")
        headers = [
            f"{method} {path} HTTP/1.1",
            "Host: testserver",
            f"Content-Length: {len(body)}",
        ]
        if payload is not None:
            headers.append("Content-Type: application/json")
        request_bytes = ("\r\n".join(headers) + "\r\n\r\n").encode("utf-8") + body
        fake_socket = _FakeSocket(request_bytes)
        self.handler_cls(fake_socket, ("127.0.0.1", 0), object())
        return self._parse_response(fake_socket._writer.getvalue())

    def _request_raw(
        self,
        method: str,
        path: str,
        body: bytes,
        content_length: str,
    ) -> tuple[int, dict[str, str], bytes]:
        request_bytes = (
            "\r\n".join(
                [
                    f"{method} {path} HTTP/1.1",
                    "Host: testserver",
                    "Content-Type: application/json",
                    f"Content-Length: {content_length}",
                ]
            )
            + "\r\n\r\n"
        ).encode("utf-8") + body
        fake_socket = _FakeSocket(request_bytes)
        self.handler_cls(fake_socket, ("127.0.0.1", 0), object())
        return self._parse_response(fake_socket._writer.getvalue())

    def _parse_response(self, response: bytes) -> tuple[int, dict[str, str], bytes]:
        header_bytes, response_body = response.split(b"\r\n\r\n", 1)
        header_lines = header_bytes.decode("iso-8859-1").split("\r\n")
        status = int(header_lines[0].split()[1])
        response_headers = {}
        for line in header_lines[1:]:
            key, value = line.split(":", 1)
            response_headers[key.lower()] = value.strip()
        return status, response_headers, response_body


def _event_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _finalize_fixture(tmp_path, pair_count: int = 2):
    pairs = [_pair(f"pair-{idx}") for idx in range(pair_count)]
    events = tmp_path / "annotation_events.jsonl"
    output = tmp_path / "annotations.jsonl"
    seal = tmp_path / "annotation_seal.json"
    _write_benchmark_files(tmp_path, pairs)
    for idx, pair in enumerate(pairs):
        append_event(
            events,
            AnnotationEvent(
                f"e{idx}",
                "label",
                pair["pair_id"],
                idx % 2,
                None,
                f"t{idx}",
            ),
        )
    finalize_annotations(
        pairs=pairs,
        event_path=events,
        output_path=output,
        manifest_path=tmp_path / "manifest.json",
        seal_path=seal,
    )
    return pairs, output, seal


def _read_jsonl_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _rewrite_annotations_and_seal_hash(output: Path, seal: Path, rows: list[dict]) -> None:
    _write_pairs(output, rows)
    seal_payload = json.loads(seal.read_text(encoding="utf-8"))
    seal_payload["annotation_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    seal_payload["count"] = len(rows)
    _write_json(seal, seal_payload)


def test_replay_supports_positive_negative_and_undo(tmp_path):
    events = tmp_path / "annotation_events.jsonl"
    append_event(events, AnnotationEvent("e1", "label", "pair-a", 1, None, "t1"))
    append_event(events, AnnotationEvent("e2", "label", "pair-b", 0, None, "t2"))
    append_event(events, AnnotationEvent("e3", "undo", "pair-b", None, "e2", "t3"))

    assert replay_events(events) == {"pair-a": 1}


def test_finalize_requires_complete_binary_labels(tmp_path):
    pairs = [{"pair_id": "pair-a"}, {"pair_id": "pair-b"}]
    events = tmp_path / "annotation_events.jsonl"
    _write_benchmark_files(tmp_path, pairs)
    append_event(events, AnnotationEvent("e1", "label", "pair-a", 1, None, "t1"))

    with pytest.raises(ValueError, match="unlabeled"):
        finalize_annotations(
            pairs=pairs,
            event_path=events,
            output_path=tmp_path / "annotations.jsonl",
            manifest_path=tmp_path / "manifest.json",
            seal_path=tmp_path / "annotation_seal.json",
        )


@pytest.mark.parametrize("label", [-1, 2, None])
def test_append_rejects_invalid_labels(tmp_path, label):
    with pytest.raises(ValueError, match="label 0 or 1"):
        append_event(
            tmp_path / "annotation_events.jsonl",
            AnnotationEvent("e1", "label", "pair-a", label, None, "t1"),
        )


@pytest.mark.parametrize("label", [False, True, 0.0, 1.0, "0", "1", None])
def test_annotation_event_creation_rejects_non_exact_integer_labels(label):
    with pytest.raises(ValueError, match="label 0 or 1"):
        AnnotationEvent("e1", "label", "pair-a", label, None, "t1")


@pytest.mark.parametrize("label", [False, True, 0.0, 1.0, "0", "1", None])
def test_replay_rejects_non_exact_integer_labels_from_disk(tmp_path, label):
    events = tmp_path / "annotation_events.jsonl"
    events.write_text(
        json.dumps(
            {
                "event_id": "e1",
                "action": "label",
                "pair_id": "pair-a",
                "label": label,
                "target_event_id": None,
                "annotated_at": "t1",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="label 0 or 1"):
        replay_events(events)


@pytest.mark.parametrize(
    "row",
    [
        {
            "action": "label",
            "pair_id": "pair-a",
            "label": 1,
            "target_event_id": None,
            "annotated_at": "t1",
        },
        {
            "event_id": 1,
            "action": "label",
            "pair_id": "pair-a",
            "label": 1,
            "target_event_id": None,
            "annotated_at": "t1",
        },
        {
            "event_id": "",
            "action": "label",
            "pair_id": "pair-a",
            "label": 1,
            "target_event_id": None,
            "annotated_at": "t1",
        },
        {
            "event_id": "e1",
            "action": 1,
            "pair_id": "pair-a",
            "label": 1,
            "target_event_id": None,
            "annotated_at": "t1",
        },
        {
            "event_id": "e1",
            "action": "label",
            "pair_id": None,
            "label": 1,
            "target_event_id": None,
            "annotated_at": "t1",
        },
        {
            "event_id": "e1",
            "action": "label",
            "pair_id": "",
            "label": 1,
            "target_event_id": None,
            "annotated_at": "t1",
        },
        {
            "event_id": "e1",
            "action": "label",
            "pair_id": "pair-a",
            "label": 1,
            "target_event_id": None,
        },
        {
            "event_id": "e1",
            "action": "label",
            "pair_id": "pair-a",
            "label": 1,
            "target_event_id": None,
            "annotated_at": 1,
        },
        {
            "event_id": "e1",
            "action": "label",
            "pair_id": "pair-a",
            "label": 1,
            "target_event_id": None,
            "annotated_at": "",
        },
    ],
)
def test_replay_rejects_malformed_persisted_event_fields(tmp_path, row):
    events = tmp_path / "annotation_events.jsonl"
    events.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="event|action|pair_id|annotated_at"):
        replay_events(events)


def test_finalize_rejects_labels_for_unknown_pair_ids(tmp_path):
    pairs = [{"pair_id": "pair-a"}]
    events = tmp_path / "annotation_events.jsonl"
    _write_benchmark_files(tmp_path, pairs)
    append_event(events, AnnotationEvent("e1", "label", "pair-a", 1, None, "t1"))
    append_event(events, AnnotationEvent("e2", "label", "pair-b", 0, None, "t2"))

    with pytest.raises(ValueError, match="unknown pair_id"):
        finalize_annotations(
            pairs=pairs,
            event_path=events,
            output_path=tmp_path / "annotations.jsonl",
            manifest_path=tmp_path / "manifest.json",
            seal_path=tmp_path / "annotation_seal.json",
        )


def test_finalize_rejects_duplicate_manifest_pair_ids(tmp_path):
    pairs = [{"pair_id": "pair-a"}, {"pair_id": "pair-a"}]
    events = tmp_path / "annotation_events.jsonl"
    _write_benchmark_files(tmp_path, pairs)

    with pytest.raises(ValueError, match="duplicate pair_id"):
        finalize_annotations(
            pairs=pairs,
            event_path=events,
            output_path=tmp_path / "annotations.jsonl",
            manifest_path=tmp_path / "manifest.json",
            seal_path=tmp_path / "annotation_seal.json",
        )


def test_append_rejects_duplicate_event_ids(tmp_path):
    events = tmp_path / "annotation_events.jsonl"
    append_event(events, AnnotationEvent("e1", "label", "pair-a", 1, None, "t1"))

    with pytest.raises(ValueError, match="duplicate event_id"):
        append_event(events, AnnotationEvent("e1", "label", "pair-b", 0, None, "t2"))


def test_append_event_physically_appends_without_replacing_existing_log(tmp_path, monkeypatch):
    events = tmp_path / "annotation_events.jsonl"
    append_event(events, AnnotationEvent("e1", "label", "pair-a", 1, None, "t1"))
    original = events.read_text(encoding="utf-8")

    def forbid_replace(_source, _destination):
        raise AssertionError("append_event must not rewrite or replace the annotation log")

    monkeypatch.setattr(rover_annotation.os, "replace", forbid_replace)

    append_event(events, AnnotationEvent("e2", "label", "pair-b", 0, None, "t2"))

    assert events.read_text(encoding="utf-8").startswith(original)
    assert replay_events(events) == {"pair-a": 1, "pair-b": 0}
    assert list(tmp_path.iterdir()) == [events]


def test_atomic_append_fsyncs_parent_directory_after_publication(tmp_path, monkeypatch):
    events = tmp_path / "annotation_events.jsonl"
    real_open = rover_annotation.os.open
    opened_paths = []

    def record_open(path, flags, mode=0o777):
        opened_paths.append(Path(path))
        return real_open(path, flags, mode)

    monkeypatch.setattr(rover_annotation.os, "open", record_open)

    append_event(events, AnnotationEvent("e1", "label", "pair-a", 1, None, "t1"))

    assert tmp_path in opened_paths


def test_atomic_append_commits_when_parent_directory_fsync_fails(tmp_path, monkeypatch):
    events = tmp_path / "annotation_events.jsonl"
    real_fsync = rover_annotation.os.fsync

    def fail_directory_fsync(descriptor):
        if stat.S_ISDIR(rover_annotation.os.fstat(descriptor).st_mode):
            raise OSError("injected directory fsync failure")
        return real_fsync(descriptor)

    monkeypatch.setattr(rover_annotation.os, "fsync", fail_directory_fsync)

    append_event(events, AnnotationEvent("e1", "label", "pair-a", 1, None, "t1"))

    assert replay_events(events) == {"pair-a": 1}


def test_replay_rejects_duplicate_event_ids_from_disk(tmp_path):
    events = tmp_path / "annotation_events.jsonl"
    rows = [
        AnnotationEvent("e1", "label", "pair-a", 1, None, "t1"),
        AnnotationEvent("e1", "label", "pair-b", 0, None, "t2"),
    ]
    events.write_text(
        "".join(json.dumps(event.__dict__, sort_keys=True) + "\n" for event in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate event_id"):
        replay_events(events)


def test_restart_replay_reads_persisted_log(tmp_path):
    events = tmp_path / "annotation_events.jsonl"
    append_event(events, AnnotationEvent("e1", "label", "pair-a", 1, None, "t1"))
    append_event(events, AnnotationEvent("e2", "label", "pair-a", 0, None, "t2"))

    assert replay_events(events) == {"pair-a": 0}
    assert replay_resolved_label_events(events)["pair-a"].annotated_at == "t2"


def test_replay_rejects_undo_targeting_unknown_event(tmp_path):
    events = tmp_path / "annotation_events.jsonl"
    append_event(events, AnnotationEvent("e1", "undo", "pair-a", None, "missing", "t1"))

    with pytest.raises(ValueError, match="invalid undo target"):
        replay_events(events)


def test_replay_rejects_undo_targeting_already_undone_event(tmp_path):
    events = tmp_path / "annotation_events.jsonl"
    append_event(events, AnnotationEvent("e1", "label", "pair-a", 1, None, "t1"))
    append_event(events, AnnotationEvent("e2", "undo", "pair-a", None, "e1", "t2"))
    append_event(events, AnnotationEvent("e3", "undo", "pair-a", None, "e1", "t3"))

    with pytest.raises(ValueError, match="already undone"):
        replay_events(events)


def test_replay_rejects_undo_targeting_superseded_label_event(tmp_path):
    events = tmp_path / "annotation_events.jsonl"
    append_event(events, AnnotationEvent("e1", "label", "pair-a", 1, None, "t1"))
    append_event(events, AnnotationEvent("e2", "label", "pair-a", 0, None, "t2"))
    append_event(events, AnnotationEvent("e3", "undo", "pair-a", None, "e1", "t3"))

    with pytest.raises(ValueError, match="superseded|effective"):
        replay_events(events)


def test_replay_requires_undo_of_globally_latest_active_label(tmp_path):
    invalid_events = tmp_path / "invalid_annotation_events.jsonl"
    append_event(invalid_events, AnnotationEvent("e1", "label", "pair-a", 1, None, "t1"))
    append_event(invalid_events, AnnotationEvent("e2", "label", "pair-b", 0, None, "t2"))
    append_event(invalid_events, AnnotationEvent("e3", "undo", "pair-a", None, "e1", "t3"))

    with pytest.raises(ValueError, match="latest|effective"):
        replay_events(invalid_events)

    valid_events = tmp_path / "valid_annotation_events.jsonl"
    append_event(valid_events, AnnotationEvent("e1", "label", "pair-a", 1, None, "t1"))
    append_event(valid_events, AnnotationEvent("e2", "label", "pair-b", 0, None, "t2"))
    append_event(valid_events, AnnotationEvent("e3", "undo", "pair-b", None, "e2", "t3"))

    assert replay_events(valid_events) == {"pair-a": 1}


def test_append_rejects_invalid_undo_shape(tmp_path):
    with pytest.raises(ValueError, match="undo events require"):
        append_event(
            tmp_path / "annotation_events.jsonl",
            AnnotationEvent("e1", "undo", "pair-a", 1, "target", "t1"),
        )

    with pytest.raises(ValueError, match="undo events require"):
        append_event(
            tmp_path / "annotation_events.jsonl",
            AnnotationEvent("e2", "undo", "pair-a", None, None, "t2"),
        )


def test_replay_rejects_undo_pair_mismatch(tmp_path):
    events = tmp_path / "annotation_events.jsonl"
    append_event(events, AnnotationEvent("e1", "label", "pair-a", 1, None, "t1"))
    append_event(events, AnnotationEvent("e2", "undo", "pair-b", None, "e1", "t2"))

    with pytest.raises(ValueError, match="pair_id"):
        replay_events(events)


def test_finalize_writes_seal_and_verifies_hashes(tmp_path):
    pairs = [_pair("pair-a"), _pair("pair-b")]
    pairs_path = tmp_path / "benchmark_pairs.jsonl"
    manifest_path = tmp_path / "manifest.json"
    events = tmp_path / "annotation_events.jsonl"
    output = tmp_path / "annotations.jsonl"
    seal = tmp_path / "annotation_seal.json"
    _write_benchmark_files(tmp_path, pairs)
    append_event(events, AnnotationEvent("e1", "label", "pair-a", 1, None, "t1"))
    append_event(events, AnnotationEvent("e2", "label", "pair-b", 0, None, "t2"))

    finalize_annotations(
        pairs=pairs,
        event_path=events,
        output_path=output,
        manifest_path=manifest_path,
        seal_path=seal,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {
            "annotated_at": "t1",
            "annotation_version": 1,
            "label": 1,
            "pair_id": "pair-a",
        },
        {
            "annotated_at": "t2",
            "annotation_version": 1,
            "label": 0,
            "pair_id": "pair-b",
        },
    ]
    seal_payload = json.loads(seal.read_text(encoding="utf-8"))
    assert seal_payload["version"] == "rover_annotation_seal_v1"
    assert seal_payload["annotation_version"] == 1
    assert seal_payload["benchmark_version"] == "benchmark_v1"
    verify_annotation_seal(manifest_path, pairs_path, output, seal)
    output.write_text(output.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="annotation hash"):
        verify_annotation_seal(manifest_path, pairs_path, output, seal)


@pytest.mark.parametrize(
    "case",
    [
        "missing_field",
        "extra_field",
        "empty_pair_id",
        "duplicate_pair_id",
        "missing_pair",
        "unknown_pair",
        "reordered_pair_ids",
        "bool_label",
        "float_label",
        "string_label",
        "missing_annotated_at",
        "empty_annotated_at",
        "numeric_annotated_at",
        "missing_annotation_version",
        "wrong_annotation_version",
    ],
)
def test_verify_annotation_seal_rejects_malformed_finalized_rows(tmp_path, case):
    pairs, output, seal = _finalize_fixture(tmp_path)
    rows = _read_jsonl_rows(output)

    if case == "missing_field":
        rows[0].pop("label")
    elif case == "extra_field":
        rows[0]["unexpected"] = "value"
    elif case == "empty_pair_id":
        rows[0]["pair_id"] = ""
    elif case == "duplicate_pair_id":
        rows[1]["pair_id"] = rows[0]["pair_id"]
    elif case == "missing_pair":
        rows.pop()
    elif case == "unknown_pair":
        rows[1]["pair_id"] = "unknown"
    elif case == "reordered_pair_ids":
        rows.reverse()
    elif case == "bool_label":
        rows[0]["label"] = True
    elif case == "float_label":
        rows[0]["label"] = 1.0
    elif case == "string_label":
        rows[0]["label"] = "1"
    elif case == "missing_annotated_at":
        rows[0].pop("annotated_at")
    elif case == "empty_annotated_at":
        rows[0]["annotated_at"] = ""
    elif case == "numeric_annotated_at":
        rows[0]["annotated_at"] = 1
    elif case == "missing_annotation_version":
        rows[0].pop("annotation_version")
    elif case == "wrong_annotation_version":
        rows[0]["annotation_version"] = 2

    _rewrite_annotations_and_seal_hash(output, seal, rows)

    with pytest.raises(
        ValueError,
        match="annotation|pair_id|label|annotated_at|coverage|order|fields",
    ):
        verify_annotation_seal(
            tmp_path / "manifest.json",
            tmp_path / "benchmark_pairs.jsonl",
            output,
            seal,
        )


@pytest.mark.parametrize("count", [True, False, 1.0, "2", None])
def test_verify_annotation_seal_rejects_invalid_count_type(tmp_path, count):
    _, output, seal = _finalize_fixture(tmp_path)
    seal_payload = json.loads(seal.read_text(encoding="utf-8"))
    seal_payload["count"] = count
    _write_json(seal, seal_payload)

    with pytest.raises(ValueError, match="count"):
        verify_annotation_seal(
            tmp_path / "manifest.json",
            tmp_path / "benchmark_pairs.jsonl",
            output,
            seal,
        )


@pytest.mark.parametrize("completed_at", [None, "", "not-a-timestamp", "2026-06-10T12:00:00"])
def test_verify_annotation_seal_rejects_invalid_completion_timestamp(tmp_path, completed_at):
    _, output, seal = _finalize_fixture(tmp_path)
    seal_payload = json.loads(seal.read_text(encoding="utf-8"))
    seal_payload["completed_at"] = completed_at
    _write_json(seal, seal_payload)

    with pytest.raises(ValueError, match="completed_at"):
        verify_annotation_seal(
            tmp_path / "manifest.json",
            tmp_path / "benchmark_pairs.jsonl",
            output,
            seal,
        )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("count", "count"),
        ("completed_at", "completed_at"),
        ("annotation_version", "version"),
        ("benchmark_version", "version"),
    ],
)
def test_verify_annotation_seal_rejects_missing_required_metadata(tmp_path, field, message):
    _, output, seal = _finalize_fixture(tmp_path)
    seal_payload = json.loads(seal.read_text(encoding="utf-8"))
    seal_payload.pop(field)
    _write_json(seal, seal_payload)

    with pytest.raises(ValueError, match=message):
        verify_annotation_seal(
            tmp_path / "manifest.json",
            tmp_path / "benchmark_pairs.jsonl",
            output,
            seal,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", "wrong"),
        ("annotation_version", 2),
        ("benchmark_version", "wrong"),
    ],
)
def test_verify_annotation_seal_rejects_all_version_mismatches(tmp_path, field, value):
    _, output, seal = _finalize_fixture(tmp_path)
    seal_payload = json.loads(seal.read_text(encoding="utf-8"))
    seal_payload[field] = value
    _write_json(seal, seal_payload)

    with pytest.raises(ValueError, match="version"):
        verify_annotation_seal(
            tmp_path / "manifest.json",
            tmp_path / "benchmark_pairs.jsonl",
            output,
            seal,
        )


def test_finalize_rolls_back_output_when_seal_publication_fails_and_retry_succeeds(
    tmp_path, monkeypatch
):
    pairs = [_pair("pair-a")]
    events = tmp_path / "annotation_events.jsonl"
    output = tmp_path / "annotations.jsonl"
    seal = tmp_path / "annotation_seal.json"
    _write_benchmark_files(tmp_path, pairs)
    append_event(events, AnnotationEvent("e1", "label", "pair-a", 1, None, "t1"))
    real_replace = rover_annotation.os.replace

    def fail_seal_replace(source, destination):
        if Path(destination) == seal:
            raise OSError("injected seal publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr(rover_annotation.os, "replace", fail_seal_replace)

    with pytest.raises(OSError, match="injected seal publication failure"):
        finalize_annotations(
            pairs=pairs,
            event_path=events,
            output_path=output,
            manifest_path=tmp_path / "manifest.json",
            seal_path=seal,
        )

    assert not output.exists()
    assert not seal.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "annotation_events.jsonl",
        "benchmark_pairs.jsonl",
        "manifest.json",
    ]

    monkeypatch.setattr(rover_annotation.os, "replace", real_replace)
    finalize_annotations(
        pairs=pairs,
        event_path=events,
        output_path=output,
        manifest_path=tmp_path / "manifest.json",
        seal_path=seal,
    )

    verify_annotation_seal(
        tmp_path / "manifest.json",
        tmp_path / "benchmark_pairs.jsonl",
        output,
        seal,
    )


def test_finalize_fsyncs_parent_after_publications_and_rollback(tmp_path, monkeypatch):
    pairs = [_pair("pair-a")]
    events = tmp_path / "annotation_events.jsonl"
    output = tmp_path / "annotations.jsonl"
    seal = tmp_path / "annotation_seal.json"
    _write_benchmark_files(tmp_path, pairs)
    append_event(events, AnnotationEvent("e1", "label", "pair-a", 1, None, "t1"))
    real_open = rover_annotation.os.open
    real_replace = rover_annotation.os.replace
    opened_paths = []

    def record_open(path, flags, mode=0o777):
        opened_paths.append(Path(path))
        return real_open(path, flags, mode)

    def fail_seal_replace(source, destination):
        if Path(destination) == seal:
            raise OSError("injected seal publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr(rover_annotation.os, "open", record_open)
    monkeypatch.setattr(rover_annotation.os, "replace", fail_seal_replace)

    with pytest.raises(OSError, match="injected seal publication failure"):
        finalize_annotations(
            pairs=pairs,
            event_path=events,
            output_path=output,
            manifest_path=tmp_path / "manifest.json",
            seal_path=seal,
        )

    assert opened_paths.count(tmp_path) >= 2
    assert not output.exists()
    assert not seal.exists()


def test_finalize_commits_when_parent_directory_fsync_fails(tmp_path, monkeypatch):
    pairs = [_pair("pair-a")]
    events = tmp_path / "annotation_events.jsonl"
    output = tmp_path / "annotations.jsonl"
    seal = tmp_path / "annotation_seal.json"
    _write_benchmark_files(tmp_path, pairs)
    append_event(events, AnnotationEvent("e1", "label", "pair-a", 1, None, "t1"))
    real_fsync = rover_annotation.os.fsync

    def fail_directory_fsync(descriptor):
        if stat.S_ISDIR(rover_annotation.os.fstat(descriptor).st_mode):
            raise OSError("injected directory fsync failure")
        return real_fsync(descriptor)

    monkeypatch.setattr(rover_annotation.os, "fsync", fail_directory_fsync)

    finalize_annotations(
        pairs=pairs,
        event_path=events,
        output_path=output,
        manifest_path=tmp_path / "manifest.json",
        seal_path=seal,
    )

    verify_annotation_seal(
        tmp_path / "manifest.json",
        tmp_path / "benchmark_pairs.jsonl",
        output,
        seal,
    )


def test_finalize_rejects_missing_manifest_hash_without_writing_outputs(tmp_path):
    pairs = [_pair("pair-a")]
    events = tmp_path / "annotation_events.jsonl"
    output = tmp_path / "annotations.jsonl"
    seal = tmp_path / "annotation_seal.json"
    _write_pairs(tmp_path / "benchmark_pairs.jsonl", pairs)
    _write_json(tmp_path / "manifest.json", {})
    append_event(events, AnnotationEvent("e1", "label", "pair-a", 1, None, "t1"))

    with pytest.raises(ValueError, match="pair_manifest_sha256"):
        finalize_annotations(
            pairs=pairs,
            event_path=events,
            output_path=output,
            manifest_path=tmp_path / "manifest.json",
            seal_path=seal,
        )

    assert not output.exists()
    assert not seal.exists()


def test_finalize_rejects_wrong_manifest_hash_without_writing_outputs(tmp_path):
    pairs = [_pair("pair-a")]
    events = tmp_path / "annotation_events.jsonl"
    output = tmp_path / "annotations.jsonl"
    seal = tmp_path / "annotation_seal.json"
    _write_pairs(tmp_path / "benchmark_pairs.jsonl", pairs)
    _write_json(tmp_path / "manifest.json", {"pair_manifest_sha256": "wrong"})
    append_event(events, AnnotationEvent("e1", "label", "pair-a", 1, None, "t1"))

    with pytest.raises(ValueError, match="pair-manifest hash"):
        finalize_annotations(
            pairs=pairs,
            event_path=events,
            output_path=output,
            manifest_path=tmp_path / "manifest.json",
            seal_path=seal,
        )

    assert not output.exists()
    assert not seal.exists()


@pytest.mark.parametrize("benchmark_version", [None, "", 1])
def test_finalize_rejects_invalid_benchmark_version_without_writing_outputs(
    tmp_path, benchmark_version
):
    pairs = [_pair("pair-a")]
    events = tmp_path / "annotation_events.jsonl"
    output = tmp_path / "annotations.jsonl"
    seal = tmp_path / "annotation_seal.json"
    manifest = _write_benchmark_files(tmp_path, pairs)
    manifest["benchmark_version"] = benchmark_version
    _write_json(tmp_path / "manifest.json", manifest)
    append_event(events, AnnotationEvent("e1", "label", "pair-a", 1, None, "t1"))

    with pytest.raises(ValueError, match="benchmark_version"):
        finalize_annotations(
            pairs=pairs,
            event_path=events,
            output_path=output,
            manifest_path=tmp_path / "manifest.json",
            seal_path=seal,
        )

    assert not output.exists()
    assert not seal.exists()


@pytest.mark.parametrize(
    "caller_pairs",
    [
        [_pair("pair-a")],
        [_pair("pair-b"), _pair("pair-a")],
        [_pair("pair-a"), _pair("pair-c")],
        [{**_pair("pair-a"), "rank": 2}, _pair("pair-b")],
    ],
)
def test_finalize_rejects_caller_pairs_that_differ_from_disk(tmp_path, caller_pairs):
    disk_pairs = [_pair("pair-a"), _pair("pair-b")]
    events = tmp_path / "annotation_events.jsonl"
    output = tmp_path / "annotations.jsonl"
    seal = tmp_path / "annotation_seal.json"
    _write_benchmark_files(tmp_path, disk_pairs)
    append_event(events, AnnotationEvent("e1", "label", "pair-a", 1, None, "t1"))
    append_event(events, AnnotationEvent("e2", "label", "pair-b", 0, None, "t2"))

    with pytest.raises(ValueError, match="caller.*benchmark_pairs|pairs.*disk"):
        finalize_annotations(
            pairs=caller_pairs,
            event_path=events,
            output_path=output,
            manifest_path=tmp_path / "manifest.json",
            seal_path=seal,
        )

    assert not output.exists()
    assert not seal.exists()


def test_verify_annotation_seal_rejects_pair_manifest_hash_mismatch(tmp_path):
    pairs = [_pair("pair-a")]
    pairs_path = tmp_path / "benchmark_pairs.jsonl"
    manifest_path = tmp_path / "manifest.json"
    output = tmp_path / "annotations.jsonl"
    seal = tmp_path / "annotation_seal.json"
    _write_pairs(pairs_path, pairs)
    _write_json(
        manifest_path,
        {"benchmark_version": "benchmark_v1", "pair_manifest_sha256": "wrong"},
    )
    _write_pairs(
        output,
        [
            {
                "pair_id": "pair-a",
                "label": 1,
                "annotated_at": "t1",
                "annotation_version": 1,
            }
        ],
    )
    _write_json(
        seal,
        {
            "version": "rover_annotation_seal_v1",
            "annotation_version": 1,
            "benchmark_version": "benchmark_v1",
            "pair_manifest_sha256": "wrong",
            "annotation_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "count": 1,
            "completed_at": "2026-06-10T12:00:00+00:00",
        },
    )

    with pytest.raises(ValueError, match="pair-manifest hash"):
        verify_annotation_seal(manifest_path, pairs_path, output, seal)


def test_verify_annotation_seal_rejects_count_mismatch(tmp_path):
    pairs = [_pair("pair-a")]
    pairs_path = tmp_path / "benchmark_pairs.jsonl"
    manifest_path = tmp_path / "manifest.json"
    output = tmp_path / "annotations.jsonl"
    seal = tmp_path / "annotation_seal.json"
    _write_pairs(pairs_path, pairs)
    pair_hash = hashlib.sha256(pairs_path.read_bytes()).hexdigest()
    _write_json(
        manifest_path,
        {"benchmark_version": "benchmark_v1", "pair_manifest_sha256": pair_hash},
    )
    _write_pairs(
        output,
        [
            {
                "pair_id": "pair-a",
                "label": 1,
                "annotated_at": "t1",
                "annotation_version": 1,
            }
        ],
    )
    _write_json(
        seal,
        {
            "version": "rover_annotation_seal_v1",
            "annotation_version": 1,
            "benchmark_version": "benchmark_v1",
            "pair_manifest_sha256": pair_hash,
            "annotation_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "count": 2,
            "completed_at": "2026-06-10T12:00:00+00:00",
        },
    )

    with pytest.raises(ValueError, match="count"):
        verify_annotation_seal(manifest_path, pairs_path, output, seal)


def test_verify_annotation_seal_rejects_version_mismatch(tmp_path):
    pairs = [_pair("pair-a")]
    pairs_path = tmp_path / "benchmark_pairs.jsonl"
    manifest_path = tmp_path / "manifest.json"
    output = tmp_path / "annotations.jsonl"
    seal = tmp_path / "annotation_seal.json"
    _write_pairs(pairs_path, pairs)
    pair_hash = hashlib.sha256(pairs_path.read_bytes()).hexdigest()
    _write_json(
        manifest_path,
        {"benchmark_version": "benchmark_v1", "pair_manifest_sha256": pair_hash},
    )
    _write_pairs(
        output,
        [
            {
                "pair_id": "pair-a",
                "label": 1,
                "annotated_at": "t1",
                "annotation_version": 1,
            }
        ],
    )
    _write_json(
        seal,
        {
            "version": "wrong",
            "annotation_version": 1,
            "benchmark_version": "benchmark_v1",
            "pair_manifest_sha256": pair_hash,
            "annotation_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "count": 1,
            "completed_at": "2026-06-10T12:00:00+00:00",
        },
    )

    with pytest.raises(ValueError, match="version"):
        verify_annotation_seal(manifest_path, pairs_path, output, seal)


def test_finalize_refuses_existing_outputs_and_append_refuses_after_sealing(tmp_path):
    events = tmp_path / "annotation_events.jsonl"
    seal = tmp_path / "annotation_seal.json"
    seal.write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="sealed"):
        append_event(events, AnnotationEvent("e1", "label", "pair-a", 1, None, "t1"))

    with pytest.raises(FileExistsError, match="already exists"):
        finalize_annotations(
            pairs=[{"pair_id": "pair-a"}],
            event_path=events,
            output_path=tmp_path / "annotations.jsonl",
            manifest_path=tmp_path / "manifest.json",
            seal_path=seal,
        )


def test_valid_image_context_keeps_sequence_boundary_nones():
    pair = _pair()

    assert valid_image_context(pair) == {
        "query_prev": None,
        "query": "images/query.png",
        "query_next": "images/query-next.png",
        "candidate_prev": "images/candidate-prev.png",
        "candidate": "images/candidate.png",
        "candidate_next": None,
    }


def test_opaque_image_tokens_are_deterministic_and_hide_paths():
    pair = _pair("pair-a")

    tokens = opaque_image_tokens(pair, token_secret=b"secret")
    assert tokens == opaque_image_tokens(pair, token_secret=b"secret")
    assert tokens["query_prev"] is None
    assert tokens["candidate_next"] is None
    assert tokens["query"] != "images/query.png"
    assert "query" not in tokens["query"]


def test_display_order_and_progress_are_deterministic():
    pair_ids = ["pair-a", "pair-b", "pair-c"]

    assert deterministic_display_order(pair_ids, seed=3) == deterministic_display_order(
        pair_ids, seed=3
    )
    assert sorted(deterministic_display_order(pair_ids, seed=3)) == pair_ids
    assert annotation_progress(pair_ids, {"pair-a": 1}) == {"completed": 1, "total": 3}


def test_stale_state_nonce_rejection(tmp_path):
    state = _cache_state(tmp_path)
    payload = state.public_state()

    assert state.validate_request(payload["pair_token"], payload["state_nonce"]) == "pair-0"
    with pytest.raises(ValueError, match="state nonce"):
        state.validate_request(payload["pair_token"], "stale")
    with pytest.raises(ValueError, match="pair token"):
        state.validate_request("wrong", payload["state_nonce"])


def test_from_benchmark_root_loads_geometry_predictions_in_exact_pair_order(tmp_path):
    direct_state = _cache_state(tmp_path, pair_count=2)
    predictions = [
        _geometry_prediction("pair-0"),
        _geometry_prediction("pair-1", automatic_label=0),
    ]
    _write_pairs(direct_state.benchmark_root / "geometry_predictions.jsonl", predictions)

    state = type(direct_state).from_benchmark_root(direct_state.benchmark_root)

    assert state.geometry_by_pair_id == {
        "pair-0": predictions[0],
        "pair-1": predictions[1],
    }


@pytest.mark.parametrize(
    "prediction_pair_ids",
    [
        ["pair-0"],
        ["pair-0", "pair-0"],
        ["pair-0", "unknown"],
        ["pair-1", "pair-0"],
    ],
)
def test_from_benchmark_root_rejects_geometry_without_exact_ordered_coverage(
    tmp_path, prediction_pair_ids
):
    direct_state = _cache_state(tmp_path, pair_count=2)
    predictions = [_geometry_prediction(pair_id) for pair_id in prediction_pair_ids]
    _write_pairs(direct_state.benchmark_root / "geometry_predictions.jsonl", predictions)

    with pytest.raises(ValueError, match="geometry|pair_id|order|coverage"):
        type(direct_state).from_benchmark_root(direct_state.benchmark_root)


def test_from_benchmark_root_rejects_partial_geometry_final_line(tmp_path):
    direct_state = _cache_state(tmp_path)
    prediction_path = direct_state.benchmark_root / "geometry_predictions.jsonl"
    prediction_path.write_text(
        json.dumps(_geometry_prediction("pair-0"), sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="geometry|partial|newline"):
        type(direct_state).from_benchmark_root(direct_state.benchmark_root)


def test_direct_construction_rejects_partial_geometry_final_line(tmp_path):
    initial_state = _cache_state(tmp_path)
    prediction_path = initial_state.benchmark_root / "geometry_predictions.jsonl"
    prediction_path.write_text(
        json.dumps(_geometry_prediction("pair-0"), sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="geometry|partial|newline"):
        type(initial_state)(
            benchmark_root=initial_state.benchmark_root,
            pairs=initial_state.pairs,
            manifest=initial_state.manifest,
            token_secret=b"secret",
        )


def test_valid_direct_construction_matches_factory_public_state(tmp_path, monkeypatch):
    initial_state = _cache_state(tmp_path)
    _write_pairs(
        initial_state.benchmark_root / "geometry_predictions.jsonl",
        [_geometry_prediction("pair-0")],
    )
    monkeypatch.setattr(
        "robust_loop_verification_scripts.annotate_rover_aligned_benchmark.secrets.token_bytes",
        lambda _length: b"secret",
    )

    direct_state = type(initial_state)(
        benchmark_root=initial_state.benchmark_root,
        pairs=initial_state.pairs,
        manifest=initial_state.manifest,
        token_secret=b"secret",
    )
    factory_state = type(initial_state).from_benchmark_root(initial_state.benchmark_root)

    assert direct_state.geometry_by_pair_id == factory_state.geometry_by_pair_id
    assert direct_state.public_state() == factory_state.public_state()


def test_explicit_geometry_cannot_bypass_invalid_disk_geometry(tmp_path):
    initial_state = _cache_state(tmp_path)
    (initial_state.benchmark_root / "geometry_predictions.jsonl").write_text(
        "{invalid json\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="geometry.*invalid JSON"):
        type(initial_state)(
            benchmark_root=initial_state.benchmark_root,
            pairs=initial_state.pairs,
            manifest=initial_state.manifest,
            token_secret=b"secret",
            geometry_by_pair_id={"pair-0": _geometry_prediction("pair-0")},
        )


def test_explicit_geometry_must_match_valid_disk_geometry(tmp_path):
    initial_state = _cache_state(tmp_path)
    _write_pairs(
        initial_state.benchmark_root / "geometry_predictions.jsonl",
        [_geometry_prediction("pair-0")],
    )
    explicit_prediction = _geometry_prediction("pair-0", automatic_label=0)

    with pytest.raises(ValueError, match="geometry.*match|mismatch"):
        type(initial_state)(
            benchmark_root=initial_state.benchmark_root,
            pairs=initial_state.pairs,
            manifest=initial_state.manifest,
            token_secret=b"secret",
            geometry_by_pair_id={"pair-0": explicit_prediction},
        )


def test_canonically_equal_explicit_and_disk_geometry_are_accepted(tmp_path):
    initial_state = _cache_state(tmp_path, pair_count=2)
    predictions = [
        _geometry_prediction("pair-0"),
        _geometry_prediction("pair-1", automatic_label=0),
    ]
    _write_pairs(
        initial_state.benchmark_root / "geometry_predictions.jsonl",
        predictions,
    )
    explicit_pair_0 = dict(predictions[0])
    explicit_pair_1 = dict(predictions[1])
    explicit_pair_0.pop("pair_id")
    explicit_pair_1.pop("pair_id")

    state = type(initial_state)(
        benchmark_root=initial_state.benchmark_root,
        pairs=initial_state.pairs,
        manifest=initial_state.manifest,
        token_secret=b"secret",
        geometry_by_pair_id={
            "pair-1": explicit_pair_1,
            "pair-0": explicit_pair_0,
        },
    )

    assert state.geometry_by_pair_id == {
        "pair-0": predictions[0],
        "pair-1": predictions[1],
    }


@pytest.mark.parametrize(
    "geometry_by_pair_id",
    [
        {"pair-0": _geometry_prediction("pair-0")},
        {
            "pair-0": _geometry_prediction("pair-0"),
            "pair-1": _geometry_prediction("pair-1"),
            "unknown": _geometry_prediction("unknown"),
        },
    ],
)
def test_direct_construction_rejects_explicit_geometry_without_exact_coverage(
    tmp_path, geometry_by_pair_id
):
    initial_state = _cache_state(tmp_path, pair_count=2)

    with pytest.raises(ValueError, match="geometry|coverage|pair_id"):
        type(initial_state)(
            benchmark_root=initial_state.benchmark_root,
            pairs=initial_state.pairs,
            manifest=initial_state.manifest,
            token_secret=b"secret",
            geometry_by_pair_id=geometry_by_pair_id,
        )


def test_direct_construction_canonicalizes_explicit_geometry_in_pair_order(tmp_path):
    initial_state = _cache_state(tmp_path, pair_count=2)
    pair_0 = _geometry_prediction("pair-0")
    pair_1 = _geometry_prediction("pair-1")
    pair_1.pop("pair_id")

    state = type(initial_state)(
        benchmark_root=initial_state.benchmark_root,
        pairs=initial_state.pairs,
        manifest=initial_state.manifest,
        token_secret=b"secret",
        geometry_by_pair_id={"pair-1": pair_1, "pair-0": pair_0},
    )

    assert list(state.geometry_by_pair_id) == ["pair-0", "pair-1"]
    assert state.geometry_by_pair_id["pair-1"]["pair_id"] == "pair-1"


def test_direct_construction_rejects_explicit_geometry_row_pair_id_mismatch(tmp_path):
    initial_state = _cache_state(tmp_path)

    with pytest.raises(ValueError, match="geometry|pair_id"):
        type(initial_state)(
            benchmark_root=initial_state.benchmark_root,
            pairs=initial_state.pairs,
            manifest=initial_state.manifest,
            token_secret=b"secret",
            geometry_by_pair_id={"pair-0": _geometry_prediction("other")},
        )


def test_direct_construction_rejects_explicit_null_geometry_row_pair_id(tmp_path):
    initial_state = _cache_state(tmp_path)
    prediction = _geometry_prediction("pair-0")
    prediction["pair_id"] = None

    with pytest.raises(ValueError, match="geometry|pair_id"):
        type(initial_state)(
            benchmark_root=initial_state.benchmark_root,
            pairs=initial_state.pairs,
            manifest=initial_state.manifest,
            token_secret=b"secret",
            geometry_by_pair_id={"pair-0": prediction},
        )


def test_direct_construction_rejects_disk_geometry_row_without_pair_id(tmp_path):
    initial_state = _cache_state(tmp_path)
    prediction = _geometry_prediction("pair-0")
    prediction.pop("pair_id")
    _write_pairs(
        initial_state.benchmark_root / "geometry_predictions.jsonl",
        [prediction],
    )

    with pytest.raises(ValueError, match="geometry|pair_id"):
        type(initial_state)(
            benchmark_root=initial_state.benchmark_root,
            pairs=initial_state.pairs,
            manifest=initial_state.manifest,
            token_secret=b"secret",
        )


@pytest.mark.parametrize("automatic_label", [True, False, 1.0, 0.0])
@pytest.mark.parametrize("source", ["disk", "mapping"])
def test_geometry_contract_rejects_non_exact_integer_automatic_labels(
    tmp_path, automatic_label, source
):
    initial_state = _cache_state(tmp_path)
    prediction = _geometry_prediction("pair-0", automatic_label=automatic_label)
    kwargs = {}
    if source == "disk":
        _write_pairs(
            initial_state.benchmark_root / "geometry_predictions.jsonl",
            [prediction],
        )
    else:
        kwargs["geometry_by_pair_id"] = {"pair-0": prediction}

    with pytest.raises(ValueError, match="automatic_label.*exact int 0 or 1"):
        type(initial_state)(
            benchmark_root=initial_state.benchmark_root,
            pairs=initial_state.pairs,
            manifest=initial_state.manifest,
            token_secret=b"secret",
            **kwargs,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("support_idx", True),
        ("factor_status", 1),
        ("gt_rotation_angle_deg", "34.1"),
        ("gt_rotation_axis", [0.0, 1.0]),
        ("estimated_translation", [0.0, float("nan"), 1.0]),
        ("automatic_label_reason", None),
    ],
)
def test_direct_construction_rejects_invalid_explicit_geometry_field_types(tmp_path, field, value):
    initial_state = _cache_state(tmp_path)
    prediction = _geometry_prediction("pair-0", **{field: value})

    with pytest.raises(ValueError, match=f"geometry.*{field}|JSON-safe"):
        type(initial_state)(
            benchmark_root=initial_state.benchmark_root,
            pairs=initial_state.pairs,
            manifest=initial_state.manifest,
            token_secret=b"secret",
            geometry_by_pair_id={"pair-0": prediction},
        )


def test_is_sealed_surfaces_incomplete_or_corrupt_finalization_state(tmp_path):
    state = _cache_state(tmp_path)
    state.seal_path.write_text("{broken", encoding="utf-8")

    with pytest.raises(ValueError, match="incomplete|corrupt"):
        state.is_sealed()


def test_is_sealed_rejects_semantically_invalid_hashed_annotations(tmp_path):
    state = _cache_state(tmp_path)
    append_event(
        state.event_path,
        AnnotationEvent("e1", "label", "pair-0", 1, None, "t1"),
    )
    state.finalize()
    rows = _read_jsonl_rows(state.output_path)
    rows[0]["label"] = True
    _rewrite_annotations_and_seal_hash(state.output_path, state.seal_path, rows)

    with pytest.raises(ValueError, match="corrupt.*label"):
        state.is_sealed()


@pytest.mark.parametrize(
    ("first_timestamp", "second_timestamp"),
    [("z-later", "a-earlier"), ("same", "same")],
)
def test_undo_targets_latest_active_label_by_log_order(
    tmp_path, first_timestamp, second_timestamp
):
    state = _cache_state(tmp_path, pair_count=2)
    append_event(
        state.event_path,
        AnnotationEvent("e1", "label", "pair-0", 1, None, first_timestamp),
    )
    append_event(
        state.event_path,
        AnnotationEvent("e2", "label", "pair-1", 0, None, second_timestamp),
    )

    state.append_undo()

    assert replay_events(state.event_path) == {"pair-0": 1}


def test_server_resolves_pair_images_from_manifest_sequence_cache(tmp_path):
    state = _cache_state(tmp_path)
    token = opaque_image_tokens(state.pairs[0], b"secret")["query"]

    assert state.image_path_for_token(token) == tmp_path / "cache" / "images" / "query.png"


def test_http_state_redacts_pair_metadata_and_returns_opaque_image_urls(tmp_path):
    state = _cache_state(tmp_path)

    status, payload = _HandlerClient(state).get_json("/api/state")

    assert status == 200
    serialized = json.dumps(payload)
    assert "rank" not in payload
    assert "dbow2_score" not in serialized
    assert "query_idx" not in serialized
    assert "candidate_idx" not in serialized
    assert payload["images"]["query"].startswith("/image/")
    assert "query.png" not in payload["images"]["query"]


def test_http_state_exposes_exact_geometry_review_fields_without_manual_labels(tmp_path):
    direct_state = _cache_state(tmp_path)
    prediction = _geometry_prediction(
        "pair-0",
        label=0,
        manual_label=1,
        annotations=[{"label": 1}],
        gt_relative_pose=[[1.0]],
    )
    _write_pairs(
        direct_state.benchmark_root / "geometry_predictions.jsonl",
        [prediction],
    )
    state = type(direct_state).from_benchmark_root(direct_state.benchmark_root)

    status, payload = _HandlerClient(state).get_json("/api/state")

    assert status == 200
    assert payload["geometry"] == {
        "automatic_label": 1,
        "support_idx": 28,
        "factor_status": "ok",
        "gt_rotation_angle_deg": 34.1,
        "gt_rotation_axis": [0.0, 0.0, 1.0],
        "gt_translation": [1.2, -0.1, 0.0],
        "gt_translation_norm_m": 1.204,
        "estimated_rotation_angle_deg": 31.2,
        "estimated_rotation_axis": [0.0, 0.0, 1.0],
        "estimated_translation": [1.1, -0.2, 0.0],
        "estimated_translation_norm_m": 1.118,
        "translation_error_m": 0.14,
        "effective_translation_error_threshold_m": 1.0,
        "rotation_error_deg": 3.2,
        "translation_direction_error_deg": 4.8,
        "automatic_label_reason": "accepted",
    }
    serialized_geometry = json.dumps(payload["geometry"])
    assert "manual_label" not in serialized_geometry
    assert "annotations" not in serialized_geometry
    assert '"label"' not in serialized_geometry
    assert "gt_relative_pose" not in serialized_geometry


def test_old_benchmark_root_without_geometry_predictions_keeps_legacy_state(tmp_path):
    direct_state = _cache_state(tmp_path)

    state = type(direct_state).from_benchmark_root(direct_state.benchmark_root)
    payload = state.public_state()

    assert "geometry" not in payload
    assert "automatic_label" not in json.dumps(payload)


def test_http_label_appends_exactly_one_event_for_valid_state(tmp_path):
    state = _cache_state(tmp_path)
    client = _HandlerClient(state)

    _, payload = client.get_json("/api/state")
    status, _ = client.post_json(
        "/api/label",
        {
            "pair_token": payload["pair_token"],
            "state_nonce": payload["state_nonce"],
            "label": 1,
        },
    )

    assert status == 200
    assert _event_count(state.event_path) == 1
    assert replay_events(state.event_path) == {"pair-0": 1}


@pytest.mark.parametrize("label", [False, True, 0.0, 1.0, "0", "1", None])
def test_http_label_rejects_non_exact_integer_labels(tmp_path, label):
    state = _cache_state(tmp_path)
    client = _HandlerClient(state)
    _, payload = client.get_json("/api/state")

    status, response = client.post_json(
        "/api/label",
        {
            "pair_token": payload["pair_token"],
            "state_nonce": payload["state_nonce"],
            "label": label,
        },
    )

    assert status == 400
    assert response == {"error": "label must be 0 or 1"}
    assert _event_count(state.event_path) == 0


def test_http_replaying_same_label_payload_returns_409_without_second_append(tmp_path):
    state = _cache_state(tmp_path)
    client = _HandlerClient(state)
    _, payload = client.get_json("/api/state")
    label_payload = {
        "pair_token": payload["pair_token"],
        "state_nonce": payload["state_nonce"],
        "label": 1,
    }

    first_status, _ = client.post_json("/api/label", label_payload)
    second_status, _ = client.post_json("/api/label", label_payload)

    assert first_status == 200
    assert second_status == 409
    assert _event_count(state.event_path) == 1
    assert replay_events(state.event_path) == {"pair-0": 1}


@pytest.mark.parametrize("path", ["/api/label", "/api/undo", "/api/finalize"])
@pytest.mark.parametrize(
    ("body", "content_length"),
    [
        (b"{", None),
        (b"[]", None),
        (b"{}", "invalid"),
    ],
)
def test_http_bad_json_requests_return_400_without_appending(tmp_path, path, body, content_length):
    state = _cache_state(tmp_path)
    client = _HandlerClient(state)

    status, payload = client.post_raw(path, body, content_length=content_length)

    assert status == 400
    assert "JSON" in payload["error"] or "Content-Length" in payload["error"]
    assert _event_count(state.event_path) == 0


def test_http_oversized_json_request_returns_structured_413_without_appending(tmp_path):
    state = _cache_state(tmp_path)
    client = _HandlerClient(state)
    body = b'{"padding":"' + (b"x" * (64 * 1024)) + b'"}'

    status, payload = client.post_raw("/api/label", body)

    assert status == 413
    assert payload == {"error": "JSON request body is too large"}
    assert _event_count(state.event_path) == 0


def test_http_short_json_body_returns_400_without_appending(tmp_path):
    state = _cache_state(tmp_path)
    client = _HandlerClient(state)
    body = b'{"label":1}'

    status, payload = client.post_raw(
        "/api/label",
        body,
        content_length=str(len(body) + 5),
    )

    assert status == 400
    assert payload == {"error": "JSON request body shorter than Content-Length"}
    assert _event_count(state.event_path) == 0


def test_http_stale_nonce_or_token_returns_409_without_appending(tmp_path):
    state = _cache_state(tmp_path)
    client = _HandlerClient(state)

    _, payload = client.get_json("/api/state")
    nonce_status, _ = client.post_json(
        "/api/label",
        {
            "pair_token": payload["pair_token"],
            "state_nonce": "stale",
            "label": 1,
        },
    )
    token_status, _ = client.post_json(
        "/api/label",
        {
            "pair_token": "wrong",
            "state_nonce": payload["state_nonce"],
            "label": 1,
        },
    )

    assert nonce_status == 409
    assert token_status == 409
    assert _event_count(state.event_path) == 0


def test_http_undo_allows_complete_state_nonce_before_finalize(tmp_path):
    state = _cache_state(tmp_path)
    client = _HandlerClient(state)
    _, payload = client.get_json("/api/state")
    label_status, _ = client.post_json(
        "/api/label",
        {
            "pair_token": payload["pair_token"],
            "state_nonce": payload["state_nonce"],
            "label": 1,
        },
    )
    _, complete_payload = client.get_json("/api/state")

    undo_status, undo_payload = client.post_json(
        "/api/undo",
        {
            "pair_token": complete_payload["pair_token"],
            "state_nonce": complete_payload["state_nonce"],
        },
    )

    assert label_status == 200
    assert complete_payload["complete"] is True
    assert undo_status == 200
    assert undo_payload["complete"] is False
    assert _event_count(state.event_path) == 2
    assert replay_events(state.event_path) == {}


def test_http_label_rejects_complete_state_nonce(tmp_path):
    state = _cache_state(tmp_path)
    client = _HandlerClient(state)
    _, payload = client.get_json("/api/state")
    client.post_json(
        "/api/label",
        {
            "pair_token": payload["pair_token"],
            "state_nonce": payload["state_nonce"],
            "label": 1,
        },
    )
    _, complete_payload = client.get_json("/api/state")

    status, _ = client.post_json(
        "/api/label",
        {
            "pair_token": complete_payload["pair_token"],
            "state_nonce": complete_payload["state_nonce"],
            "label": 0,
        },
    )

    assert status == 409
    assert _event_count(state.event_path) == 1
    assert replay_events(state.event_path) == {"pair-0": 1}


@pytest.mark.parametrize("finalize_payload", [{}, {"state_nonce": "stale"}])
def test_http_finalize_requires_current_complete_state_nonce(tmp_path, finalize_payload):
    state = _cache_state(tmp_path)
    client = _HandlerClient(state)
    _, payload = client.get_json("/api/state")
    client.post_json(
        "/api/label",
        {
            "pair_token": payload["pair_token"],
            "state_nonce": payload["state_nonce"],
            "label": 1,
        },
    )

    status, _ = client.post_json("/api/finalize", finalize_payload)

    assert status == 409
    assert not state.output_path.exists()
    assert not state.seal_path.exists()


@pytest.mark.parametrize("path", ["/api/label", "/api/undo", "/api/finalize"])
def test_http_sealed_state_returns_409_for_mutating_requests(tmp_path, path):
    state = _cache_state(tmp_path)
    append_event(
        state.event_path,
        AnnotationEvent("e1", "label", "pair-0", 1, None, "t1"),
    )
    state.finalize()
    client = _HandlerClient(state)

    _, payload = client.get_json("/api/state")
    status, _ = client.post_json(
        path,
        {
            "pair_token": payload.get("pair_token"),
            "state_nonce": payload.get("state_nonce"),
            "label": 1,
        },
    )

    assert status == 409
    assert _event_count(state.event_path) == 1
    assert replay_events(state.event_path) == {"pair-0": 1}


def test_http_image_endpoint_serves_existing_cache_root_image(tmp_path):
    state = _cache_state(tmp_path)
    client = _HandlerClient(state)

    _, payload = client.get_json("/api/state")
    status, body = client.get_bytes(payload["images"]["query"])

    assert status == 200
    assert body == b"image:query.png"


def test_http_image_endpoint_serves_cache_root_symlinked_image(tmp_path):
    state = _cache_state(tmp_path)
    outside = tmp_path / "raw_images"
    outside.mkdir()
    target = outside / "query.png"
    target.write_bytes(b"raw-query-image")
    cache_image = tmp_path / "cache" / "images" / "query.png"
    cache_image.unlink()
    cache_image.symlink_to(target)
    client = _HandlerClient(state)

    _, payload = client.get_json("/api/state")
    status, body = client.get_bytes(payload["images"]["query"])

    assert status == 200
    assert body == b"raw-query-image"


@pytest.mark.parametrize(
    "image_path",
    [
        "../outside.png",
        "/tmp/outside.png",
    ],
)
def test_server_rejects_image_context_path_traversal(tmp_path, image_path):
    state = _cache_state(tmp_path)
    state.pairs[0]["query_context"] = (None, image_path, None)
    token = opaque_image_tokens(state.pairs[0], b"secret")["query"]

    with pytest.raises(ValueError, match="escapes sequence cache root"):
        state.image_path_for_token(token)


def test_http_finalize_writes_annotations_and_seal_at_complete_coverage(tmp_path):
    state = _cache_state(tmp_path)
    client = _HandlerClient(state)
    _, payload = client.get_json("/api/state")
    client.post_json(
        "/api/label",
        {
            "pair_token": payload["pair_token"],
            "state_nonce": payload["state_nonce"],
            "label": 1,
        },
    )

    _, complete_payload = client.get_json("/api/state")
    status, payload = client.post_json(
        "/api/finalize",
        {"state_nonce": complete_payload["state_nonce"]},
    )

    assert status == 200
    assert payload["finalized"] is True
    assert payload["state"]["sealed"] is True
    assert state.output_path.exists()
    assert state.seal_path.exists()
    verify_annotation_seal(
        state.manifest_path,
        state.benchmark_root / "benchmark_pairs.jsonl",
        state.output_path,
        state.seal_path,
    )


def test_public_state_reports_unsealed_and_sealed_states(tmp_path):
    state = _cache_state(tmp_path)

    assert state.public_state()["sealed"] is False
    append_event(
        state.event_path,
        AnnotationEvent("e1", "label", "pair-0", 1, None, "t1"),
    )
    state.finalize()

    sealed_state = state.public_state()
    assert sealed_state["sealed"] is True
    assert sealed_state["complete"] is True


def test_html_complete_state_keeps_undo_available_and_sends_finalize_nonce():
    annotator = pytest.importorskip(
        "robust_loop_verification_scripts.annotate_rover_aligned_benchmark"
    )

    assert "data-label-button" in annotator.HTML_PAGE
    assert "button.disabled = requestActive || complete || sealed;" in annotator.HTML_PAGE
    assert "stateGuardedPayload({ allowComplete: true })" in annotator.HTML_PAGE
    assert (
        "postJSON('/api/finalize', { state_nonce: currentState.state_nonce });"
        in annotator.HTML_PAGE
    )


def test_html_sealed_state_disables_mutations_and_shows_finalized_message():
    annotator = pytest.importorskip(
        "robust_loop_verification_scripts.annotate_rover_aligned_benchmark"
    )

    assert "currentState.sealed" in annotator.HTML_PAGE
    assert "state.sealed ? 'Annotations finalized.'" in annotator.HTML_PAGE
    assert "requestActive || currentState.sealed" in annotator.HTML_PAGE


def test_html_keeps_images_dominant_and_renders_plain_compact_geometry_card():
    annotator = pytest.importorskip(
        "robust_loop_verification_scripts.annotate_rover_aligned_benchmark"
    )
    html = annotator.HTML_PAGE

    assert html.index('class="grid"') < html.index('id="geometry-card"')
    assert 'id="query-track"' in html
    assert 'id="candidate-track"' in html
    assert ".geometry-card" in html
    assert "DA3 AUTO LABEL:" in html
    assert "POSITIVE" in html
    assert "NEGATIVE" in html
    for label in [
        "GT rotation",
        "GT translation",
        "Estimated rotation",
        "Estimated translation",
        "Rotation error",
        "Metric translation error",
        "Translation-direction error",
        "Support index",
        "Factor status",
        "Automatic reason",
    ]:
        assert label in html
    assert "support + DA3 + Sim3" in html
    assert "not the final human label" in html
    assert "N/A" in html
    assert "MathJax" not in html
    assert "katex" not in html.lower()
    assert "\\(" not in html
    assert "\\[" not in html


def test_html_keyboard_accepts_auto_label_and_preserves_overrides_undo_and_request_lock():
    annotator = pytest.importorskip(
        "robust_loop_verification_scripts.annotate_rover_aligned_benchmark"
    )
    html = annotator.HTML_PAGE

    assert "function acceptAutomaticLabel()" in html
    assert "currentState.geometry" in html
    assert "Number.isInteger(geometry.automatic_label)" in html
    assert "return submitLabel(geometry.automatic_label);" in html
    assert "event.key === ' ' || event.key === 'Enter'" in html
    accept_branch = html.split("if (event.key === ' ' || event.key === 'Enter')", 1)[1]
    assert "if (event.target instanceof HTMLButtonElement) return;" in accept_branch
    assert "if (acceptAutomaticLabel()) event.preventDefault();" in accept_branch
    assert "event.key.toLowerCase() === 'p'" in html
    assert "event.key.toLowerCase() === 'n'" in html
    assert "event.key === 'Backspace'" in html
    assert "if (requestActive) return;" in html
    assert "if (!payload || requestActive) return;" in html
    assert "setBusy(true);" in html
    assert "setBusy(false);" in html


def test_html_keydown_filters_repeats_before_all_annotation_shortcuts():
    annotator = pytest.importorskip(
        "robust_loop_verification_scripts.annotate_rover_aligned_benchmark"
    )
    handler = annotator.HTML_PAGE.split("document.addEventListener('keydown', event => {", 1)[
        1
    ].split("});", 1)[0]

    repeat_guard = handler.index("if (event.repeat) return;")
    assert repeat_guard < handler.index("if (requestActive) return;")
    assert repeat_guard < handler.index("event.key === ' '")
    assert repeat_guard < handler.index("event.key.toLowerCase() === 'p'")
    assert repeat_guard < handler.index("event.key.toLowerCase() === 'n'")
    assert repeat_guard < handler.index("event.key === 'Backspace'")


def test_html_automatic_label_acceptance_returns_whether_submission_started():
    annotator = pytest.importorskip(
        "robust_loop_verification_scripts.annotate_rover_aligned_benchmark"
    )
    html = annotator.HTML_PAGE
    submit_body = html.split("function submitLabel(label) {", 1)[1].split("\n}", 1)[0]
    accept_body = html.split("function acceptAutomaticLabel() {", 1)[1].split("\n}", 1)[0]

    assert "if (!payload || requestActive) return false;" in submit_body
    assert "postJSON('/api/label', payload);" in submit_body
    assert submit_body.index("postJSON('/api/label', payload);") < submit_body.index(
        "return true;"
    )
    assert "currentState.sealed || currentState.complete || requestActive" in accept_body
    assert "return false;" in accept_body
    assert "Number.isInteger(geometry.automatic_label)" in accept_body
    assert "return submitLabel(geometry.automatic_label);" in accept_body
