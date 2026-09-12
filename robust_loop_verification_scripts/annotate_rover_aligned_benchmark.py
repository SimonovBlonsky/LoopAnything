#!/usr/bin/env python3
"""Serve a local binary annotation UI for the frozen ROVER-aligned benchmark."""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import secrets
import sys
import uuid
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from json import JSONDecodeError
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlparse

LOOPANYTHING_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = LOOPANYTHING_ROOT / "src"
for import_path in (LOOPANYTHING_ROOT, SRC_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from robust_loop_verifier.io import read_json, read_jsonl
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

MAX_JSON_BODY_BYTES = 64 * 1024
GEOMETRY_PUBLIC_FIELDS = (
    "automatic_label",
    "support_idx",
    "factor_status",
    "gt_rotation_angle_deg",
    "gt_rotation_axis",
    "gt_translation",
    "gt_translation_norm_m",
    "estimated_rotation_angle_deg",
    "estimated_rotation_axis",
    "estimated_translation",
    "estimated_translation_norm_m",
    "translation_error_m",
    "effective_translation_error_threshold_m",
    "rotation_error_deg",
    "translation_direction_error_deg",
    "automatic_label_reason",
)
GEOMETRY_NUMERIC_FIELDS = (
    "gt_rotation_angle_deg",
    "gt_translation_norm_m",
    "estimated_rotation_angle_deg",
    "estimated_translation_norm_m",
    "translation_error_m",
    "effective_translation_error_threshold_m",
    "rotation_error_deg",
    "translation_direction_error_deg",
)
GEOMETRY_VECTOR_FIELDS = (
    "gt_rotation_axis",
    "gt_translation",
    "estimated_rotation_axis",
    "estimated_translation",
)


class RequestBodyTooLarge(ValueError):
    pass


@dataclass
class AnnotationServerState:
    benchmark_root: Path
    pairs: Sequence[Mapping[str, object]]
    manifest: Mapping[str, object]
    token_secret: bytes
    geometry_by_pair_id: Mapping[str, Mapping[str, object]] | None = None

    def __post_init__(self) -> None:
        self.benchmark_root = Path(self.benchmark_root)
        self.event_path = self.benchmark_root / "annotation_events.jsonl"
        self.output_path = self.benchmark_root / "annotations.jsonl"
        self.manifest_path = self.benchmark_root / "manifest.json"
        self.seal_path = self.benchmark_root / "annotation_seal.json"
        self.pairs_by_id = {str(pair["pair_id"]): pair for pair in self.pairs}
        if len(self.pairs_by_id) != len(self.pairs):
            raise ValueError("duplicate pair_id in benchmark pairs")
        geometry_path = self.benchmark_root / "geometry_predictions.jsonl"
        if geometry_path.exists():
            disk_geometry = _load_geometry_predictions(geometry_path, self.pairs)
            if self.geometry_by_pair_id is not None:
                explicit_geometry = _validate_geometry_mapping(
                    self.geometry_by_pair_id,
                    self.pairs,
                )
                if explicit_geometry != disk_geometry:
                    raise ValueError(
                        "explicit geometry_by_pair_id does not match " "geometry_predictions.jsonl"
                    )
            self.geometry_by_pair_id = disk_geometry
        elif self.geometry_by_pair_id is not None:
            self.geometry_by_pair_id = _validate_geometry_mapping(
                self.geometry_by_pair_id,
                self.pairs,
            )
        else:
            self.geometry_by_pair_id = {}
        self.sequence_cache_roots = self._sequence_cache_roots()
        seed = int(self.manifest.get("annotation_shuffle_seed", 20260610))
        self.display_order = deterministic_display_order(self.pairs_by_id, seed=seed)

    @classmethod
    def from_benchmark_root(cls, benchmark_root: Path) -> "AnnotationServerState":
        benchmark_root = Path(benchmark_root)
        pairs = list(read_jsonl(benchmark_root / "benchmark_pairs.jsonl"))
        manifest = read_json(benchmark_root / "manifest.json")
        return cls(
            benchmark_root=benchmark_root,
            pairs=pairs,
            manifest=manifest,
            token_secret=secrets.token_bytes(32),
        )

    def public_state(self) -> dict[str, object]:
        sealed = self.is_sealed()
        resolved = replay_events(self.event_path)
        progress = annotation_progress(self.display_order, resolved)
        pair_id = self.current_pair_id(resolved)
        if pair_id is None:
            return {
                "complete": True,
                "sealed": sealed,
                "pair_token": None,
                "state_nonce": self._state_nonce(None, resolved),
                "images": {},
                "progress": progress,
            }
        pair = self.pairs_by_id[pair_id]
        tokens = opaque_image_tokens(pair, self.token_secret)
        state: dict[str, object] = {
            "complete": False,
            "sealed": sealed,
            "pair_token": self._pair_token(pair_id),
            "state_nonce": self._state_nonce(pair_id, resolved),
            "images": {
                slot: None if token is None else f"/image/{token}"
                for slot, token in tokens.items()
            },
            "progress": progress,
        }
        geometry = self.geometry_by_pair_id.get(pair_id)
        if geometry is not None:
            state["geometry"] = {field: geometry.get(field) for field in GEOMETRY_PUBLIC_FIELDS}
        return state

    def current_pair_id(self, resolved: Mapping[str, int] | None = None) -> str | None:
        resolved = replay_events(self.event_path) if resolved is None else resolved
        for pair_id in self.display_order:
            if pair_id not in resolved:
                return pair_id
        return None

    def validate_request(self, pair_token: str, state_nonce: str) -> str:
        resolved = replay_events(self.event_path)
        pair_id = self.current_pair_id(resolved)
        if pair_id is None:
            raise ValueError("annotation is already complete")
        if pair_token != self._pair_token(pair_id):
            raise ValueError("pair token does not match current state")
        if state_nonce != self._state_nonce(pair_id, resolved):
            raise ValueError("state nonce does not match current state")
        return pair_id

    def validate_undo_request(self, pair_token: object, state_nonce: object) -> None:
        resolved = replay_events(self.event_path)
        pair_id = self.current_pair_id(resolved)
        if str(state_nonce) != self._state_nonce(pair_id, resolved):
            raise ValueError("state nonce does not match current state")
        if pair_id is None:
            if pair_token not in (None, ""):
                raise ValueError("pair token does not match current state")
            return
        if str(pair_token) != self._pair_token(pair_id):
            raise ValueError("pair token does not match current state")

    def validate_finalize_request(self, state_nonce: object) -> None:
        resolved = replay_events(self.event_path)
        pair_id = self.current_pair_id(resolved)
        if pair_id is not None:
            raise ValueError("annotation is not complete")
        if not isinstance(state_nonce, str) or state_nonce != self._state_nonce(None, resolved):
            raise ValueError("state nonce does not match current state")

    def image_path_for_token(self, token: str) -> Path | None:
        for pair in self.pairs:
            tokens = opaque_image_tokens(pair, self.token_secret)
            context = valid_image_context(pair)
            for slot, candidate_token in tokens.items():
                if candidate_token == token and context[slot] is not None:
                    return self._safe_image_path(pair, str(context[slot]))
        return None

    def append_label(self, pair_id: str, label: int) -> None:
        append_event(
            self.event_path,
            AnnotationEvent(
                event_id=uuid.uuid4().hex,
                action="label",
                pair_id=pair_id,
                label=label,
                target_event_id=None,
                annotated_at=datetime.now(timezone.utc).isoformat(),
            ),
        )

    def append_undo(self) -> None:
        active_events = replay_resolved_label_events(self.event_path)
        if not active_events:
            raise ValueError("no annotation event to undo")
        target_event = list(active_events.values())[-1]
        append_event(
            self.event_path,
            AnnotationEvent(
                event_id=uuid.uuid4().hex,
                action="undo",
                pair_id=target_event.pair_id,
                label=None,
                target_event_id=target_event.event_id,
                annotated_at=datetime.now(timezone.utc).isoformat(),
            ),
        )

    def finalize(self) -> None:
        finalize_annotations(
            pairs=self.pairs,
            event_path=self.event_path,
            output_path=self.output_path,
            manifest_path=self.manifest_path,
            seal_path=self.seal_path,
        )

    def is_sealed(self) -> bool:
        output_exists = self.output_path.exists()
        seal_exists = self.seal_path.exists()
        if not output_exists and not seal_exists:
            return False
        if output_exists != seal_exists:
            raise ValueError("incomplete annotation finalization state")
        try:
            verify_annotation_seal(
                self.manifest_path,
                self.benchmark_root / "benchmark_pairs.jsonl",
                self.output_path,
                self.seal_path,
            )
        except (OSError, ValueError, JSONDecodeError) as exc:
            raise ValueError(f"corrupt annotation finalization state: {exc}") from exc
        return True

    def _pair_token(self, pair_id: str) -> str:
        return _digest(self.token_secret, f"pair\0{pair_id}")

    def _state_nonce(self, pair_id: str | None, resolved: Mapping[str, int]) -> str:
        event_size = self.event_path.stat().st_size if self.event_path.exists() else 0
        payload = json.dumps(
            {
                "pair_id": pair_id,
                "event_size": event_size,
                "resolved": sorted((str(key), int(value)) for key, value in resolved.items()),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return _digest(self.token_secret, f"state\0{payload}")

    def _sequence_cache_roots(self) -> dict[tuple[str, str, str], Path]:
        rows = self.manifest.get("sequences")
        if not isinstance(rows, list):
            raise ValueError("manifest must contain sequences list")
        cache_roots: dict[tuple[str, str, str], Path] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("manifest sequence rows must be mappings")
            key = (str(row["dataset"]), str(row["platform"]), str(row["sequence"]))
            if key in cache_roots:
                raise ValueError(f"duplicate manifest sequence cache root: {key}")
            cache_roots[key] = Path(str(row["cache"])).resolve()
        return cache_roots

    def _safe_image_path(self, pair: Mapping[str, object], relative_path: str) -> Path:
        key = (str(pair["dataset"]), str(pair["platform"]), str(pair["sequence"]))
        if key not in self.sequence_cache_roots:
            raise ValueError(f"missing cache root for sequence: {key}")
        root = self.sequence_cache_roots[key]
        image_relative_path = Path(relative_path)
        if image_relative_path.is_absolute() or ".." in image_relative_path.parts:
            raise ValueError("image path escapes sequence cache root")
        image_path = root / image_relative_path
        return image_path


def build_handler(state: AnnotationServerState) -> type[BaseHTTPRequestHandler]:
    class AnnotationHandler(BaseHTTPRequestHandler):
        server_version = "RoverAnnotationHTTP/1.0"

        def do_GET(self) -> None:  # noqa: N802
            route = urlparse(self.path).path
            if route == "/":
                self._send_html(HTML_PAGE)
                return
            if route == "/api/state":
                self._send_json(HTTPStatus.OK, state.public_state())
                return
            if route.startswith("/image/"):
                self._send_image(unquote(route.removeprefix("/image/")))
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            route = urlparse(self.path).path
            if route in {"/api/label", "/api/undo", "/api/finalize"}:
                try:
                    sealed = state.is_sealed()
                except ValueError as exc:
                    self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
                    return
                if sealed:
                    self._send_json(HTTPStatus.CONFLICT, {"error": "annotation output is sealed"})
                    return
            if route == "/api/label":
                self._handle_label()
                return
            if route == "/api/undo":
                self._handle_undo()
                return
            if route == "/api/finalize":
                self._handle_finalize()
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def _handle_label(self) -> None:
            payload = self._read_json_body_or_400()
            if payload is None:
                return
            label = payload.get("label")
            if type(label) is not int or label not in (0, 1):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "label must be 0 or 1"})
                return
            pair_id = self._validate_state_payload(payload)
            if pair_id is None:
                return
            state.append_label(pair_id, label)
            self._send_json(HTTPStatus.OK, state.public_state())

        def _handle_undo(self) -> None:
            payload = self._read_json_body_or_400()
            if payload is None:
                return
            if not self._validate_undo_payload(payload):
                return
            try:
                state.append_undo()
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._send_json(HTTPStatus.OK, state.public_state())

        def _handle_finalize(self) -> None:
            payload = self._read_json_body_or_400()
            if payload is None:
                return
            try:
                state.validate_finalize_request(payload.get("state_nonce"))
            except ValueError as exc:
                self._send_json(
                    HTTPStatus.CONFLICT,
                    {"error": str(exc), "state": state.public_state()},
                )
                return
            try:
                state.finalize()
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            except FileExistsError as exc:
                self._send_json(HTTPStatus.CONFLICT, {"error": str(exc)})
                return
            self._send_json(HTTPStatus.OK, {"finalized": True, "state": state.public_state()})

        def _validate_state_payload(self, payload: Mapping[str, object]) -> str | None:
            try:
                return state.validate_request(
                    str(payload.get("pair_token", "")),
                    str(payload.get("state_nonce", "")),
                )
            except ValueError as exc:
                self._send_json(
                    HTTPStatus.CONFLICT,
                    {"error": str(exc), "state": state.public_state()},
                )
                return None

        def _validate_undo_payload(self, payload: Mapping[str, object]) -> bool:
            try:
                state.validate_undo_request(
                    payload.get("pair_token"),
                    payload.get("state_nonce"),
                )
            except ValueError as exc:
                self._send_json(
                    HTTPStatus.CONFLICT,
                    {"error": str(exc), "state": state.public_state()},
                )
                return False
            return True

        def _send_image(self, token: str) -> None:
            try:
                image_path = state.image_path_for_token(token)
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            if image_path is None or not image_path.is_file():
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "image not found"})
                return
            content_type = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
            data = image_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_json_body_or_400(self) -> Mapping[str, object] | None:
            try:
                return self._read_json_body()
            except RequestBodyTooLarge as exc:
                self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": str(exc)})
                return None
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return None

        def _read_json_body(self) -> Mapping[str, object]:
            raw_length = self.headers.get("Content-Length", "0")
            try:
                length = int(raw_length)
            except ValueError as exc:
                raise ValueError("invalid Content-Length header") from exc
            if length < 0:
                raise ValueError("invalid Content-Length header")
            if length > MAX_JSON_BODY_BYTES:
                raise RequestBodyTooLarge("JSON request body is too large")
            data = self.rfile.read(length) if length else b"{}"
            if len(data) != length:
                raise ValueError("JSON request body shorter than Content-Length")
            try:
                payload = json.loads(data.decode("utf-8"))
            except (JSONDecodeError, UnicodeDecodeError) as exc:
                raise ValueError("invalid JSON request body") from exc
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
            return payload

        def _send_html(self, html: str) -> None:
            data = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_json(self, status: HTTPStatus, payload: Mapping[str, object]) -> None:
            data = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: object) -> None:
            sys.stderr.write(
                "%s - - [%s] %s\n"
                % (self.client_address[0], self.log_date_time_string(), format % args)
            )

    return AnnotationHandler


def _digest(secret: bytes, message: str) -> str:
    import hashlib
    import hmac

    return hmac.new(secret, message.encode("utf-8"), hashlib.sha256).hexdigest()


def _load_geometry_predictions(
    path: Path,
    pairs: Sequence[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    if not path.exists():
        return {}
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        raise ValueError("geometry_predictions.jsonl has a partial final line")

    rows: list[Mapping[str, Any]] = []
    for line_number, raw_line in enumerate(data.splitlines(), start=1):
        if not raw_line.strip():
            raise ValueError(f"geometry_predictions.jsonl line {line_number} is blank")
        try:
            row = json.loads(raw_line)
        except (JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(
                f"geometry_predictions.jsonl line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(f"geometry_predictions.jsonl line {line_number} must be an object")
        rows.append(row)

    expected_pair_ids = [str(pair["pair_id"]) for pair in pairs]
    prediction_pair_ids = []
    for line_number, row in enumerate(rows, start=1):
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError(f"geometry_predictions.jsonl line {line_number} has invalid pair_id")
        prediction_pair_ids.append(pair_id)
    if len(set(prediction_pair_ids)) != len(prediction_pair_ids):
        raise ValueError("geometry_predictions.jsonl contains duplicate pair_id")
    if prediction_pair_ids != expected_pair_ids:
        raise ValueError(
            "geometry_predictions.jsonl pair_id order and coverage must exactly match "
            "benchmark_pairs.jsonl"
        )
    return {
        pair_id: _validate_geometry_row(row, pair_id, allow_missing_pair_id=False)
        for pair_id, row in zip(expected_pair_ids, rows)
    }


def _validate_geometry_mapping(
    geometry_by_pair_id: Mapping[str, Mapping[str, object]],
    pairs: Sequence[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    if not isinstance(geometry_by_pair_id, Mapping):
        raise ValueError("geometry_by_pair_id must be a mapping")
    expected_pair_ids = [str(pair["pair_id"]) for pair in pairs]
    supplied_pair_ids = list(geometry_by_pair_id)
    if any(not isinstance(pair_id, str) or not pair_id for pair_id in supplied_pair_ids):
        raise ValueError("geometry_by_pair_id keys must be non-empty pair_id strings")
    if set(supplied_pair_ids) != set(expected_pair_ids):
        raise ValueError("geometry_by_pair_id pair_id coverage must exactly match benchmark pairs")
    return {
        pair_id: _validate_geometry_row(
            geometry_by_pair_id[pair_id],
            pair_id,
            allow_missing_pair_id=True,
        )
        for pair_id in expected_pair_ids
    }


def _validate_geometry_row(
    row: Mapping[str, object],
    expected_pair_id: str,
    *,
    allow_missing_pair_id: bool,
) -> dict[str, object]:
    if not isinstance(row, Mapping):
        raise ValueError(f"geometry row for {expected_pair_id} must be a mapping")
    normalized = dict(row)
    if "pair_id" not in normalized and allow_missing_pair_id:
        normalized["pair_id"] = expected_pair_id
    elif normalized.get("pair_id") != expected_pair_id:
        raise ValueError(
            f"geometry row pair_id {normalized.get('pair_id')!r} "
            f"does not match {expected_pair_id!r}"
        )

    missing_fields = [field for field in GEOMETRY_PUBLIC_FIELDS if field not in normalized]
    if missing_fields:
        raise ValueError(
            f"geometry row {expected_pair_id} is missing required fields: {missing_fields}"
        )
    try:
        json.dumps(normalized, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"geometry row {expected_pair_id} is not JSON-safe") from exc

    automatic_label = normalized["automatic_label"]
    if type(automatic_label) is not int or automatic_label not in (0, 1):
        raise ValueError("geometry automatic_label must be exact int 0 or 1")

    support_idx = normalized["support_idx"]
    if support_idx is not None and type(support_idx) is not int:
        raise ValueError("geometry support_idx must be an integer or None")
    for field in ("factor_status", "automatic_label_reason"):
        value = normalized[field]
        if not isinstance(value, str) or not value:
            raise ValueError(f"geometry {field} must be a non-empty string")
    for field in GEOMETRY_NUMERIC_FIELDS:
        value = normalized[field]
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
            raise ValueError(f"geometry {field} must be a finite number or None")
    for field in GEOMETRY_VECTOR_FIELDS:
        value = normalized[field]
        if value is None:
            continue
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(f"geometry {field} must be a finite length-3 vector or None")
        if any(
            type(component) not in (int, float) or not math.isfinite(component)
            for component in value
        ):
            raise ValueError(f"geometry {field} must be a finite length-3 vector or None")
        normalized[field] = list(value)
    return normalized


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_root", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true")
    return parser


def run_server(benchmark_root: Path, host: str, port: int, open_browser: bool) -> None:
    state = AnnotationServerState.from_benchmark_root(benchmark_root)
    server = HTTPServer((host, port), build_handler(state))
    url = f"http://{host}:{port}/"
    if open_browser:
        webbrowser.open(url)
    print(f"Serving annotation UI at {url}", flush=True)
    server.serve_forever()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_server(args.benchmark_root, args.host, args.port, args.open)
    return 0


HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ROVER Binary Annotation</title>
  <style>
    :root { color-scheme: light; --ink: #1f2922; --paper: #f4efe3; --accent: #bf5a28; --positive: #236b4b; --negative: #8f332b; }
    body { margin: 0; font-family: Georgia, 'Times New Roman', serif; background: radial-gradient(circle at top left, #fff7de, var(--paper)); color: var(--ink); }
    main { max-width: 1180px; margin: 0 auto; padding: 28px; }
    header { display: flex; justify-content: space-between; align-items: baseline; gap: 18px; }
    h1 { font-size: clamp(28px, 4vw, 54px); margin: 0; letter-spacing: -0.04em; }
    #progress { font-size: 20px; color: var(--accent); }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 26px; margin-top: 28px; }
    .track { display: grid; grid-template-columns: 0.7fr 1.3fr 0.7fr; gap: 12px; align-items: center; }
    figure { margin: 0; padding: 10px; background: rgba(255,255,255,0.68); border: 1px solid rgba(31,41,34,0.15); border-radius: 18px; box-shadow: 0 18px 50px rgba(31,41,34,0.12); }
    figure.primary { padding: 14px; border-color: rgba(191,90,40,0.45); transform: translateY(-4px); }
    img { display: block; width: 100%; aspect-ratio: 4 / 3; object-fit: contain; background: #211f1b; border-radius: 12px; }
    figcaption { margin-top: 8px; text-align: center; font-size: 13px; text-transform: uppercase; letter-spacing: 0.08em; }
    .geometry-card { max-width: 860px; margin-top: 22px; padding: 14px 16px; background: rgba(255,255,255,0.72); border: 1px solid rgba(31,41,34,0.18); border-radius: 14px; box-shadow: 0 10px 28px rgba(31,41,34,0.08); }
    .geometry-card[hidden] { display: none; }
    .geometry-heading { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
    .geometry-heading p { margin: 0; font-size: 13px; color: #596159; }
    .auto-badge { display: inline-block; padding: 7px 11px; border-radius: 999px; color: white; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; font-weight: 800; letter-spacing: 0.04em; }
    .auto-badge.positive { background: var(--positive); }
    .auto-badge.negative { background: var(--negative); }
    .geometry-details { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px 18px; margin-top: 12px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
    .geometry-item { min-width: 0; }
    .geometry-item strong { display: block; margin-bottom: 2px; font-family: Georgia, 'Times New Roman', serif; font-size: 11px; letter-spacing: 0.04em; text-transform: uppercase; }
    .geometry-item span { overflow-wrap: anywhere; }
    .actions { display: flex; gap: 14px; flex-wrap: wrap; margin-top: 26px; }
    button { border: 0; border-radius: 999px; padding: 14px 22px; background: var(--ink); color: white; font-size: 17px; cursor: pointer; }
    button.positive { background: var(--positive); }
    button.negative { background: var(--negative); }
    button:disabled { opacity: 0.45; cursor: wait; }
    #message { margin-top: 16px; min-height: 24px; color: var(--accent); }
    @media (max-width: 760px) { .grid { grid-template-columns: 1fr; } .geometry-details { grid-template-columns: 1fr 1fr; } main { padding: 16px; } }
  </style>
</head>
<body>
<main>
  <header><h1>Loop Pair Annotation</h1><div id="progress">Loading...</div></header>
  <section class="grid">
    <div><h2>Query</h2><div class="track" id="query-track"></div></div>
    <div><h2>Candidate</h2><div class="track" id="candidate-track"></div></div>
  </section>
  <aside id="geometry-card" class="geometry-card" hidden>
    <div class="geometry-heading">
      <span id="auto-label-badge" class="auto-badge"></span>
      <p>Automated support + DA3 + Sim3 review aid; not the final human label. Space or Enter accepts it, while P/N overrides it.</p>
    </div>
    <div class="geometry-details">
      <div class="geometry-item"><strong>GT rotation</strong><span id="gt-rotation">N/A</span></div>
      <div class="geometry-item"><strong>GT translation</strong><span id="gt-translation">N/A</span></div>
      <div class="geometry-item"><strong>Estimated rotation</strong><span id="estimated-rotation">N/A</span></div>
      <div class="geometry-item"><strong>Estimated translation</strong><span id="estimated-translation">N/A</span></div>
      <div class="geometry-item"><strong>Rotation error</strong><span id="rotation-error">N/A</span></div>
      <div class="geometry-item"><strong>Metric translation error</strong><span id="translation-error">N/A</span></div>
      <div class="geometry-item"><strong>Translation error threshold</strong><span id="translation-error-threshold">N/A</span></div>
      <div class="geometry-item"><strong>Translation-direction error</strong><span id="translation-direction-error">N/A</span></div>
      <div class="geometry-item"><strong>Support index</strong><span id="support-index">N/A</span></div>
      <div class="geometry-item"><strong>Factor status</strong><span id="factor-status">N/A</span></div>
      <div class="geometry-item"><strong>Automatic reason</strong><span id="automatic-reason">N/A</span></div>
    </div>
  </aside>
  <div class="actions">
    <button data-label-button class="positive" onclick="submitLabel(1)">Positive override (P)</button>
    <button data-label-button class="negative" onclick="submitLabel(0)">Negative override (N)</button>
    <button id="undo-button" onclick="undoLast()">Undo (Backspace)</button>
    <button id="finalize-button" onclick="finalizeAnnotations()">Finalize</button>
  </div>
  <div id="message"></div>
</main>
<script>
let currentState = null;
let requestActive = false;
function updateButtonState() {
  const complete = Boolean(currentState && currentState.complete);
  const sealed = Boolean(currentState && currentState.sealed);
  document.querySelectorAll('[data-label-button]').forEach(button => {
    button.disabled = requestActive || complete || sealed;
  });
  document.getElementById('undo-button').disabled =
    requestActive || currentState.sealed || currentState.progress.completed === 0;
  document.getElementById('finalize-button').disabled =
    requestActive || currentState.sealed || !complete;
}
function setBusy(value) {
  requestActive = value;
  updateButtonState();
}
function figure(src, label, primary) {
  const el = document.createElement('figure');
  if (primary) el.className = 'primary';
  el.innerHTML = src ? `<img src="${src}" alt="${label}"><figcaption>${label}</figcaption>` : `<img alt="${label}"><figcaption>${label}: boundary</figcaption>`;
  return el;
}
function formatNumber(value, digits = 3) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 'N/A';
  return String(Number(value.toFixed(digits)));
}
function formatMeasure(value, unit) {
  const formatted = formatNumber(value);
  return formatted === 'N/A' ? formatted : `${formatted} ${unit}`;
}
function formatVector(value) {
  if (!Array.isArray(value) || value.length !== 3) return 'N/A';
  const components = value.map(component => formatNumber(component));
  return components.includes('N/A') ? 'N/A' : `[${components.join(', ')}]`;
}
function formatRotation(angle, axis) {
  const formattedAngle = formatMeasure(angle, 'deg');
  const formattedAxis = formatVector(axis);
  if (formattedAngle === 'N/A' && formattedAxis === 'N/A') return 'N/A';
  return `${formattedAngle}; axis ${formattedAxis}`;
}
function formatTranslation(vector, norm) {
  const formattedVector = formatVector(vector);
  const formattedNorm = formatMeasure(norm, 'm');
  if (formattedVector === 'N/A' && formattedNorm === 'N/A') return 'N/A';
  return `${formattedVector}; norm ${formattedNorm}`;
}
function formatText(value) {
  return value === null || value === undefined || value === '' ? 'N/A' : String(value);
}
function renderGeometry(geometry) {
  const card = document.getElementById('geometry-card');
  if (!geometry) {
    card.hidden = true;
    return;
  }
  card.hidden = false;
  const positive = geometry.automatic_label === 1;
  const badge = document.getElementById('auto-label-badge');
  badge.textContent = `DA3 AUTO LABEL: ${positive ? 'POSITIVE' : 'NEGATIVE'}`;
  badge.className = `auto-badge ${positive ? 'positive' : 'negative'}`;
  document.getElementById('gt-rotation').textContent =
    formatRotation(geometry.gt_rotation_angle_deg, geometry.gt_rotation_axis);
  document.getElementById('gt-translation').textContent =
    formatTranslation(geometry.gt_translation, geometry.gt_translation_norm_m);
  document.getElementById('estimated-rotation').textContent =
    formatRotation(geometry.estimated_rotation_angle_deg, geometry.estimated_rotation_axis);
  document.getElementById('estimated-translation').textContent =
    formatTranslation(geometry.estimated_translation, geometry.estimated_translation_norm_m);
  document.getElementById('rotation-error').textContent =
    formatMeasure(geometry.rotation_error_deg, 'deg');
  document.getElementById('translation-error').textContent =
    formatMeasure(geometry.translation_error_m, 'm');
  document.getElementById('translation-error-threshold').textContent =
    formatMeasure(geometry.effective_translation_error_threshold_m, 'm');
  document.getElementById('translation-direction-error').textContent =
    formatMeasure(geometry.translation_direction_error_deg, 'deg');
  document.getElementById('support-index').textContent = formatText(geometry.support_idx);
  document.getElementById('factor-status').textContent = formatText(geometry.factor_status);
  document.getElementById('automatic-reason').textContent =
    formatText(geometry.automatic_label_reason);
}
function render(state) {
  currentState = state;
  document.getElementById('progress').textContent = `${state.progress.completed} / ${state.progress.total}`;
  document.getElementById('message').textContent = state.sealed ? 'Annotations finalized.' : state.complete ? 'All pairs labeled. Finalize when ready.' : '';
  const q = document.getElementById('query-track');
  const c = document.getElementById('candidate-track');
  q.replaceChildren(figure(state.images.query_prev, 'previous', false), figure(state.images.query, 'query', true), figure(state.images.query_next, 'next', false));
  c.replaceChildren(figure(state.images.candidate_prev, 'previous', false), figure(state.images.candidate, 'candidate', true), figure(state.images.candidate_next, 'next', false));
  renderGeometry(state.geometry);
  updateButtonState();
}
async function loadState() {
  const response = await fetch('/api/state');
  render(await response.json());
}
async function postJSON(url, payload) {
  setBusy(true);
  try {
    const response = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload || {}) });
    const data = await response.json();
    if (data.state) render(data.state);
    else if (response.ok) render(data);
    else await loadState();
    if (!response.ok) document.getElementById('message').textContent = data.error || 'request failed';
  } finally {
    setBusy(false);
  }
}
function stateGuardedPayload({ allowComplete = false } = {}) {
  if (!currentState || currentState.sealed || (!allowComplete && currentState.complete)) return null;
  return { pair_token: currentState.pair_token, state_nonce: currentState.state_nonce };
}
function submitLabel(label) {
  const payload = stateGuardedPayload();
  if (!payload || requestActive) return false;
  payload.label = label;
  postJSON('/api/label', payload);
  return true;
}
function acceptAutomaticLabel() {
  if (
    !currentState ||
    currentState.sealed || currentState.complete || requestActive
  ) return false;
  const geometry = currentState.geometry;
  if (
    !geometry ||
    !Number.isInteger(geometry.automatic_label) ||
    ![0, 1].includes(geometry.automatic_label)
  ) return false;
  return submitLabel(geometry.automatic_label);
}
function undoLast() {
  const payload = stateGuardedPayload({ allowComplete: true });
  if (!payload || requestActive) return;
  postJSON('/api/undo', payload);
}
function finalizeAnnotations() {
  if (!currentState || currentState.sealed || !currentState.complete || requestActive) return;
  postJSON('/api/finalize', { state_nonce: currentState.state_nonce });
}
document.addEventListener('keydown', event => {
  if (event.repeat) return;
  if (requestActive) return;
  if (event.key === ' ' || event.key === 'Enter') {
    if (event.target instanceof HTMLButtonElement) return;
    if (acceptAutomaticLabel()) event.preventDefault();
    return;
  }
  if (event.key.toLowerCase() === 'p') submitLabel(1);
  if (event.key.toLowerCase() === 'n') submitLabel(0);
  if (event.key === 'Backspace') { event.preventDefault(); undoLast(); }
});
loadState();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
