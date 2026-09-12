#!/usr/bin/env python3
"""Build or resume a frozen DA3 geometry-prediction annotation bundle."""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import math
import os
import stat
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

LOOPANYTHING_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = LOOPANYTHING_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for import_path in (LOOPANYTHING_ROOT, SRC_ROOT, SCRIPT_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from robust_loop_verifier.geometry_annotation import (  # noqa: E402
    GeometryLabelThresholds,
    evaluate_metric_loop_factor,
)
from robust_loop_verifier.rover_pair_scoring import (  # noqa: E402
    iter_geometry_factor_batches,
)
from score_rover_aligned_benchmark import (  # noqa: E402
    _load_verifier_configs,
    _make_da3_runner,
    _verifier_manifest,
)

PREDICTION_FILENAME = "geometry_predictions.jsonl"
PREDICTION_MANIFEST_FILENAME = "geometry_prediction_manifest.json"
LOCK_FILENAME = ".build_da3_geometry_annotation.lock"
FORMAT_VERSION = 1
FROZEN_SOURCE_FILENAMES = ("benchmark_pairs.jsonl", "manifest.json")
CRITICAL_OUTPUT_FILENAMES = (
    *FROZEN_SOURCE_FILENAMES,
    PREDICTION_FILENAME,
    PREDICTION_MANIFEST_FILENAME,
    LOCK_FILENAME,
)
_UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS = {
    errno.EINVAL,
    errno.ENOSYS,
    getattr(errno, "ENOTSUP", errno.EINVAL),
    getattr(errno, "EOPNOTSUPP", errno.EINVAL),
}
COMMON_DA3_CONFIG_FIELDS = {
    "backend",
    "device",
    "verifier_configs",
    "verifier_config_sha256",
    "salad_score_file",
    "salad_score_file_sha256",
    "salad_score_manifest",
    "salad_score_manifest_sha256",
    "da3_model_name",
    "da3_model_dir",
    "da3_model_path",
    "da3_model_path_sha256",
    "da3_checkpoint",
    "da3_checkpoint_sha256",
    "da3_snapshot",
    "da3_snapshot_sha256",
    "da3_runtime_by_dataset",
}
REAL_DA3_CONFIG_FIELDS = {
    "da3_cache_dir",
    "da3_cache_source",
    "da3_process_res",
    "da3_ref_view_strategy",
    "da3_triplet_batch_size",
}
REAL_DATASET_RUNTIME_FIELDS = {
    "da3_model_name",
    "da3_model_dir",
    "da3_model_path",
    "da3_model_path_sha256",
    "da3_checkpoint",
    "da3_checkpoint_sha256",
    "da3_snapshot",
    "da3_snapshot_sha256",
    "da3_cache_dir",
    "da3_cache_source",
}
IDENTITY_FIELDS = (
    "pair_id",
    "dataset",
    "platform",
    "sequence",
    "sequence_key",
    "query_idx",
    "candidate_idx",
    "rank",
)
SUPPORT_FIELDS = (
    "support_idx",
    "support_rejection_reason",
    "support_baseline_m",
    "support_ensemble_enabled",
    "support_ensemble_support_count_requested",
    "support_ensemble_support_count_used",
    "support_ensemble_supports",
    "support_ensemble_effective_support_count",
    "support_ensemble_loop_sigmas",
    "support_ensemble_sigma_rot",
    "support_ensemble_sigma_trans",
    "support_ensemble_uncertainty_logdet_penalty",
)
SIM3_FIELDS = (
    "sim3_valid",
    "sim3_scale",
    "sim3_support_alignment_residual_m",
    "sim3_direction_error_deg",
    "sim3_rejection_reason",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--backend", choices=("real", "mock"), default="real")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pair-limit", type=_positive_int)
    parser.add_argument("--min-translation-error-m", type=float, default=1.0)
    parser.add_argument("--max-translation-error-m", type=float, default=5.0)
    parser.add_argument("--translation-error-scale-ratio", type=float, default=0.2)
    parser.add_argument("--max-rotation-error-deg", type=float, default=15.0)
    parser.add_argument(
        "--max-translation-direction-error-deg",
        type=float,
        default=20.0,
    )
    parser.add_argument("--min-direction-baseline-m", type=float, default=0.5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_arg_parser().parse_args(effective_argv)
    benchmark_root = Path(args.benchmark_root)
    output_root = Path(args.output_root)
    _validate_output_paths(benchmark_root, output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    with _bundle_lock(output_root):
        return _build_bundle_locked(args, effective_argv)


def _build_bundle_locked(args: argparse.Namespace, effective_argv: Sequence[str]) -> int:
    benchmark_root = Path(args.benchmark_root)
    output_root = Path(args.output_root)
    command = [str(Path(__file__).resolve()), *effective_argv]
    _validate_critical_output_paths(benchmark_root, output_root)

    source_contents = {
        filename: _read_stable_bytes(benchmark_root / filename)
        for filename in FROZEN_SOURCE_FILENAMES
    }
    for filename in FROZEN_SOURCE_FILENAMES:
        _copy_frozen_source_once(
            benchmark_root / filename,
            output_root / filename,
            content=source_contents[filename],
        )
    _validate_critical_output_paths(benchmark_root, output_root)

    snapshot_root = output_root
    source_pair_path = snapshot_root / "benchmark_pairs.jsonl"
    source_manifest_path = snapshot_root / "manifest.json"
    source_pairs = _read_jsonl_strict(source_pair_path)
    source_pair_ids = _validated_pair_ids(source_pairs, context="source pair manifest")
    source_pair_hash = _sha256_file(source_pair_path)
    source_manifest = _read_required_json(source_manifest_path)
    _validate_source_pair_hash(source_manifest, source_pair_hash)
    target_pairs = source_pairs[: args.pair_limit] if args.pair_limit is not None else source_pairs
    target_pair_ids = [str(pair["pair_id"]) for pair in target_pairs]
    thresholds = GeometryLabelThresholds(
        min_translation_error_m=args.min_translation_error_m,
        max_translation_error_m=args.max_translation_error_m,
        translation_error_scale_ratio=args.translation_error_scale_ratio,
        max_rotation_error_deg=args.max_rotation_error_deg,
        max_translation_direction_error_deg=(args.max_translation_direction_error_deg),
        min_direction_baseline_m=args.min_direction_baseline_m,
    )
    threshold_values = asdict(thresholds)
    request_contract = {
        "backend": args.backend,
        "device": args.device,
        "pair_limit": args.pair_limit,
        "thresholds": threshold_values,
    }
    static_manifest = {
        "format_version": FORMAT_VERSION,
        "prediction_file": PREDICTION_FILENAME,
        "source_pair_manifest_sha256": source_pair_hash,
        "source_manifest_sha256": _sha256_file(source_manifest_path),
        "source_pair_count": len(source_pair_ids),
        "target_pair_count": len(target_pair_ids),
        "target_pair_ids_sha256": _pair_ids_sha256(target_pair_ids),
        "pair_limit": args.pair_limit,
        "thresholds": threshold_values,
        "request_contract": request_contract,
    }

    prediction_path = output_root / PREDICTION_FILENAME
    prediction_manifest_path = output_root / PREDICTION_MANIFEST_FILENAME
    existing_predictions = _read_jsonl_strict(prediction_path, missing_ok=True)
    _validate_prediction_prefix(existing_predictions, target_pair_ids)
    existing_manifest = _read_optional_json(prediction_manifest_path)
    if existing_predictions and existing_manifest is None:
        raise ValueError("existing geometry predictions require geometry_prediction_manifest.json")

    if existing_manifest is not None:
        committed_count = _validate_existing_generation_manifest(
            existing_manifest,
            static_manifest,
            prediction_path,
            existing_predictions,
            target_pair_ids,
        )
        if not existing_manifest["complete"] and committed_count < len(existing_predictions):
            _truncate_jsonl_rows(prediction_path, committed_count)
            existing_predictions = existing_predictions[:committed_count]
        if existing_manifest["complete"]:
            return 0
        if len(existing_predictions) == len(target_pair_ids):
            complete_manifest = {
                **existing_manifest,
                "complete": True,
            }
            _write_json_atomic(prediction_manifest_path, complete_manifest)
            return 0

    completed_pair_ids = _validate_prediction_prefix(existing_predictions, target_pair_ids)
    configs, da3_runner, da3_config = _build_runtime(
        snapshot_root,
        args.backend,
        args.device,
    )
    _validate_da3_config(da3_config, request_contract)
    new_manifest = {
        **static_manifest,
        "da3_config": da3_config,
        "da3_config_sha256": _canonical_json_sha256(da3_config),
        "command": command,
        **_source_provenance(),
        "prediction_record_count": len(existing_predictions),
        "prediction_file_sha256": (
            _sha256_file(prediction_path) if prediction_path.exists() else None
        ),
        "complete": False,
    }
    if existing_manifest is not None:
        if existing_manifest["da3_config"] != da3_config:
            raise ValueError("existing geometry prediction bundle runtime mismatch: da3_config")
        manifest = dict(existing_manifest)
    else:
        manifest = new_manifest
        _write_json_atomic(prediction_manifest_path, manifest)

    excluded_pair_ids = set(source_pair_ids) - set(target_pair_ids)
    iterator_completed_ids = set(completed_pair_ids) | excluded_pair_ids
    next_pair_offset = len(existing_predictions)
    for factor_batch in iter_geometry_factor_batches(
        snapshot_root,
        configs,
        da3_runner,
        completed_pair_ids=iterator_completed_ids,
    ):
        prediction_batch = [
            _geometry_prediction_row(record, thresholds) for record in factor_batch
        ]
        batch_pair_ids = [str(row["pair_id"]) for row in prediction_batch]
        expected_pair_ids = target_pair_ids[
            next_pair_offset : next_pair_offset + len(batch_pair_ids)
        ]
        if not prediction_batch or batch_pair_ids != expected_pair_ids:
            raise ValueError(
                "geometry factor iterator did not preserve target pair order: "
                f"expected {expected_pair_ids[:5]}, got {batch_pair_ids[:5]}"
            )
        _append_prediction_batch(prediction_path, prediction_batch)
        next_pair_offset += len(prediction_batch)
        manifest = {
            **manifest,
            "prediction_record_count": next_pair_offset,
            "prediction_file_sha256": _sha256_file(prediction_path),
            "complete": False,
        }
        _write_json_atomic(prediction_manifest_path, manifest)

    final_predictions = _read_jsonl_strict(prediction_path, missing_ok=True)
    final_pair_ids = [str(row.get("pair_id")) for row in final_predictions]
    if final_pair_ids != target_pair_ids:
        incomplete_manifest = {
            **manifest,
            "prediction_record_count": len(final_predictions),
            "prediction_file_sha256": (
                _sha256_file(prediction_path) if prediction_path.exists() else None
            ),
            "complete": False,
        }
        _write_json_atomic(prediction_manifest_path, incomplete_manifest)
        raise RuntimeError(
            "incomplete geometry prediction coverage: "
            f"expected {len(target_pair_ids)}, got {len(final_pair_ids)}"
        )

    complete_manifest = {
        **manifest,
        "prediction_record_count": len(final_predictions),
        "prediction_file_sha256": _sha256_file(prediction_path),
        "complete": True,
    }
    _write_json_atomic(prediction_manifest_path, complete_manifest)
    return 0


def _build_runtime(
    benchmark_root: Path,
    backend: str,
    device: str,
) -> tuple[Mapping[str, object], object, dict[str, object]]:
    configs = _load_verifier_configs(benchmark_root)
    runner = _make_da3_runner(benchmark_root, configs, backend, device)
    args = argparse.Namespace(backend=backend, device=device)
    config = _verifier_manifest(benchmark_root, args, da3_runner=runner)
    recorded_runtime = config.get("da3_runtime_by_dataset", {})
    runtime_by_dataset = {}
    for dataset, verifier_config in configs.items():
        existing_runtime = (
            recorded_runtime.get(str(dataset), {}) if isinstance(recorded_runtime, Mapping) else {}
        )
        runtime_by_dataset[str(dataset)] = {
            **(dict(existing_runtime) if isinstance(existing_runtime, Mapping) else {}),
            "process_res": int(verifier_config.da3.process_res),
            "ref_view_strategy": str(verifier_config.da3.ref_view_strategy),
            "triplet_batch_size": int(verifier_config.da3.triplet_batch_size),
        }
    config["da3_runtime_by_dataset"] = runtime_by_dataset
    return configs, runner, _json_safe_mapping(config)


def _geometry_prediction_row(
    record: Mapping[str, Any],
    thresholds: GeometryLabelThresholds,
) -> dict[str, object]:
    estimate, factor_status = _loop_factor_and_status(record)
    gt_query = _pose_matrix(record.get("gt_query_pose"), "gt_query_pose")
    gt_candidate = _pose_matrix(record.get("gt_candidate_pose"), "gt_candidate_pose")
    evaluated = evaluate_metric_loop_factor(
        gt_query,
        gt_candidate,
        estimate,
        factor_status=factor_status,
        thresholds=thresholds,
    )
    output = {
        key: record[key]
        for key in (*IDENTITY_FIELDS, *SUPPORT_FIELDS, *SIM3_FIELDS)
        if key in record
    }
    if "failure_reasons" in record:
        output["failure_reasons"] = list(record["failure_reasons"])
    output["gt_query_pose"] = gt_query.tolist()
    output["gt_candidate_pose"] = gt_candidate.tolist()
    output.update(evaluated)
    return output


def _loop_factor_and_status(
    record: Mapping[str, Any],
) -> tuple[np.ndarray | None, str]:
    raw_factor = record.get("loop_factor")
    estimate = None
    if raw_factor is not None:
        try:
            parsed = np.asarray(raw_factor, dtype=np.float64)
            if parsed.size == 16:
                estimate = parsed.reshape(4, 4)
        except (TypeError, ValueError, OverflowError):
            estimate = None
    if estimate is not None and bool(record.get("sim3_valid")):
        return estimate, "ok"
    failure_reasons = [str(reason) for reason in record.get("failure_reasons", [])]
    if record.get("support_idx") is None:
        return estimate, "support_failed"
    if any(reason.startswith("da3:") for reason in failure_reasons):
        return estimate, "da3_failed"
    if not bool(record.get("sim3_valid")):
        return estimate, "sim3_failed"
    return estimate, "invalid_loop_factor"


def _pose_matrix(value: object, field_name: str) -> np.ndarray:
    try:
        pose = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be a finite 4x4 pose") from exc
    if pose.size == 16:
        pose = pose.reshape(4, 4)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError(f"{field_name} must be a finite 4x4 pose")
    return pose


def _append_prediction_batch(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    path = Path(path)
    content = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for row in rows
    )
    descriptor = _open_regular_file_no_symlink(
        path,
        os.O_WRONLY | os.O_APPEND | os.O_CREAT,
        mode=0o644,
    )
    with os.fdopen(descriptor, "a", encoding="utf-8", newline="") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _copy_frozen_source_once(
    source: Path,
    destination: Path,
    *,
    content: bytes | None = None,
) -> None:
    source = Path(source)
    destination = Path(destination)
    if content is None:
        content = _read_stable_bytes(source)
    _ensure_regular_or_missing(destination)
    if destination.exists():
        if os.path.samefile(source, destination):
            raise ValueError(
                f"frozen source destination is a hardlink to the same file: {destination.name}"
            )
        if _read_regular_bytes_no_symlink(destination) != content:
            raise ValueError(f"frozen source copy mismatch: {destination.name}")
        return

    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_path, destination)
        except FileExistsError:
            _ensure_regular_or_missing(destination)
            if os.path.samefile(source, destination):
                raise ValueError(
                    "frozen source destination is a hardlink to the same file: "
                    f"{destination.name}"
                )
            if _read_regular_bytes_no_symlink(destination) != content:
                raise ValueError(f"frozen source copy mismatch: {destination.name}")
        else:
            temp_path.unlink()
        if _read_regular_bytes_no_symlink(destination) != content:
            raise OSError(f"frozen source copy verification failed: {destination.name}")
        _fsync_directory(destination.parent)
    finally:
        temp_path.unlink(missing_ok=True)


def _validate_output_paths(benchmark_root: Path, output_root: Path) -> None:
    if Path(output_root).is_symlink():
        raise ValueError("output_root must not be a symlink")
    benchmark_resolved = Path(benchmark_root).resolve()
    output_resolved = Path(output_root).resolve()
    if output_resolved == benchmark_resolved:
        raise ValueError("output_root must not resolve to benchmark_root")
    for filename in FROZEN_SOURCE_FILENAMES:
        source = (Path(benchmark_root) / filename).resolve()
        destination = (Path(output_root) / filename).resolve()
        if destination == source:
            raise ValueError(
                "frozen source destination must not resolve to source itself: " f"{filename}"
            )


def _validate_critical_output_paths(benchmark_root: Path, output_root: Path) -> None:
    for filename in CRITICAL_OUTPUT_FILENAMES:
        _ensure_regular_or_missing(Path(output_root) / filename)
    for filename in FROZEN_SOURCE_FILENAMES:
        source = Path(benchmark_root) / filename
        destination = Path(output_root) / filename
        if destination.exists() and os.path.samefile(source, destination):
            raise ValueError(
                f"frozen source destination is a hardlink to the same file: {filename}"
            )


@contextmanager
def _bundle_lock(output_root: Path) -> Iterator[None]:
    output_root = Path(output_root)
    lock_path = output_root / LOCK_FILENAME
    _ensure_regular_or_missing(lock_path)
    descriptor = _open_regular_file_no_symlink(
        lock_path,
        os.O_RDWR | os.O_CREAT,
        mode=0o600,
    )
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise RuntimeError(
                    f"output bundle is already locked by another builder: {output_root}"
                ) from exc
            raise
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _read_stable_bytes(path: Path) -> bytes:
    path = Path(path)
    descriptor = os.open(path, os.O_RDONLY)
    try:
        before = os.fstat(descriptor)
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            content = handle.read()
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after or len(content) != before.st_size:
        raise RuntimeError(f"source changed during stable read: {path}")
    return content


def _ensure_regular_or_missing(path: Path) -> None:
    path = Path(path)
    try:
        file_stat = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(file_stat.st_mode):
        raise ValueError(f"critical output path must not be a symlink: {path}")
    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(f"critical output path must be a regular file: {path}")
    if file_stat.st_nlink != 1:
        raise ValueError(f"critical output path must not be a hardlink: {path}")


def _open_regular_file_no_symlink(path: Path, flags: int, *, mode: int) -> int:
    path = Path(path)
    _ensure_regular_or_missing(path)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags | nofollow, mode)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError(f"critical output path must not be a symlink: {path}") from exc
        raise
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError(f"critical output path must be a regular file: {path}")
    if os.fstat(descriptor).st_nlink != 1:
        os.close(descriptor)
        raise ValueError(f"critical output path must not be a hardlink: {path}")
    return descriptor


def _read_regular_bytes_no_symlink(path: Path) -> bytes:
    descriptor = _open_regular_file_no_symlink(Path(path), os.O_RDONLY, mode=0)
    with os.fdopen(descriptor, "rb") as handle:
        return handle.read()


def _read_jsonl_strict(path: Path, *, missing_ok: bool = False) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        if missing_ok:
            return []
        raise FileNotFoundError(path)
    content = path.read_bytes()
    if content and not content.endswith(b"\n"):
        raise ValueError(f"partial final line in JSONL file: {path}")
    rows = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line:
            raise ValueError(f"blank JSONL line at {path}:{line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL row at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
        rows.append(row)
    return rows


def _validated_pair_ids(
    rows: Sequence[Mapping[str, object]],
    *,
    context: str,
) -> list[str]:
    pair_ids = []
    for index, row in enumerate(rows):
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError(f"{context} row {index} has invalid pair_id")
        pair_ids.append(pair_id)
    duplicates = sorted(pair_id for pair_id in set(pair_ids) if pair_ids.count(pair_id) > 1)
    if duplicates:
        raise ValueError(f"{context} contains duplicate pair IDs: {duplicates[:5]}")
    return pair_ids


def _validate_prediction_prefix(
    predictions: Sequence[Mapping[str, object]],
    target_pair_ids: Sequence[str],
) -> set[str]:
    observed = _validated_pair_ids(predictions, context="geometry predictions")
    expected = list(target_pair_ids[: len(observed)])
    if observed != expected:
        raise ValueError(
            "existing geometry predictions are not an ordered target prefix: "
            f"expected {expected[:5]}, got {observed[:5]}"
        )
    return set(observed)


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON manifest must be an object: {path}")
    return payload


def _read_required_json(path: Path) -> dict[str, Any]:
    payload = _read_optional_json(path)
    if payload is None:
        raise FileNotFoundError(path)
    return payload


def _validate_source_pair_hash(
    source_manifest: Mapping[str, object],
    actual_pair_hash: str,
) -> None:
    declared_pair_hash = source_manifest.get("pair_manifest_sha256")
    if not isinstance(declared_pair_hash, str) or not declared_pair_hash:
        raise ValueError("source manifest pair_manifest_sha256 must be a non-empty string")
    if declared_pair_hash != actual_pair_hash:
        raise ValueError(
            "source manifest pair_manifest_sha256 does not match " "benchmark_pairs.jsonl"
        )


def _validate_existing_generation_manifest(
    existing: Mapping[str, object],
    expected_static: Mapping[str, object],
    prediction_path: Path,
    predictions: Sequence[Mapping[str, object]],
    target_pair_ids: Sequence[str],
) -> int:
    _require_exact_int(
        existing,
        "format_version",
        int(expected_static["format_version"]),
    )
    if existing.get("prediction_file") != PREDICTION_FILENAME:
        raise ValueError(
            "existing geometry prediction manifest prediction_file must be "
            f"{PREDICTION_FILENAME}"
        )
    complete = existing.get("complete")
    if not isinstance(complete, bool):
        raise ValueError("existing geometry prediction manifest complete must be a bool")

    for field in ("source_pair_count", "target_pair_count"):
        _require_exact_int(existing, field, int(expected_static[field]))
    for field in (
        "source_pair_manifest_sha256",
        "source_manifest_sha256",
        "target_pair_ids_sha256",
    ):
        value = existing.get(field)
        if not isinstance(value, str) or value != expected_static[field]:
            raise ValueError("existing geometry prediction bundle contract mismatch: " f"{field}")

    _validate_pair_limit(
        existing.get("pair_limit"),
        expected_static["pair_limit"],
        "pair_limit",
    )
    _validate_thresholds(
        existing.get("thresholds"),
        expected_static["thresholds"],
        "thresholds",
    )
    _validate_request_contract(
        existing.get("request_contract"),
        expected_static["request_contract"],
    )

    da3_config = existing.get("da3_config")
    da3_config_hash = existing.get("da3_config_sha256")
    _validate_sha256(da3_config_hash, "da3_config_sha256")
    if da3_config_hash != _canonical_json_sha256(da3_config):
        raise ValueError(
            "existing geometry prediction manifest da3_config_sha256 " "does not match da3_config"
        )
    _validate_da3_config(da3_config, expected_static["request_contract"])
    source_commit = existing.get("source_commit")
    if not isinstance(source_commit, str) or not source_commit:
        raise ValueError(
            "existing geometry prediction manifest source_commit must be " "a non-empty string"
        )
    source_commit_error = existing.get("source_commit_error")
    if source_commit_error is not None and (
        not isinstance(source_commit_error, str) or not source_commit_error
    ):
        raise ValueError(
            "existing geometry prediction manifest source_commit_error must be "
            "a non-empty string"
        )
    command = existing.get("command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(item, str) or not item for item in command)
    ):
        raise ValueError(
            "existing geometry prediction manifest command must be a non-empty "
            "list of non-empty strings"
        )

    recorded_count = existing.get("prediction_record_count")
    if type(recorded_count) is not int or recorded_count < 0 or recorded_count > len(predictions):
        raise ValueError(
            "existing geometry prediction bundle contract mismatch: " "prediction_record_count"
        )
    if complete and recorded_count != len(predictions):
        raise ValueError("complete geometry prediction manifest prediction_record_count mismatch")

    recorded_prediction_hash = existing.get("prediction_file_sha256")
    checkpoint_hash = _jsonl_prefix_sha256(prediction_path, recorded_count)
    if recorded_prediction_hash != checkpoint_hash:
        raise ValueError(
            "existing geometry prediction manifest prediction_file_sha256 "
            "does not match its claimed prediction prefix"
        )

    pair_ids = [str(row["pair_id"]) for row in predictions]
    if complete and pair_ids != list(target_pair_ids):
        raise ValueError("complete geometry prediction manifest lacks full target coverage")
    return recorded_count


def _validate_request_contract(
    value: object,
    expected: object,
) -> None:
    if not isinstance(value, Mapping) or not isinstance(expected, Mapping):
        raise ValueError("existing geometry prediction bundle contract mismatch: request_contract")
    expected_fields = {"backend", "device", "pair_limit", "thresholds"}
    if set(value) != expected_fields:
        raise ValueError("existing geometry prediction bundle contract mismatch: request_contract")
    for field in ("backend", "device"):
        actual_value = value.get(field)
        if (
            not isinstance(actual_value, str)
            or not actual_value
            or actual_value != expected[field]
        ):
            raise ValueError(
                "existing geometry prediction bundle contract mismatch: "
                f"request_contract.{field}"
            )
    _validate_pair_limit(
        value.get("pair_limit"),
        expected["pair_limit"],
        "request_contract.pair_limit",
    )
    _validate_thresholds(
        value.get("thresholds"),
        expected["thresholds"],
        "request_contract.thresholds",
    )


def _validate_pair_limit(
    value: object,
    expected: object,
    field_name: str,
) -> None:
    valid = value is None if expected is None else type(value) is int and value > 0
    if not valid or value != expected:
        raise ValueError("existing geometry prediction bundle contract mismatch: " f"{field_name}")


def _validate_thresholds(
    value: object,
    expected: object,
    field_name: str,
) -> None:
    if not isinstance(value, Mapping) or not isinstance(expected, Mapping):
        raise ValueError("existing geometry prediction bundle contract mismatch: " f"{field_name}")
    if set(value) != set(expected):
        raise ValueError("existing geometry prediction bundle contract mismatch: " f"{field_name}")
    for threshold_name, expected_value in expected.items():
        threshold = value.get(threshold_name)
        if (
            type(threshold) is not float
            or not math.isfinite(threshold)
            or threshold < 0.0
            or threshold != expected_value
        ):
            raise ValueError(
                "existing geometry prediction bundle contract mismatch: "
                f"{field_name}.{threshold_name}"
            )


def _validate_da3_config(
    value: object,
    request_contract: object,
) -> None:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(
            "existing geometry prediction manifest da3_config must be " "a non-empty mapping"
        )
    if not isinstance(request_contract, Mapping):
        raise ValueError("request_contract must be a mapping")
    missing_fields = COMMON_DA3_CONFIG_FIELDS - set(value)
    if missing_fields:
        raise ValueError(
            "geometry prediction manifest da3_config missing required field: "
            f"{sorted(missing_fields)[0]}"
        )
    for field in ("backend", "device"):
        config_value = value.get(field)
        if (
            not isinstance(config_value, str)
            or not config_value
            or config_value != request_contract.get(field)
        ):
            raise ValueError(f"geometry prediction manifest da3_config {field} mismatch")
    backend = str(value["backend"])
    if backend == "real":
        missing_real_fields = REAL_DA3_CONFIG_FIELDS - set(value)
        if missing_real_fields:
            raise ValueError(
                "geometry prediction manifest real da3_config missing required "
                f"field: {sorted(missing_real_fields)[0]}"
            )

    verifier_configs = value.get("verifier_configs")
    if not isinstance(verifier_configs, Mapping) or not verifier_configs:
        raise ValueError(
            "geometry prediction manifest da3_config verifier_configs must be "
            "a non-empty mapping"
        )
    for dataset, config_path in verifier_configs.items():
        if (
            not isinstance(dataset, str)
            or not dataset
            or not isinstance(config_path, str)
            or not config_path
        ):
            raise ValueError(
                "geometry prediction manifest da3_config verifier_configs "
                "must map non-empty strings to non-empty strings"
            )

    config_hashes = value.get("verifier_config_sha256")
    if not isinstance(config_hashes, Mapping) or set(config_hashes) != set(verifier_configs):
        raise ValueError(
            "geometry prediction manifest da3_config verifier_config_sha256 "
            "must match verifier_configs"
        )
    for dataset, config_hash in config_hashes.items():
        if config_hash is not None:
            _validate_sha256(
                config_hash,
                f"da3_config.verifier_config_sha256[{dataset!r}]",
            )

    runtime_by_dataset = value.get("da3_runtime_by_dataset")
    if not isinstance(runtime_by_dataset, Mapping) or set(runtime_by_dataset) != set(
        verifier_configs
    ):
        raise ValueError(
            "geometry prediction manifest da3_config da3_runtime_by_dataset "
            "must match verifier_configs"
        )
    for dataset, runtime in runtime_by_dataset.items():
        if not isinstance(runtime, Mapping):
            raise ValueError(
                "geometry prediction manifest da3_config runtime for "
                f"{dataset!r} must be a mapping"
            )
        process_res = runtime.get("process_res")
        if type(process_res) is not int or process_res <= 0:
            raise ValueError(
                "geometry prediction manifest da3_config "
                f"da3_runtime_by_dataset[{dataset!r}].process_res must be "
                "a positive integer"
            )
        ref_view_strategy = runtime.get("ref_view_strategy")
        if not isinstance(ref_view_strategy, str) or not ref_view_strategy:
            raise ValueError(
                "geometry prediction manifest da3_config "
                f"da3_runtime_by_dataset[{dataset!r}].ref_view_strategy must "
                "be a non-empty string"
            )
        triplet_batch_size = runtime.get("triplet_batch_size")
        if type(triplet_batch_size) is not int or triplet_batch_size <= 0:
            raise ValueError(
                "geometry prediction manifest da3_config "
                f"da3_runtime_by_dataset[{dataset!r}].triplet_batch_size must "
                "be a positive integer"
            )
        if backend == "real":
            missing_runtime_fields = REAL_DATASET_RUNTIME_FIELDS - set(runtime)
            if missing_runtime_fields:
                raise ValueError(
                    "geometry prediction manifest real da3 runtime missing "
                    f"required field for {dataset!r}: "
                    f"{sorted(missing_runtime_fields)[0]}"
                )
        _validate_optional_da3_provenance(runtime, f"runtime[{dataset!r}]")

    if backend == "real":
        for field in ("da3_process_res", "da3_triplet_batch_size"):
            field_value = value.get(field)
            if type(field_value) is not int or field_value <= 0:
                raise ValueError(
                    f"geometry prediction manifest real da3_config {field} "
                    "must be a positive integer"
                )
        ref_view_strategy = value.get("da3_ref_view_strategy")
        if not isinstance(ref_view_strategy, str) or not ref_view_strategy:
            raise ValueError(
                "geometry prediction manifest real da3_config "
                "da3_ref_view_strategy must be a non-empty string"
            )
    _validate_optional_da3_provenance(value, "da3_config")


def _validate_optional_da3_provenance(
    value: Mapping[str, object],
    context: str,
) -> None:
    optional_string_fields = (
        "da3_model_name",
        "da3_model_dir",
        "da3_model_path",
        "da3_checkpoint",
        "da3_snapshot",
        "da3_cache_dir",
        "da3_cache_source",
        "salad_score_file",
        "salad_score_manifest",
    )
    optional_hash_fields = (
        "da3_model_path_sha256",
        "da3_checkpoint_sha256",
        "da3_snapshot_sha256",
        "salad_score_file_sha256",
        "salad_score_manifest_sha256",
    )
    for field in (*optional_string_fields, *optional_hash_fields):
        if field not in value:
            continue
        field_value = value[field]
        if field_value is None:
            continue
        if field in optional_hash_fields:
            _validate_sha256(field_value, f"{context}.{field}")
        elif not isinstance(field_value, str) or not field_value:
            raise ValueError(f"geometry prediction manifest {context}.{field} has invalid type")


def _validate_sha256(value: object, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError(f"{field_name} must be a non-empty 64-character hex SHA256")


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jsonl_prefix_sha256(path: Path, row_count: int) -> str | None:
    if row_count == 0:
        return None
    lines = Path(path).read_bytes().splitlines(keepends=True)
    if row_count > len(lines):
        raise ValueError("prediction_record_count exceeds geometry_predictions.jsonl")
    return hashlib.sha256(b"".join(lines[:row_count])).hexdigest()


def _truncate_jsonl_rows(path: Path, row_count: int) -> None:
    path = Path(path)
    content = _read_regular_bytes_no_symlink(path)
    lines = content.splitlines(keepends=True)
    if row_count < 0 or row_count > len(lines):
        raise ValueError("invalid geometry prediction truncation boundary")
    retained_size = sum(len(line) for line in lines[:row_count])
    descriptor = _open_regular_file_no_symlink(path, os.O_WRONLY, mode=0)
    try:
        os.ftruncate(descriptor, retained_size)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _require_exact_int(
    mapping: Mapping[str, object],
    field: str,
    expected: int,
) -> None:
    value = mapping.get(field)
    if type(value) is not int or value != expected:
        raise ValueError("existing geometry prediction bundle contract mismatch: " f"{field}")


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    _ensure_regular_or_missing(path)
    content = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        _ensure_regular_or_missing(path)
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    finally:
        temp_path.unlink(missing_ok=True)


def _source_provenance() -> dict[str, str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=LOOPANYTHING_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception as exc:
        detail = getattr(exc, "stderr", None)
        suffix = f": {str(detail).strip()}" if detail else f": {exc}"
        return {
            "source_commit": "unknown",
            "source_commit_error": f"{type(exc).__name__}{suffix}",
        }
    commit = result.stdout.strip()
    if commit:
        return {"source_commit": commit}
    return {
        "source_commit": "unknown",
        "source_commit_error": "git rev-parse HEAD returned empty output",
    }


def _json_safe_mapping(value: Mapping[str, object]) -> dict[str, object]:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _pair_ids_sha256(pair_ids: Sequence[str]) -> str:
    content = "".join(f"{pair_id}\n" for pair_id in pair_ids).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        if exc.errno in _UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS:
            return
        raise
    try:
        try:
            os.fsync(descriptor)
        except OSError as exc:
            if exc.errno not in _UNSUPPORTED_DIRECTORY_FSYNC_ERRNOS:
                raise
    finally:
        os.close(descriptor)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
