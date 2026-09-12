#!/usr/bin/env python3
"""Score frozen ROVER-aligned benchmark pairs without reretrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

LOOPANYTHING_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = LOOPANYTHING_ROOT / "src"
for import_path in (LOOPANYTHING_ROOT, SRC_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from robust_loop_verifier.da3_runner import (  # noqa: E402
    MockDa3Runner,
    RealDa3Runner,
    RealDa3RunnerConfig,
    resolve_local_da3_model_name_or_path,
)
from robust_loop_verifier.io import read_json, read_yaml  # noqa: E402
from robust_loop_verifier.rover_benchmark import sha256_file  # noqa: E402
from robust_loop_verifier.rover_pair_scoring import (  # noqa: E402
    build_netvlad_backend,
    build_salad_backend,
    load_verifier_config_for_dataset,
    records_to_named_score_rows,
    score_netvlad_pairs,
    score_salad_pairs,
    score_verifier_pairs,
    validate_score_row_order,
)


METHOD_ROVER = "ROVER deformation only"
METHOD_LOOPANYTHING = "query_gate_graph:def=0.5,res=0.25,margin=0"
SOURCE_IMPLEMENTATION_PATHS = (
    "robust_loop_verification_scripts/score_rover_aligned_benchmark.py",
    "src/robust_loop_verifier/rover_pair_scoring.py",
    "src/robust_loop_verifier/pipeline.py",
    "src/robust_loop_verifier/score_sweep.py",
    "src/robust_loop_verifier/da3_runner.py",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_root", type=Path)
    parser.add_argument("--method", choices=("netvlad", "salad", "verifier"), required=True)
    parser.add_argument("--backend", choices=("real", "mock"), default="real")
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    effective_command = [str(Path(__file__).resolve()), *effective_argv]
    args = build_arg_parser().parse_args(effective_argv)
    benchmark_root = Path(args.benchmark_root)
    pairs = list(_read_pairs(benchmark_root))

    if args.method == "netvlad":
        backend = _make_netvlad_backend(benchmark_root, args.backend, args.device)
        rows = score_netvlad_pairs(benchmark_root, backend)
        validate_score_row_order(pairs, rows)
        payloads = _score_output_payloads(
            benchmark_root,
            score_name="netvlad",
            method="NetVLAD score only",
            evaluation_name="NetVLAD",
            rows=rows,
            extra_manifest=_netvlad_manifest(benchmark_root, args),
            command=effective_command,
        )
    elif args.method == "salad":
        backend = _make_salad_backend(benchmark_root, args.backend, args.device)
        rows = score_salad_pairs(benchmark_root, backend)
        validate_score_row_order(pairs, rows)
        payloads = _score_output_payloads(
            benchmark_root,
            score_name="salad",
            method="SALAD score only",
            evaluation_name="SALAD",
            rows=rows,
            extra_manifest=_salad_manifest(benchmark_root, args),
            command=effective_command,
        )
    else:
        configs = _load_verifier_configs(benchmark_root)
        da3_runner = _make_da3_runner(benchmark_root, configs, args.backend, args.device)
        verifier_manifest = _verifier_manifest(
            benchmark_root,
            args,
            da3_runner=da3_runner,
        )
        records = score_verifier_pairs(benchmark_root, configs, da3_runner)
        validate_score_row_order(pairs, records)
        rover_rows = records_to_named_score_rows(records, METHOD_ROVER)
        loopanything_rows = records_to_named_score_rows(records, METHOD_LOOPANYTHING)
        validate_score_row_order(pairs, rover_rows)
        validate_score_row_order(pairs, loopanything_rows)
        payloads = {
            benchmark_root / "candidate_records.jsonl": _jsonl_content(records),
            **_score_output_payloads(
                benchmark_root,
                score_name="rover_like",
                method=METHOD_ROVER,
                evaluation_name="ROVER-like",
                rows=rover_rows,
                extra_manifest=verifier_manifest,
                command=effective_command,
            ),
            **_score_output_payloads(
                benchmark_root,
                score_name="loopanything",
                method=METHOD_LOOPANYTHING,
                evaluation_name="LoopAnything",
                rows=loopanything_rows,
                extra_manifest=verifier_manifest,
                command=effective_command,
            ),
        }
    _publish_output_set(payloads)
    return 0


def _score_output_payloads(
    benchmark_root: Path,
    *,
    score_name: str,
    method: str,
    evaluation_name: str,
    rows: Sequence[Mapping[str, object]],
    extra_manifest: Mapping[str, object],
    command: Sequence[str],
) -> dict[Path, str]:
    score_path = Path(benchmark_root) / "scores" / f"{score_name}.jsonl"
    score_content = _jsonl_content(rows)
    manifest = {
        "method": method,
        "evaluation_name": evaluation_name,
        "score_file": f"scores/{score_name}.jsonl",
        "pair_manifest_sha256": sha256_file(Path(benchmark_root) / "benchmark_pairs.jsonl"),
        "score_file_sha256": hashlib.sha256(score_content.encode("utf-8")).hexdigest(),
        "command": list(command),
    }
    manifest.update(_source_provenance())
    manifest.update(extra_manifest)
    return {
        score_path: score_content,
        score_path.with_suffix(".manifest.json"): _json_content(manifest),
    }


def _write_score_outputs(
    benchmark_root: Path,
    *,
    score_name: str,
    method: str,
    evaluation_name: str,
    rows: Sequence[Mapping[str, object]],
    extra_manifest: Mapping[str, object],
    command: Sequence[str] | None = None,
) -> None:
    effective_command = (
        list(command)
        if command is not None
        else [str(Path(__file__).resolve()), *sys.argv[1:]]
    )
    _publish_output_set(
        _score_output_payloads(
            benchmark_root,
            score_name=score_name,
            method=method,
            evaluation_name=evaluation_name,
            rows=rows,
            extra_manifest=extra_manifest,
            command=effective_command,
        )
    )


def _read_pairs(benchmark_root: Path):
    from robust_loop_verifier.io import read_jsonl

    return read_jsonl(Path(benchmark_root) / "benchmark_pairs.jsonl")


def _make_netvlad_backend(benchmark_root: Path, backend: str, device: str):
    if backend == "mock":
        return _MockDescriptorBackend()
    manifest = read_json(Path(benchmark_root) / "manifest.json")
    netvlad_root = manifest.get("netvlad_root")
    if not isinstance(netvlad_root, str) or not netvlad_root:
        raise ValueError("manifest does not record a netvlad_root")
    return build_netvlad_backend(Path(netvlad_root), device)


def _make_salad_backend(benchmark_root: Path, backend: str, device: str):
    if backend == "mock":
        return _MockDescriptorBackend()
    config = _salad_config(benchmark_root, device)
    return build_salad_backend(
        device=str(config["device"]),
        salad_repo=Path(str(config["repo"])),
        checkpoint_path=Path(str(config["checkpoint"])),
        backbone=str(config["backbone"]),
        batch_size=int(config["batch_size"]),
    )


def _load_verifier_configs(benchmark_root: Path) -> dict[str, object]:
    manifest = read_json(Path(benchmark_root) / "manifest.json")
    datasets = {
        str(row["dataset"])
        for row in manifest.get("sequences", [])
        if isinstance(row, Mapping)
    }
    return {
        dataset: load_verifier_config_for_dataset(benchmark_root, dataset)
        for dataset in sorted(datasets)
    }


def _make_da3_runner(
    benchmark_root: Path,
    configs: Mapping[str, object],
    backend: str,
    device: str,
):
    if backend == "mock":
        return MockDa3Runner()
    if not configs:
        raise ValueError("real verifier requires at least one dataset config")
    manifest = read_json(Path(benchmark_root) / "manifest.json")
    verifier_configs = _verifier_config_paths(manifest)
    runtime = _da3_runtime_settings(manifest, verifier_configs)
    effective_cache_dir, _ = _effective_da3_cache_dir(runtime["cache_dir"])
    model_name = runtime["model_name"]
    if model_name is None:
        model_name = RealDa3RunnerConfig().model_name
    resolved_model_name = resolve_local_da3_model_name_or_path(
        str(model_name),
        cache_dir=effective_cache_dir,
        require_local=True,
    )
    common_config_kwargs = {
        "model_name": resolved_model_name,
        "device": device,
        "cache_dir": effective_cache_dir,
    }
    return {
        str(dataset): RealDa3Runner(
            RealDa3RunnerConfig(
                **common_config_kwargs,
                process_res=config.da3.process_res,
                ref_view_strategy=config.da3.ref_view_strategy,
                triplet_batch_size=config.da3.triplet_batch_size,
            )
        )
        for dataset, config in configs.items()
    }


def _write_jsonl_if_identical_or_absent(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    _publish_output_set({Path(path): _jsonl_content(rows)})


def _jsonl_content(rows: Sequence[Mapping[str, object]]) -> str:
    return "".join(
        json.dumps(
            _sanitize_json_value(dict(row)),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for row in rows
    )


def _write_json_if_identical_or_absent(path: Path, payload: Mapping[str, object]) -> None:
    _publish_output_set({Path(path): _json_content(payload)})


def _json_content(payload: Mapping[str, object]) -> str:
    return json.dumps(
        _sanitize_json_value(dict(payload)),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"


def _sanitize_json_value(value):
    if isinstance(value, (float, np.floating)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    if isinstance(value, Mapping):
        return {key: _sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json_value(item) for item in value]
    return value


def _write_text_if_identical_or_absent(path: Path, content: str) -> None:
    _publish_output_set({Path(path): content})


def _publish_output_set(payloads: Mapping[Path, str]) -> None:
    normalized = {Path(path): content for path, content in payloads.items()}
    missing: list[tuple[Path, str]] = []
    ordered_paths = sorted(
        normalized,
        key=lambda path: path.name.endswith(".manifest.json"),
    )
    for path in ordered_paths:
        content = normalized[path]
        if not path.exists():
            missing.append((path, content))
            continue
        existing = path.read_text(encoding="utf-8")
        if existing != content:
            raise FileExistsError(f"refusing to overwrite changed output: {path}")
    if not missing:
        return

    staged: dict[Path, Path] = {}
    published: list[Path] = []
    created_dirs: list[Path] = []
    try:
        for path, content in missing:
            _ensure_parent_dirs(path.parent, created_dirs)
            descriptor, temp_name = tempfile.mkstemp(
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
            )
            temp_path = Path(temp_name)
            staged[path] = temp_path
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())

        for path, temp_path in staged.items():
            try:
                os.link(temp_path, path)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"refusing to overwrite changed output: {path}"
                ) from exc
            published.append(path)
            _fsync_directory(path.parent)

        for temp_path in staged.values():
            temp_path.unlink()
    except BaseException:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        for path, temp_path in staged.items():
            if path.exists() and temp_path.exists():
                try:
                    if os.path.samefile(path, temp_path):
                        path.unlink()
                except OSError:
                    pass
            temp_path.unlink(missing_ok=True)
        for parent in {path.parent for path in staged}:
            _fsync_directory(parent)
        for directory in reversed(created_dirs):
            try:
                directory.rmdir()
            except OSError:
                pass
            else:
                _fsync_directory(directory.parent)
        raise


def _ensure_parent_dirs(path: Path, created_dirs: list[Path]) -> None:
    pending = []
    current = Path(path)
    while not current.exists():
        pending.append(current)
        current = current.parent
    for directory in reversed(pending):
        directory.mkdir()
        created_dirs.append(directory)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
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


def _netvlad_manifest(benchmark_root: Path, args) -> dict[str, object]:
    manifest = read_json(Path(benchmark_root) / "manifest.json")
    netvlad = manifest.get("netvlad")
    netvlad_config = netvlad if isinstance(netvlad, Mapping) else {}
    netvlad_root = _manifest_value(
        manifest,
        netvlad_config,
        flat_keys=("netvlad_root",),
        nested_keys=("root",),
        default=None,
    )
    config_path = _manifest_value(
        manifest,
        netvlad_config,
        flat_keys=("netvlad_config", "netvlad_config_path"),
        nested_keys=("config", "config_path"),
        default=None,
    )
    resolved_config_path = _optional_resolved_path(config_path)
    resolved_root = _optional_resolved_path(netvlad_root)
    resolved_checkpoint_path = _netvlad_checkpoint_path()
    return {
        "backend": args.backend,
        "device": args.device,
        "netvlad_root": str(resolved_root) if resolved_root is not None else None,
        "netvlad_root_git_commit": _optional_git_commit(resolved_root),
        "netvlad_config": str(resolved_config_path) if resolved_config_path is not None else None,
        "netvlad_config_sha256": (
            _optional_file_sha256(resolved_config_path)
            if resolved_config_path is not None
            else None
        ),
        "netvlad_checkpoint": (
            str(resolved_checkpoint_path) if resolved_checkpoint_path is not None else None
        ),
        "netvlad_checkpoint_sha256": (
            _optional_file_sha256(resolved_checkpoint_path)
            if resolved_checkpoint_path is not None
            else None
        ),
    }


def _salad_manifest(benchmark_root: Path, args) -> dict[str, object]:
    config = _salad_config(benchmark_root, args.device)
    checkpoint_path = Path(str(config["checkpoint"]))
    config_path = config.get("config")
    resolved_config_path = Path(str(config_path)) if config_path is not None else None
    return {
        "backend": args.backend,
        "device": str(config["device"]),
        "salad_repo": str(config["repo"]),
        "salad_repo_git_commit": _optional_git_commit(Path(str(config["repo"]))),
        "salad_checkpoint": str(checkpoint_path),
        "salad_checkpoint_sha256": _optional_file_sha256(checkpoint_path),
        "salad_config": str(resolved_config_path) if resolved_config_path is not None else None,
        "salad_config_sha256": (
            _optional_file_sha256(resolved_config_path)
            if resolved_config_path is not None
            else None
        ),
        "salad_backbone": str(config["backbone"]),
        "salad_batch_size": int(config["batch_size"]),
    }


def _verifier_manifest(
    benchmark_root: Path,
    args,
    *,
    da3_runner=None,
) -> dict[str, object]:
    manifest = read_json(Path(benchmark_root) / "manifest.json")
    verifier_configs = _verifier_config_paths(manifest)
    output = {
        "backend": args.backend,
        "device": args.device,
        "verifier_configs": verifier_configs,
        "verifier_config_sha256": {
            dataset: _optional_file_sha256(_resolve_repo_path(path))
            for dataset, path in verifier_configs.items()
        },
        **_frozen_salad_provenance(benchmark_root),
    }
    if args.backend == "real":
        if da3_runner is None:
            raise ValueError("real verifier manifest requires the constructed DA3 runner")
        runtime = _da3_runtime_settings(manifest, verifier_configs)
        _, cache_source = _effective_da3_cache_dir(runtime["cache_dir"])
        if isinstance(da3_runner, Mapping):
            runtime_by_dataset = {
                str(dataset): _real_da3_runtime_manifest(runner.config, cache_source)
                for dataset, runner in da3_runner.items()
            }
            output["da3_runtime_by_dataset"] = runtime_by_dataset
            output.update(_common_runtime_fields(runtime_by_dataset))
        else:
            runner_runtime = _real_da3_runtime_manifest(
                da3_runner.config,
                cache_source,
            )
            output["da3_runtime_by_dataset"] = {
                str(dataset): dict(runner_runtime) for dataset in verifier_configs
            }
            output.update(runner_runtime)
    else:
        output.update(_da3_provenance_manifest(manifest, verifier_configs))
    return output


def _optional_file_sha256(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def _frozen_salad_provenance(benchmark_root: Path) -> dict[str, object]:
    score_relative = Path("scores") / "salad.jsonl"
    manifest_relative = score_relative.with_suffix(".manifest.json")
    score_path = Path(benchmark_root) / score_relative
    manifest_path = Path(benchmark_root) / manifest_relative
    return {
        "salad_score_file": score_relative.as_posix(),
        "salad_score_file_sha256": _optional_file_sha256(score_path),
        "salad_score_manifest": manifest_relative.as_posix(),
        "salad_score_manifest_sha256": _optional_file_sha256(manifest_path),
    }


def _verifier_config_paths(manifest: Mapping[str, object]) -> dict[str, object]:
    verifier_configs = manifest.get("verifier_configs", {})
    if not isinstance(verifier_configs, Mapping):
        raise ValueError("manifest verifier_configs must be a mapping")
    return dict(verifier_configs)


def _netvlad_checkpoint_path() -> Path:
    import torch

    return (
        Path(torch.hub.get_dir()).expanduser()
        / "netvlad"
        / "VGG16-NetVLAD-Pitts30K.mat"
    )


def _da3_provenance_manifest(
    benchmark_manifest: Mapping[str, object],
    verifier_configs: Mapping[str, object],
) -> dict[str, object]:
    sources = _da3_provenance_sources(benchmark_manifest, verifier_configs)
    model_name = _first_manifest_value(
        sources,
        (
            "da3_model_name",
            "model_name",
            "name",
        ),
    )
    model_name_or_path = _first_manifest_value(
        sources,
        (
            "da3_model_name_or_path",
            "model_name_or_path",
        ),
    )
    if model_name is None and model_name_or_path is not None:
        parsed_model_path = _optional_model_name_or_path_path(model_name_or_path)
        if parsed_model_path is None:
            model_name = str(model_name_or_path)

    model_dir = _resolve_first_path(
        sources,
        (
            "da3_model_dir",
            "model_dir",
        ),
    )
    model_path = _resolve_first_path(
        sources,
        (
            "da3_model_path",
            "model_path",
            "path",
        ),
    )
    if model_path is None and model_name_or_path is not None:
        model_path = _optional_model_name_or_path_path(model_name_or_path)

    checkpoint_path = _resolve_first_path(
        sources,
        (
            "da3_checkpoint",
            "da3_checkpoint_path",
            "checkpoint",
            "checkpoint_path",
        ),
    )
    snapshot_path = _resolve_first_path(
        sources,
        (
            "da3_snapshot",
            "da3_snapshot_path",
            "snapshot",
            "snapshot_path",
        ),
    )
    return {
        "da3_model_name": str(model_name) if model_name is not None else None,
        "da3_model_dir": str(model_dir) if model_dir is not None else None,
        "da3_model_path": str(model_path) if model_path is not None else None,
        "da3_model_path_sha256": _optional_file_sha256(model_path) if model_path else None,
        "da3_checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "da3_checkpoint_sha256": (
            _optional_file_sha256(checkpoint_path) if checkpoint_path is not None else None
        ),
        "da3_snapshot": str(snapshot_path) if snapshot_path is not None else None,
        "da3_snapshot_sha256": (
            _optional_file_sha256(snapshot_path) if snapshot_path is not None else None
        ),
    }


def _real_da3_runtime_manifest(
    runner_config: RealDa3RunnerConfig,
    cache_source: str,
) -> dict[str, object]:
    resolved_model = Path(runner_config.model_name).expanduser()
    if resolved_model.is_dir():
        snapshot_path = resolved_model
        model_path = snapshot_path / "model.safetensors"
        if not model_path.is_file():
            raise FileNotFoundError(
                f"resolved DA3 snapshot is missing model.safetensors: {snapshot_path}"
            )
        model_name = None
    elif resolved_model.is_file():
        snapshot_path = None
        model_path = resolved_model
        model_name = None
    else:
        snapshot_path = None
        model_path = None
        model_name = runner_config.model_name
    return {
        "da3_model_name": model_name,
        "da3_model_dir": None,
        "da3_model_path": str(model_path) if model_path is not None else None,
        "da3_model_path_sha256": (
            _optional_file_sha256(model_path) if model_path is not None else None
        ),
        "da3_checkpoint": None,
        "da3_checkpoint_sha256": None,
        "da3_snapshot": str(snapshot_path) if snapshot_path is not None else None,
        "da3_snapshot_sha256": (
            _optional_file_sha256(snapshot_path) if snapshot_path is not None else None
        ),
        "da3_cache_dir": str(runner_config.cache_dir),
        "da3_cache_source": cache_source,
        "da3_process_res": int(runner_config.process_res),
        "da3_ref_view_strategy": str(runner_config.ref_view_strategy),
        "da3_triplet_batch_size": int(runner_config.triplet_batch_size),
    }


def _common_runtime_fields(
    runtime_by_dataset: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    runtimes = list(runtime_by_dataset.values())
    if not runtimes:
        return {}
    return {
        key: value
        for key, value in runtimes[0].items()
        if all(runtime.get(key) == value for runtime in runtimes[1:])
    }


def _effective_da3_cache_dir(configured_cache_dir: object) -> tuple[Path, str]:
    if configured_cache_dir is not None:
        return Path(str(configured_cache_dir)).expanduser(), "manifest"
    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub_cache:
        return Path(hub_cache).expanduser(), "HUGGINGFACE_HUB_CACHE"
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub", "HF_HOME"
    return Path.home() / ".cache" / "huggingface" / "hub", "default"


def _da3_runtime_settings(
    benchmark_manifest: Mapping[str, object],
    verifier_configs: Mapping[str, object],
) -> dict[str, object]:
    sources = _da3_provenance_sources(benchmark_manifest, verifier_configs)
    cache_dir = _resolve_first_path(
        sources,
        (
            "da3_cache_dir",
            "cache_dir",
            "hf_cache_dir",
            "huggingface_cache_dir",
        ),
    )

    snapshot_path = _resolve_first_path(
        sources,
        (
            "da3_snapshot",
            "da3_snapshot_path",
            "snapshot",
            "snapshot_path",
        ),
    )
    if snapshot_path is not None:
        return {
            "model_name": str(snapshot_path),
            "model_source": "snapshot",
            "cache_dir": cache_dir,
        }

    model_path = _resolve_first_path(
        sources,
        (
            "da3_model_path",
            "model_path",
            "path",
        ),
    )
    if model_path is not None:
        return {
            "model_name": str(model_path),
            "model_source": "model_path",
            "cache_dir": cache_dir,
        }

    model_dir = _resolve_first_path(
        sources,
        (
            "da3_model_dir",
            "model_dir",
        ),
    )
    if model_dir is not None:
        return {
            "model_name": str(model_dir),
            "model_source": "model_dir",
            "cache_dir": cache_dir,
        }

    model_name_or_path = _first_manifest_value(
        sources,
        (
            "da3_model_name_or_path",
            "model_name_or_path",
        ),
    )
    if model_name_or_path is not None:
        parsed_path = _optional_model_name_or_path_path(model_name_or_path)
        if parsed_path is not None:
            return {
                "model_name": str(parsed_path),
                "model_source": "model_path",
                "cache_dir": cache_dir,
            }
        return {
            "model_name": str(model_name_or_path),
            "model_source": "model_name",
            "cache_dir": cache_dir,
        }

    model_name = _first_manifest_value(
        sources,
        (
            "da3_model_name",
            "model_name",
            "name",
        ),
    )
    if model_name is not None:
        return {
            "model_name": str(model_name),
            "model_source": "model_name",
            "cache_dir": cache_dir,
        }

    return {
        "model_name": None,
        "model_source": None,
        "cache_dir": cache_dir,
    }


def _da3_provenance_sources(
    benchmark_manifest: Mapping[str, object],
    verifier_configs: Mapping[str, object],
) -> list[Mapping[str, object]]:
    sources: list[Mapping[str, object]] = [benchmark_manifest]
    for key in ("da3", "model"):
        nested = benchmark_manifest.get(key)
        if isinstance(nested, Mapping):
            sources.append(nested)
    for _, config_path in sorted(verifier_configs.items()):
        config = _read_optional_yaml_mapping(_resolve_repo_path(config_path))
        if config:
            sources.append(config)
        for key in ("da3", "model"):
            nested = config.get(key)
            if isinstance(nested, Mapping):
                sources.append(nested)
    return sources


def _read_optional_yaml_mapping(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        return {}
    try:
        data = read_yaml(path)
    except Exception:
        return {}
    return data if isinstance(data, Mapping) else {}


def _first_manifest_value(
    sources: Sequence[Mapping[str, object]],
    keys: Sequence[str],
) -> object | None:
    for source in sources:
        for key in keys:
            value = source.get(key)
            if value is not None:
                return value
    return None


def _resolve_first_path(
    sources: Sequence[Mapping[str, object]],
    keys: Sequence[str],
) -> Path | None:
    value = _first_manifest_value(sources, keys)
    if value is None:
        return None
    return _resolve_manifest_path(value)


def _optional_model_name_or_path_path(value: object) -> Path | None:
    text = str(value)
    path = Path(text).expanduser()
    if path.exists() or path.is_absolute() or text.startswith((".", "~")):
        return _resolve_manifest_path(value)
    return None


def _salad_config(benchmark_root: Path, device: str) -> dict[str, object]:
    manifest = read_json(Path(benchmark_root) / "manifest.json")
    salad = manifest.get("salad")
    salad_manifest = salad if isinstance(salad, Mapping) else {}
    default_repo = LOOPANYTHING_ROOT / "da3_streaming" / "loop_utils" / "salad"
    repo = _resolve_manifest_path(
        _manifest_value(
            manifest,
            salad_manifest,
            flat_keys=("salad_repo",),
            nested_keys=("repo", "salad_repo"),
            default=default_repo,
        )
    )
    checkpoint = _resolve_manifest_path(
        _manifest_value(
            manifest,
            salad_manifest,
            flat_keys=("salad_checkpoint", "salad_checkpoint_path"),
            nested_keys=("checkpoint", "checkpoint_path"),
            default=repo / "weights" / "dino_salad.ckpt",
        )
    )
    config_path = _manifest_value(
        manifest,
        salad_manifest,
        flat_keys=("salad_config", "salad_config_path"),
        nested_keys=("config", "config_path"),
        default=None,
    )
    resolved_config_path = (
        _resolve_manifest_path(config_path) if config_path is not None else None
    )
    return {
        "repo": repo,
        "checkpoint": checkpoint,
        "backbone": _manifest_value(
            manifest,
            salad_manifest,
            flat_keys=("salad_backbone",),
            nested_keys=("backbone",),
            default="dinov2_vitb14",
        ),
        "batch_size": int(
            _manifest_value(
                manifest,
                salad_manifest,
                flat_keys=("salad_batch_size",),
                nested_keys=("batch_size",),
                default=32,
            )
        ),
        "device": device,
        "config": resolved_config_path,
    }


def _manifest_value(
    manifest: Mapping[str, object],
    nested: Mapping[str, object],
    *,
    flat_keys: Sequence[str],
    nested_keys: Sequence[str],
    default: object,
) -> object:
    for key in flat_keys:
        value = manifest.get(key)
        if value is not None:
            return value
    for key in nested_keys:
        value = nested.get(key)
        if value is not None:
            return value
    return default


def _optional_resolved_path(path: object) -> Path | None:
    if path is None:
        return None
    return _resolve_manifest_path(path)


def _resolve_manifest_path(path: object) -> Path:
    parsed = Path(str(path))
    return parsed if parsed.is_absolute() else LOOPANYTHING_ROOT / parsed


def _optional_git_commit(path: Path | None) -> str | None:
    if path is None or not Path(path).exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _resolve_repo_path(path: object) -> Path:
    parsed = Path(str(path))
    return parsed if parsed.is_absolute() else LOOPANYTHING_ROOT / parsed


def _source_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=LOOPANYTHING_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _source_provenance() -> dict[str, object]:
    provenance = {
        "source_commit": _source_commit(),
        "source_file_sha256": {
            relative_path: sha256_file(LOOPANYTHING_ROOT / relative_path)
            for relative_path in SOURCE_IMPLEMENTATION_PATHS
        },
    }
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=LOOPANYTHING_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception as exc:
        provenance["source_tree_dirty"] = "unknown"
        provenance["source_tree_status_error"] = f"{type(exc).__name__}: {exc}"
    else:
        provenance["source_tree_dirty"] = bool(result.stdout.strip())
    return provenance


class _MockDescriptorBackend:
    def compute(self, image_paths: Sequence[str], keyframe_indices: Sequence[int]):
        from robust_loop_verifier.retrieval import DescriptorSet

        descriptors = []
        for index in keyframe_indices:
            phase = float(int(index) + 1)
            descriptors.append([math.cos(phase), math.sin(phase)])
        return DescriptorSet(
            keyframe_indices=[int(index) for index in keyframe_indices],
            descriptors=np.asarray(descriptors, dtype=np.float64),
        )


if __name__ == "__main__":
    raise SystemExit(main())
