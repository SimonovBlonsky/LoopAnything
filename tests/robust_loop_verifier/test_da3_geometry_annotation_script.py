from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _load_script():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "robust_loop_verification_scripts"
        / "build_da3_geometry_annotation.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_da3_geometry_annotation",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_source_bundle(
    tmp_path: Path,
    pair_count: int = 3,
    *,
    declared_pair_hash: object = ...,
) -> tuple[Path, list[dict]]:
    benchmark_root = tmp_path / "benchmark"
    benchmark_root.mkdir()
    pairs = [
        {
            "pair_id": f"p{index + 1}",
            "dataset": "dataset",
            "platform": "platform",
            "sequence": "sequence",
            "query_idx": 10 + index // 2,
            "candidate_idx": index,
            "rank": index % 2 + 1,
            "dbow2_score": 0.9 - 0.1 * index,
        }
        for index in range(pair_count)
    ]
    pair_content = "".join(
        json.dumps(pair, separators=(",", ":"), sort_keys=False) + "\n" for pair in pairs
    )
    pair_path = benchmark_root / "benchmark_pairs.jsonl"
    pair_path.write_text(
        pair_content,
        encoding="utf-8",
    )
    actual_pair_hash = hashlib.sha256(pair_path.read_bytes()).hexdigest()
    manifest = {
        "benchmark_version": "test-v1",
        "sequences": [],
    }
    if declared_pair_hash is ...:
        manifest["pair_manifest_sha256"] = actual_pair_hash
    elif declared_pair_hash is not None:
        manifest["pair_manifest_sha256"] = declared_pair_hash
    (benchmark_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return benchmark_root, pairs


def _record(pair: dict, *, loop_factor=None, **extra) -> dict:
    identity = np.eye(4, dtype=np.float64)
    payload = {
        **pair,
        "sequence_key": "dataset/platform/sequence",
        "gt_query_pose": identity.reshape(-1).tolist(),
        "gt_candidate_pose": identity.reshape(-1).tolist(),
        "support_idx": pair["candidate_idx"] + 1,
        "support_rejection_reason": None,
        "support_baseline_m": 1.5,
        "sim3_valid": loop_factor is not None,
        "sim3_scale": 1.0 if loop_factor is not None else None,
        "sim3_support_alignment_residual_m": 0.02 if loop_factor is not None else None,
        "sim3_direction_error_deg": 1.0 if loop_factor is not None else None,
        "sim3_rejection_reason": None if loop_factor is not None else "failed",
        "loop_factor": loop_factor,
        "failure_reasons": [] if loop_factor is not None else ["sim3: failed"],
    }
    payload.update(extra)
    return payload


def _runtime(monkeypatch, module) -> None:
    monkeypatch.setattr(
        module,
        "_build_runtime",
        lambda benchmark_root, backend, device: (
            {"dataset": object()},
            object(),
            {
                "backend": backend,
                "device": device,
                "verifier_configs": {"dataset": "config.yaml"},
                "verifier_config_sha256": {"dataset": "a" * 64},
                "salad_score_file": "scores/salad.jsonl",
                "salad_score_file_sha256": "b" * 64,
                "salad_score_manifest": "scores/salad.manifest.json",
                "salad_score_manifest_sha256": "c" * 64,
                "da3_model_name": None,
                "da3_model_dir": None,
                "da3_model_path": None,
                "da3_model_path_sha256": None,
                "da3_checkpoint": None,
                "da3_checkpoint_sha256": None,
                "da3_snapshot": None,
                "da3_snapshot_sha256": None,
                "da3_runtime_by_dataset": {
                    "dataset": {
                        "process_res": 504,
                        "ref_view_strategy": "first",
                        "triplet_batch_size": 4,
                    }
                },
            },
        ),
    )
    monkeypatch.setattr(
        module,
        "_source_provenance",
        lambda: {"source_commit": "test-commit"},
    )


def _run(
    module,
    benchmark_root: Path,
    output_root: Path,
    *extra_args: str,
) -> int:
    return module.main(
        [
            str(benchmark_root),
            "--output-root",
            str(output_root),
            "--backend",
            "mock",
            "--device",
            "cpu",
            *extra_args,
        ]
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _read_manifest(output_root: Path) -> dict:
    return json.loads(
        (output_root / "geometry_prediction_manifest.json").read_text(encoding="utf-8")
    )


def _write_manifest(output_root: Path, manifest: dict) -> None:
    (output_root / "geometry_prediction_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_complete_bundle(tmp_path, monkeypatch, module, *, pair_count=1):
    benchmark_root, pairs = _write_source_bundle(tmp_path, pair_count=pair_count)
    output_root = tmp_path / "output"
    _runtime(monkeypatch, module)
    monkeypatch.setattr(
        module,
        "iter_geometry_factor_batches",
        lambda *args, **kwargs: [
            [_record(pair, loop_factor=np.eye(4).reshape(-1)) for pair in pairs]
        ],
    )
    assert _run(module, benchmark_root, output_root) == 0
    return benchmark_root, output_root, pairs


def test_rejects_output_root_equal_to_benchmark_before_write_or_runtime(
    tmp_path,
    monkeypatch,
):
    module = _load_script()
    benchmark_root, _ = _write_source_bundle(tmp_path, pair_count=1)
    source_snapshot = {
        path.name: path.read_bytes() for path in benchmark_root.iterdir() if path.is_file()
    }
    monkeypatch.setattr(
        module,
        "_copy_frozen_source_once",
        lambda *args, **kwargs: pytest.fail("must not copy frozen sources"),
    )
    monkeypatch.setattr(
        module,
        "_build_runtime",
        lambda *args, **kwargs: pytest.fail("runtime must not be built"),
    )

    with pytest.raises(ValueError, match="output_root.*benchmark_root"):
        _run(module, benchmark_root, benchmark_root)

    assert {
        path.name: path.read_bytes() for path in benchmark_root.iterdir() if path.is_file()
    } == source_snapshot


@pytest.mark.parametrize(
    ("declared_pair_hash", "error"),
    [
        (None, "pair_manifest_sha256.*non-empty string"),
        (123, "pair_manifest_sha256.*non-empty string"),
        ("wrong-hash", "pair_manifest_sha256.*does not match"),
    ],
)
def test_rejects_invalid_source_manifest_pair_hash_before_runtime(
    tmp_path,
    monkeypatch,
    declared_pair_hash,
    error,
):
    module = _load_script()
    benchmark_root, _ = _write_source_bundle(
        tmp_path,
        pair_count=1,
        declared_pair_hash=declared_pair_hash,
    )
    monkeypatch.setattr(
        module,
        "_build_runtime",
        lambda *args, **kwargs: pytest.fail("runtime must not be built"),
    )

    with pytest.raises(ValueError, match=error):
        _run(module, benchmark_root, tmp_path / "output")


def test_copies_frozen_source_files_exactly_once(tmp_path, monkeypatch):
    module = _load_script()
    benchmark_root, pairs = _write_source_bundle(tmp_path, pair_count=1)
    output_root = tmp_path / "output"
    _runtime(monkeypatch, module)
    monkeypatch.setattr(
        module,
        "iter_geometry_factor_batches",
        lambda *args, **kwargs: [[_record(pairs[0], loop_factor=np.eye(4).reshape(-1))]],
    )

    assert _run(module, benchmark_root, output_root) == 0
    copied_stats = {
        name: (output_root / name).stat() for name in ("benchmark_pairs.jsonl", "manifest.json")
    }
    assert _run(module, benchmark_root, output_root) == 0

    for name in ("benchmark_pairs.jsonl", "manifest.json"):
        source_path = benchmark_root / name
        output_path = output_root / name
        assert output_path.read_bytes() == source_path.read_bytes()
        assert output_path.stat().st_ino == copied_stats[name].st_ino
        assert output_path.stat().st_mtime_ns == copied_stats[name].st_mtime_ns


def test_complete_resume_validates_and_returns_without_building_runtime(
    tmp_path,
    monkeypatch,
):
    module = _load_script()
    benchmark_root, output_root, _ = _build_complete_bundle(
        tmp_path,
        monkeypatch,
        module,
    )
    monkeypatch.setattr(
        module,
        "_build_runtime",
        lambda *args, **kwargs: pytest.fail("runtime must not be built"),
    )
    monkeypatch.setattr(
        module,
        "iter_geometry_factor_batches",
        lambda *args, **kwargs: pytest.fail("iterator must not run"),
    )

    assert _run(module, benchmark_root, output_root) == 0


@pytest.mark.parametrize(
    "changed_args",
    [
        ("--backend", "real"),
        ("--device", "cuda"),
        ("--pair-limit", "1"),
        ("--min-translation-error-m", "0.5"),
        ("--max-translation-error-m", "6.0"),
        ("--translation-error-scale-ratio", "0.3"),
        ("--max-rotation-error-deg", "16.0"),
        ("--max-translation-direction-error-deg", "21.0"),
        ("--min-direction-baseline-m", "0.6"),
    ],
)
def test_complete_resume_rejects_changed_request_before_runtime(
    tmp_path,
    monkeypatch,
    changed_args,
):
    module = _load_script()
    benchmark_root, output_root, _ = _build_complete_bundle(
        tmp_path,
        monkeypatch,
        module,
    )
    monkeypatch.setattr(
        module,
        "_build_runtime",
        lambda *args, **kwargs: pytest.fail("runtime must not be built"),
    )

    with pytest.raises(ValueError, match="request_contract|contract mismatch"):
        _run(module, benchmark_root, output_root, *changed_args)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("complete", "false", "complete must be a bool"),
        ("format_version", 2, "format_version"),
        ("prediction_file", "other.jsonl", "prediction_file"),
        ("source_pair_count", 2, "source_pair_count"),
        ("target_pair_count", 2, "target_pair_count"),
        ("target_pair_ids_sha256", "wrong", "target_pair_ids_sha256"),
        ("source_pair_manifest_sha256", "wrong", "source_pair_manifest_sha256"),
        ("source_manifest_sha256", "wrong", "source_manifest_sha256"),
        ("pair_limit", 1, "pair_limit"),
        ("source_commit", 123, "source_commit"),
        ("command", "pytest", "command"),
        ("prediction_record_count", 0, "prediction_record_count"),
        ("prediction_file_sha256", "wrong", "prediction_file_sha256"),
    ],
)
def test_complete_resume_strictly_validates_manifest_before_runtime(
    tmp_path,
    monkeypatch,
    field,
    value,
    error,
):
    module = _load_script()
    benchmark_root, output_root, _ = _build_complete_bundle(
        tmp_path,
        monkeypatch,
        module,
    )
    manifest = _read_manifest(output_root)
    manifest[field] = value
    _write_manifest(output_root, manifest)
    monkeypatch.setattr(
        module,
        "_build_runtime",
        lambda *args, **kwargs: pytest.fail("runtime must not be built"),
    )

    with pytest.raises(ValueError, match=error):
        _run(module, benchmark_root, output_root)


def test_complete_resume_allows_changed_valid_provenance_without_runtime(
    tmp_path,
    monkeypatch,
):
    module = _load_script()
    benchmark_root, output_root, _ = _build_complete_bundle(
        tmp_path,
        monkeypatch,
        module,
    )
    manifest = _read_manifest(output_root)
    manifest["source_commit"] = "another-valid-commit"
    manifest["command"] = ["another", "valid", "command"]
    _write_manifest(output_root, manifest)
    monkeypatch.setattr(
        module,
        "_build_runtime",
        lambda *args, **kwargs: pytest.fail("runtime must not be built"),
    )

    assert _run(module, benchmark_root, output_root) == 0


@pytest.mark.parametrize("field", ["thresholds", "request_contract"])
def test_complete_resume_rejects_changed_nested_contract_before_runtime(
    tmp_path,
    monkeypatch,
    field,
):
    module = _load_script()
    benchmark_root, output_root, _ = _build_complete_bundle(
        tmp_path,
        monkeypatch,
        module,
    )
    manifest = _read_manifest(output_root)
    if field == "thresholds":
        manifest[field]["max_translation_error_m"] = 2.0
    else:
        manifest[field]["backend"] = "real"
    _write_manifest(output_root, manifest)
    monkeypatch.setattr(
        module,
        "_build_runtime",
        lambda *args, **kwargs: pytest.fail("runtime must not be built"),
    )

    with pytest.raises(ValueError, match=field):
        _run(module, benchmark_root, output_root)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda config: config.clear(), "da3_config"),
        (lambda config: config.pop("da3_model_name"), "da3_model_name"),
        (lambda config: config.update(backend="real"), "backend"),
        (lambda config: config.update(device="cuda"), "device"),
        (
            lambda config: config["da3_runtime_by_dataset"]["dataset"].update(process_res=True),
            "process_res",
        ),
        (
            lambda config: config["da3_runtime_by_dataset"]["dataset"].update(
                ref_view_strategy=""
            ),
            "ref_view_strategy",
        ),
        (
            lambda config: config["da3_runtime_by_dataset"]["dataset"].update(
                triplet_batch_size=1.0
            ),
            "triplet_batch_size",
        ),
        (
            lambda config: config.update(da3_model_path=123),
            "da3_model_path",
        ),
        (
            lambda config: config["verifier_config_sha256"].update(dataset=123),
            "verifier_config_sha256",
        ),
    ],
)
def test_complete_resume_strictly_validates_da3_config_before_runtime(
    tmp_path,
    monkeypatch,
    mutation,
    error,
):
    module = _load_script()
    benchmark_root, output_root, _ = _build_complete_bundle(
        tmp_path,
        monkeypatch,
        module,
    )
    manifest = _read_manifest(output_root)
    mutation(manifest["da3_config"])
    manifest["da3_config_sha256"] = _canonical_sha256(manifest["da3_config"])
    _write_manifest(output_root, manifest)
    monkeypatch.setattr(
        module,
        "_build_runtime",
        lambda *args, **kwargs: pytest.fail("runtime must not be built"),
    )

    with pytest.raises(ValueError, match=error):
        _run(module, benchmark_root, output_root)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda config: config.pop("da3_model_name"),
        lambda config: config["da3_runtime_by_dataset"]["dataset"].update(process_res=672),
        lambda config: config["verifier_config_sha256"].update(dataset="d" * 64),
    ],
)
def test_complete_resume_rejects_da3_config_tamper_without_updated_hash(
    tmp_path,
    monkeypatch,
    mutation,
):
    module = _load_script()
    benchmark_root, output_root, _ = _build_complete_bundle(
        tmp_path,
        monkeypatch,
        module,
    )
    manifest = _read_manifest(output_root)
    original_hash = manifest["da3_config_sha256"]
    mutation(manifest["da3_config"])
    assert _canonical_sha256(manifest["da3_config"]) != original_hash
    _write_manifest(output_root, manifest)
    monkeypatch.setattr(
        module,
        "_build_runtime",
        lambda *args, **kwargs: pytest.fail("runtime must not be built"),
    )

    with pytest.raises(ValueError, match="da3_config_sha256"):
        _run(module, benchmark_root, output_root)


@pytest.mark.parametrize("bad_hash", ["", "not-hex", "a" * 63, "g" * 64])
def test_complete_resume_rejects_invalid_da3_config_hash(
    tmp_path,
    monkeypatch,
    bad_hash,
):
    module = _load_script()
    benchmark_root, output_root, _ = _build_complete_bundle(
        tmp_path,
        monkeypatch,
        module,
    )
    manifest = _read_manifest(output_root)
    manifest["da3_config_sha256"] = bad_hash
    _write_manifest(output_root, manifest)
    monkeypatch.setattr(
        module,
        "_build_runtime",
        lambda *args, **kwargs: pytest.fail("runtime must not be built"),
    )

    with pytest.raises(ValueError, match="da3_config_sha256"):
        _run(module, benchmark_root, output_root)


def test_complete_resume_rejects_float_pair_limit_even_when_numerically_equal(
    tmp_path,
    monkeypatch,
):
    module = _load_script()
    benchmark_root, pairs = _write_source_bundle(tmp_path, pair_count=2)
    output_root = tmp_path / "output"
    _runtime(monkeypatch, module)
    monkeypatch.setattr(
        module,
        "iter_geometry_factor_batches",
        lambda *args, **kwargs: [[_record(pairs[0], loop_factor=np.eye(4).reshape(-1))]],
    )
    assert _run(module, benchmark_root, output_root, "--pair-limit", "1") == 0
    manifest = _read_manifest(output_root)
    manifest["pair_limit"] = 1.0
    manifest["request_contract"]["pair_limit"] = 1.0
    _write_manifest(output_root, manifest)
    monkeypatch.setattr(
        module,
        "_build_runtime",
        lambda *args, **kwargs: pytest.fail("runtime must not be built"),
    )

    with pytest.raises(ValueError, match="pair_limit"):
        _run(module, benchmark_root, output_root, "--pair-limit", "1")


def test_complete_resume_rejects_integer_threshold_even_when_numerically_equal(
    tmp_path,
    monkeypatch,
):
    module = _load_script()
    benchmark_root, output_root, _ = _build_complete_bundle(
        tmp_path,
        monkeypatch,
        module,
    )
    manifest = _read_manifest(output_root)
    manifest["thresholds"]["max_translation_error_m"] = 1
    manifest["request_contract"]["thresholds"]["max_translation_error_m"] = 1
    _write_manifest(output_root, manifest)
    monkeypatch.setattr(
        module,
        "_build_runtime",
        lambda *args, **kwargs: pytest.fail("runtime must not be built"),
    )

    with pytest.raises(ValueError, match="max_translation_error_m"):
        _run(module, benchmark_root, output_root)


@pytest.mark.parametrize("filename", ["benchmark_pairs.jsonl", "manifest.json"])
def test_rejects_existing_output_bundle_that_differs_from_source(
    tmp_path,
    monkeypatch,
    filename,
):
    module = _load_script()
    benchmark_root, _ = _write_source_bundle(tmp_path, pair_count=1)
    output_root = tmp_path / "output"
    output_root.mkdir()
    for source_name in ("benchmark_pairs.jsonl", "manifest.json"):
        content = (benchmark_root / source_name).read_bytes()
        (output_root / source_name).write_bytes(
            b"changed\n" if source_name == filename else content
        )
    _runtime(monkeypatch, module)

    with pytest.raises(ValueError, match=f"frozen source copy mismatch.*{filename}"):
        _run(module, benchmark_root, output_root)


@pytest.mark.parametrize(
    "filename",
    [
        "benchmark_pairs.jsonl",
        "manifest.json",
        "geometry_predictions.jsonl",
        "geometry_prediction_manifest.json",
        ".build_da3_geometry_annotation.lock",
    ],
)
def test_rejects_symlinked_critical_output_paths_without_write_through(
    tmp_path,
    monkeypatch,
    filename,
):
    module = _load_script()
    benchmark_root, _ = _write_source_bundle(tmp_path, pair_count=1)
    output_root = tmp_path / "output"
    output_root.mkdir()
    external = tmp_path / f"external-{filename.replace('/', '-')}"
    external.write_text("sentinel\n", encoding="utf-8")
    (output_root / filename).symlink_to(external)
    monkeypatch.setattr(
        module,
        "_build_runtime",
        lambda *args, **kwargs: pytest.fail("runtime must not be built"),
    )

    with pytest.raises(ValueError, match="symlink"):
        _run(module, benchmark_root, output_root)

    assert external.read_text(encoding="utf-8") == "sentinel\n"


@pytest.mark.parametrize("filename", ["benchmark_pairs.jsonl", "manifest.json"])
def test_rejects_frozen_source_hardlinks(tmp_path, monkeypatch, filename):
    module = _load_script()
    benchmark_root, _ = _write_source_bundle(tmp_path, pair_count=1)
    output_root = tmp_path / "output"
    output_root.mkdir()
    os.link(benchmark_root / filename, output_root / filename)
    monkeypatch.setattr(
        module,
        "_build_runtime",
        lambda *args, **kwargs: pytest.fail("runtime must not be built"),
    )

    with pytest.raises(ValueError, match="hardlink|same file"):
        _run(module, benchmark_root, output_root)


def test_rejects_prediction_hardlink_without_write_through(tmp_path, monkeypatch):
    module = _load_script()
    benchmark_root, _ = _write_source_bundle(tmp_path, pair_count=1)
    output_root = tmp_path / "output"
    output_root.mkdir()
    external = tmp_path / "external-predictions.jsonl"
    external.write_bytes(b"")
    os.link(external, output_root / "geometry_predictions.jsonl")
    monkeypatch.setattr(
        module,
        "_build_runtime",
        lambda *args, **kwargs: pytest.fail("runtime must not be built"),
    )

    with pytest.raises(ValueError, match="hardlink"):
        _run(module, benchmark_root, output_root)

    assert external.read_bytes() == b""


def test_second_builder_fails_while_bundle_lock_is_held(tmp_path, monkeypatch):
    module = _load_script()
    benchmark_root, _ = _write_source_bundle(tmp_path, pair_count=1)
    output_root = tmp_path / "output"
    output_root.mkdir()
    monkeypatch.setattr(
        module,
        "_build_runtime",
        lambda *args, **kwargs: pytest.fail("runtime must not be built"),
    )

    with module._bundle_lock(output_root):
        with pytest.raises(RuntimeError, match="already locked|another builder"):
            _run(module, benchmark_root, output_root)


def test_uses_frozen_output_snapshot_after_copy(tmp_path, monkeypatch):
    module = _load_script()
    benchmark_root, pairs = _write_source_bundle(tmp_path, pair_count=1)
    output_root = tmp_path / "output"
    original_pair_bytes = (benchmark_root / "benchmark_pairs.jsonl").read_bytes()
    original_pair_hash = hashlib.sha256(original_pair_bytes).hexdigest()
    real_copy = module._copy_frozen_source_once
    copy_count = 0

    def copy_then_mutate_source(*args, **kwargs):
        nonlocal copy_count
        result = real_copy(*args, **kwargs)
        copy_count += 1
        if copy_count == 2:
            changed_pair = {**pairs[0], "pair_id": "changed-after-snapshot"}
            changed_bytes = (json.dumps(changed_pair, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
            (benchmark_root / "benchmark_pairs.jsonl").write_bytes(changed_bytes)
            changed_manifest = {
                "benchmark_version": "changed",
                "sequences": [],
                "pair_manifest_sha256": hashlib.sha256(changed_bytes).hexdigest(),
            }
            (benchmark_root / "manifest.json").write_text(
                json.dumps(changed_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return result

    monkeypatch.setattr(module, "_copy_frozen_source_once", copy_then_mutate_source)
    monkeypatch.setattr(
        module,
        "_build_runtime",
        lambda root, backend, device: (
            pytest.fail("runtime did not receive output snapshot")
            if Path(root) != output_root
            else (
                {"dataset": object()},
                object(),
                {
                    "backend": backend,
                    "device": device,
                    "verifier_configs": {"dataset": "config.yaml"},
                    "verifier_config_sha256": {"dataset": "a" * 64},
                    "salad_score_file": None,
                    "salad_score_file_sha256": None,
                    "salad_score_manifest": None,
                    "salad_score_manifest_sha256": None,
                    "da3_model_name": None,
                    "da3_model_dir": None,
                    "da3_model_path": None,
                    "da3_model_path_sha256": None,
                    "da3_checkpoint": None,
                    "da3_checkpoint_sha256": None,
                    "da3_snapshot": None,
                    "da3_snapshot_sha256": None,
                    "da3_runtime_by_dataset": {
                        "dataset": {
                            "process_res": 504,
                            "ref_view_strategy": "first",
                            "triplet_batch_size": 4,
                        }
                    },
                },
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "_source_provenance",
        lambda: {"source_commit": "test-commit"},
    )

    def batches(root, *args, **kwargs):
        assert Path(root) == output_root
        yield [_record(pairs[0], loop_factor=np.eye(4).reshape(-1))]

    monkeypatch.setattr(module, "iter_geometry_factor_batches", batches)

    assert _run(module, benchmark_root, output_root) == 0
    assert (output_root / "benchmark_pairs.jsonl").read_bytes() == original_pair_bytes
    manifest = _read_manifest(output_root)
    assert manifest["source_pair_manifest_sha256"] == original_pair_hash


def test_fsyncs_each_query_batch_before_requesting_the_next(
    tmp_path,
    monkeypatch,
):
    module = _load_script()
    benchmark_root, pairs = _write_source_bundle(tmp_path, pair_count=3)
    output_root = tmp_path / "output"
    _runtime(monkeypatch, module)
    prediction_fsyncs = []
    real_fsync = os.fsync

    def tracking_fsync(descriptor):
        try:
            target = os.readlink(f"/proc/self/fd/{descriptor}")
        except OSError:
            target = ""
        if target.endswith("/geometry_predictions.jsonl"):
            prediction_fsyncs.append(target)
        return real_fsync(descriptor)

    monkeypatch.setattr(module.os, "fsync", tracking_fsync)

    def batches(*args, **kwargs):
        yield [_record(pair, loop_factor=np.eye(4).reshape(-1)) for pair in pairs[:2]]
        assert len(prediction_fsyncs) == 1
        yield [_record(pairs[2], loop_factor=np.eye(4).reshape(-1))]

    monkeypatch.setattr(module, "iter_geometry_factor_batches", batches)

    assert _run(module, benchmark_root, output_root) == 0
    assert len(prediction_fsyncs) == 2


def test_append_prediction_batch_fsyncs_parent_directory(tmp_path, monkeypatch):
    module = _load_script()
    prediction_path = tmp_path / "geometry_predictions.jsonl"
    fsynced_directories = []
    monkeypatch.setattr(
        module,
        "_fsync_directory",
        lambda path: fsynced_directories.append(Path(path)),
    )

    module._append_prediction_batch(prediction_path, [{"pair_id": "p1"}])

    assert fsynced_directories == [tmp_path]


def test_append_prediction_batch_propagates_parent_directory_fsync_failure(
    tmp_path,
    monkeypatch,
):
    module = _load_script()
    prediction_path = tmp_path / "geometry_predictions.jsonl"

    def fail_directory_fsync(path):
        raise OSError("injected directory fsync failure")

    monkeypatch.setattr(module, "_fsync_directory", fail_directory_fsync)

    with pytest.raises(OSError, match="injected directory fsync failure"):
        module._append_prediction_batch(prediction_path, [{"pair_id": "p1"}])


def test_restart_skips_completed_pair_ids_without_duplicates(tmp_path, monkeypatch):
    module = _load_script()
    benchmark_root, pairs = _write_source_bundle(tmp_path, pair_count=3)
    output_root = tmp_path / "output"
    _runtime(monkeypatch, module)

    def interrupted(*args, **kwargs):
        yield [_record(pair, loop_factor=np.eye(4).reshape(-1)) for pair in pairs[:2]]
        raise RuntimeError("injected interruption")

    monkeypatch.setattr(module, "iter_geometry_factor_batches", interrupted)
    with pytest.raises(RuntimeError, match="injected interruption"):
        _run(module, benchmark_root, output_root)

    observed_completed = []

    def resumed(*args, **kwargs):
        observed_completed.append(set(kwargs["completed_pair_ids"]))
        yield [_record(pairs[2], loop_factor=np.eye(4).reshape(-1))]

    monkeypatch.setattr(module, "iter_geometry_factor_batches", resumed)
    assert _run(module, benchmark_root, output_root) == 0

    rows = _read_jsonl(output_root / "geometry_predictions.jsonl")
    assert observed_completed == [{"p1", "p2"}]
    assert [row["pair_id"] for row in rows] == ["p1", "p2", "p3"]
    assert len({row["pair_id"] for row in rows}) == 3


def test_each_checkpoint_updates_incomplete_manifest_count_and_hash(
    tmp_path,
    monkeypatch,
):
    module = _load_script()
    benchmark_root, pairs = _write_source_bundle(tmp_path, pair_count=2)
    output_root = tmp_path / "output"
    _runtime(monkeypatch, module)

    def interrupted(*args, **kwargs):
        yield [_record(pairs[0], loop_factor=np.eye(4).reshape(-1))]
        raise RuntimeError("stop")

    monkeypatch.setattr(module, "iter_geometry_factor_batches", interrupted)
    with pytest.raises(RuntimeError, match="stop"):
        _run(module, benchmark_root, output_root)

    manifest = _read_manifest(output_root)
    prediction_path = output_root / "geometry_predictions.jsonl"
    assert manifest["complete"] is False
    assert manifest["prediction_record_count"] == 1
    assert (
        manifest["prediction_file_sha256"]
        == hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize("durable_row_count", [1, 2])
def test_resume_discards_uncommitted_query_batch_rows(
    tmp_path,
    monkeypatch,
    durable_row_count,
):
    module = _load_script()
    benchmark_root, pairs = _write_source_bundle(tmp_path, pair_count=3)
    output_root = tmp_path / "output"
    _runtime(monkeypatch, module)
    real_append_prediction_batch = module._append_prediction_batch
    real_write_json_atomic = module._write_json_atomic
    if durable_row_count == 1:

        def append_partial_batch(path, rows):
            real_append_prediction_batch(path, rows[:1])
            raise OSError("injected append failure")

        monkeypatch.setattr(module, "_append_prediction_batch", append_partial_batch)
    else:
        manifest_writes = 0

        def fail_first_checkpoint(path, payload):
            nonlocal manifest_writes
            manifest_writes += 1
            if manifest_writes == 2:
                raise OSError("injected checkpoint failure")
            return real_write_json_atomic(path, payload)

        monkeypatch.setattr(module, "_write_json_atomic", fail_first_checkpoint)
    monkeypatch.setattr(
        module,
        "iter_geometry_factor_batches",
        lambda *args, **kwargs: [
            [_record(pair, loop_factor=np.eye(4).reshape(-1)) for pair in pairs[:2]]
        ],
    )

    with pytest.raises(OSError, match="injected (append|checkpoint) failure"):
        _run(module, benchmark_root, output_root)

    assert [row["pair_id"] for row in _read_jsonl(output_root / "geometry_predictions.jsonl")] == [
        pair["pair_id"] for pair in pairs[:durable_row_count]
    ]
    stale_manifest = _read_manifest(output_root)
    assert stale_manifest["prediction_record_count"] == 0
    assert stale_manifest["prediction_file_sha256"] is None

    observed_completed = []

    def resumed(*args, **kwargs):
        observed_completed.append(set(kwargs["completed_pair_ids"]))
        assert _read_jsonl(output_root / "geometry_predictions.jsonl") == []
        yield [_record(pair, loop_factor=np.eye(4).reshape(-1)) for pair in pairs[:2]]
        yield [_record(pairs[2], loop_factor=np.eye(4).reshape(-1))]

    monkeypatch.setattr(module, "_append_prediction_batch", real_append_prediction_batch)
    monkeypatch.setattr(module, "_write_json_atomic", real_write_json_atomic)
    monkeypatch.setattr(module, "iter_geometry_factor_batches", resumed)

    assert _run(module, benchmark_root, output_root) == 0
    assert observed_completed == [set()]
    rows = _read_jsonl(output_root / "geometry_predictions.jsonl")
    assert [row["pair_id"] for row in rows] == ["p1", "p2", "p3"]
    assert len({row["pair_id"] for row in rows}) == 3


def test_incomplete_resume_strictly_validates_checkpoint_before_runtime(
    tmp_path,
    monkeypatch,
):
    module = _load_script()
    benchmark_root, pairs = _write_source_bundle(tmp_path, pair_count=2)
    output_root = tmp_path / "output"
    _runtime(monkeypatch, module)

    def interrupted(*args, **kwargs):
        yield [_record(pairs[0], loop_factor=np.eye(4).reshape(-1))]
        raise RuntimeError("stop")

    monkeypatch.setattr(module, "iter_geometry_factor_batches", interrupted)
    with pytest.raises(RuntimeError, match="stop"):
        _run(module, benchmark_root, output_root)
    manifest = _read_manifest(output_root)
    manifest["prediction_file_sha256"] = "wrong"
    _write_manifest(output_root, manifest)
    monkeypatch.setattr(
        module,
        "_build_runtime",
        lambda *args, **kwargs: pytest.fail("runtime must not be built"),
    )

    with pytest.raises(ValueError, match="prediction_file_sha256"):
        _run(module, benchmark_root, output_root)


def test_incomplete_resume_rejects_manifest_count_ahead_of_predictions(
    tmp_path,
    monkeypatch,
):
    module = _load_script()
    benchmark_root, pairs = _write_source_bundle(tmp_path, pair_count=2)
    output_root = tmp_path / "output"
    _runtime(monkeypatch, module)

    def interrupted(*args, **kwargs):
        yield [_record(pairs[0], loop_factor=np.eye(4).reshape(-1))]
        raise RuntimeError("stop")

    monkeypatch.setattr(module, "iter_geometry_factor_batches", interrupted)
    with pytest.raises(RuntimeError, match="stop"):
        _run(module, benchmark_root, output_root)
    manifest = _read_manifest(output_root)
    manifest["prediction_record_count"] = 2
    _write_manifest(output_root, manifest)
    monkeypatch.setattr(
        module,
        "_build_runtime",
        lambda *args, **kwargs: pytest.fail("runtime must not be built"),
    )

    with pytest.raises(ValueError, match="prediction_record_count"):
        _run(module, benchmark_root, output_root)


def test_rejects_partial_final_prediction_line(tmp_path, monkeypatch):
    module = _load_script()
    benchmark_root, pairs = _write_source_bundle(tmp_path, pair_count=2)
    output_root = tmp_path / "output"
    _runtime(monkeypatch, module)

    def interrupted(*args, **kwargs):
        yield [_record(pairs[0], loop_factor=np.eye(4).reshape(-1))]
        raise RuntimeError("stop")

    monkeypatch.setattr(module, "iter_geometry_factor_batches", interrupted)
    with pytest.raises(RuntimeError, match="stop"):
        _run(module, benchmark_root, output_root)
    with (output_root / "geometry_predictions.jsonl").open("ab") as handle:
        handle.write(b'{"pair_id":"partial"}')

    monkeypatch.setattr(
        module,
        "iter_geometry_factor_batches",
        lambda *args, **kwargs: pytest.fail("iterator must not run"),
    )
    with pytest.raises(ValueError, match="partial final line"):
        _run(module, benchmark_root, output_root)


def test_only_marks_complete_after_target_coverage_and_pair_limit_is_the_target(
    tmp_path,
    monkeypatch,
):
    module = _load_script()
    benchmark_root, pairs = _write_source_bundle(tmp_path, pair_count=3)
    output_root = tmp_path / "output"
    _runtime(monkeypatch, module)
    monkeypatch.setattr(
        module,
        "iter_geometry_factor_batches",
        lambda *args, **kwargs: [[_record(pairs[0], loop_factor=np.eye(4).reshape(-1))]],
    )

    with pytest.raises(RuntimeError, match="incomplete geometry prediction coverage"):
        _run(module, benchmark_root, output_root)
    incomplete = json.loads(
        (output_root / "geometry_prediction_manifest.json").read_text(encoding="utf-8")
    )
    assert incomplete["complete"] is False
    assert incomplete["target_pair_count"] == 3

    limited_output = tmp_path / "limited-output"
    assert (
        _run(
            module,
            benchmark_root,
            limited_output,
            "--pair-limit",
            "1",
        )
        == 0
    )
    complete = json.loads(
        (limited_output / "geometry_prediction_manifest.json").read_text(encoding="utf-8")
    )
    assert complete["complete"] is True
    assert complete["target_pair_count"] == 1
    assert complete["prediction_record_count"] == 1


def test_records_provenance_geometry_fields_and_excludes_verifier_labels(
    tmp_path,
    monkeypatch,
):
    module = _load_script()
    benchmark_root, pairs = _write_source_bundle(tmp_path, pair_count=1)
    output_root = tmp_path / "output"
    _runtime(monkeypatch, module)
    raw_record = _record(
        pairs[0],
        loop_factor=np.eye(4).reshape(-1),
        pgo_error_after=0.0,
        trajectory_deformation_rmse=0.0,
        score_query_gate_graph=1.0,
        query_gate_graph_score=1.0,
        label=True,
        manual_label=1,
    )
    monkeypatch.setattr(
        module,
        "iter_geometry_factor_batches",
        lambda *args, **kwargs: [[raw_record]],
    )
    argv = [
        str(benchmark_root),
        "--output-root",
        str(output_root),
        "--backend",
        "mock",
        "--device",
        "cpu",
        "--min-translation-error-m",
        "0.75",
        "--max-translation-error-m",
        "1.25",
        "--translation-error-scale-ratio",
        "0.3",
        "--max-rotation-error-deg",
        "12",
        "--max-translation-direction-error-deg",
        "18",
        "--min-direction-baseline-m",
        "0.75",
    ]

    assert module.main(argv) == 0

    row = _read_jsonl(output_root / "geometry_predictions.jsonl")[0]
    assert row["pair_id"] == "p1"
    assert row["support_idx"] == 1
    assert row["sim3_valid"] is True
    assert row["factor_status"] == "ok"
    assert row["estimated_relative_pose"] == np.eye(4).tolist()
    assert row["automatic_label"] == 1
    for forbidden in (
        "pgo_error_after",
        "trajectory_deformation_rmse",
        "score_query_gate_graph",
        "query_gate_graph_score",
        "label",
        "manual_label",
    ):
        assert forbidden not in row

    manifest = json.loads(
        (output_root / "geometry_prediction_manifest.json").read_text(encoding="utf-8")
    )
    assert (
        manifest["source_pair_manifest_sha256"]
        == hashlib.sha256((benchmark_root / "benchmark_pairs.jsonl").read_bytes()).hexdigest()
    )
    assert manifest["da3_config"]["backend"] == "mock"
    assert manifest["da3_config"]["device"] == "cpu"
    assert manifest["request_contract"] == {
        "backend": "mock",
        "device": "cpu",
        "pair_limit": None,
        "thresholds": {
            "max_rotation_error_deg": 12.0,
            "max_translation_direction_error_deg": 18.0,
            "max_translation_error_m": 1.25,
            "min_direction_baseline_m": 0.75,
            "min_translation_error_m": 0.75,
            "translation_error_scale_ratio": 0.3,
        },
    }
    assert manifest["thresholds"] == {
        "max_rotation_error_deg": 12.0,
        "max_translation_direction_error_deg": 18.0,
        "max_translation_error_m": 1.25,
        "min_direction_baseline_m": 0.75,
        "min_translation_error_m": 0.75,
        "translation_error_scale_ratio": 0.3,
    }
    assert manifest["command"] == [str(Path(module.__file__).resolve()), *argv]
    assert manifest["source_commit"] == "test-commit"
    assert (
        manifest["prediction_file_sha256"]
        == hashlib.sha256((output_root / "geometry_predictions.jsonl").read_bytes()).hexdigest()
    )


def test_source_commit_failure_is_explicit_and_offline(tmp_path, monkeypatch):
    module = _load_script()

    def fail_git(*args, **kwargs):
        raise subprocess.CalledProcessError(128, args[0], stderr="not a git repository")

    monkeypatch.setattr(module.subprocess, "run", fail_git)

    provenance = module._source_provenance()

    assert provenance["source_commit"] == "unknown"
    assert "CalledProcessError" in provenance["source_commit_error"]


def test_build_runtime_records_per_dataset_da3_settings(tmp_path, monkeypatch):
    module = _load_script()
    config = SimpleNamespace(
        da3=SimpleNamespace(
            process_res=504,
            ref_view_strategy="first",
            triplet_batch_size=8,
        )
    )
    monkeypatch.setattr(
        module,
        "_load_verifier_configs",
        lambda benchmark_root: {"dataset": config},
    )
    monkeypatch.setattr(
        module,
        "_make_da3_runner",
        lambda benchmark_root, configs, backend, device: object(),
    )
    monkeypatch.setattr(
        module,
        "_verifier_manifest",
        lambda benchmark_root, args, da3_runner: {
            "backend": args.backend,
            "device": args.device,
        },
    )

    _, _, da3_config = module._build_runtime(tmp_path, "mock", "cpu")

    assert da3_config["da3_runtime_by_dataset"] == {
        "dataset": {
            "process_res": 504,
            "ref_view_strategy": "first",
            "triplet_batch_size": 8,
        }
    }
