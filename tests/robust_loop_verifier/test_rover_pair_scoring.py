from __future__ import annotations

import hashlib
import json
import importlib.util
import math
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robust_loop_verifier.pgo import PgoResult
from robust_loop_verifier.retrieval import DescriptorSet
from robust_loop_verifier.rover_pair_scoring import (
    candidate_records_to_score_rows,
    iter_geometry_factor_batches,
    load_frozen_pairs_by_sequence,
    records_to_named_score_rows,
    score_descriptor_pairs,
    score_salad_pairs,
    score_verifier_pairs,
    validate_score_row_order,
)
from robust_loop_verifier.schema import RobustLoopVerifierConfig
from robust_loop_verifier.score_sweep import compute_named_scores
from robust_loop_verifier.sim3_factor import Sim3LoopFactorResult


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_score_artifact(
    benchmark_root: Path,
    score_name: str,
    rows: list[dict],
    *,
    evaluation_name: str,
) -> None:
    score_path = benchmark_root / "scores" / f"{score_name}.jsonl"
    _write_jsonl(score_path, rows)
    _write_json(
        score_path.with_suffix(".manifest.json"),
        {
            "method": f"{evaluation_name} score only",
            "evaluation_name": evaluation_name,
            "score_file": f"scores/{score_name}.jsonl",
            "pair_manifest_sha256": _sha256(benchmark_root / "benchmark_pairs.jsonl"),
            "score_file_sha256": _sha256(score_path),
            "source_commit": "test-commit",
            "command": ["pytest"],
        },
    )


def _load_score_script():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "robust_loop_verification_scripts"
        / "score_rover_aligned_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location("score_rover_aligned_benchmark", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_benchmark_with_cache(tmp_path: Path) -> Path:
    benchmark_root = tmp_path / "benchmark"
    sequence_cache = tmp_path / "sequence_cache"
    (sequence_cache / "images").mkdir(parents=True)
    for idx in range(4):
        (sequence_cache / "images" / f"{idx:06d}.png").write_bytes(b"image")
    _write_jsonl(
        sequence_cache / "keyframes.jsonl",
        [
            {
                "idx": idx,
                "image_path": f"images/{idx:06d}.png",
                "odom_pose": np.eye(4).reshape(-1).tolist(),
            }
            for idx in range(4)
        ],
    )
    _write_jsonl(
        benchmark_root / "benchmark_pairs.jsonl",
        [
            {
                "pair_id": "p1",
                "dataset": "d",
                "platform": "p",
                "sequence": "s",
                "query_idx": 3,
                "candidate_idx": 0,
                "rank": 1,
                "dbow2_score": 0.9,
            },
            {
                "pair_id": "p2",
                "dataset": "d",
                "platform": "p",
                "sequence": "s",
                "query_idx": 3,
                "candidate_idx": 1,
                "rank": 2,
                "dbow2_score": 0.8,
            },
        ],
    )
    _write_json(
        benchmark_root / "manifest.json",
        {
            "netvlad_root": str(tmp_path / "netvlad"),
            "sequences": [
                {
                    "dataset": "d",
                    "platform": "p",
                    "sequence": "s",
                    "cache": str(sequence_cache),
                }
            ],
        },
    )
    return benchmark_root


def _verifier_config(tmp_path: Path, **overrides) -> RobustLoopVerifierConfig:
    data = {
        "dataset_name": "d",
        "platform": "p",
        "input_root": str(tmp_path / "input"),
        "output_root": str(tmp_path / "cache-root"),
        "gt_root": str(tmp_path / "gt"),
        "positive_radius_m": 0.5,
        "recent_exclusion_keyframes": 0,
        "retrieval_top_k_main": 2,
        "retrieval_top_k_ablations": [1],
        "support_window": 4,
        "support_count": 1,
        "min_support_baseline_m": 0.3,
        "pgo_noise": {
            "prior_sigmas": [0.01, 0.01, 0.01, 0.1, 0.1, 0.1],
            "odom_sigmas": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
            "loop_sigmas": [0.1, 0.1, 0.1, 1.0, 1.0, 1.0],
        },
        "da3": {
            "process_res": 504,
            "ref_view_strategy": "first",
            "triplet_batch_size": 2,
        },
    }
    data.update(overrides)
    return RobustLoopVerifierConfig.from_mapping(data)


def test_scoring_cli_writes_manifest_and_refuses_changed_overwrite(tmp_path):
    module = _load_score_script()
    benchmark_root = _write_benchmark_with_cache(tmp_path)

    assert module.main([str(benchmark_root), "--method", "salad", "--backend", "mock"]) == 0
    score_path = benchmark_root / "scores" / "salad.jsonl"
    manifest_path = benchmark_root / "scores" / "salad.manifest.json"
    assert score_path.is_file()
    assert manifest_path.is_file()
    rows = [json.loads(line) for line in score_path.read_text(encoding="utf-8").splitlines()]
    assert [row["pair_id"] for row in rows] == ["p1", "p2"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["pair_manifest_sha256"]
    assert manifest["score_file_sha256"]

    assert module.main([str(benchmark_root), "--method", "salad", "--backend", "mock"]) == 0
    score_path.write_text('{"pair_id":"p1","score":0.0,"status":"ok"}\n', encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        module.main([str(benchmark_root), "--method", "salad", "--backend", "mock"])


def test_programmatic_main_records_exact_effective_command_arguments(
    tmp_path,
    monkeypatch,
):
    module = _load_score_script()
    benchmark_root = _write_benchmark_with_cache(tmp_path)
    argv = [
        str(benchmark_root),
        "--method",
        "salad",
        "--backend",
        "mock",
        "--device",
        "cpu",
    ]
    monkeypatch.setattr(sys, "argv", ["pytest", "--ambient-argument"])

    assert module.main(argv) == 0

    manifest = json.loads(
        (benchmark_root / "scores" / "salad.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["command"] == [str(Path(module.__file__).resolve()), *argv]


def test_verifier_cli_serializes_non_finite_values_as_strict_json_null(
    tmp_path,
    monkeypatch,
):
    module = _load_score_script()
    benchmark_root = _write_benchmark_with_cache(tmp_path)
    monkeypatch.setattr(module, "_load_verifier_configs", lambda _root: {"d": object()})
    monkeypatch.setattr(module, "_make_da3_runner", lambda *args: object())
    monkeypatch.setattr(
        module,
        "_verifier_manifest",
        lambda *args, **kwargs: {"runtime_seconds": float("inf")},
    )
    monkeypatch.setattr(
        module,
        "score_verifier_pairs",
        lambda *args, **kwargs: [
            {
                "pair_id": "p1",
                "sequence_key": "d/p/s",
                "query_idx": 3,
                "trajectory_deformation_rmse": 0.2,
                "pgo_error_after": float("inf"),
            },
            {
                "pair_id": "p2",
                "sequence_key": "d/p/s",
                "query_idx": 3,
                "trajectory_deformation_rmse": 0.5,
                "pgo_error_after": 0.2,
            },
        ],
    )

    assert module.main([str(benchmark_root), "--method", "verifier", "--backend", "mock"]) == 0

    output_paths = [
        benchmark_root / "candidate_records.jsonl",
        benchmark_root / "scores" / "rover_like.jsonl",
        benchmark_root / "scores" / "rover_like.manifest.json",
        benchmark_root / "scores" / "loopanything.jsonl",
        benchmark_root / "scores" / "loopanything.manifest.json",
    ]

    def reject_constant(value):
        raise ValueError(f"non-standard JSON constant: {value}")

    for path in output_paths:
        content = path.read_text(encoding="utf-8")
        payloads = content.splitlines() if path.suffix == ".jsonl" else [content]
        for payload in payloads:
            json.loads(payload, parse_constant=reject_constant)
    candidate = json.loads(
        (benchmark_root / "candidate_records.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert candidate["pgo_error_after"] is None
    manifest = json.loads(
        (benchmark_root / "scores" / "rover_like.manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["runtime_seconds"] is None


def test_verifier_cli_late_manifest_conflict_leaves_output_set_unchanged(
    tmp_path,
    monkeypatch,
):
    module = _load_score_script()
    benchmark_root = _write_benchmark_with_cache(tmp_path)
    conflict_path = benchmark_root / "scores" / "loopanything.manifest.json"
    conflict_path.parent.mkdir(parents=True)
    conflict_content = '{"existing":"preserve"}\n'
    conflict_path.write_text(conflict_content, encoding="utf-8")
    monkeypatch.setattr(module, "_load_verifier_configs", lambda _root: {"d": object()})
    monkeypatch.setattr(module, "_make_da3_runner", lambda *args: object())
    monkeypatch.setattr(module, "_verifier_manifest", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        module,
        "score_verifier_pairs",
        lambda *args, **kwargs: [
            {
                "pair_id": "p1",
                "sequence_key": "d/p/s",
                "query_idx": 3,
                "trajectory_deformation_rmse": 0.2,
                "pgo_error_after": 0.1,
            },
            {
                "pair_id": "p2",
                "sequence_key": "d/p/s",
                "query_idx": 3,
                "trajectory_deformation_rmse": 0.5,
                "pgo_error_after": 0.2,
            },
        ],
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        module.main([str(benchmark_root), "--method", "verifier", "--backend", "mock"])

    assert conflict_path.read_text(encoding="utf-8") == conflict_content
    assert not (benchmark_root / "candidate_records.jsonl").exists()
    assert not (benchmark_root / "scores" / "rover_like.jsonl").exists()
    assert not (benchmark_root / "scores" / "rover_like.manifest.json").exists()
    assert not (benchmark_root / "scores" / "loopanything.jsonl").exists()


def test_verifier_score_manifests_bind_frozen_salad_artifacts(
    tmp_path,
    monkeypatch,
):
    module = _load_score_script()
    benchmark_root = _write_benchmark_with_cache(tmp_path)
    _write_score_artifact(
        benchmark_root,
        "salad",
        [
            {"pair_id": "p1", "score": 0.125, "status": "ok"},
            {"pair_id": "p2", "score": 0.25, "status": "ok"},
        ],
        evaluation_name="SALAD",
    )
    monkeypatch.setattr(module, "_load_verifier_configs", lambda _root: {"d": object()})
    monkeypatch.setattr(module, "_make_da3_runner", lambda *args: object())
    monkeypatch.setattr(
        module,
        "score_verifier_pairs",
        lambda *args, **kwargs: [
            {
                "pair_id": "p1",
                "sequence_key": "d/p/s",
                "query_idx": 3,
                "trajectory_deformation_rmse": 0.2,
                "pgo_error_after": 0.1,
            },
            {
                "pair_id": "p2",
                "sequence_key": "d/p/s",
                "query_idx": 3,
                "trajectory_deformation_rmse": 0.5,
                "pgo_error_after": 0.2,
            },
        ],
    )

    assert module.main(
        [str(benchmark_root), "--method", "verifier", "--backend", "mock"]
    ) == 0

    salad_score_path = benchmark_root / "scores" / "salad.jsonl"
    salad_manifest_path = benchmark_root / "scores" / "salad.manifest.json"
    for manifest_name in ("rover_like.manifest.json", "loopanything.manifest.json"):
        manifest = json.loads(
            (benchmark_root / "scores" / manifest_name).read_text(encoding="utf-8")
        )
        assert manifest["salad_score_file"] == "scores/salad.jsonl"
        assert manifest["salad_score_file_sha256"] == _sha256(salad_score_path)
        assert manifest["salad_score_manifest"] == "scores/salad.manifest.json"
        assert manifest["salad_score_manifest_sha256"] == _sha256(
            salad_manifest_path
        )


def test_score_manifest_records_dirty_source_tree_and_implementation_hashes(
    tmp_path,
    monkeypatch,
):
    module = _load_score_script()
    benchmark_root = _write_benchmark_with_cache(tmp_path)

    def fake_git_run(command, **kwargs):
        if command == ["git", "rev-parse", "HEAD"]:
            return SimpleNamespace(stdout="test-commit\n")
        if command == ["git", "status", "--porcelain=v1", "--untracked-files=all"]:
            return SimpleNamespace(
                stdout=(
                    " M src/robust_loop_verifier/pipeline.py\n"
                    "?? robust_loop_verification_scripts/"
                    "score_rover_aligned_benchmark.py\n"
                )
            )
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(module.subprocess, "run", fake_git_run)
    payloads = module._score_output_payloads(
        benchmark_root,
        score_name="test",
        method="Test method",
        evaluation_name="Test",
        rows=[],
        extra_manifest={},
        command=["pytest"],
    )
    manifest = json.loads(
        payloads[benchmark_root / "scores" / "test.manifest.json"]
    )

    expected_paths = (
        "robust_loop_verification_scripts/score_rover_aligned_benchmark.py",
        "src/robust_loop_verifier/rover_pair_scoring.py",
        "src/robust_loop_verifier/pipeline.py",
        "src/robust_loop_verifier/score_sweep.py",
        "src/robust_loop_verifier/da3_runner.py",
    )
    assert manifest["source_commit"] == "test-commit"
    assert manifest["source_tree_dirty"] is True
    assert manifest["source_file_sha256"] == {
        relative_path: _sha256(module.LOOPANYTHING_ROOT / relative_path)
        for relative_path in expected_paths
    }


def test_score_manifest_records_unknown_when_git_status_cannot_be_read(
    tmp_path,
    monkeypatch,
):
    module = _load_score_script()
    benchmark_root = _write_benchmark_with_cache(tmp_path)

    def fake_git_run(command, **kwargs):
        if command == ["git", "rev-parse", "HEAD"]:
            return SimpleNamespace(stdout="test-commit\n")
        if command == ["git", "status", "--porcelain=v1", "--untracked-files=all"]:
            raise OSError("git status unavailable")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(module.subprocess, "run", fake_git_run)
    payloads = module._score_output_payloads(
        benchmark_root,
        score_name="test",
        method="Test method",
        evaluation_name="Test",
        rows=[],
        extra_manifest={},
        command=["pytest"],
    )
    manifest = json.loads(
        payloads[benchmark_root / "scores" / "test.manifest.json"]
    )

    assert manifest["source_commit"] == "test-commit"
    assert manifest["source_tree_dirty"] == "unknown"
    assert manifest["source_tree_status_error"] == "OSError: git status unavailable"


def test_publish_output_set_publishes_data_before_manifests(tmp_path, monkeypatch):
    module = _load_score_script()
    data_a = tmp_path / "scores" / "a.jsonl"
    data_b = tmp_path / "scores" / "b.jsonl"
    manifest_a = data_a.with_suffix(".manifest.json")
    manifest_b = data_b.with_suffix(".manifest.json")
    real_link = module.os.link
    published = []

    def record_link(source, destination):
        published.append(Path(destination))
        real_link(source, destination)

    monkeypatch.setattr(module.os, "link", record_link)
    module._publish_output_set(
        {
            manifest_a: '{"method":"a"}\n',
            data_a: '{"pair_id":"a"}\n',
            manifest_b: '{"method":"b"}\n',
            data_b: '{"pair_id":"b"}\n',
        }
    )

    assert published == [data_a, data_b, manifest_a, manifest_b]


def test_publish_output_set_completes_partial_data_only_publication(tmp_path):
    module = _load_score_script()
    data_path = tmp_path / "scores" / "method.jsonl"
    manifest_path = data_path.with_suffix(".manifest.json")
    data_content = '{"pair_id":"p1","score":0.5,"status":"ok"}\n'
    manifest_content = '{"method":"Method"}\n'
    data_path.parent.mkdir(parents=True)
    data_path.write_text(data_content, encoding="utf-8")

    module._publish_output_set(
        {
            manifest_path: manifest_content,
            data_path: data_content,
        }
    )

    assert data_path.read_text(encoding="utf-8") == data_content
    assert manifest_path.read_text(encoding="utf-8") == manifest_content


def test_publish_output_set_fsyncs_destination_parent_after_each_link(
    tmp_path,
    monkeypatch,
):
    module = _load_score_script()
    real_fsync = module.os.fsync
    directory_fsyncs = []

    def record_fsync(descriptor):
        if stat.S_ISDIR(module.os.fstat(descriptor).st_mode):
            directory_fsyncs.append(descriptor)
        return real_fsync(descriptor)

    monkeypatch.setattr(module.os, "fsync", record_fsync)
    module._publish_output_set(
        {
            tmp_path / "a.jsonl": "a\n",
            tmp_path / "b.jsonl": "b\n",
        }
    )

    assert len(directory_fsyncs) == 2


def test_publish_output_set_fsyncs_destination_parent_after_rollback(
    tmp_path,
    monkeypatch,
):
    module = _load_score_script()
    real_fsync = module.os.fsync
    real_link = module.os.link
    directory_fsyncs = []
    link_count = 0

    def record_fsync(descriptor):
        if stat.S_ISDIR(module.os.fstat(descriptor).st_mode):
            directory_fsyncs.append(descriptor)
        return real_fsync(descriptor)

    def fail_second_link(source, destination):
        nonlocal link_count
        link_count += 1
        if link_count == 2:
            raise OSError("injected publication failure")
        real_link(source, destination)

    monkeypatch.setattr(module.os, "fsync", record_fsync)
    monkeypatch.setattr(module.os, "link", fail_second_link)

    with pytest.raises(OSError, match="injected publication failure"):
        module._publish_output_set(
            {
                tmp_path / "a.jsonl": "a\n",
                tmp_path / "b.jsonl": "b\n",
            }
        )

    assert not (tmp_path / "a.jsonl").exists()
    assert not (tmp_path / "b.jsonl").exists()
    assert len(directory_fsyncs) >= 2


def test_salad_score_manifest_records_model_config_and_hashes(tmp_path):
    module = _load_score_script()
    benchmark_root = _write_benchmark_with_cache(tmp_path)
    salad_repo = tmp_path / "salad_repo"
    salad_checkpoint = tmp_path / "weights" / "dino_salad.ckpt"
    salad_config = tmp_path / "salad_config.yaml"
    salad_repo.mkdir()
    salad_checkpoint.parent.mkdir()
    salad_checkpoint.write_bytes(b"salad checkpoint")
    salad_config.write_text("backbone: dinov2_vits14\n", encoding="utf-8")
    benchmark_manifest_path = benchmark_root / "manifest.json"
    benchmark_manifest = json.loads(benchmark_manifest_path.read_text(encoding="utf-8"))
    benchmark_manifest["salad"] = {
        "repo": str(salad_repo),
        "checkpoint": str(salad_checkpoint),
        "backbone": "dinov2_vits14",
        "batch_size": 7,
        "config": str(salad_config),
    }
    _write_json(benchmark_manifest_path, benchmark_manifest)

    assert (
        module.main(
            [
                str(benchmark_root),
                "--method",
                "salad",
                "--backend",
                "mock",
                "--device",
                "cpu",
            ]
        )
        == 0
    )

    manifest = json.loads(
        (benchmark_root / "scores" / "salad.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["salad_repo"] == str(salad_repo)
    assert manifest["salad_checkpoint"] == str(salad_checkpoint)
    assert manifest["salad_backbone"] == "dinov2_vits14"
    assert manifest["salad_batch_size"] == 7
    assert manifest["device"] == "cpu"
    assert manifest["salad_config"] == str(salad_config)
    assert manifest["salad_checkpoint_sha256"] == _sha256(salad_checkpoint)
    assert manifest["salad_config_sha256"] == _sha256(salad_config)
    assert manifest["pair_manifest_sha256"]
    assert manifest["score_file_sha256"]
    assert manifest["command"]
    assert manifest["source_commit"]


def test_netvlad_score_manifest_records_config_hash_when_available(tmp_path):
    module = _load_score_script()
    benchmark_root = _write_benchmark_with_cache(tmp_path)
    netvlad_root = tmp_path / "netvlad"
    netvlad_config = tmp_path / "netvlad_config.yaml"
    netvlad_root.mkdir()
    netvlad_config.write_text("descriptor: netvlad\n", encoding="utf-8")
    benchmark_manifest_path = benchmark_root / "manifest.json"
    benchmark_manifest = json.loads(benchmark_manifest_path.read_text(encoding="utf-8"))
    benchmark_manifest["netvlad_root"] = str(netvlad_root)
    benchmark_manifest["netvlad_config"] = str(netvlad_config)
    _write_json(benchmark_manifest_path, benchmark_manifest)

    assert (
        module.main(
            [
                str(benchmark_root),
                "--method",
                "netvlad",
                "--backend",
                "mock",
                "--device",
                "cpu",
            ]
        )
        == 0
    )

    manifest = json.loads(
        (benchmark_root / "scores" / "netvlad.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["netvlad_root"] == str(netvlad_root)
    assert manifest["netvlad_config"] == str(netvlad_config)
    assert manifest["netvlad_config_sha256"] == _sha256(netvlad_config)
    assert "netvlad_root_git_commit" in manifest
    assert manifest["pair_manifest_sha256"]
    assert manifest["score_file_sha256"]
    assert manifest["command"]
    assert manifest["source_commit"]


def test_netvlad_score_manifest_records_actual_torch_hub_checkpoint(
    tmp_path,
    monkeypatch,
):
    import torch

    module = _load_score_script()
    benchmark_root = _write_benchmark_with_cache(tmp_path)
    torch_hub = tmp_path / "torch-hub"
    actual_checkpoint = torch_hub / "netvlad" / "VGG16-NetVLAD-Pitts30K.mat"
    actual_checkpoint.parent.mkdir(parents=True)
    actual_checkpoint.write_bytes(b"actual netvlad checkpoint")
    unsupported_checkpoint = tmp_path / "weights" / "unsupported.mat"
    unsupported_checkpoint.parent.mkdir()
    unsupported_checkpoint.write_bytes(b"manifest checkpoint is not used")
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(torch_hub))
    benchmark_manifest_path = benchmark_root / "manifest.json"
    benchmark_manifest = json.loads(benchmark_manifest_path.read_text(encoding="utf-8"))
    benchmark_manifest["netvlad"] = {"model_path": str(unsupported_checkpoint)}
    _write_json(benchmark_manifest_path, benchmark_manifest)

    assert (
        module.main(
            [
                str(benchmark_root),
                "--method",
                "netvlad",
                "--backend",
                "mock",
                "--device",
                "cpu",
            ]
        )
        == 0
    )

    manifest = json.loads(
        (benchmark_root / "scores" / "netvlad.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["netvlad_checkpoint"] == str(actual_checkpoint)
    assert manifest["netvlad_checkpoint_sha256"] == _sha256(actual_checkpoint)


def test_descriptor_scorer_scores_manifest_pairs_without_retrieval():
    pairs = [
        {"pair_id": "p1", "query_idx": 2, "candidate_idx": 0},
        {"pair_id": "p2", "query_idx": 2, "candidate_idx": 1},
    ]
    descriptors = DescriptorSet(
        keyframe_indices=[0, 1, 2],
        descriptors=np.asarray([[1.0, 0.0], [0.0, 1.0], [0.8, 0.6]]),
    )

    rows = score_descriptor_pairs(pairs, descriptors)

    assert rows == [
        {"pair_id": "p1", "score": 0.8, "status": "ok"},
        {"pair_id": "p2", "score": 0.6, "status": "ok"},
    ]


def test_descriptor_scorer_marks_missing_descriptors_as_failed():
    rows = score_descriptor_pairs(
        [{"pair_id": "p1", "query_idx": 2, "candidate_idx": 99}],
        DescriptorSet(
            keyframe_indices=[0, 2],
            descriptors=np.asarray([[1.0, 0.0], [0.8, 0.6]]),
        ),
    )

    assert rows == [{"pair_id": "p1", "score": None, "status": "failed"}]


def test_verifier_scorer_preserves_pair_ids_and_explicit_failures():
    records = [
        {"pair_id": "p1", "score_rover": -0.1, "pgo_converged": True},
        {"pair_id": "p2", "score_rover": None, "pgo_converged": False},
    ]

    rows = candidate_records_to_score_rows(records, "score_rover")

    assert rows == [
        {"pair_id": "p1", "score": -0.1, "status": "ok"},
        {"pair_id": "p2", "score": None, "status": "failed"},
    ]


def test_candidate_records_invalid_scores_become_explicit_failures():
    records = [
        {"pair_id": "missing"},
        {"pair_id": "none", "score_rover": None},
        {"pair_id": "nan", "score_rover": float("nan")},
        {"pair_id": "inf", "score_rover": float("inf")},
        {"pair_id": "text", "score_rover": "not-a-number"},
    ]

    rows = candidate_records_to_score_rows(records, "score_rover")

    assert rows == [
        {"pair_id": "missing", "score": None, "status": "failed"},
        {"pair_id": "none", "score": None, "status": "failed"},
        {"pair_id": "nan", "score": None, "status": "failed"},
        {"pair_id": "inf", "score": None, "status": "failed"},
        {"pair_id": "text", "score": None, "status": "failed"},
    ]


@pytest.mark.parametrize(
    "rows,match",
    [
        ([{"pair_id": "p2"}, {"pair_id": "p1"}], "order"),
        ([{"pair_id": "p1"}], "count"),
        ([{"pair_id": "p1"}, {"pair_id": "p1"}], "duplicate"),
        ([{"pair_id": "p1"}, {"pair_id": "unknown"}], "unknown"),
    ],
)
def test_score_row_order_rejects_reordered_missing_duplicate_and_unknown(rows, match):
    pairs = [{"pair_id": "p1"}, {"pair_id": "p2"}]

    with pytest.raises(ValueError, match=match):
        validate_score_row_order(pairs, rows)


def test_load_frozen_pairs_groups_by_sequence_without_reading_labels(tmp_path, monkeypatch):
    benchmark_root = tmp_path / "benchmark"
    pairs = [
        {
            "pair_id": "a",
            "dataset": "d",
            "platform": "p",
            "sequence": "s1",
            "query_idx": 2,
            "candidate_idx": 0,
        },
        {
            "pair_id": "b",
            "dataset": "d",
            "platform": "p",
            "sequence": "s2",
            "query_idx": 3,
            "candidate_idx": 1,
        },
    ]
    _write_jsonl(benchmark_root / "benchmark_pairs.jsonl", pairs)
    for forbidden in ("positives.jsonl", "annotations.jsonl", "annotation_seal.json"):
        (benchmark_root / forbidden).write_text("must not be read", encoding="utf-8")

    opened: list[Path] = []
    real_open = Path.open

    def tracking_open(self, *args, **kwargs):
        opened.append(Path(self))
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracking_open)

    grouped = load_frozen_pairs_by_sequence(benchmark_root)

    assert list(grouped) == ["d/p/s1", "d/p/s2"]
    assert grouped["d/p/s1"][0]["pair_id"] == "a"
    assert not any(path.name in {"positives.jsonl", "annotations.jsonl", "annotation_seal.json"} for path in opened)


def test_geometry_factor_batches_preserve_order_skip_completed_and_attach_gt(
    tmp_path,
    monkeypatch,
):
    benchmark_root = tmp_path / "benchmark"
    sequence_cache = tmp_path / "sequence_cache"
    (sequence_cache / "images").mkdir(parents=True)

    def pose_at_x(x):
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = float(x)
        return pose.reshape(-1).tolist()

    for idx in range(5):
        (sequence_cache / "images" / f"{idx:06d}.png").write_bytes(b"image")
    _write_jsonl(
        sequence_cache / "keyframes.jsonl",
        [
            {
                "idx": idx,
                "image_path": f"images/{idx:06d}.png",
                "odom_pose": pose_at_x(idx),
                "gt_pose": pose_at_x(100 + idx),
            }
            for idx in range(5)
        ],
    )
    pairs = [
        {
            "pair_id": pair_id,
            "dataset": "d",
            "platform": "p",
            "sequence": "s",
            "query_idx": query_idx,
            "candidate_idx": candidate_idx,
            "rank": rank,
            "dbow2_score": score,
        }
        for pair_id, query_idx, candidate_idx, rank, score in (
            ("p1", 3, 0, 1, 0.9),
            ("p2", 3, 1, 2, 0.8),
            ("p3", 3, 2, 3, 0.7),
            ("p4", 4, 0, 1, 0.6),
        )
    ]
    _write_jsonl(benchmark_root / "benchmark_pairs.jsonl", pairs)
    _write_json(
        benchmark_root / "manifest.json",
        {
            "sequences": [
                {
                    "dataset": "d",
                    "platform": "p",
                    "sequence": "s",
                    "cache": str(sequence_cache),
                }
            ]
        },
    )
    (benchmark_root / "annotations.jsonl").write_text("must not be read", encoding="utf-8")
    (sequence_cache / "positives.jsonl").write_text("must not be read", encoding="utf-8")

    calls = []

    def fake_score_frozen_query_candidates(**kwargs):
        calls.append(kwargs)
        assert kwargs["run_pgo"] is False
        assert kwargs["cache_order"] == [0, 1, 2, 3, 4]
        assert set(kwargs["image_by_idx"]) == {0, 1, 2, 3, 4}
        assert all(Path(path).is_absolute() for path in kwargs["image_by_idx"].values())
        assert all(pose.shape == (4, 4) for pose in kwargs["odom_by_idx"].values())
        return [
            {
                "pair_id": candidate["pair_id"],
                "query_idx": kwargs["query_idx"],
                "candidate_idx": candidate["candidate_idx"],
                "rank": candidate["rank"],
                "support_idx": candidate["candidate_idx"] + 1,
                "sim3_valid": True,
                "score_da3_sim3": -0.1,
                "loop_factor": np.eye(4).reshape(-1).tolist(),
            }
            for candidate in kwargs["frozen_candidates"]
        ]

    monkeypatch.setattr(
        "robust_loop_verifier.rover_pair_scoring.score_frozen_query_candidates",
        fake_score_frozen_query_candidates,
    )
    opened = []
    real_open = Path.open

    def tracking_open(self, *args, **kwargs):
        opened.append(Path(self))
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracking_open)

    batches = list(
        iter_geometry_factor_batches(
            benchmark_root,
            config=object(),
            da3_runner=object(),
            completed_pair_ids={"p2"},
        )
    )

    assert [[record["pair_id"] for record in batch] for batch in batches] == [
        ["p1", "p3"],
        ["p4"],
    ]
    assert [call["query_idx"] for call in calls] == [3, 4]
    assert [candidate["pair_id"] for call in calls for candidate in call["frozen_candidates"]] == [
        "p1",
        "p3",
        "p4",
    ]
    flattened = [record for batch in batches for record in batch]
    assert flattened[0]["gt_query_pose"] == pose_at_x(103)
    assert flattened[0]["gt_candidate_pose"] == pose_at_x(100)
    assert flattened[1]["gt_candidate_pose"] == pose_at_x(102)
    assert flattened[2]["gt_query_pose"] == pose_at_x(104)
    assert all(record["sequence_key"] == "d/p/s" for record in flattened)
    assert not any(path.name in {"annotations.jsonl", "positives.jsonl"} for path in opened)
    json.dumps(batches, allow_nan=False)


def test_geometry_factor_batches_skip_completed_query_and_preserve_cross_sequence_order(
    tmp_path,
    monkeypatch,
):
    benchmark_root = tmp_path / "benchmark"

    def pose_at_x(x):
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = float(x)
        return pose.reshape(-1).tolist()

    def write_sequence_cache(cache: Path, gt_offset: int) -> None:
        (cache / "images").mkdir(parents=True)
        for idx in range(5):
            (cache / "images" / f"{idx:06d}.png").write_bytes(b"image")
        _write_jsonl(
            cache / "keyframes.jsonl",
            [
                {
                    "idx": idx,
                    "image_path": f"images/{idx:06d}.png",
                    "odom_pose": pose_at_x(idx),
                    "gt_pose": pose_at_x(gt_offset + idx),
                }
                for idx in range(5)
            ],
        )

    cache_s1 = tmp_path / "cache_s1"
    cache_s2 = tmp_path / "cache_s2"
    write_sequence_cache(cache_s1, gt_offset=100)
    write_sequence_cache(cache_s2, gt_offset=200)
    pairs = [
        {
            "pair_id": "s1-complete",
            "dataset": "d",
            "platform": "p",
            "sequence": "s1",
            "query_idx": 3,
            "candidate_idx": 0,
            "rank": 1,
            "dbow2_score": 0.9,
        },
        {
            "pair_id": "s2-q2",
            "dataset": "d",
            "platform": "p",
            "sequence": "s2",
            "query_idx": 2,
            "candidate_idx": 0,
            "rank": 1,
            "dbow2_score": 0.8,
        },
        {
            "pair_id": "s1-q4",
            "dataset": "d",
            "platform": "p",
            "sequence": "s1",
            "query_idx": 4,
            "candidate_idx": 1,
            "rank": 1,
            "dbow2_score": 0.7,
        },
        {
            "pair_id": "s2-q3",
            "dataset": "d",
            "platform": "p",
            "sequence": "s2",
            "query_idx": 3,
            "candidate_idx": 1,
            "rank": 1,
            "dbow2_score": 0.6,
        },
    ]
    _write_jsonl(benchmark_root / "benchmark_pairs.jsonl", pairs)
    _write_json(
        benchmark_root / "manifest.json",
        {
            "sequences": [
                {"dataset": "d", "platform": "p", "sequence": "s1", "cache": str(cache_s1)},
                {"dataset": "d", "platform": "p", "sequence": "s2", "cache": str(cache_s2)},
            ]
        },
    )
    calls = []

    def fake_score_frozen_query_candidates(**kwargs):
        calls.append(
            {
                "query_idx": kwargs["query_idx"],
                "pair_ids": [candidate["pair_id"] for candidate in kwargs["frozen_candidates"]],
            }
        )
        return [
            {
                "pair_id": candidate["pair_id"],
                "query_idx": kwargs["query_idx"],
                "candidate_idx": candidate["candidate_idx"],
                "rank": candidate["rank"],
                "loop_factor": np.eye(4).reshape(-1).tolist(),
            }
            for candidate in kwargs["frozen_candidates"]
        ]

    monkeypatch.setattr(
        "robust_loop_verifier.rover_pair_scoring.score_frozen_query_candidates",
        fake_score_frozen_query_candidates,
    )

    batches = list(
        iter_geometry_factor_batches(
            benchmark_root,
            config=object(),
            da3_runner=object(),
            completed_pair_ids={"s1-complete"},
        )
    )

    assert [[record["pair_id"] for record in batch] for batch in batches] == [
        ["s2-q2"],
        ["s1-q4"],
        ["s2-q3"],
    ]
    assert calls == [
        {"query_idx": 2, "pair_ids": ["s2-q2"]},
        {"query_idx": 4, "pair_ids": ["s1-q4"]},
        {"query_idx": 3, "pair_ids": ["s2-q3"]},
    ]
    flattened = [record for batch in batches for record in batch]
    assert [record["sequence_key"] for record in flattened] == [
        "d/p/s2",
        "d/p/s1",
        "d/p/s2",
    ]
    assert all(record["pair_id"] != "s1-complete" for record in flattened)
    assert flattened[0]["gt_query_pose"] == pose_at_x(202)
    assert flattened[1]["gt_query_pose"] == pose_at_x(104)
    assert flattened[2]["gt_query_pose"] == pose_at_x(203)


def test_geometry_factor_batches_reject_missing_pair_keyframe_with_context(
    tmp_path,
    monkeypatch,
):
    benchmark_root = tmp_path / "benchmark"
    sequence_cache = tmp_path / "sequence_cache"
    (sequence_cache / "images").mkdir(parents=True)

    pose = np.eye(4, dtype=np.float64).reshape(-1).tolist()
    (sequence_cache / "images" / "000003.png").write_bytes(b"image")
    _write_jsonl(
        sequence_cache / "keyframes.jsonl",
        [
            {
                "idx": 3,
                "image_path": "images/000003.png",
                "odom_pose": pose,
                "gt_pose": pose,
            }
        ],
    )
    _write_jsonl(
        benchmark_root / "benchmark_pairs.jsonl",
        [
            {
                "pair_id": "p1",
                "dataset": "d",
                "platform": "p",
                "sequence": "s",
                "query_idx": 3,
                "candidate_idx": 0,
                "rank": 1,
                "dbow2_score": 0.9,
            }
        ],
    )
    _write_json(
        benchmark_root / "manifest.json",
        {
            "sequences": [
                {"dataset": "d", "platform": "p", "sequence": "s", "cache": str(sequence_cache)}
            ]
        },
    )
    monkeypatch.setattr(
        "robust_loop_verifier.rover_pair_scoring.score_frozen_query_candidates",
        lambda **kwargs: [
            {
                "pair_id": "p1",
                "query_idx": kwargs["query_idx"],
                "candidate_idx": 0,
                "rank": 1,
            }
        ],
    )

    with pytest.raises(ValueError, match="d/p/s.*candidate.*0"):
        list(iter_geometry_factor_batches(benchmark_root, object(), object()))


@pytest.mark.parametrize(
    "field_name,bad_pose",
    [
        ("odom_pose", [float("nan")] * 16),
        ("gt_pose", [1.0, 2.0, 3.0]),
    ],
)
def test_geometry_factor_batches_reject_malformed_required_pose_with_keyframe_context(
    tmp_path,
    monkeypatch,
    field_name,
    bad_pose,
):
    benchmark_root = tmp_path / "benchmark"
    sequence_cache = tmp_path / "sequence_cache"
    (sequence_cache / "images").mkdir(parents=True)
    good_pose = np.eye(4, dtype=np.float64).reshape(-1).tolist()
    for idx in (0, 3):
        (sequence_cache / "images" / f"{idx:06d}.png").write_bytes(b"image")
    rows = []
    for idx in (0, 3):
        row = {
            "idx": idx,
            "image_path": f"images/{idx:06d}.png",
            "odom_pose": good_pose,
            "gt_pose": good_pose,
        }
        if idx == 3:
            row[field_name] = bad_pose
        rows.append(row)
    _write_jsonl(sequence_cache / "keyframes.jsonl", rows)
    _write_jsonl(
        benchmark_root / "benchmark_pairs.jsonl",
        [
            {
                "pair_id": "p1",
                "dataset": "d",
                "platform": "p",
                "sequence": "s",
                "query_idx": 3,
                "candidate_idx": 0,
                "rank": 1,
                "dbow2_score": 0.9,
            }
        ],
    )
    _write_json(
        benchmark_root / "manifest.json",
        {
            "sequences": [
                {"dataset": "d", "platform": "p", "sequence": "s", "cache": str(sequence_cache)}
            ]
        },
    )
    monkeypatch.setattr(
        "robust_loop_verifier.rover_pair_scoring.score_frozen_query_candidates",
        lambda **kwargs: [],
    )

    with pytest.raises(ValueError, match=f"keyframe 3.*{field_name}"):
        list(iter_geometry_factor_batches(benchmark_root, object(), object()))


def test_descriptor_backend_scoring_preserves_interleaved_manifest_order(tmp_path):
    benchmark_root = tmp_path / "benchmark"
    cache_s1 = tmp_path / "cache_s1"
    cache_s2 = tmp_path / "cache_s2"
    for cache in (cache_s1, cache_s2):
        (cache / "images").mkdir(parents=True)
        for idx in range(3):
            (cache / "images" / f"{idx:06d}.png").write_bytes(b"image")
        _write_jsonl(
            cache / "keyframes.jsonl",
            [
                {"idx": idx, "image_path": f"images/{idx:06d}.png"}
                for idx in range(3)
            ],
        )

    pairs = [
        {
            "pair_id": "s1-a",
            "dataset": "d",
            "platform": "p",
            "sequence": "s1",
            "query_idx": 2,
            "candidate_idx": 0,
        },
        {
            "pair_id": "s2-a",
            "dataset": "d",
            "platform": "p",
            "sequence": "s2",
            "query_idx": 2,
            "candidate_idx": 0,
        },
        {
            "pair_id": "s1-b",
            "dataset": "d",
            "platform": "p",
            "sequence": "s1",
            "query_idx": 2,
            "candidate_idx": 1,
        },
    ]
    _write_jsonl(benchmark_root / "benchmark_pairs.jsonl", pairs)
    _write_json(
        benchmark_root / "manifest.json",
        {
            "sequences": [
                {"dataset": "d", "platform": "p", "sequence": "s1", "cache": str(cache_s1)},
                {"dataset": "d", "platform": "p", "sequence": "s2", "cache": str(cache_s2)},
            ]
        },
    )

    class FakeDescriptorBackend:
        def compute(self, image_paths, keyframe_indices):
            return DescriptorSet(
                keyframe_indices=[int(index) for index in keyframe_indices],
                descriptors=np.asarray(
                    [[float(int(index) + 1), 1.0] for index in keyframe_indices],
                    dtype=np.float64,
                ),
            )

    rows = score_salad_pairs(benchmark_root, FakeDescriptorBackend())

    assert [row["pair_id"] for row in rows] == ["s1-a", "s2-a", "s1-b"]


def test_compute_named_scores_is_label_free_for_main_verifier_methods():
    records = [
        {
            "query_idx": 10,
            "pair_id": "p1",
            "trajectory_deformation_rmse": 0.2,
            "pgo_error_after": 0.1,
        },
        {
            "query_idx": 10,
            "pair_id": "p2",
            "trajectory_deformation_rmse": 2.0,
            "pgo_error_after": 1.0,
        },
    ]

    assert compute_named_scores(records, "ROVER deformation only") == [-0.2, -2.0]
    query_gate = compute_named_scores(records, "query_gate_graph:def=0.5,res=0.25,margin=0")

    assert len(query_gate) == 2
    assert query_gate[0] > query_gate[1]


def test_query_gate_graph_groups_same_query_idx_by_sequence_key():
    records = [
        {
            "sequence_key": "dataset/platform/sequence-a",
            "query_idx": 10,
            "pair_id": "a",
            "trajectory_deformation_rmse": 1.0,
            "pgo_error_after": 0.0,
        },
        {
            "sequence_key": "dataset/platform/sequence-b",
            "query_idx": 10,
            "pair_id": "b",
            "trajectory_deformation_rmse": 10.0,
            "pgo_error_after": 0.0,
        },
    ]

    scores = compute_named_scores(records, "query_gate_graph:def=1,res=0,margin=1")

    assert scores == [-2.0, -20.0]


def test_score_verifier_pairs_groups_frozen_query_without_retrieval(monkeypatch, tmp_path):
    benchmark_root = tmp_path / "benchmark"
    sequence_cache = tmp_path / "sequence_cache"
    (sequence_cache / "images").mkdir(parents=True)
    for idx in range(4):
        (sequence_cache / "images" / f"{idx:06d}.png").write_bytes(b"image")
    _write_jsonl(
        sequence_cache / "keyframes.jsonl",
        [
            {
                "idx": idx,
                "image_path": f"images/{idx:06d}.png",
                "odom_pose": np.eye(4).reshape(-1).tolist(),
            }
            for idx in range(4)
        ],
    )
    pairs = [
        {
            "pair_id": "p1",
            "dataset": "d",
            "platform": "p",
            "sequence": "s",
            "query_idx": 3,
            "candidate_idx": 0,
            "rank": 1,
            "dbow2_score": 0.9,
        },
        {
            "pair_id": "p2",
            "dataset": "d",
            "platform": "p",
            "sequence": "s",
            "query_idx": 3,
            "candidate_idx": 1,
            "rank": 2,
            "dbow2_score": 0.8,
        },
    ]
    _write_jsonl(benchmark_root / "benchmark_pairs.jsonl", pairs)
    _write_json(
        benchmark_root / "manifest.json",
        {
            "sequences": [
                {
                    "dataset": "d",
                    "platform": "p",
                    "sequence": "s",
                    "cache": str(sequence_cache),
                }
            ]
        },
    )
    _write_score_artifact(
        benchmark_root,
        "salad",
        [
            {"pair_id": "p1", "score": 0.125, "status": "ok"},
            {"pair_id": "p2", "score": 0.25, "status": "ok"},
        ],
        evaluation_name="SALAD",
    )

    def forbidden_retrieval(*args, **kwargs):
        raise AssertionError("retrieval must not run for frozen pairs")

    calls = []

    def fake_score_frozen_query_candidates(**kwargs):
        assert kwargs["cache_order"] == [0, 1, 2, 3]
        assert set(kwargs["image_by_idx"]) == {0, 1, 2, 3}
        assert set(kwargs["odom_by_idx"]) == {0, 1, 2, 3}
        calls.append(kwargs)
        return [
            {
                "pair_id": candidate["pair_id"],
                "query_idx": kwargs["query_idx"],
                "candidate_idx": candidate["candidate_idx"],
                "rank": candidate["rank"],
                "score_rover": -float(candidate["rank"]),
                "trajectory_deformation_rmse": float(candidate["rank"]),
                "pgo_error_after": 0.0,
            }
            for candidate in kwargs["frozen_candidates"]
        ]

    monkeypatch.setattr("robust_loop_verifier.pipeline.retrieve_historical_topk", forbidden_retrieval)
    monkeypatch.setattr(
        "robust_loop_verifier.rover_pair_scoring.score_frozen_query_candidates",
        fake_score_frozen_query_candidates,
    )

    records = score_verifier_pairs(benchmark_root, config=object(), da3_backend=object())

    assert [record["pair_id"] for record in records] == ["p1", "p2"]
    assert len(calls) == 1
    assert calls[0]["query_idx"] == 3
    assert [candidate["pair_id"] for candidate in calls[0]["frozen_candidates"]] == ["p1", "p2"]
    assert [candidate["score"] for candidate in calls[0]["frozen_candidates"]] == [
        0.125,
        0.25,
    ]


def test_score_verifier_pairs_isolates_failed_frozen_salad_pair(monkeypatch, tmp_path):
    benchmark_root = _write_benchmark_with_cache(tmp_path)
    _write_score_artifact(
        benchmark_root,
        "salad",
        [
            {"pair_id": "p1", "score": 0.125, "status": "ok"},
            {"pair_id": "p2", "score": None, "status": "failed"},
        ],
        evaluation_name="SALAD",
    )
    calls = []

    def fake_score_frozen_query_candidates(**kwargs):
        calls.append(kwargs)
        return [
            {
                "pair_id": "p1",
                "query_idx": 3,
                "candidate_idx": 0,
                "rank": 1,
                "score_rover": -0.2,
                "trajectory_deformation_rmse": 0.2,
                "pgo_error_after": 0.1,
                "pgo_converged": True,
            }
        ]

    monkeypatch.setattr(
        "robust_loop_verifier.rover_pair_scoring.score_frozen_query_candidates",
        fake_score_frozen_query_candidates,
    )

    records = score_verifier_pairs(benchmark_root, config=object(), da3_backend=object())

    assert len(calls) == 1
    assert [candidate["pair_id"] for candidate in calls[0]["frozen_candidates"]] == ["p1"]
    assert records[0]["pair_id"] == "p1"
    failed = records[1]
    assert {
        "pair_id": "p2",
        "dataset": "d",
        "platform": "p",
        "sequence": "s",
        "sequence_key": "d/p/s",
        "query_idx": 3,
        "candidate_idx": 1,
        "rank": 2,
        "salad_score": None,
        "score_salad": None,
        "score_da3_sim3": None,
        "score_rover": None,
        "score_rover_source": None,
        "pgo_converged": False,
    }.items() <= failed.items()
    assert any("frozen SALAD" in reason for reason in failed["failure_reasons"])
    assert records_to_named_score_rows(records, "ROVER deformation only")[1] == {
        "pair_id": "p2",
        "score": None,
        "status": "failed",
    }
    assert records_to_named_score_rows(
        records,
        "query_gate_graph:def=0.5,res=0.25,margin=0",
    )[1] == {"pair_id": "p2", "score": None, "status": "failed"}


def test_score_verifier_pairs_preserves_image_less_odom_and_file_order_in_pgo(
    monkeypatch,
    tmp_path,
):
    benchmark_root = tmp_path / "benchmark"
    sequence_cache = tmp_path / "sequence_cache"
    (sequence_cache / "images").mkdir(parents=True)
    for idx in (0, 2, 3):
        (sequence_cache / "images" / f"{idx:06d}.png").write_bytes(b"image")
    ordered_indices = [2, 1, 0, 3]
    _write_jsonl(
        sequence_cache / "keyframes.jsonl",
        [
            {
                "idx": idx,
                "image_path": (
                    None if idx == 1 else f"images/{idx:06d}.png"
                ),
                "odom_pose": (
                    np.eye(4, dtype=np.float64)
                    + np.asarray(
                        [
                            [0.0, 0.0, 0.0, float(idx)],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ]
                    )
                ).reshape(-1).tolist(),
            }
            for idx in ordered_indices
        ],
    )
    pairs = [
        {
            "pair_id": "p1",
            "dataset": "d",
            "platform": "p",
            "sequence": "s",
            "query_idx": 3,
            "candidate_idx": 0,
            "rank": 1,
        }
    ]
    _write_jsonl(benchmark_root / "benchmark_pairs.jsonl", pairs)
    _write_json(
        benchmark_root / "manifest.json",
        {
            "sequences": [
                {
                    "dataset": "d",
                    "platform": "p",
                    "sequence": "s",
                    "cache": str(sequence_cache),
                }
            ]
        },
    )
    _write_score_artifact(
        benchmark_root,
        "salad",
        [{"pair_id": "p1", "score": 0.125, "status": "ok"}],
        evaluation_name="SALAD",
    )
    captured = {}

    class FakeRunner:
        def run_triplets(self, triplets):
            return [SimpleNamespace(predicted_c2w=np.empty((3, 4, 4))) for _ in triplets]

    monkeypatch.setattr(
        "robust_loop_verifier.pipeline.align_triplet_to_candidate_support",
        lambda *args, **kwargs: Sim3LoopFactorResult(
            valid=True,
            loop_factor=np.eye(4),
            sim3_scale=1.0,
            support_alignment_residual_m=0.0,
            direction_error_deg=0.0,
            rejection_reason=None,
        ),
    )

    def capture_pgo(prefix_indices, odom_poses, *args, **kwargs):
        captured["prefix_indices"] = list(prefix_indices)
        captured["odom_x"] = [float(pose[0, 3]) for pose in odom_poses]
        return PgoResult(
            converged=True,
            optimized_poses=[np.asarray(pose).copy() for pose in odom_poses],
            error_before=0.0,
            error_after=0.0,
            failure_reason=None,
        )

    monkeypatch.setattr(
        "robust_loop_verifier.pipeline.run_full_prefix_pgo",
        capture_pgo,
    )

    records = score_verifier_pairs(
        benchmark_root,
        config=_verifier_config(tmp_path),
        da3_backend=FakeRunner(),
    )

    assert records[0]["pgo_converged"] is True, records[0]["failure_reasons"]
    assert captured["prefix_indices"] == ordered_indices
    assert captured["odom_x"] == [2.0, 1.0, 0.0, 3.0]


def test_score_verifier_pairs_uses_manifest_recent_exclusion_for_support_selection(
    tmp_path,
    monkeypatch,
):
    benchmark_root = tmp_path / "benchmark"
    sequence_cache = tmp_path / "sequence_cache"
    (sequence_cache / "images").mkdir(parents=True)
    for idx in (5, 6, 7, 35):
        (sequence_cache / "images" / f"{idx:06d}.png").write_bytes(b"image")
    _write_jsonl(
        sequence_cache / "keyframes.jsonl",
        [
            {
                "idx": idx,
                "image_path": f"images/{idx:06d}.png",
                "odom_pose": (
                    np.eye(4, dtype=np.float64)
                    + np.asarray(
                        [
                            [0.0, 0.0, 0.0, float(idx)],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ]
                    )
                ).reshape(-1).tolist(),
            }
            for idx in (5, 6, 7, 35)
        ],
    )
    _write_jsonl(
        benchmark_root / "benchmark_pairs.jsonl",
        [
            {
                "pair_id": "p1",
                "dataset": "d",
                "platform": "p",
                "sequence": "s",
                "query_idx": 35,
                "candidate_idx": 6,
                "rank": 1,
            }
        ],
    )
    _write_json(
        benchmark_root / "manifest.json",
        {
            "sequences": [
                {
                    "dataset": "d",
                    "platform": "p",
                    "sequence": "s",
                    "cache": str(sequence_cache),
                    "recent_exclusion_keyframes": 5,
                }
            ]
        },
    )
    _write_score_artifact(
        benchmark_root,
        "salad",
        [{"pair_id": "p1", "score": 0.125, "status": "ok"}],
        evaluation_name="SALAD",
    )

    class FakeRunner:
        def run_triplets(self, triplets):
            return [SimpleNamespace(predicted_c2w=np.empty((3, 4, 4))) for _ in triplets]

    monkeypatch.setattr(
        "robust_loop_verifier.pipeline.align_triplet_to_candidate_support",
        lambda *args, **kwargs: Sim3LoopFactorResult(
            valid=True,
            loop_factor=np.eye(4),
            sim3_scale=1.0,
            support_alignment_residual_m=0.0,
            direction_error_deg=0.0,
            rejection_reason=None,
        ),
    )
    monkeypatch.setattr(
        "robust_loop_verifier.pipeline.run_full_prefix_pgo",
        lambda prefix_indices, odom_poses, *args, **kwargs: PgoResult(
            converged=True,
            optimized_poses=[np.asarray(pose).copy() for pose in odom_poses],
            error_before=0.0,
            error_after=0.0,
            failure_reason=None,
        ),
    )

    records = score_verifier_pairs(
        benchmark_root,
        config=_verifier_config(tmp_path, recent_exclusion_keyframes=30),
        da3_backend=FakeRunner(),
    )

    assert records[0]["support_idx"] == 5
    assert records[0]["support_rejection_reason"] is None
    assert records[0]["pgo_converged"] is True, records[0]["failure_reasons"]


@pytest.mark.parametrize(
    "salad_rows,match",
    [
        (
            [{"pair_id": "p1", "score": 0.125, "status": "ok"}],
            "count",
        ),
        (
            [
                {"pair_id": "p2", "score": 0.25, "status": "ok"},
                {"pair_id": "p1", "score": 0.125, "status": "ok"},
            ],
            "order",
        ),
        (
            [
                {"pair_id": "p1", "score": 0.125, "status": "ok"},
                {"pair_id": "wrong", "score": 0.25, "status": "ok"},
            ],
            "unknown",
        ),
    ],
)
def test_score_verifier_pairs_rejects_invalid_frozen_salad_rows(
    tmp_path,
    salad_rows,
    match,
):
    benchmark_root = _write_benchmark_with_cache(tmp_path)
    _write_score_artifact(
        benchmark_root,
        "salad",
        salad_rows,
        evaluation_name="SALAD",
    )

    with pytest.raises(ValueError, match=match):
        score_verifier_pairs(benchmark_root, config=object(), da3_backend=object())


@pytest.mark.parametrize(
    "salad_rows,match",
    [
        (
            [
                {"pair_id": "p1", "score": 0.125, "status": "bad"},
                {"pair_id": "p2", "score": 0.25, "status": "ok"},
            ],
            "status",
        ),
        (
            [
                {"pair_id": "p1", "score": 0.125, "status": "failed"},
                {"pair_id": "p2", "score": 0.25, "status": "ok"},
            ],
            "failed.*score",
        ),
        (
            [
                {"pair_id": "p1", "score": None, "status": "ok"},
                {"pair_id": "p2", "score": 0.25, "status": "ok"},
            ],
            "ok.*finite",
        ),
    ],
)
def test_score_verifier_pairs_rejects_invalid_frozen_salad_score_schema(
    monkeypatch,
    tmp_path,
    salad_rows,
    match,
):
    benchmark_root = _write_benchmark_with_cache(tmp_path)
    _write_score_artifact(
        benchmark_root,
        "salad",
        salad_rows,
        evaluation_name="SALAD",
    )

    monkeypatch.setattr(
        "robust_loop_verifier.rover_pair_scoring.score_frozen_query_candidates",
        lambda **_kwargs: [],
    )

    with pytest.raises(ValueError, match=match):
        score_verifier_pairs(benchmark_root, config=object(), da3_backend=object())


def test_score_verifier_pairs_rejects_frozen_salad_manifest_pair_hash_mismatch(
    tmp_path,
):
    benchmark_root = _write_benchmark_with_cache(tmp_path)
    _write_score_artifact(
        benchmark_root,
        "salad",
        [
            {"pair_id": "p1", "score": 0.125, "status": "ok"},
            {"pair_id": "p2", "score": 0.25, "status": "ok"},
        ],
        evaluation_name="SALAD",
    )
    manifest_path = benchmark_root / "scores" / "salad.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["pair_manifest_sha256"] = "not-the-benchmark-hash"
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="candidate-manifest hash"):
        score_verifier_pairs(benchmark_root, config=object(), da3_backend=object())


def test_verifier_score_manifests_record_da3_model_provenance(
    tmp_path,
    monkeypatch,
):
    module = _load_score_script()
    benchmark_root = _write_benchmark_with_cache(tmp_path)
    da3_model_dir = tmp_path / "models" / "da3-large"
    da3_checkpoint = tmp_path / "weights" / "da3_model.safetensors"
    da3_snapshot = tmp_path / "snapshots" / "da3_snapshot.bin"
    verifier_config = tmp_path / "configs" / "verifier.yaml"
    da3_model_dir.mkdir(parents=True)
    da3_checkpoint.parent.mkdir()
    da3_checkpoint.write_bytes(b"da3 checkpoint")
    da3_snapshot.parent.mkdir()
    da3_snapshot.write_bytes(b"da3 snapshot")
    _write_json(
        verifier_config,
        {
            "dataset_name": "d",
            "platform": "p",
            "input_root": str(tmp_path / "input"),
            "output_root": str(tmp_path / "cache"),
            "gt_root": str(tmp_path / "gt"),
            "positive_radius_m": 0.5,
            "recent_exclusion_keyframes": 1,
            "retrieval_top_k_main": 2,
            "retrieval_top_k_ablations": [1],
            "support_window": 4,
            "support_count": 1,
            "min_support_baseline_m": 0.3,
            "pgo_noise": {
                "prior_sigmas": [0.01, 0.01, 0.01, 0.1, 0.1, 0.1],
                "odom_sigmas": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                "loop_sigmas": [0.1, 0.1, 0.1, 1.0, 1.0, 1.0],
            },
            "da3": {
                "process_res": 504,
                "ref_view_strategy": "first",
                "checkpoint": str(da3_checkpoint),
                "snapshot_path": str(da3_snapshot),
            },
        },
    )
    benchmark_manifest_path = benchmark_root / "manifest.json"
    benchmark_manifest = json.loads(benchmark_manifest_path.read_text(encoding="utf-8"))
    benchmark_manifest["verifier_configs"] = {"d": str(verifier_config)}
    benchmark_manifest["da3"] = {
        "model_name": "depth-anything/DA3-LARGE-1.1",
        "model_dir": str(da3_model_dir),
    }
    _write_json(benchmark_manifest_path, benchmark_manifest)

    def fake_score_verifier_pairs(*args, **kwargs):
        return [
            {
                "pair_id": "p1",
                "sequence_key": "d/p/s",
                "query_idx": 3,
                "trajectory_deformation_rmse": 0.2,
                "pgo_error_after": 0.1,
            },
            {
                "pair_id": "p2",
                "sequence_key": "d/p/s",
                "query_idx": 3,
                "trajectory_deformation_rmse": 0.5,
                "pgo_error_after": 0.2,
            },
        ]

    monkeypatch.setattr(module, "score_verifier_pairs", fake_score_verifier_pairs)

    assert module.main([str(benchmark_root), "--method", "verifier", "--backend", "mock"]) == 0

    for manifest_name in ("rover_like.manifest.json", "loopanything.manifest.json"):
        manifest = json.loads(
            (benchmark_root / "scores" / manifest_name).read_text(encoding="utf-8")
        )
        assert manifest["da3_model_name"] == "depth-anything/DA3-LARGE-1.1"
        assert manifest["da3_model_dir"] == str(da3_model_dir)
        assert manifest["da3_checkpoint"] == str(da3_checkpoint)
        assert manifest["da3_checkpoint_sha256"] == _sha256(da3_checkpoint)
        assert manifest["da3_snapshot"] == str(da3_snapshot)
        assert manifest["da3_snapshot_sha256"] == _sha256(da3_snapshot)


def test_real_verifier_backend_uses_device_and_truthful_da3_runtime_provenance(
    tmp_path,
    monkeypatch,
):
    module = _load_score_script()
    benchmark_root = _write_benchmark_with_cache(tmp_path)
    hf_cache = tmp_path / "huggingface" / "hub"
    revision = "test-revision"
    da3_snapshot = (
        hf_cache
        / "models--depth-anything--DA3-SMALL"
        / "snapshots"
        / revision
    )
    da3_model_file = da3_snapshot / "model.safetensors"
    da3_model_file.parent.mkdir(parents=True)
    da3_model_file.write_bytes(b"resolved da3 model")
    refs_dir = hf_cache / "models--depth-anything--DA3-SMALL" / "refs"
    refs_dir.mkdir()
    (refs_dir / "main").write_text(revision + "\n", encoding="utf-8")
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hf_cache))
    monkeypatch.delenv("HF_HOME", raising=False)
    da3_checkpoint = tmp_path / "weights" / "ignored_checkpoint.safetensors"
    verifier_config = tmp_path / "configs" / "verifier.yaml"
    da3_checkpoint.parent.mkdir()
    da3_checkpoint.write_bytes(b"checkpoint not supported by RealDa3RunnerConfig")
    _write_json(
        verifier_config,
        {
            "dataset_name": "d",
            "platform": "p",
            "input_root": str(tmp_path / "input"),
            "output_root": str(tmp_path / "cache"),
            "gt_root": str(tmp_path / "gt"),
            "positive_radius_m": 0.5,
            "recent_exclusion_keyframes": 1,
            "retrieval_top_k_main": 2,
            "retrieval_top_k_ablations": [1],
            "support_window": 4,
            "support_count": 1,
            "min_support_baseline_m": 0.3,
            "pgo_noise": {
                "prior_sigmas": [0.01, 0.01, 0.01, 0.1, 0.1, 0.1],
                "odom_sigmas": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                "loop_sigmas": [0.1, 0.1, 0.1, 1.0, 1.0, 1.0],
            },
            "da3": {
                "process_res": 672,
                "ref_view_strategy": "first",
                "triplet_batch_size": 3,
                "checkpoint": str(da3_checkpoint),
            },
        },
    )
    benchmark_manifest_path = benchmark_root / "manifest.json"
    benchmark_manifest = json.loads(benchmark_manifest_path.read_text(encoding="utf-8"))
    benchmark_manifest["verifier_configs"] = {"d": str(verifier_config)}
    benchmark_manifest["da3"] = {
        "model_name": "depth-anything/DA3-SMALL",
    }
    _write_json(benchmark_manifest_path, benchmark_manifest)
    captured = {}

    def fake_score_verifier_pairs(_benchmark_root, _configs, da3_runner):
        captured["runner"] = da3_runner
        return [
            {
                "pair_id": "p1",
                "sequence_key": "d/p/s",
                "query_idx": 3,
                "trajectory_deformation_rmse": 0.2,
                "pgo_error_after": 0.1,
            },
            {
                "pair_id": "p2",
                "sequence_key": "d/p/s",
                "query_idx": 3,
                "trajectory_deformation_rmse": 0.5,
                "pgo_error_after": 0.2,
            },
        ]

    monkeypatch.setattr(module, "score_verifier_pairs", fake_score_verifier_pairs)

    assert (
        module.main(
            [
                str(benchmark_root),
                "--method",
                "verifier",
                "--backend",
                "real",
                "--device",
                "cpu",
            ]
        )
        == 0
    )

    assert set(captured["runner"]) == {"d"}
    config = captured["runner"]["d"].config
    assert config.device == "cpu"
    assert config.model_name == str(da3_snapshot)
    assert config.cache_dir == hf_cache
    assert config.process_res == 672
    assert config.triplet_batch_size == 3

    manifest = json.loads(
        (benchmark_root / "scores" / "loopanything.manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["device"] == "cpu"
    assert manifest["da3_model_name"] is None
    assert manifest["da3_model_dir"] is None
    assert manifest["da3_model_path"] == str(da3_model_file)
    assert manifest["da3_model_path_sha256"] == _sha256(da3_model_file)
    assert manifest["da3_checkpoint"] is None
    assert manifest["da3_snapshot"] == str(da3_snapshot)
    assert manifest["da3_snapshot_sha256"] is None
    assert manifest["da3_cache_dir"] == str(hf_cache)
    assert manifest["da3_cache_source"] == "HUGGINGFACE_HUB_CACHE"
    assert manifest["da3_runtime_by_dataset"]["d"]["da3_process_res"] == 672
    assert manifest["da3_runtime_by_dataset"]["d"]["da3_triplet_batch_size"] == 3


def test_real_verifier_uses_and_records_dataset_specific_da3_runners(
    tmp_path,
    monkeypatch,
):
    module = _load_score_script()
    benchmark_root = tmp_path / "benchmark"
    pairs = []
    sequence_rows = []
    for dataset, sequence, pair_id in (
        ("d1", "s1", "p1"),
        ("d2", "s2", "p2"),
    ):
        cache = tmp_path / f"cache_{dataset}"
        (cache / "images").mkdir(parents=True)
        for idx in range(4):
            (cache / "images" / f"{idx:06d}.png").write_bytes(b"image")
        _write_jsonl(
            cache / "keyframes.jsonl",
            [
                {
                    "idx": idx,
                    "image_path": f"images/{idx:06d}.png",
                    "odom_pose": np.eye(4).reshape(-1).tolist(),
                }
                for idx in range(4)
            ],
        )
        pairs.append(
            {
                "pair_id": pair_id,
                "dataset": dataset,
                "platform": "p",
                "sequence": sequence,
                "query_idx": 3,
                "candidate_idx": 0,
                "rank": 1,
            }
        )
        sequence_rows.append(
            {
                "dataset": dataset,
                "platform": "p",
                "sequence": sequence,
                "cache": str(cache),
            }
        )
    _write_jsonl(benchmark_root / "benchmark_pairs.jsonl", pairs)
    model_path = tmp_path / "model.safetensors"
    model_path.write_bytes(b"model")
    _write_json(
        benchmark_root / "manifest.json",
        {
            "sequences": sequence_rows,
            "verifier_configs": {
                "d1": "configs/d1.yaml",
                "d2": "configs/d2.yaml",
            },
            "da3_model_path": str(model_path),
        },
    )
    _write_score_artifact(
        benchmark_root,
        "salad",
        [
            {"pair_id": "p1", "score": 0.1, "status": "ok"},
            {"pair_id": "p2", "score": 0.2, "status": "ok"},
        ],
        evaluation_name="SALAD",
    )
    configs = {
        "d1": _verifier_config(
            tmp_path,
            dataset_name="d1",
            da3={
                "process_res": 448,
                "ref_view_strategy": "first",
                "triplet_batch_size": 2,
            },
        ),
        "d2": _verifier_config(
            tmp_path,
            dataset_name="d2",
            da3={
                "process_res": 672,
                "ref_view_strategy": "first",
                "triplet_batch_size": 5,
            },
        ),
    }

    class FakeRealDa3Runner:
        def __init__(self, config):
            self.config = config

    monkeypatch.setattr(module, "RealDa3Runner", FakeRealDa3Runner)
    monkeypatch.setattr(
        module,
        "resolve_local_da3_model_name_or_path",
        lambda *args, **kwargs: str(model_path),
    )
    runners = module._make_da3_runner(
        benchmark_root,
        configs,
        backend="real",
        device="cpu",
    )

    assert set(runners) == {"d1", "d2"}
    assert runners["d1"].config.process_res == 448
    assert runners["d1"].config.triplet_batch_size == 2
    assert runners["d2"].config.process_res == 672
    assert runners["d2"].config.triplet_batch_size == 5
    selected = {}

    def fake_score_frozen_query_candidates(**kwargs):
        candidate = kwargs["frozen_candidates"][0]
        selected[candidate["pair_id"]] = kwargs["da3_runner"].config.process_res
        return [
            {
                "pair_id": candidate["pair_id"],
                "query_idx": kwargs["query_idx"],
                "candidate_idx": candidate["candidate_idx"],
                "rank": candidate["rank"],
                "trajectory_deformation_rmse": 0.0,
                "pgo_error_after": 0.0,
            }
        ]

    monkeypatch.setattr(
        "robust_loop_verifier.rover_pair_scoring.score_frozen_query_candidates",
        fake_score_frozen_query_candidates,
    )

    score_verifier_pairs(benchmark_root, configs, runners)

    assert selected == {"p1": 448, "p2": 672}
    verifier_manifest = module._verifier_manifest(
        benchmark_root,
        SimpleNamespace(backend="real", device="cpu"),
        da3_runner=runners,
    )
    assert verifier_manifest["da3_runtime_by_dataset"]["d1"]["da3_process_res"] == 448
    assert (
        verifier_manifest["da3_runtime_by_dataset"]["d1"]["da3_triplet_batch_size"]
        == 2
    )
    assert verifier_manifest["da3_runtime_by_dataset"]["d2"]["da3_process_res"] == 672
    assert (
        verifier_manifest["da3_runtime_by_dataset"]["d2"]["da3_triplet_batch_size"]
        == 5
    )
    assert verifier_manifest["da3_model_path"] == str(model_path)
    assert "da3_process_res" not in verifier_manifest


def test_compute_named_scores_rejects_unknown_method():
    with pytest.raises(ValueError, match="unknown named score"):
        compute_named_scores([], "not-a-method")
