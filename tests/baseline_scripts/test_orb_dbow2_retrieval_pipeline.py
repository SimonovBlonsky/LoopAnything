from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _load_module():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "baseline_scripts"
        / "orb_dbow2_retrieval_pipeline.py"
    )
    spec = importlib.util.spec_from_file_location("orb_dbow2_retrieval_pipeline", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_sequence_cache(root: Path, platform: str, sequence: str) -> Path:
    sequence_cache = root / platform / sequence
    (sequence_cache / "images").mkdir(parents=True)
    keyframes = []
    for idx in range(5):
        image_path = f"images/{idx:06d}.png"
        (sequence_cache / image_path).write_bytes(b"fake-image")
        keyframes.append({"idx": idx, "image_path": image_path})
    _write_jsonl(sequence_cache / "keyframes.jsonl", keyframes)
    _write_jsonl(
        sequence_cache / "positives.jsonl",
        [
            {"query_idx": 0, "positive_indices": []},
            {"query_idx": 1, "positive_indices": []},
            {"query_idx": 2, "positive_indices": []},
            {"query_idx": 3, "positive_indices": [0]},
            {"query_idx": 4, "positive_indices": [1]},
        ],
    )
    (sequence_cache / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_name": "FusionPortableV2",
                "platform": platform,
                "sequence_name": sequence,
                "recent_exclusion_keyframes": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return sequence_cache


def test_fusionportablev2_defaults_match_main_experiment_sequences(tmp_path):
    module = _load_module()

    assert module.DEFAULT_SEQUENCES_BY_PLATFORM["handheld"] == [
        "handheld_escalator00",
        "handheld_escalator01",
        "handheld_grass00",
        "handheld_room00",
        "handheld_room01",
    ]
    assert module.DEFAULT_SEQUENCES_BY_PLATFORM["ugv"] == [
        "ugv_campus01",
        "ugv_parking00",
        "ugv_parking01",
        "ugv_parking02",
        "ugv_parking03",
    ]

    existing = _write_sequence_cache(tmp_path, "ugv", "ugv_parking01")
    specs = module.resolve_sequence_specs(
        dataset="fusionportablev2",
        cache_root=tmp_path,
        platforms=["ugv"],
        strict=False,
    )

    assert [spec.sequence_cache for spec in specs] == [existing]
    assert specs[0].platform == "ugv"
    assert specs[0].sequence == "ugv_parking01"


def test_defaults_include_geode_and_ntu_viral_main_experiment_sequences(tmp_path):
    module = _load_module()

    geode_cache = _write_sequence_cache(tmp_path, "Offroad", "Offroad02_beta")
    ntu_cache = _write_sequence_cache(tmp_path, "NTU-VIRAL", "eee_02")

    geode_specs = module.resolve_sequence_specs(
        dataset="geode",
        cache_root=tmp_path,
        platforms=["Offroad"],
        strict=True,
    )
    ntu_specs = module.resolve_sequence_specs(
        dataset="ntu_viral",
        cache_root=tmp_path,
        platforms=["NTU-VIRAL"],
        strict=False,
    )

    assert [spec.sequence_cache for spec in geode_specs] == [geode_cache]
    assert [spec.sequence_cache for spec in ntu_specs] == [ntu_cache]


def test_evaluate_sequence_specs_keep_going_records_failed_sequence(tmp_path):
    module = _load_module()
    specs = [
        module.SequenceSpec("ugv", "bad", tmp_path / "bad"),
        module.SequenceSpec("ugv", "good", tmp_path / "good"),
    ]

    def evaluate(spec):
        if spec.sequence == "bad":
            raise RuntimeError("broken sequence")
        return spec.sequence

    results, failures = module.evaluate_sequence_specs(specs, evaluate, keep_going=True)

    assert results == ["good"]
    assert failures == [{"platform": "ugv", "sequence": "bad", "error": "broken sequence"}]


def test_parse_helper_output_rejects_malformed_rows():
    module = _load_module()

    records = module.parse_helper_output(
        "3\t0\t1\t0.9\n"
        "4\t1\t1\t0.8\n"
        "4\t0\t2\t0.7\n"
    )

    assert [candidate.candidate_idx for candidate in records[4].candidates] == [1, 0]
    assert records[3].candidates[0].score == 0.9

    try:
        module.parse_helper_output("3\t0\t1\n")
    except ValueError as exc:
        assert "must have 4 tab-separated fields" in str(exc)
    else:
        raise AssertionError("malformed helper output was accepted")


def test_helper_source_emits_full_precision_scores():
    module = _load_module()

    source = module._helper_source()

    assert "#include <iomanip>" in source
    assert "std::setprecision(17)" in source


def test_existing_helper_binary_rebuilds_when_generated_source_changes(tmp_path, monkeypatch):
    module = _load_module()
    orb_root = tmp_path / "ORB_SLAM3"
    (orb_root / "Thirdparty" / "DBoW2" / "lib").mkdir(parents=True)
    (orb_root / "src").mkdir()
    (orb_root / "include").mkdir()
    vocabulary_path = orb_root / "Vocabulary" / "ORBvoc.txt"
    vocabulary_path.parent.mkdir()
    vocabulary_path.write_text("vocabulary\n", encoding="utf-8")
    helper_build_dir = tmp_path / "helper_build"
    helper_build_dir.mkdir()
    (helper_build_dir / "orb_dbow2_retrieval_helper.cpp").write_text(
        "old source\n",
        encoding="utf-8",
    )
    (helper_build_dir / "orb_dbow2_retrieval_helper").write_bytes(b"old binary")
    compile_calls = []

    def fake_compile(source_path, binary_path):
        compile_calls.append((source_path, binary_path))
        binary_path.write_bytes(b"new binary")

    backend = module.OrbDbow2RetrievalBackend(
        orb_slam3_root=orb_root,
        vocabulary_path=vocabulary_path,
        helper_build_dir=helper_build_dir,
        rebuild_helper=False,
        nfeatures=1000,
        scale_factor=1.2,
        nlevels=8,
        ini_fast=20,
        min_fast=7,
    )
    monkeypatch.setattr(backend, "_compile_helper", fake_compile)

    backend.ensure_helper_built()

    assert compile_calls == [
        (
            helper_build_dir / "orb_dbow2_retrieval_helper.cpp",
            helper_build_dir / "orb_dbow2_retrieval_helper",
        )
    ]


def test_existing_helper_binary_rebuilds_when_orb_slam3_commit_changes(
    tmp_path, monkeypatch
):
    module = _load_module()
    orb_root = tmp_path / "ORB_SLAM3"
    (orb_root / "Thirdparty" / "DBoW2" / "lib").mkdir(parents=True)
    (orb_root / "src").mkdir()
    (orb_root / "include").mkdir()
    vocabulary_path = orb_root / "Vocabulary" / "ORBvoc.txt"
    vocabulary_path.parent.mkdir()
    vocabulary_path.write_text("vocabulary\n", encoding="utf-8")
    helper_build_dir = tmp_path / "helper_build"
    helper_build_dir.mkdir()
    source_path = helper_build_dir / "orb_dbow2_retrieval_helper.cpp"
    binary_path = helper_build_dir / "orb_dbow2_retrieval_helper"
    source_path.write_text(module._helper_source(), encoding="utf-8")
    binary_path.write_bytes(b"old binary")
    (helper_build_dir / "orb_dbow2_retrieval_helper.build.json").write_text(
        json.dumps(
            {
                "helper_source_sha256": module._sha256_text(module._helper_source()),
                "orb_slam3_git_commit": "old-commit",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    compile_calls = []

    def fake_compile(source_path, binary_path):
        compile_calls.append((source_path, binary_path))
        binary_path.write_bytes(b"new binary")

    backend = module.OrbDbow2RetrievalBackend(
        orb_slam3_root=orb_root,
        vocabulary_path=vocabulary_path,
        helper_build_dir=helper_build_dir,
        rebuild_helper=False,
        nfeatures=1000,
        scale_factor=1.2,
        nlevels=8,
        ini_fast=20,
        min_fast=7,
    )
    monkeypatch.setattr(backend, "_compile_helper", fake_compile)
    monkeypatch.setattr(module, "_git_commit_best_effort", lambda repo_root: "new-commit")

    backend.ensure_helper_built()

    assert compile_calls == [(source_path, binary_path)]
    metadata = json.loads((helper_build_dir / "orb_dbow2_retrieval_helper.build.json").read_text())
    assert metadata["orb_slam3_git_commit"] == "new-commit"
    assert metadata["helper_source_sha256"] == module._sha256_text(module._helper_source())


def test_failed_helper_rebuild_removes_stale_binary(tmp_path, monkeypatch):
    module = _load_module()
    orb_root = tmp_path / "ORB_SLAM3"
    (orb_root / "Thirdparty" / "DBoW2" / "lib").mkdir(parents=True)
    (orb_root / "src").mkdir()
    (orb_root / "include").mkdir()
    vocabulary_path = orb_root / "Vocabulary" / "ORBvoc.txt"
    vocabulary_path.parent.mkdir()
    vocabulary_path.write_text("vocabulary\n", encoding="utf-8")
    helper_build_dir = tmp_path / "helper_build"
    helper_build_dir.mkdir()
    binary_path = helper_build_dir / "orb_dbow2_retrieval_helper"
    (helper_build_dir / "orb_dbow2_retrieval_helper.cpp").write_text(
        "old source\n",
        encoding="utf-8",
    )
    binary_path.write_bytes(b"old binary")

    def fail_compile(source_path, binary_path):
        raise subprocess.CalledProcessError(1, ["g++"])

    backend = module.OrbDbow2RetrievalBackend(
        orb_slam3_root=orb_root,
        vocabulary_path=vocabulary_path,
        helper_build_dir=helper_build_dir,
        rebuild_helper=False,
        nfeatures=1000,
        scale_factor=1.2,
        nlevels=8,
        ini_fast=20,
        min_fast=7,
    )
    monkeypatch.setattr(backend, "_compile_helper", fail_compile)

    with pytest.raises(subprocess.CalledProcessError):
        backend.ensure_helper_built()

    assert not binary_path.exists()


def test_resolve_ldd_library_reports_not_found(monkeypatch, tmp_path):
    module = _load_module()
    binary_path = tmp_path / "helper"
    binary_path.write_bytes(b"fake")

    def fake_check_output(command, text):
        assert command == ["ldd", str(binary_path)]
        assert text is True
        return "\tlibDBoW2.so => not found\n"

    monkeypatch.setattr(module.subprocess, "check_output", fake_check_output)

    with pytest.raises(FileNotFoundError, match="libDBoW2.so.*not found"):
        module._resolve_ldd_library(binary_path, "libDBoW2.so")


def test_evaluate_sequence_uses_orb_dbow2_scores_and_main_metrics(tmp_path):
    module = _load_module()
    sequence_cache = _write_sequence_cache(tmp_path, "handheld", "handheld_escalator00")

    class FakeBackend:
        def retrieve(self, keyframes, top_k, recent_exclusion_keyframes):
            assert top_k == 2
            assert recent_exclusion_keyframes == 1
            return {
                3: module.RetrievalRecord(
                    query_idx=3,
                    candidates=[
                        module.RetrievalCandidate(3, 0, 1, 0.9),
                        module.RetrievalCandidate(3, 1, 2, 0.1),
                    ],
                ),
                4: module.RetrievalRecord(
                    query_idx=4,
                    candidates=[
                        module.RetrievalCandidate(4, 1, 1, 0.8),
                        module.RetrievalCandidate(4, 0, 2, 0.2),
                    ],
                ),
            }

    result = module.evaluate_sequence(
        module.SequenceSpec(
            platform="handheld",
            sequence="handheld_escalator00",
            sequence_cache=sequence_cache,
        ),
        retrieval_backend=FakeBackend(),
        top_k=2,
        recent_exclusion_keyframes=None,
    )

    assert result.metrics == {"AP": 1.0, "MR@100P": 1.0}
    assert result.candidate_count == 4
    assert result.positive_candidate_count == 2
    assert result.query_count == 5
