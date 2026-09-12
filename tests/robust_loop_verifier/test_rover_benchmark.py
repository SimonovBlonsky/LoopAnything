from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from robust_loop_verifier.rover_benchmark import (
    BenchmarkPair,
    compute_recent_exclusion,
    sample_eligible_queries,
    validate_pair_manifest,
)


def _pair(query_idx: int, candidate_idx: int, rank: int, pair_id: str | None = None):
    return BenchmarkPair(
        pair_id=pair_id or f"d_p_s_q{query_idx:06d}_c{candidate_idx:06d}",
        dataset="d",
        platform="p",
        sequence="s",
        query_idx=query_idx,
        candidate_idx=candidate_idx,
        rank=rank,
        dbow2_score=1.0 / rank,
        query_image=f"images/{query_idx:06d}.png",
        candidate_image=f"images/{candidate_idx:06d}.png",
        query_context=(
            f"images/{query_idx - 1:06d}.png",
            f"images/{query_idx:06d}.png",
            f"images/{query_idx + 1:06d}.png",
        ),
        candidate_context=(None, f"images/{candidate_idx:06d}.png", None),
    )


def _legal_pairs(query_idx: int = 20):
    return [_pair(query_idx, candidate, rank) for rank, candidate in enumerate(range(10), start=1)]


def _sequence_config(count: int = 10):
    return [
        {
            "dataset": f"d{idx}",
            "platform": "p",
            "sequence": f"s{idx}",
            "cache": f"/tmp/cache/{idx}",
        }
        for idx in range(count)
    ]


def _write_sequence_cache(root, sequence_name: str):
    sequence_cache = root / sequence_name
    (sequence_cache / "images").mkdir(parents=True)
    rows = []
    for index in range(20):
        image_path = f"images/{index:06d}.png"
        (sequence_cache / image_path).write_bytes(b"fake-image")
        rows.append({"idx": index, "image_path": image_path})
    (sequence_cache / "keyframes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return sequence_cache


def _benchmark_config(tmp_path, sequences, **overrides):
    config = {
        "benchmark_version": "benchmark_v1",
        "query_limit_per_sequence": 40,
        "retrieval_top_k": 10,
        "annotation_shuffle_seed": 20260610,
        "orb_slam3_root": str(tmp_path / "ORB_SLAM3"),
        "helper_build_dir": str(tmp_path / "helper_build"),
        "vocabulary": str(tmp_path / "ORBvoc.txt"),
        "orb": {
            "nfeatures": 1000,
            "scale_factor": 1.2,
            "nlevels": 8,
            "ini_fast": 20,
            "min_fast": 7,
        },
        "sequences": sequences,
    }
    config.update(overrides)
    return config


def test_recent_exclusion_uses_single_clipped_sequence_rule():
    assert compute_recent_exclusion(50) == 5
    assert compute_recent_exclusion(116) == 9
    assert compute_recent_exclusion(184) == 15
    assert compute_recent_exclusion(1000) == 30


def test_query_sampling_is_deterministic_and_covers_ordinal_range():
    eligible = list(range(10, 110))
    selected = sample_eligible_queries(eligible, limit=40)

    assert selected == [
        11,
        13,
        16,
        18,
        21,
        23,
        26,
        28,
        31,
        33,
        36,
        38,
        41,
        43,
        46,
        48,
        51,
        53,
        56,
        58,
        61,
        63,
        66,
        68,
        71,
        73,
        76,
        78,
        81,
        83,
        86,
        88,
        91,
        93,
        96,
        98,
        101,
        103,
        106,
        108,
    ]


def test_query_sampling_prefers_smaller_index_on_exact_tie():
    eligible = [10, 20, 30, 40, 50, 60, 70, 80]
    assert sample_eligible_queries(eligible, limit=4) == [10, 30, 50, 70]


def test_query_sampling_keeps_all_when_fewer_than_limit():
    assert sample_eligible_queries([7, 9, 12], limit=40) == [7, 9, 12]


def test_manifest_requires_exactly_ten_legal_candidates_per_query():
    validate_pair_manifest(_legal_pairs(), recent_exclusion_by_sequence={("d", "p", "s"): 5})


def test_manifest_rejects_empty_pairs():
    with pytest.raises(ValueError, match="empty pair manifest"):
        validate_pair_manifest([], recent_exclusion_by_sequence={("d", "p", "s"): 5})


def test_manifest_rejects_duplicate_pair_id():
    pairs = _legal_pairs()
    pairs[1] = _pair(20, 1, 2, pair_id=pairs[0].pair_id)

    with pytest.raises(ValueError, match="duplicate pair_id"):
        validate_pair_manifest(pairs, recent_exclusion_by_sequence={("d", "p", "s"): 5})


def test_manifest_rejects_duplicate_rank():
    pairs = _legal_pairs()
    pairs[1] = _pair(20, 1, 1)

    with pytest.raises(ValueError, match="duplicate rank"):
        validate_pair_manifest(pairs, recent_exclusion_by_sequence={("d", "p", "s"): 5})


def test_manifest_rejects_non_causal_candidate():
    pairs = _legal_pairs()
    pairs[0] = _pair(20, 20, 1)

    with pytest.raises(ValueError, match="non-causal"):
        validate_pair_manifest(pairs, recent_exclusion_by_sequence={("d", "p", "s"): 5})


def test_manifest_rejects_recent_candidate():
    pairs = _legal_pairs()
    pairs[0] = _pair(20, 16, 1)

    with pytest.raises(ValueError, match="recent"):
        validate_pair_manifest(pairs, recent_exclusion_by_sequence={("d", "p", "s"): 5})


def test_manifest_rejects_missing_rank():
    pairs = _legal_pairs()
    pairs[9] = _pair(20, 9, 11)

    with pytest.raises(ValueError, match="ranks 1..10"):
        validate_pair_manifest(pairs, recent_exclusion_by_sequence={("d", "p", "s"): 5})


@pytest.mark.parametrize("count", [9, 11])
def test_manifest_rejects_wrong_candidate_count(count):
    pairs = [_pair(30, candidate, rank) for rank, candidate in enumerate(range(count), start=1)]

    with pytest.raises(ValueError, match="exactly 10"):
        validate_pair_manifest(pairs, recent_exclusion_by_sequence={("d", "p", "s"): 5})


@pytest.mark.parametrize("count", [9, 11])
def test_builder_rejects_non_ten_sequence_config(count):
    builder = importlib.import_module(
        "robust_loop_verification_scripts.build_rover_aligned_benchmark"
    )
    config = {"sequences": _sequence_config(count)}

    with pytest.raises(ValueError, match="exactly 10"):
        builder.validate_benchmark_config(config)


def test_builder_rejects_empty_sequence_group():
    builder = importlib.import_module(
        "robust_loop_verification_scripts.build_rover_aligned_benchmark"
    )

    with pytest.raises(ValueError, match="sequences"):
        builder.validate_benchmark_config({"sequences": []})


def test_builder_rejects_existing_non_empty_output_root(tmp_path):
    builder = importlib.import_module(
        "robust_loop_verification_scripts.build_rover_aligned_benchmark"
    )
    output_root = tmp_path / "benchmark"
    output_root.mkdir()
    (output_root / "existing.txt").write_text("already here\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="non-empty"):
        builder.refuse_non_empty_output_root(output_root)


def test_builder_script_help_works_from_repo_root():
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "robust_loop_verification_scripts/build_rover_aligned_benchmark.py", "--help"],
        cwd=str(Path(__file__).resolve().parents[2]),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--config" in completed.stdout
    assert "--output-root" in completed.stdout


def test_builder_rejects_sequence_with_no_selected_queries_before_writing(tmp_path, monkeypatch):
    builder = importlib.import_module(
        "robust_loop_verification_scripts.build_rover_aligned_benchmark"
    )
    sequences = []
    for index in range(10):
        cache = _write_sequence_cache(tmp_path / "cache", f"s{index}")
        sequences.append(
            {
                "dataset": f"d{index}",
                "platform": "p",
                "sequence": f"s{index}",
                "cache": str(cache),
            }
        )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        json.dumps(_benchmark_config(tmp_path, sequences)),
        encoding="utf-8",
    )
    output_root = tmp_path / "benchmark"

    class EmptyRetrievalBackend:
        def __init__(self, *args, **kwargs):
            pass

        def helper_fingerprints(self):
            return {
                "vocabulary_path": str(tmp_path / "ORBvoc.txt"),
                "vocabulary_sha256": "vocab-hash",
                "orb_slam3_git_commit": "orb-commit",
                "helper_source_path": str(tmp_path / "helper.cpp"),
                "helper_source_sha256": "source-hash",
                "helper_binary_path": str(tmp_path / "helper"),
                "helper_binary_sha256": "binary-hash",
                "dbow2_shared_library_path": str(tmp_path / "libDBoW2.so"),
                "dbow2_shared_library_sha256": "lib-hash",
            }

        def retrieve(self, keyframes, top_k, recent_exclusion_keyframes):
            return {}

    monkeypatch.setattr(builder, "OrbDbow2RetrievalBackend", EmptyRetrievalBackend)

    with pytest.raises(ValueError, match="no selected queries|empty sequence group"):
        builder.build_rover_aligned_benchmark(
            config_path=config_path,
            output_root=output_root,
            rebuild_dbow2_helper=False,
        )

    assert not output_root.exists() or not any(output_root.iterdir())


def test_builder_rejects_retrieval_top_k_other_than_ten_before_backend(tmp_path, monkeypatch):
    builder = importlib.import_module(
        "robust_loop_verification_scripts.build_rover_aligned_benchmark"
    )
    sequences = _sequence_config(10)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        json.dumps(_benchmark_config(tmp_path, sequences, retrieval_top_k=9)),
        encoding="utf-8",
    )

    def fail_backend(*args, **kwargs):
        raise AssertionError("backend should not be constructed")

    monkeypatch.setattr(builder, "OrbDbow2RetrievalBackend", fail_backend)

    with pytest.raises(ValueError, match="retrieval_top_k.*10"):
        builder.build_rover_aligned_benchmark(
            config_path=config_path,
            output_root=tmp_path / "benchmark",
            rebuild_dbow2_helper=False,
        )


def test_builder_rejects_helper_build_dir_inside_output_root_before_backend(
    tmp_path, monkeypatch
):
    builder = importlib.import_module(
        "robust_loop_verification_scripts.build_rover_aligned_benchmark"
    )
    sequences = _sequence_config(10)
    output_root = tmp_path / "benchmark"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        json.dumps(
            _benchmark_config(
                tmp_path,
                sequences,
                helper_build_dir=str(output_root / "helper_build"),
            )
        ),
        encoding="utf-8",
    )

    def fail_backend(*args, **kwargs):
        raise AssertionError("backend should not be constructed")

    monkeypatch.setattr(builder, "OrbDbow2RetrievalBackend", fail_backend)

    with pytest.raises(ValueError, match="helper_build_dir.*output_root"):
        builder.build_rover_aligned_benchmark(
            config_path=config_path,
            output_root=output_root,
            rebuild_dbow2_helper=False,
        )


def test_benchmark_manifest_contains_required_fingerprints():
    builder = importlib.import_module(
        "robust_loop_verification_scripts.build_rover_aligned_benchmark"
    )
    sequences = validate_config = builder.validate_benchmark_config(
        {"sequences": _sequence_config(10)}
    )
    assert validate_config == sequences

    manifest = builder.build_benchmark_manifest(
        config={
            "benchmark_version": "benchmark_v1",
            "annotation_shuffle_seed": 20260610,
            "vocabulary": "/tmp/ORBvoc.txt",
        },
        sequences=sequences,
        helper_fingerprints={
            "vocabulary_path": "/tmp/ORBvoc.txt",
            "vocabulary_sha256": "vocab-hash",
            "orb_slam3_git_commit": "orb-commit",
            "helper_source_path": "/tmp/helper.cpp",
            "helper_source_sha256": "source-hash",
            "helper_binary_path": "/tmp/helper",
            "helper_binary_sha256": "binary-hash",
            "dbow2_shared_library_path": "/tmp/libDBoW2.so",
            "dbow2_shared_library_sha256": "lib-hash",
        },
        recent_exclusion_by_sequence={
            (sequence.dataset, sequence.platform, sequence.sequence): 5 for sequence in sequences
        },
        eligible_query_counts={
            (sequence.dataset, sequence.platform, sequence.sequence): 40 for sequence in sequences
        },
        selected_query_counts={
            (sequence.dataset, sequence.platform, sequence.sequence): 40 for sequence in sequences
        },
        pair_counts={
            (sequence.dataset, sequence.platform, sequence.sequence): 400 for sequence in sequences
        },
        pair_manifest_sha256="pairs-hash",
    )

    assert len(manifest["cache_roots"]) == 10
    assert manifest["vocabulary_sha256"] == "vocab-hash"
    assert manifest["orb_slam3_git_commit"] == "orb-commit"
    assert manifest["helper_source_sha256"] == "source-hash"
    assert manifest["helper_binary_sha256"] == "binary-hash"
    assert manifest["dbow2_shared_library_path"] == "/tmp/libDBoW2.so"
    assert manifest["dbow2_shared_library_sha256"] == "lib-hash"
