from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from robust_loop_verifier.rover_annotation import verify_annotation_seal


def test_converts_da3_geometry_predictions_to_sealed_annotations_and_copies_scores(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script = (
        repo_root
        / "robust_loop_verification_scripts"
        / "convert_da3_geometry_predictions_to_rover_annotations.py"
    )
    benchmark_root = tmp_path / "benchmark"
    geometry_root = tmp_path / "geometry"
    output_root = tmp_path / "auto_labels"
    benchmark_root.mkdir()
    geometry_root.mkdir()
    (benchmark_root / "scores").mkdir()

    pairs = [
        {
            "pair_id": "pair-a",
            "dataset": "Dataset",
            "platform": "Platform",
            "sequence": "Seq",
            "query_idx": 10,
            "candidate_idx": 1,
            "rank": 1,
            "dbow2_score": 0.9,
        },
        {
            "pair_id": "pair-b",
            "dataset": "Dataset",
            "platform": "Platform",
            "sequence": "Seq",
            "query_idx": 11,
            "candidate_idx": 2,
            "rank": 2,
            "dbow2_score": 0.8,
        },
    ]
    pair_text = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in pairs)
    (benchmark_root / "benchmark_pairs.jsonl").write_text(pair_text, encoding="utf-8")
    pair_hash = hashlib.sha256(pair_text.encode("utf-8")).hexdigest()
    manifest = {
        "benchmark_version": "test-benchmark",
        "pair_manifest_sha256": pair_hash,
        "sequence_order": ["Dataset/Platform/Seq"],
        "sequences": [
            {
                "dataset": "Dataset",
                "platform": "Platform",
                "sequence": "Seq",
                "cache": str(tmp_path / "cache"),
            }
        ],
    }
    (benchmark_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (benchmark_root / "scores" / "dbow2.jsonl").write_text(
        '{"pair_id":"pair-a","score":1.0,"status":"ok"}\n'
        '{"pair_id":"pair-b","score":0.5,"status":"ok"}\n',
        encoding="utf-8",
    )
    (benchmark_root / "scores" / "dbow2.manifest.json").write_text(
        '{"method":"DBoW2"}\n',
        encoding="utf-8",
    )
    for filename in ("benchmark_pairs.jsonl", "manifest.json"):
        (geometry_root / filename).write_bytes((benchmark_root / filename).read_bytes())
    (geometry_root / "geometry_predictions.jsonl").write_text(
        json.dumps({"pair_id": "pair-a", "automatic_label": 1}, sort_keys=True) + "\n"
        + json.dumps({"pair_id": "pair-b", "automatic_label": 0}, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            str(script),
            str(benchmark_root),
            "--geometry-root",
            str(geometry_root),
            "--output-root",
            str(output_root),
            "--copy-scores",
        ],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    verify_annotation_seal(
        output_root / "manifest.json",
        output_root / "benchmark_pairs.jsonl",
        output_root / "annotations.jsonl",
        output_root / "annotation_seal.json",
    )
    annotations = [
        json.loads(line)
        for line in (output_root / "annotations.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [(row["pair_id"], row["label"]) for row in annotations] == [
        ("pair-a", 1),
        ("pair-b", 0),
    ]
    assert (output_root / "geometry_predictions.jsonl").read_bytes() == (
        geometry_root / "geometry_predictions.jsonl"
    ).read_bytes()
    assert (output_root / "scores" / "dbow2.jsonl").read_text(encoding="utf-8").startswith(
        '{"pair_id":"pair-a"'
    )
    conversion = json.loads(
        (output_root / "auto_label_conversion_manifest.json").read_text(encoding="utf-8")
    )
    assert conversion["positive_count"] == 1
    assert conversion["negative_count"] == 1


def test_converter_rejects_output_root_equal_to_source_benchmark(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script = (
        repo_root
        / "robust_loop_verification_scripts"
        / "convert_da3_geometry_predictions_to_rover_annotations.py"
    )
    benchmark_root = tmp_path / "benchmark"
    geometry_root = tmp_path / "geometry"
    benchmark_root.mkdir()
    geometry_root.mkdir()

    result = subprocess.run(
        [
            str(script),
            str(benchmark_root),
            "--geometry-root",
            str(geometry_root),
            "--output-root",
            str(benchmark_root),
        ],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    assert "must differ from benchmark_root" in result.stderr


def test_prepares_unsealed_review_root_with_geometry_and_scores(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script = (
        repo_root
        / "robust_loop_verification_scripts"
        / "convert_da3_geometry_predictions_to_rover_annotations.py"
    )
    benchmark_root = tmp_path / "benchmark"
    geometry_root = tmp_path / "geometry"
    output_root = tmp_path / "review"
    benchmark_root.mkdir()
    geometry_root.mkdir()
    (benchmark_root / "scores").mkdir()

    pair_text = (
        '{"candidate_idx":1,"dataset":"Dataset","dbow2_score":0.9,'
        '"pair_id":"pair-a","platform":"Platform","query_idx":10,"rank":1,'
        '"sequence":"Seq"}\n'
    )
    (benchmark_root / "benchmark_pairs.jsonl").write_text(pair_text, encoding="utf-8")
    pair_hash = hashlib.sha256(pair_text.encode("utf-8")).hexdigest()
    manifest = {
        "benchmark_version": "test-benchmark",
        "pair_manifest_sha256": pair_hash,
        "sequence_order": ["Dataset/Platform/Seq"],
        "sequences": [
            {
                "dataset": "Dataset",
                "platform": "Platform",
                "sequence": "Seq",
                "cache": str(tmp_path / "cache"),
            }
        ],
    }
    (benchmark_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (benchmark_root / "scores" / "dbow2.jsonl").write_text(
        '{"pair_id":"pair-a","score":1.0,"status":"ok"}\n',
        encoding="utf-8",
    )
    for filename in ("benchmark_pairs.jsonl", "manifest.json"):
        (geometry_root / filename).write_bytes((benchmark_root / filename).read_bytes())
    (geometry_root / "geometry_predictions.jsonl").write_text(
        '{"automatic_label":1,"pair_id":"pair-a"}\n',
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            str(script),
            str(benchmark_root),
            "--geometry-root",
            str(geometry_root),
            "--output-root",
            str(output_root),
            "--copy-scores",
            "--review-only",
        ],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (output_root / "benchmark_pairs.jsonl").is_file()
    assert (output_root / "manifest.json").is_file()
    assert (output_root / "geometry_predictions.jsonl").is_file()
    assert (output_root / "scores" / "dbow2.jsonl").is_file()
    assert not (output_root / "annotations.jsonl").exists()
    assert not (output_root / "annotation_seal.json").exists()
