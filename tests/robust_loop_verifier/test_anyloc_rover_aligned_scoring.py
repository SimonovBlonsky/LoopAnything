from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from baseline_scripts.score_anyloc_rover_aligned_benchmark import (
    AnyLocConfig,
    AnyLocDescriptorResult,
    AnyLocImageDescriptor,
    score_benchmark_pairs,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class _FakeDescriptor(AnyLocImageDescriptor):
    def __init__(self) -> None:
        self.calls: list[Path] = []
        self.descriptors = {
            "query_b.png": np.array([1.0, 0.0], dtype=np.float32),
            "candidate_b.png": np.array([0.0, 1.0], dtype=np.float32),
            "query_a.png": np.array([3.0, 4.0], dtype=np.float32),
            "candidate_a.png": np.array([6.0, 8.0], dtype=np.float32),
        }

    def describe(self, image_path: Path) -> AnyLocDescriptorResult:
        self.calls.append(image_path)
        return AnyLocDescriptorResult(
            descriptor=self.descriptors[image_path.name],
            num_patches=64,
            resized_height=224,
            resized_width=336,
        )


def test_score_benchmark_pairs_preserves_pair_order_and_reuses_unique_image_descriptors(
    tmp_path,
):
    cache = tmp_path / "cache"
    (cache / "images").mkdir(parents=True)
    _write_jsonl(
        tmp_path / "benchmark" / "benchmark_pairs.jsonl",
        [
            {
                "pair_id": "pair-b",
                "dataset": "d",
                "platform": "p",
                "sequence": "s",
                "query_image": "images/query_b.png",
                "candidate_image": "images/candidate_b.png",
            },
            {
                "pair_id": "pair-a",
                "dataset": "d",
                "platform": "p",
                "sequence": "s",
                "query_image": "images/query_a.png",
                "candidate_image": "images/candidate_a.png",
            },
            {
                "pair_id": "pair-a-repeat-query",
                "dataset": "d",
                "platform": "p",
                "sequence": "s",
                "query_image": "images/query_a.png",
                "candidate_image": "images/candidate_b.png",
            },
        ],
    )
    _write_json(
        tmp_path / "benchmark" / "manifest.json",
        {
            "sequences": [
                {
                    "dataset": "d",
                    "platform": "p",
                    "sequence": "s",
                    "cache": str(cache),
                }
            ]
        },
    )

    descriptor = _FakeDescriptor()
    rows = score_benchmark_pairs(tmp_path / "benchmark", descriptor)

    assert [row["pair_id"] for row in rows] == [
        "pair-b",
        "pair-a",
        "pair-a-repeat-query",
    ]
    assert rows[0]["status"] == "ok"
    assert rows[0]["score"] == 0.0
    assert rows[0]["num_patches_query"] == 64
    assert rows[0]["num_patches_candidate"] == 64
    assert rows[1]["status"] == "ok"
    assert np.isclose(rows[1]["score"], 1.0)
    assert np.isclose(rows[2]["score"], 0.8)
    assert sorted(path.name for path in descriptor.calls) == [
        "candidate_a.png",
        "candidate_b.png",
        "query_a.png",
        "query_b.png",
    ]


def test_anyloc_descriptor_prepare_fails_fast_for_missing_vocabulary(tmp_path):
    anyloc_root = tmp_path / "AnyLoc"
    anyloc_root.mkdir()
    descriptor = AnyLocImageDescriptor(
        anyloc_root=anyloc_root,
        cache_root=tmp_path / "missing_cache",
        device="cpu",
        config=AnyLocConfig(),
    )

    try:
        descriptor.prepare()
    except FileNotFoundError as exc:
        assert "AnyLoc vocabulary not found" in str(exc)
    else:
        raise AssertionError("expected missing vocabulary to fail before scoring pairs")
