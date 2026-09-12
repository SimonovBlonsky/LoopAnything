from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from baseline_scripts.score_dust3r_rover_aligned_benchmark import (
    DUSt3RMatchResult,
    RansacConfig,
    compute_ransac_summary,
    score_benchmark_pairs,
    trusted_torch_load_wrapper,
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


class _FakeMatcher:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, Path]] = []

    def match(self, query_image: Path, candidate_image: Path) -> DUSt3RMatchResult:
        self.calls.append((query_image, candidate_image))
        count = 10 if candidate_image.name == "candidate_a.png" else 6
        points = np.column_stack(
            [
                np.arange(count, dtype=np.float32),
                np.arange(count, dtype=np.float32) * 3.0,
            ]
        )
        return DUSt3RMatchResult(
            mkpts0=points,
            mkpts1=points + 2.0,
            num_reciprocal_matches=count,
            mean_confidence=3.5,
            valid_ratio=0.25,
        )


def _fake_ransac(mkpts0, mkpts1, config):
    return {
        "num_matches": int(len(mkpts0)),
        "num_inliers": 4,
        "inlier_ratio": 4.0 / float(len(mkpts0)),
    }


def test_score_benchmark_pairs_preserves_frozen_pair_order_and_paths(tmp_path):
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

    matcher = _FakeMatcher()
    rows = score_benchmark_pairs(
        tmp_path / "benchmark",
        matcher,
        ransac_config=RansacConfig(min_matches=8),
        ransac_summary_fn=_fake_ransac,
    )

    assert [row["pair_id"] for row in rows] == ["pair-b", "pair-a"]
    assert rows[0]["status"] == "ok"
    assert rows[0]["score"] == 0.0
    assert rows[0]["num_matches"] == 6
    assert rows[0]["num_inliers"] == 0
    assert rows[1]["status"] == "ok"
    assert rows[1]["score"] == 4.0
    assert rows[1]["num_matches"] == 10
    assert rows[1]["num_inliers"] == 4
    assert rows[1]["num_reciprocal_matches"] == 10
    assert rows[1]["mean_confidence"] == 3.5
    assert rows[1]["valid_ratio"] == 0.25
    assert matcher.calls == [
        (cache / "images/query_b.png", cache / "images/candidate_b.png"),
        (cache / "images/query_a.png", cache / "images/candidate_a.png"),
    ]


def test_compute_ransac_summary_returns_zero_for_too_few_matches():
    points = np.zeros((7, 2), dtype=np.float32)

    summary = compute_ransac_summary(points, points, RansacConfig(min_matches=8))

    assert summary == {
        "num_matches": 7,
        "num_inliers": 0,
        "inlier_ratio": 0.0,
    }


def test_trusted_torch_load_wrapper_sets_weights_only_false_by_default():
    calls = []

    def fake_load(*args, **kwargs):
        calls.append((args, kwargs))
        return "loaded"

    wrapped = trusted_torch_load_wrapper(fake_load)

    assert wrapped("checkpoint.pth", map_location="cpu") == "loaded"
    assert calls[0][1]["weights_only"] is False


def test_trusted_torch_load_wrapper_respects_explicit_weights_only_value():
    calls = []

    def fake_load(*args, **kwargs):
        calls.append((args, kwargs))
        return "loaded"

    wrapped = trusted_torch_load_wrapper(fake_load)

    assert wrapped("checkpoint.pth", weights_only=True) == "loaded"
    assert calls[0][1]["weights_only"] is True
