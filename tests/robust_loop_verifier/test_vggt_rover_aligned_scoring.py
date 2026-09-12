from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from baseline_scripts.score_vggt_rover_aligned_benchmark import (
    RansacConfig,
    VGGTConfig,
    VGGTMatchResult,
    VGGTTrackMatcher,
    compute_ransac_summary,
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


class _FakeMatcher:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, Path]] = []

    def match(self, query_image: Path, candidate_image: Path) -> VGGTMatchResult:
        self.calls.append((query_image, candidate_image))
        count = 12 if candidate_image.name == "candidate_a.png" else 5
        points = np.column_stack(
            [
                np.arange(count, dtype=np.float32),
                np.arange(count, dtype=np.float32) * 2.0,
            ]
        )
        return VGGTMatchResult(
            mkpts0=points,
            mkpts1=points + 1.0,
            num_query_points=16,
            num_visible_tracks=count,
            mean_track_confidence=0.8,
            mean_visibility=0.7,
        )


def _fake_ransac(mkpts0, mkpts1, config):
    return {
        "num_matches": int(len(mkpts0)),
        "num_inliers": 8,
        "inlier_ratio": 8.0 / float(len(mkpts0)),
    }


def test_score_benchmark_pairs_preserves_frozen_pair_order_and_vggt_fields(tmp_path):
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
    assert rows[0]["num_matches"] == 5
    assert rows[0]["num_inliers"] == 0
    assert rows[0]["num_query_points"] == 16
    assert rows[0]["num_visible_tracks"] == 5
    assert rows[1]["status"] == "ok"
    assert rows[1]["score"] == 8.0
    assert rows[1]["num_matches"] == 12
    assert rows[1]["num_inliers"] == 8
    assert rows[1]["num_query_points"] == 16
    assert rows[1]["num_visible_tracks"] == 12
    assert rows[1]["mean_track_confidence"] == 0.8
    assert rows[1]["mean_visibility"] == 0.7
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


def test_vggt_matcher_prepare_fails_before_pair_loop_for_missing_repo(tmp_path):
    matcher = VGGTTrackMatcher(
        vggt_root=tmp_path / "missing_vggt",
        device="cpu",
        vggt_config=VGGTConfig(model_path="local-model"),
    )

    try:
        matcher.prepare()
    except FileNotFoundError as exc:
        assert "VGGT repository not found" in str(exc)
    else:
        raise AssertionError("expected prepare() to fail before scoring pairs")
