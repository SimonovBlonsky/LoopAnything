import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robust_loop_verifier.pgo import PgoResult
from robust_loop_verifier.pipeline import (
    run_cached_sequence,
    run_mock_sequence_evaluation,
    score_frozen_query_candidates,
)
from robust_loop_verifier.sim3_factor import Sim3LoopFactorResult


def _cached_config(tmp_path: Path, **overrides):
    from robust_loop_verifier.schema import RobustLoopVerifierConfig

    data = {
        "dataset_name": "unit",
        "platform": "tiny",
        "input_root": str(tmp_path / "input"),
        "output_root": str(tmp_path / "cache-root"),
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
        },
    }
    data.update(overrides)
    return RobustLoopVerifierConfig.from_mapping(data)


def _pose_at_x(x: float) -> list[float]:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = x
    return pose.reshape(-1).tolist()


def _write_tiny_sequence_cache(
    cache_dir: Path,
    keyframe_count: int = 6,
    *,
    missing_images: bool = False,
    positives_by_query: dict[int, list[int]] | None = None,
) -> None:
    positives_by_query = positives_by_query or {}
    cache_dir.mkdir(parents=True)
    (cache_dir / "images").mkdir()
    (cache_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_name": "unit",
                "platform": "tiny",
                "sequence_name": "sequence",
                "keyframe_count": keyframe_count,
            }
        ),
        encoding="utf-8",
    )
    with (cache_dir / "keyframes.jsonl").open("w", encoding="utf-8") as handle:
        for idx in range(keyframe_count):
            image_path = None if missing_images else f"images/{idx:06d}.png"
            if image_path is not None:
                (cache_dir / image_path).write_bytes(b"not used by mock backends")
            handle.write(
                json.dumps(
                    {
                        "idx": idx,
                        "timestamp": float(idx),
                        "image_path": image_path,
                        "odom_pose": _pose_at_x(float(idx)),
                        "gt_pose": _pose_at_x(0.1 if idx == keyframe_count - 1 else float(idx)),
                    }
                )
            )
            handle.write("\n")
    with (cache_dir / "positives.jsonl").open("w", encoding="utf-8") as handle:
        for idx in range(keyframe_count):
            positives = positives_by_query.get(idx, [])
            handle.write(json.dumps({"query_idx": idx, "positive_indices": positives}))
            handle.write("\n")


def _read_candidate_records(run_root: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (run_root / "candidate_records.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_mock_pipeline_writes_candidate_records_and_metrics(tmp_path: Path):
    run_root = tmp_path / "run"

    result = run_mock_sequence_evaluation(run_root)

    assert result["candidate_count"] == 8
    assert (run_root / "candidate_records.jsonl").is_file()
    assert (run_root / "metrics.json").is_file()
    assert (run_root / "metrics.md").is_file()


def test_mock_pipeline_candidate_records_include_expected_fields(tmp_path: Path):
    run_root = tmp_path / "run"

    run_mock_sequence_evaluation(run_root)

    first_record = json.loads(
        (run_root / "candidate_records.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert {
        "query_idx",
        "candidate_idx",
        "label",
        "salad_score",
        "trajectory_deformation_rmse",
        "score_rover",
        "pgo_converged",
        "pgo_error_before",
        "pgo_error_after",
    } <= set(first_record)
    assert type(first_record["label"]) is bool


def test_mock_pipeline_metrics_json_contains_da3_rover_metrics(tmp_path: Path):
    run_root = tmp_path / "run"

    result = run_mock_sequence_evaluation(run_root)

    metrics = json.loads((run_root / "metrics.json").read_text(encoding="utf-8"))
    assert "da3_rover" in metrics
    assert set(metrics["da3_rover"]) == {"AP", "MR@100P"}
    assert result["metrics"] == metrics


def test_mock_pipeline_deterministic_content_and_metrics(tmp_path: Path):
    run_root = tmp_path / "run"

    result = run_mock_sequence_evaluation(run_root)
    records = _read_candidate_records(run_root)
    metrics = json.loads((run_root / "metrics.json").read_text(encoding="utf-8"))
    positives = [record for record in records if record["label"]]
    negatives = [record for record in records if not record["label"]]

    assert result["candidate_count"] == 8
    assert len(records) == 8
    assert [(record["query_idx"], record["candidate_idx"]) for record in positives] == [(7, 0)]
    assert negatives
    assert all(record["pgo_converged"] for record in records)
    assert max(record["trajectory_deformation_rmse"] for record in negatives) > 0.1
    assert metrics["da3_rover"]["AP"] == 1.0
    assert metrics["da3_rover"]["MR@100P"] == 1.0


def test_mock_pipeline_false_loop_can_deform_bent_trajectory(tmp_path: Path):
    run_root = tmp_path / "run"

    run_mock_sequence_evaluation(run_root)

    records = _read_candidate_records(run_root)
    false_deformations = [
        record["trajectory_deformation_rmse"] for record in records if not record["label"]
    ]
    assert max(false_deformations) > 0.1


def test_mock_pipeline_serializes_failed_pgo_candidates_and_finite_metrics(
    tmp_path: Path,
    monkeypatch,
):
    run_root = tmp_path / "run"

    def fail_pgo(prefix_indices, odom_poses, *args, **kwargs):
        return PgoResult(
            converged=False,
            optimized_poses=[np.asarray(pose, dtype=np.float64).copy() for pose in odom_poses],
            error_before=float("inf"),
            error_after=float("inf"),
            failure_reason="mock pgo failure",
        )

    monkeypatch.setattr("robust_loop_verifier.pipeline.run_full_prefix_pgo", fail_pgo)

    result = run_mock_sequence_evaluation(run_root)

    records = _read_candidate_records(run_root)
    assert records
    assert all(record["pgo_converged"] is False for record in records)
    assert all(record["score_rover"] is None for record in records)
    assert all(record["score_rover_source"] is None for record in records)
    assert all(record["pgo_failure_reason"] == "mock pgo failure" for record in records)
    assert all("pgo_error_before" in record and "pgo_error_after" in record for record in records)
    assert all(record["pgo_error_before"] is None for record in records)
    assert all(record["pgo_error_after"] is None for record in records)
    assert math.isfinite(result["metrics"]["da3_rover"]["AP"])
    assert math.isfinite(result["metrics"]["da3_rover"]["MR@100P"])


def test_mock_pipeline_accepts_preexisting_empty_run_root(tmp_path: Path):
    run_root = tmp_path / "run"
    run_root.mkdir()

    result = run_mock_sequence_evaluation(run_root)

    assert result["candidate_count"] > 0
    assert (run_root / "candidate_records.jsonl").is_file()


def test_mock_pipeline_rejects_preexisting_nonempty_run_root_without_writing(
    tmp_path: Path,
):
    run_root = tmp_path / "run"
    run_root.mkdir()
    stale_file = run_root / "stale.txt"
    stale_file.write_text("keep me\n", encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty"):
        run_mock_sequence_evaluation(run_root)

    assert stale_file.read_text(encoding="utf-8") == "keep me\n"
    assert not (run_root / "candidate_records.jsonl").exists()
    assert not (run_root / "metrics.json").exists()
    assert not (run_root / "metrics.md").exists()


def test_run_cached_sequence_mock_writes_metrics_records_and_artifact_dirs(tmp_path: Path):
    sequence_cache = tmp_path / "cache"
    output_root = tmp_path / "run"
    _write_tiny_sequence_cache(sequence_cache, positives_by_query={5: [1]})

    result = run_cached_sequence(
        _cached_config(tmp_path),
        sequence_cache=sequence_cache,
        output_root=output_root,
        query_limit=6,
        backend="mock",
    )

    records = _read_candidate_records(output_root)
    assert result["candidate_count"] == len(records)
    assert records
    assert (output_root / "candidate_records.jsonl").is_file()
    assert (output_root / "metrics.json").is_file()
    assert (output_root / "metrics.md").is_file()
    assert (output_root / "pr_curves").is_dir()
    assert (output_root / "visual_records").is_dir()
    assert (output_root / "trajectory_plots").is_dir()

    metrics = json.loads((output_root / "metrics.json").read_text(encoding="utf-8"))
    expected_methods = {
        "SALAD score only",
        "SALAD + DA3/Sim3 self-consistency score",
        "SALAD + DA3-ROVER full-prefix trajectory score",
    }
    assert set(metrics) == expected_methods
    assert result["metrics"] == metrics
    metrics_markdown = (output_root / "metrics.md").read_text(encoding="utf-8")
    assert all(method in metrics_markdown for method in expected_methods)
    assert {
        "query_idx",
        "candidate_idx",
        "rank",
        "label",
        "salad_score",
        "score_salad",
        "score_da3_sim3",
        "score_rover",
        "score_rover_source",
        "support_idx",
        "support_rejection_reason",
        "support_baseline_m",
        "sim3_valid",
        "sim3_rejection_reason",
        "pgo_converged",
        "failure_reasons",
    } <= set(records[0])
    labeled_pairs = [
        (record["query_idx"], record["candidate_idx"]) for record in records if record["label"]
    ]
    assert labeled_pairs == [(5, 1)]
    assert any(record["query_idx"] == 5 and record["candidate_idx"] == 0 for record in records)


def test_run_cached_sequence_does_not_emit_internal_pair_ids(tmp_path: Path):
    sequence_cache = tmp_path / "cache"
    output_root = tmp_path / "run"
    _write_tiny_sequence_cache(sequence_cache, positives_by_query={5: [1]})

    run_cached_sequence(
        _cached_config(tmp_path),
        sequence_cache=sequence_cache,
        output_root=output_root,
        query_limit=6,
        backend="mock",
    )

    records = _read_candidate_records(output_root)
    assert records
    assert all("pair_id" not in record for record in records)


def test_score_frozen_query_candidates_preserves_caller_pair_id(tmp_path, monkeypatch):
    def fake_score_candidates_with_batched_triplets(**kwargs):
        assert kwargs["candidates"][0].score == 0.42
        return [
            {
                "query_idx": 3,
                "candidate_idx": 0,
                "rank": 1,
                "label": False,
            }
        ]

    monkeypatch.setattr(
        "robust_loop_verifier.pipeline._score_candidates_with_batched_triplets",
        fake_score_candidates_with_batched_triplets,
    )

    records = score_frozen_query_candidates(
        config=_cached_config(tmp_path),
        query_idx=3,
        frozen_candidates=[
            {
                "pair_id": "caller-pair",
                "candidate_idx": 0,
                "rank": 1,
                "score": 0.42,
            }
        ],
        image_by_idx={},
        odom_by_idx={},
        cache_order=[],
        da3_runner=object(),
    )

    assert records == [
        {
            "pair_id": "caller-pair",
            "query_idx": 3,
            "candidate_idx": 0,
            "rank": 1,
        }
    ]


def test_score_frozen_query_candidates_exports_batched_loop_factors_without_pgo(
    tmp_path,
    monkeypatch,
):
    batch_sizes = []
    alignment_calls = []
    loop_factor = np.eye(4, dtype=np.float64)
    loop_factor[:3, 3] = [1.0, 2.0, 3.0]

    class BatchOnlyRunner:
        def run_triplet(self, triplet):
            raise AssertionError("single triplet inference should not be used")

        def run_triplets(self, triplets):
            batch_sizes.append(len(triplets))
            return [SimpleNamespace(predicted_c2w=np.empty((3, 4, 4))) for _ in triplets]

    def fake_align(predicted_c2w, odom_candidate, odom_support):
        alignment_calls.append((odom_candidate.copy(), odom_support.copy()))
        return Sim3LoopFactorResult(
            valid=True,
            loop_factor=loop_factor.copy(),
            sim3_scale=2.0,
            support_alignment_residual_m=0.25,
            direction_error_deg=3.0,
            rejection_reason=None,
        )

    def forbidden_pgo(*args, **kwargs):
        raise AssertionError("run_full_prefix_pgo must not run")

    monkeypatch.setattr(
        "robust_loop_verifier.pipeline.align_triplet_to_candidate_support",
        fake_align,
    )
    monkeypatch.setattr(
        "robust_loop_verifier.pipeline.run_full_prefix_pgo",
        forbidden_pgo,
    )

    image_by_idx = {idx: tmp_path / f"{idx}.png" for idx in range(6)}
    odom_by_idx = {
        idx: np.asarray(_pose_at_x(float(idx)), dtype=np.float64).reshape(4, 4) for idx in range(6)
    }
    records = score_frozen_query_candidates(
        config=_cached_config(
            tmp_path,
            da3={
                "process_res": 504,
                "ref_view_strategy": "first",
                "triplet_batch_size": 4,
            },
        ),
        query_idx=5,
        frozen_candidates=[
            {"pair_id": "p1", "candidate_idx": 0, "rank": 1, "score": 0.9},
            {"pair_id": "p2", "candidate_idx": 1, "rank": 2, "score": 0.8},
        ],
        image_by_idx=image_by_idx,
        odom_by_idx=odom_by_idx,
        cache_order=list(range(6)),
        da3_runner=BatchOnlyRunner(),
        run_pgo=False,
    )

    assert batch_sizes == [2]
    assert len(alignment_calls) == 2
    assert [record["pair_id"] for record in records] == ["p1", "p2"]
    assert all(record["support_idx"] is not None for record in records)
    assert all(record["score_da3_sim3"] == -0.25 for record in records)
    assert all(record["loop_factor"] == loop_factor.reshape(-1).tolist() for record in records)
    assert all(record["pgo_converged"] is None for record in records)
    assert all(record["pgo_error_before"] is None for record in records)
    assert all(record["pgo_error_after"] is None for record in records)
    assert all(record["pgo_failure_reason"] is None for record in records)
    assert all(record["trajectory_deformation_rmse"] is None for record in records)
    assert all(record["score_rover"] is None for record in records)
    assert all(record["score_rover_source"] is None for record in records)
    json.dumps(records, allow_nan=False)


def test_score_frozen_query_candidates_clears_pose_only_pgo_fields_on_failures(
    tmp_path,
    monkeypatch,
):
    image_by_idx = {idx: tmp_path / f"{idx}.png" for idx in range(4)}
    odom_by_idx = {
        idx: np.asarray(_pose_at_x(float(idx)), dtype=np.float64).reshape(4, 4)
        for idx in range(4)
    }
    frozen_candidates = [
        {"pair_id": "p1", "candidate_idx": 0, "rank": 1, "score": 0.9},
    ]

    class FailingDa3Runner:
        def run_triplets(self, triplets):
            raise RuntimeError("forced da3 failure")

    class PassingDa3Runner:
        def run_triplets(self, triplets):
            return [SimpleNamespace(predicted_c2w=np.empty((3, 4, 4))) for _ in triplets]

    def score_with(**kwargs):
        return score_frozen_query_candidates(
            config=kwargs.pop("config", _cached_config(tmp_path)),
            query_idx=3,
            frozen_candidates=frozen_candidates,
            image_by_idx=image_by_idx,
            odom_by_idx=odom_by_idx,
            cache_order=list(range(4)),
            run_pgo=False,
            **kwargs,
        )[0]

    support_failure = score_with(
        config=_cached_config(tmp_path, min_support_baseline_m=99.0),
        da3_runner=PassingDa3Runner(),
    )
    da3_failure = score_with(da3_runner=FailingDa3Runner())
    monkeypatch.setattr(
        "robust_loop_verifier.pipeline.align_triplet_to_candidate_support",
        lambda *args, **kwargs: Sim3LoopFactorResult(
            valid=False,
            loop_factor=None,
            sim3_scale=None,
            support_alignment_residual_m=None,
            direction_error_deg=None,
            rejection_reason="forced_sim3_failure",
        ),
    )
    sim3_failure = score_with(da3_runner=PassingDa3Runner())

    for record in (support_failure, da3_failure, sim3_failure):
        assert record["pgo_converged"] is None
        assert record["pgo_error_before"] is None
        assert record["pgo_error_after"] is None
        assert record["pgo_failure_reason"] is None
        assert record["trajectory_deformation_rmse"] is None
        assert record["score_rover"] is None
        assert record["score_rover_source"] is None


@pytest.mark.parametrize(
    "bad_loop_factor",
    [
        np.full((4, 4), np.nan, dtype=np.float64),
        np.eye(3, dtype=np.float64),
        np.full((4, 4), 10**1000, dtype=object),
    ],
)
def test_score_frozen_query_candidates_rejects_invalid_pose_only_loop_factor(
    tmp_path,
    monkeypatch,
    bad_loop_factor,
):
    class PassingDa3Runner:
        def run_triplets(self, triplets):
            return [SimpleNamespace(predicted_c2w=np.empty((3, 4, 4))) for _ in triplets]

    monkeypatch.setattr(
        "robust_loop_verifier.pipeline.align_triplet_to_candidate_support",
        lambda *args, **kwargs: Sim3LoopFactorResult(
            valid=True,
            loop_factor=bad_loop_factor,
            sim3_scale=1.0,
            support_alignment_residual_m=0.25,
            direction_error_deg=0.0,
            rejection_reason=None,
        ),
    )

    image_by_idx = {idx: tmp_path / f"{idx}.png" for idx in range(4)}
    odom_by_idx = {
        idx: np.asarray(_pose_at_x(float(idx)), dtype=np.float64).reshape(4, 4)
        for idx in range(4)
    }
    records = score_frozen_query_candidates(
        config=_cached_config(tmp_path),
        query_idx=3,
        frozen_candidates=[
            {"pair_id": "p1", "candidate_idx": 0, "rank": 1, "score": 0.9},
        ],
        image_by_idx=image_by_idx,
        odom_by_idx=odom_by_idx,
        cache_order=list(range(4)),
        da3_runner=PassingDa3Runner(),
        run_pgo=False,
    )

    assert records[0]["loop_factor"] is None
    assert records[0]["score_da3_sim3"] is None
    assert records[0]["sim3_valid"] is False
    assert records[0]["sim3_rejection_reason"] == "invalid_loop_factor"
    assert "sim3: invalid_loop_factor" in records[0]["failure_reasons"]
    assert records[0]["pgo_converged"] is None


def test_score_frozen_query_candidates_isolates_missing_image_triplet_failure(
    tmp_path,
    monkeypatch,
):
    loop_factor = np.eye(4, dtype=np.float64)

    class PassingDa3Runner:
        def run_triplets(self, triplets):
            return [SimpleNamespace(predicted_c2w=np.empty((3, 4, 4))) for _ in triplets]

    monkeypatch.setattr(
        "robust_loop_verifier.pipeline.align_triplet_to_candidate_support",
        lambda *args, **kwargs: Sim3LoopFactorResult(
            valid=True,
            loop_factor=loop_factor.copy(),
            sim3_scale=1.0,
            support_alignment_residual_m=0.0,
            direction_error_deg=0.0,
            rejection_reason=None,
        ),
    )

    image_by_idx = {idx: tmp_path / f"{idx}.png" for idx in (1, 2, 3, 4)}
    odom_by_idx = {
        idx: np.asarray(_pose_at_x(float(idx)), dtype=np.float64).reshape(4, 4)
        for idx in range(5)
    }
    records = score_frozen_query_candidates(
        config=_cached_config(tmp_path),
        query_idx=4,
        frozen_candidates=[
            {"pair_id": "missing-image", "candidate_idx": 0, "rank": 1, "score": 0.9},
            {"pair_id": "ok", "candidate_idx": 1, "rank": 2, "score": 0.8},
        ],
        image_by_idx=image_by_idx,
        odom_by_idx=odom_by_idx,
        cache_order=list(range(5)),
        da3_runner=PassingDa3Runner(),
        run_pgo=False,
    )

    assert [record["pair_id"] for record in records] == ["missing-image", "ok"]
    failed, succeeded = records
    assert failed["loop_factor"] is None
    assert failed["score_da3_sim3"] is None
    assert failed["pgo_converged"] is None
    assert any("triplet:" in reason for reason in failed["failure_reasons"])
    assert succeeded["loop_factor"] == loop_factor.reshape(-1).tolist()
    assert succeeded["score_da3_sim3"] == -0.0


def test_score_frozen_query_candidates_rejects_scored_record_count_mismatch(
    tmp_path,
    monkeypatch,
):
    def fake_score_candidates_with_batched_triplets(**kwargs):
        return [
            {
                "query_idx": 3,
                "candidate_idx": 0,
                "rank": 1,
                "label": False,
            }
        ]

    monkeypatch.setattr(
        "robust_loop_verifier.pipeline._score_candidates_with_batched_triplets",
        fake_score_candidates_with_batched_triplets,
    )

    with pytest.raises(ValueError, match="scored record count"):
        score_frozen_query_candidates(
            config=_cached_config(tmp_path),
            query_idx=3,
            frozen_candidates=[
                {"pair_id": "p1", "candidate_idx": 0, "rank": 1, "score": 0.9},
                {"pair_id": "p2", "candidate_idx": 1, "rank": 2, "score": 0.8},
            ],
            image_by_idx={},
            odom_by_idx={},
            cache_order=[],
            da3_runner=object(),
        )


def test_run_cached_sequence_batches_single_support_candidates_by_query(tmp_path: Path):
    from robust_loop_verifier.da3_runner import MockDa3Runner

    sequence_cache = tmp_path / "cache"
    output_root = tmp_path / "run"
    _write_tiny_sequence_cache(sequence_cache, keyframe_count=8, positives_by_query={7: [0]})

    class BatchOnlyRunner:
        def __init__(self):
            self.batch_sizes = []
            self.single_triplet_calls = 0

        def run_triplet(self, triplet):
            self.single_triplet_calls += 1
            raise AssertionError("single triplet inference should not be used")

        def run_triplets(self, triplets):
            self.batch_sizes.append(len(triplets))
            return [MockDa3Runner().run_triplet(triplet) for triplet in triplets]

    runner = BatchOnlyRunner()

    result = run_cached_sequence(
        _cached_config(
            tmp_path,
            retrieval_top_k_main=3,
            da3={
                "process_res": 504,
                "ref_view_strategy": "first",
                "triplet_batch_size": 4,
            },
        ),
        sequence_cache=sequence_cache,
        output_root=output_root,
        query_limit=8,
        backend=runner,
    )

    records = _read_candidate_records(output_root)
    assert result["candidate_count"] == len(records)
    assert records
    assert runner.single_triplet_calls == 0
    assert runner.batch_sizes
    assert max(runner.batch_sizes) > 1
    assert all(size <= 4 for size in runner.batch_sizes)
    assert all(record["sim3_valid"] for record in records if record["support_idx"] is not None)


def test_run_cached_sequence_collect_timing_writes_efficiency_artifact(tmp_path: Path):
    sequence_cache = tmp_path / "cache"
    output_root = tmp_path / "run"
    _write_tiny_sequence_cache(sequence_cache, positives_by_query={5: [1]})

    result = run_cached_sequence(
        _cached_config(tmp_path),
        sequence_cache=sequence_cache,
        output_root=output_root,
        query_limit=6,
        backend="mock",
        collect_timing=True,
    )

    timing = json.loads((output_root / "efficiency_timing.json").read_text(encoding="utf-8"))
    records = _read_candidate_records(output_root)
    assert result["timing"] == timing
    assert timing["candidate_count"] == len(records)
    assert timing["query_count"] > 0
    assert timing["component_totals_sec"]["descriptor_compute"] >= 0.0
    assert timing["component_totals_sec"]["retrieval_search"] >= 0.0
    assert timing["component_totals_sec"]["da3_triplet"] >= 0.0
    assert timing["component_totals_sec"]["sim3_alignment"] >= 0.0
    assert timing["component_totals_sec"]["pgo"] >= 0.0
    assert timing["component_totals_sec"]["metrics"] >= 0.0
    assert timing["per_candidate_sec"]["total"]["count"] == len(records)
    assert {
        "timing_total_candidate_sec",
        "timing_support_selection_sec",
        "timing_da3_triplet_sec",
        "timing_sim3_alignment_sec",
        "timing_pgo_sec",
    } <= set(records[0])


def test_support_ensemble_mock_pipeline_emits_fields(tmp_path: Path):
    sequence_cache = tmp_path / "cache"
    output_root = tmp_path / "run"
    _write_tiny_sequence_cache(sequence_cache, keyframe_count=8, positives_by_query={7: [0]})

    result = run_cached_sequence(
        _cached_config(
            tmp_path,
            retrieval_top_k_main=3,
            support_ensemble={
                "enabled": True,
                "support_count": 4,
            },
        ),
        sequence_cache=sequence_cache,
        output_root=output_root,
        query_limit=8,
        backend="mock",
    )

    records = _read_candidate_records(output_root)
    assert records
    scored_records = [
        record for record in records if record["score_support_ensemble"] is not None
    ]
    assert scored_records
    record = scored_records[0]
    assert record["support_ensemble_enabled"] is True
    assert record["support_ensemble_support_count_requested"] == 4
    assert record["support_ensemble_support_count_used"] >= 1
    assert isinstance(record["support_ensemble_supports"], list)
    assert any(
        "weight" in support and "residual_norm" in support
        for support in record["support_ensemble_supports"]
    )
    assert any(
        support["weight"] is not None and support["residual_norm"] is not None
        for support in record["support_ensemble_supports"]
    )
    assert len(record["support_ensemble_loop_sigmas"]) == 6
    assert record["support_ensemble_effective_support_count"] >= 1.0
    assert record["support_ensemble_graph_evidence_nll"] >= 0.0
    assert record["score_rover"] is not None
    assert record["score_rover_source"] == "support_ensemble_loop_factor"
    assert "DA3-ROVER++ support ensemble graph evidence" in result["metrics"]


def test_support_ensemble_calls_da3_once_per_support_triplet(tmp_path: Path):
    from robust_loop_verifier.da3_runner import MockDa3Runner

    sequence_cache = tmp_path / "cache"
    output_root = tmp_path / "run"
    _write_tiny_sequence_cache(sequence_cache, keyframe_count=8, positives_by_query={7: [0]})

    calls = []

    class RecordingRunner:
        def run_triplet(self, triplet):
            calls.append(
                (
                    tuple(triplet.view_roles),
                    len(triplet.image_paths),
                    len(triplet.keyframe_indices),
                    triplet.keyframe_indices,
                )
            )
            return MockDa3Runner().run_triplet(triplet)

    run_cached_sequence(
        _cached_config(
            tmp_path,
            retrieval_top_k_main=1,
            support_ensemble={
                "enabled": True,
                "support_count": 3,
            },
        ),
        sequence_cache=sequence_cache,
        output_root=output_root,
        query_limit=8,
        backend=RecordingRunner(),
    )

    assert calls
    assert all(call[:3] == (("query", "candidate", "support"), 3, 3) for call in calls)
    assert len({call[3] for call in calls}) == len(calls)


def test_support_ensemble_pgo_failure_serializes_sanitized_records(
    tmp_path: Path,
    monkeypatch,
):
    sequence_cache = tmp_path / "cache"
    output_root = tmp_path / "run"
    _write_tiny_sequence_cache(sequence_cache, keyframe_count=8, positives_by_query={7: [0]})

    def fail_support_ensemble_pgo(prefix_indices, odom_poses, *args, **kwargs):
        return PgoResult(
            converged=False,
            optimized_poses=[np.asarray(pose, dtype=np.float64).copy() for pose in odom_poses],
            error_before=float("inf"),
            error_after=float("inf"),
            failure_reason="mock support ensemble pgo failure",
        )

    monkeypatch.setattr(
        "robust_loop_verifier.pipeline.run_full_prefix_pgo",
        fail_support_ensemble_pgo,
    )

    result = run_cached_sequence(
        _cached_config(
            tmp_path,
            retrieval_top_k_main=3,
            support_ensemble={
                "enabled": True,
                "support_count": 4,
            },
        ),
        sequence_cache=sequence_cache,
        output_root=output_root,
        query_limit=8,
        backend="mock",
    )

    records = _read_candidate_records(output_root)
    assert result["candidate_count"] == len(records)
    assert records
    assert all(record["score_support_ensemble"] is None for record in records)
    assert all(record["score_rover"] is None for record in records)
    assert all(record["score_rover_source"] is None for record in records)
    assert all(record["pgo_converged"] is False for record in records)
    assert all(
        record["pgo_failure_reason"] == "mock support ensemble pgo failure"
        for record in records
    )
    assert all(
        "pgo: mock support ensemble pgo failure" in record["failure_reasons"]
        for record in records
    )
    assert all(record["pgo_error_before"] is None for record in records)
    assert all(record["pgo_error_after"] is None for record in records)


def test_support_ensemble_success_exports_json_safe_loop_factor(
    tmp_path,
    monkeypatch,
):
    loop_factor = np.eye(4, dtype=np.float64)
    loop_factor[:3, 3] = [4.0, 5.0, 6.0]
    captured_pgo = {}

    class FakeRunner:
        def run_triplet(self, triplet):
            return SimpleNamespace(predicted_c2w=np.empty((3, 4, 4)))

    monkeypatch.setattr(
        "robust_loop_verifier.pipeline.align_triplet_to_candidate_support",
        lambda *args, **kwargs: Sim3LoopFactorResult(
            valid=True,
            loop_factor=loop_factor.copy(),
            sim3_scale=1.0,
            support_alignment_residual_m=0.0,
            direction_error_deg=0.0,
            rejection_reason=None,
        ),
    )

    def capture_pgo(prefix_indices, odom_poses, *args, **kwargs):
        captured_pgo["loop_factor"] = np.asarray(kwargs["loop_factor"], dtype=np.float64)
        return PgoResult(
            converged=True,
            optimized_poses=[np.asarray(pose, dtype=np.float64).copy() for pose in odom_poses],
            error_before=0.0,
            error_after=0.0,
            failure_reason=None,
            loop_chi2_after=0.0,
            odom_strain_chi2_after=0.0,
        )

    monkeypatch.setattr(
        "robust_loop_verifier.pipeline.run_full_prefix_pgo",
        capture_pgo,
    )

    image_by_idx = {idx: tmp_path / f"{idx}.png" for idx in range(4)}
    odom_by_idx = {
        idx: np.asarray(_pose_at_x(float(idx)), dtype=np.float64).reshape(4, 4)
        for idx in range(4)
    }
    records = score_frozen_query_candidates(
        config=_cached_config(
            tmp_path,
            support_ensemble={
                "enabled": True,
                "support_count": 1,
            },
        ),
        query_idx=3,
        frozen_candidates=[
            {"pair_id": "p1", "candidate_idx": 0, "rank": 1, "score": 0.9},
        ],
        image_by_idx=image_by_idx,
        odom_by_idx=odom_by_idx,
        cache_order=list(range(4)),
        da3_runner=FakeRunner(),
    )

    assert len(records) == 1
    record = records[0]
    assert record["pgo_converged"] is True, record["failure_reasons"]
    assert np.allclose(captured_pgo["loop_factor"], loop_factor)
    assert record["loop_factor"] == loop_factor.reshape(-1).tolist()
    json.dumps(records, allow_nan=False)


def test_support_ensemble_disabled_records_keep_default_fields_and_metrics(
    tmp_path: Path,
):
    sequence_cache = tmp_path / "cache"
    output_root = tmp_path / "run"
    _write_tiny_sequence_cache(sequence_cache, positives_by_query={5: [1]})

    result = run_cached_sequence(
        _cached_config(tmp_path),
        sequence_cache=sequence_cache,
        output_root=output_root,
        query_limit=6,
        backend="mock",
    )

    records = _read_candidate_records(output_root)
    assert records
    record = records[0]
    assert record["support_ensemble_enabled"] is False
    assert record["support_ensemble_support_count_requested"] == 1
    assert record["support_ensemble_support_count_used"] == 0
    assert record["support_ensemble_supports"] == []
    assert record["support_ensemble_effective_support_count"] is None
    assert record["support_ensemble_loop_sigmas"] is None
    assert record["support_ensemble_sigma_rot"] is None
    assert record["support_ensemble_sigma_trans"] is None
    assert record["support_ensemble_uncertainty_logdet_penalty"] is None
    assert record["support_ensemble_loop_chi2_after"] is None
    assert record["support_ensemble_odom_strain_chi2_after"] is None
    assert record["support_ensemble_graph_evidence_nll"] is None
    assert record["score_support_ensemble"] is None
    assert any(
        item["score_rover_source"] == "single_support_loop_factor"
        for item in records
        if item["score_rover"] is not None
    )
    assert all(
        item["score_rover_source"] is None
        for item in records
        if item["score_rover"] is None
    )
    assert set(result["metrics"]) == {
        "SALAD score only",
        "SALAD + DA3/Sim3 self-consistency score",
        "SALAD + DA3-ROVER full-prefix trajectory score",
    }


def test_run_cached_sequence_invalid_backend_fails_before_writing(tmp_path: Path):
    sequence_cache = tmp_path / "cache"
    output_root = tmp_path / "run"
    _write_tiny_sequence_cache(sequence_cache)

    with pytest.raises(ValueError, match="backend"):
        run_cached_sequence(
            _cached_config(tmp_path),
            sequence_cache=sequence_cache,
            output_root=output_root,
            query_limit=6,
            backend="invalid",
        )

    assert not output_root.exists()


def test_run_cached_sequence_query_limit_zero_skips_real_backends_and_writes_artifacts(
    tmp_path: Path,
    monkeypatch,
):
    sequence_cache = tmp_path / "cache"
    output_root = tmp_path / "run"
    _write_tiny_sequence_cache(sequence_cache)

    def fail_make_backends(*args, **kwargs):
        raise AssertionError("real backends should not be constructed")

    monkeypatch.setattr("robust_loop_verifier.pipeline._make_backends", fail_make_backends)

    result = run_cached_sequence(
        _cached_config(tmp_path),
        sequence_cache=sequence_cache,
        output_root=output_root,
        query_limit=0,
        backend="real",
    )

    assert result["candidate_count"] == 0
    assert _read_candidate_records(output_root) == []
    assert (output_root / "metrics.json").is_file()
    assert (output_root / "metrics.md").is_file()
    assert (output_root / "pr_curves").is_dir()
    assert (output_root / "visual_records").is_dir()
    assert (output_root / "trajectory_plots").is_dir()


def test_run_cached_sequence_all_missing_images_skips_backends_and_writes_artifacts(
    tmp_path: Path,
    monkeypatch,
):
    sequence_cache = tmp_path / "cache"
    output_root = tmp_path / "run"
    _write_tiny_sequence_cache(sequence_cache, missing_images=True)

    def fail_make_backends(*args, **kwargs):
        raise AssertionError("real backends should not be constructed")

    monkeypatch.setattr("robust_loop_verifier.pipeline._make_backends", fail_make_backends)

    result = run_cached_sequence(
        _cached_config(tmp_path),
        sequence_cache=sequence_cache,
        output_root=output_root,
        query_limit=6,
        backend="real",
    )

    assert result["candidate_count"] == 0
    assert _read_candidate_records(output_root) == []
    assert (output_root / "metrics.json").is_file()
    assert (output_root / "metrics.md").is_file()
    assert (output_root / "pr_curves").is_dir()
    assert (output_root / "visual_records").is_dir()
    assert (output_root / "trajectory_plots").is_dir()


def test_run_cached_sequence_no_legal_historical_candidates_skips_real_backends(
    tmp_path: Path,
    monkeypatch,
):
    sequence_cache = tmp_path / "cache"
    output_root = tmp_path / "run"
    _write_tiny_sequence_cache(sequence_cache, keyframe_count=20)

    def fail_make_backends(*args, **kwargs):
        raise AssertionError("real backends should not be constructed")

    monkeypatch.setattr("robust_loop_verifier.pipeline._make_backends", fail_make_backends)

    result = run_cached_sequence(
        _cached_config(tmp_path, recent_exclusion_keyframes=30),
        sequence_cache=sequence_cache,
        output_root=output_root,
        query_limit=20,
        backend="real",
    )

    assert result["candidate_count"] == 0
    assert _read_candidate_records(output_root) == []
    assert (output_root / "metrics.json").is_file()
    assert (output_root / "metrics.md").is_file()
    assert (output_root / "pr_curves").is_dir()
    assert (output_root / "visual_records").is_dir()
    assert (output_root / "trajectory_plots").is_dir()


def test_run_cached_sequence_preserves_candidates_when_scoring_stage_fails(
    tmp_path: Path,
    monkeypatch,
):
    sequence_cache = tmp_path / "cache"
    output_root = tmp_path / "run"
    _write_tiny_sequence_cache(sequence_cache)

    def fail_support(*args, **kwargs):
        from robust_loop_verifier.support import SupportSelection

        return SupportSelection(
            query_idx=kwargs["query_idx"],
            candidate_idx=kwargs["candidate_idx"],
            support_idx=None,
            support_baseline_m=None,
            rejection_reason="forced support failure",
        )

    monkeypatch.setattr("robust_loop_verifier.pipeline.select_support", fail_support)

    result = run_cached_sequence(
        _cached_config(tmp_path),
        sequence_cache=sequence_cache,
        output_root=output_root,
        query_limit=6,
        backend="mock",
    )

    records = _read_candidate_records(output_root)
    assert result["candidate_count"] == len(records)
    assert records
    assert all(
        record["support_rejection_reason"] == "forced support failure" for record in records
    )
    assert all(record["score_da3_sim3"] is None for record in records)
    assert all(record["score_rover"] is None for record in records)
    assert all(
        "support: forced support failure" in record["failure_reasons"] for record in records
    )
    for method_values in result["metrics"].values():
        assert math.isfinite(method_values["AP"])
        assert math.isfinite(method_values["MR@100P"])


def test_run_cached_sequence_preserves_candidates_when_sim3_stage_raises(
    tmp_path: Path,
    monkeypatch,
):
    sequence_cache = tmp_path / "cache"
    output_root = tmp_path / "run"
    _write_tiny_sequence_cache(sequence_cache)

    def raise_sim3(*args, **kwargs):
        raise ValueError("SE3 rotation block must be orthonormal")

    monkeypatch.setattr(
        "robust_loop_verifier.pipeline.align_triplet_to_candidate_support",
        raise_sim3,
    )

    result = run_cached_sequence(
        _cached_config(tmp_path),
        sequence_cache=sequence_cache,
        output_root=output_root,
        query_limit=6,
        backend="mock",
    )

    records = _read_candidate_records(output_root)
    assert result["candidate_count"] == len(records)
    assert records
    assert all(record["score_da3_sim3"] is None for record in records)
    assert all(record["score_rover"] is None for record in records)
    assert all(
        "sim3: ValueError: SE3 rotation block must be orthonormal"
        in record["failure_reasons"]
        for record in records
    )


def test_real_pipeline_backend_defaults_to_da3_large_local_files(tmp_path: Path):
    from robust_loop_verifier.pipeline import _make_backends

    _, da3_runner = _make_backends(_cached_config(tmp_path), "real")

    assert da3_runner.config.model_name == "depth-anything/DA3-LARGE-1.1"
    assert da3_runner.config.local_files_only
