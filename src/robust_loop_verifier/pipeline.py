"""End-to-end pipeline orchestration for robust loop verifier runs."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from robust_loop_verifier.artifacts import write_json, write_metrics_markdown
from robust_loop_verifier.da3_runner import (
    MockDa3Runner,
    RealDa3Runner,
    RealDa3RunnerConfig,
    build_da3_triplet,
)
from robust_loop_verifier.geometry import make_transform, pose_between
from robust_loop_verifier.io import read_jsonl
from robust_loop_verifier.metrics import (
    assign_failure_worst_scores,
    average_precision,
    max_recall_at_100_precision,
)
from robust_loop_verifier.pgo import (
    PgoNoise,
    run_full_prefix_pgo,
    trajectory_deformation_rmse,
)
from robust_loop_verifier.retrieval import (
    DescriptorSet,
    RetrievalCandidate,
    SaladDescriptorBackend,
    SaladDescriptorBackendConfig,
    retrieve_historical_topk,
)
from robust_loop_verifier.schema import RobustLoopVerifierConfig
from robust_loop_verifier.sim3_factor import align_triplet_to_candidate_support
from robust_loop_verifier.support import select_support, select_supports
from robust_loop_verifier.support_ensemble import (
    SupportEnsembleConfig,
    SupportLoopFactor,
    aggregate_support_loop_factors,
    graph_evidence_nll,
)


METHOD_SALAD = "SALAD score only"
METHOD_DA3_SIM3 = "SALAD + DA3/Sim3 self-consistency score"
METHOD_ROVER = "SALAD + DA3-ROVER full-prefix trajectory score"
METHOD_SUPPORT_ENSEMBLE = "DA3-ROVER++ support ensemble graph evidence"


def run_mock_sequence_evaluation(run_root: Path) -> dict[str, Any]:
    """Run a deterministic synthetic retrieval and PGO evaluation.

    The mock sequence is intentionally small and bent so false loop factors can
    deform the trajectory in a way Sim3 alignment does not fully absorb.
    """

    run_root = Path(run_root)
    _prepare_run_root(run_root)

    poses = _synthetic_bent_poses()
    descriptors = _synthetic_descriptors()
    records = _evaluate_candidates(poses, descriptors)
    labels = [record["label"] for record in records]
    rover_scores = assign_failure_worst_scores(
        record["score_rover"] if record["pgo_converged"] else None for record in records
    )

    metrics = {
        "da3_rover": {
            "AP": average_precision(labels, rover_scores),
            "MR@100P": max_recall_at_100_precision(labels, rover_scores),
        }
    }

    _write_jsonl(run_root / "candidate_records.jsonl", records)
    write_json(run_root / "metrics.json", metrics)
    write_metrics_markdown(run_root / "metrics.md", metrics)

    return {
        "candidate_count": len(records),
        "metrics": metrics,
    }


def run_cached_sequence(
    config: RobustLoopVerifierConfig,
    sequence_cache: Path,
    output_root: Path,
    query_limit: int = 20,
    backend: str | Any = "real",
    collect_timing: bool = False,
) -> dict[str, Any]:
    """Run the offline verifier against a preprocessed sequence cache.

    Non-string ``backend`` values with ``run_triplet`` are a narrow test seam for
    DA3 runner injection; descriptors still use the deterministic mock backend.
    """

    injected_da3_runner = _is_da3_runner_injection(backend)
    if not injected_da3_runner and backend not in {"mock", "real"}:
        raise ValueError("backend must be one of {'mock', 'real'}")
    if query_limit < 0:
        raise ValueError("query_limit must be non-negative")
    timing = _new_timing_accumulator() if collect_timing else None

    output_root = Path(output_root)
    _prepare_run_root(output_root)

    sequence_cache = Path(sequence_cache)
    keyframes = [
        _parse_keyframe_row(row, sequence_cache)
        for row in read_jsonl(sequence_cache / "keyframes.jsonl")
    ]
    positives_by_query = _read_positives(sequence_cache / "positives.jsonl")
    selected_keyframes = keyframes[:query_limit] if query_limit is not None else keyframes
    descriptor_rows = [keyframe for keyframe in keyframes if keyframe["image_path"] is not None]

    if not selected_keyframes or not descriptor_rows:
        return _write_zero_candidate_run(output_root, config, timing=timing)

    image_indices = [int(keyframe["idx"]) for keyframe in descriptor_rows]
    selected_query_indices = [
        int(keyframe["idx"])
        for keyframe in selected_keyframes
        if keyframe["image_path"] is not None
    ]
    if not _has_legal_historical_image_candidate(
        selected_query_indices,
        image_indices,
        config.recent_exclusion_keyframes,
    ):
        return _write_zero_candidate_run(output_root, config, timing=timing)

    if injected_da3_runner:
        descriptor_backend, da3_runner = _MockDescriptorBackend(), backend
    else:
        descriptor_backend, da3_runner = _make_backends(config, backend)
    descriptor_start = _timing_start(timing)
    descriptor_set = descriptor_backend.compute(
        [str(keyframe["image_path"]) for keyframe in descriptor_rows],
        image_indices,
    )
    _record_timing(timing, "descriptor_compute", descriptor_start)

    image_by_idx = {
        int(keyframe["idx"]): Path(keyframe["image_path"])
        for keyframe in descriptor_rows
    }
    odom_by_idx = {
        int(keyframe["idx"]): np.asarray(keyframe["odom_pose"], dtype=np.float64)
        for keyframe in keyframes
    }
    cache_order = [int(keyframe["idx"]) for keyframe in keyframes]
    labels: list[bool] = []
    records: list[dict[str, Any]] = []

    for query in selected_keyframes:
        query_idx = int(query["idx"])
        if query_idx not in descriptor_set.keyframe_indices:
            continue
        retrieval_start = _timing_start(timing)
        retrieval = retrieve_historical_topk(
            query_idx=query_idx,
            descriptors=descriptor_set,
            top_k=config.retrieval_top_k_main,
            recent_exclusion_keyframes=config.recent_exclusion_keyframes,
        )
        _record_timing(timing, "retrieval_search", retrieval_start)
        positive_indices = positives_by_query.get(query_idx, set())
        candidates = list(retrieval.candidates)
        if config.support_ensemble.enabled:
            for candidate in candidates:
                label = candidate.candidate_idx in positive_indices
                record = _score_candidate(
                    config=config,
                    candidate=candidate,
                    label=label,
                    image_by_idx=image_by_idx,
                    odom_by_idx=odom_by_idx,
                    cache_order=cache_order,
                    da3_runner=da3_runner,
                    timing=timing,
                )
                labels.append(label)
                records.append(record)
            continue

        frozen_candidates = [
            {
                "pair_id": (
                    f"runtime_q{query_idx:06d}_c{int(candidate.candidate_idx):06d}"
                ),
                "candidate_idx": int(candidate.candidate_idx),
                "rank": int(candidate.rank),
                "score": float(candidate.score),
            }
            for candidate in candidates
        ]
        query_records = score_frozen_query_candidates(
            config=config,
            query_idx=query_idx,
            frozen_candidates=frozen_candidates,
            image_by_idx=image_by_idx,
            odom_by_idx=odom_by_idx,
            cache_order=cache_order,
            da3_runner=da3_runner,
            timing=timing,
        )
        for record in query_records:
            label = int(record["candidate_idx"]) in positive_indices
            record["label"] = label
        records.extend(query_records)
        labels.extend(bool(record["label"]) for record in query_records)

    for record in records:
        record.pop("pair_id", None)

    method_scores = {
        METHOD_SALAD: [record["score_salad"] for record in records],
        METHOD_DA3_SIM3: [record["score_da3_sim3"] for record in records],
        METHOD_ROVER: [record["score_rover"] for record in records],
    }
    if config.support_ensemble.enabled or any(
        record["support_ensemble_enabled"] for record in records
    ):
        method_scores[METHOD_SUPPORT_ENSEMBLE] = [
            record["score_support_ensemble"] for record in records
        ]
    metrics_start = _timing_start(timing)
    metrics = _compute_method_metrics(labels, method_scores)
    _record_timing(timing, "metrics", metrics_start)
    timing_summary = _timing_summary(
        timing,
        query_count=len(selected_query_indices),
        candidate_count=len(records),
    )
    _write_run_artifacts(output_root, config, records, metrics, timing_summary=timing_summary)

    result = {
        "candidate_count": len(records),
        "metrics": metrics,
    }
    if timing_summary is not None:
        result["timing"] = timing_summary
    return result


def score_frozen_query_candidates(
    *,
    config,
    query_idx: int,
    frozen_candidates: Sequence[Mapping[str, Any]],
    image_by_idx: Mapping[int, Path],
    odom_by_idx: Mapping[int, np.ndarray],
    cache_order: Sequence[int],
    da3_runner,
    timing: dict[str, list[float]] | None = None,
    run_pgo: bool = True,
) -> list[dict[str, Any]]:
    """Score caller-supplied candidates without retrieval or labels."""

    candidates = [
        RetrievalCandidate(
            query_idx=int(query_idx),
            candidate_idx=int(candidate["candidate_idx"]),
            rank=int(candidate["rank"]),
            score=float(candidate.get("score", candidate.get("salad_score", 0.0))),
        )
        for candidate in frozen_candidates
    ]
    pair_ids = [str(candidate["pair_id"]) for candidate in frozen_candidates]

    if config.support_ensemble.enabled and run_pgo:
        records = []
        for pair_id, candidate in zip(pair_ids, candidates):
            record = _score_candidate(
                config=config,
                candidate=candidate,
                label=False,
                image_by_idx=image_by_idx,
                odom_by_idx=odom_by_idx,
                cache_order=cache_order,
                da3_runner=da3_runner,
                timing=timing,
            )
            record["pair_id"] = pair_id
            record.pop("label", None)
            records.append(record)
        return _finalize_frozen_candidate_records(
            records,
            pair_ids,
            run_pgo=run_pgo,
        )

    records = _score_candidates_with_batched_triplets(
        config=config,
        candidates=candidates,
        positive_indices=set(),
        image_by_idx=image_by_idx,
        odom_by_idx=odom_by_idx,
        cache_order=cache_order,
        da3_runner=da3_runner,
        timing=timing,
        run_pgo=run_pgo,
    )
    return _finalize_frozen_candidate_records(
        records,
        pair_ids,
        run_pgo=run_pgo,
    )


def _finalize_frozen_candidate_records(
    records: Sequence[dict[str, Any]],
    pair_ids: Sequence[str],
    *,
    run_pgo: bool,
) -> list[dict[str, Any]]:
    if len(records) != len(pair_ids):
        raise ValueError(
            f"scored record count mismatch: expected {len(pair_ids)}, got {len(records)}"
        )
    for pair_id, record in zip(pair_ids, records):
        record["pair_id"] = pair_id
        record.pop("label", None)
        if not run_pgo:
            _clear_pose_only_verifier_fields(record)
    return [_jsonable(record) for record in records]


def _clear_pose_only_verifier_fields(record: dict[str, Any]) -> None:
    for field in (
        "pgo_converged",
        "pgo_error_before",
        "pgo_error_after",
        "pgo_failure_reason",
        "trajectory_deformation_rmse",
        "score_rover",
        "score_rover_source",
        "support_ensemble_loop_chi2_after",
        "support_ensemble_odom_strain_chi2_after",
        "support_ensemble_graph_evidence_nll",
        "score_support_ensemble",
    ):
        if field in record:
            record[field] = None


def _synthetic_bent_poses() -> list[np.ndarray]:
    translations = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [2.0, 2.0, 0.0],
        [1.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    return [make_transform(np.eye(3), translation) for translation in translations]


def _synthetic_descriptors() -> DescriptorSet:
    return DescriptorSet(
        keyframe_indices=list(range(8)),
        descriptors=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.7, 0.3, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.7, 0.3],
                [0.0, 0.0, 1.0],
                [0.2, 0.0, 0.8],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
    )


def _evaluate_candidates(
    poses: list[np.ndarray],
    descriptors: DescriptorSet,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for query_idx in range(4, len(poses)):
        retrieval = retrieve_historical_topk(
            query_idx=query_idx,
            descriptors=descriptors,
            top_k=2,
            recent_exclusion_keyframes=2,
        )
        for candidate in retrieval.candidates:
            prefix_indices = list(range(query_idx + 1))
            prefix_poses = poses[: query_idx + 1]
            label = _is_positive_loop(query_idx, candidate.candidate_idx)
            loop_factor = _mock_loop_factor(poses, query_idx, candidate.candidate_idx, label)
            pgo_result = run_full_prefix_pgo(
                prefix_indices=prefix_indices,
                odom_poses=prefix_poses,
                loop_from_idx=query_idx,
                loop_to_idx=candidate.candidate_idx,
                loop_factor=loop_factor,
                noise=PgoNoise.default_for_tests(),
            )
            deformation_rmse = trajectory_deformation_rmse(
                prefix_poses,
                pgo_result.optimized_poses,
            )
            score_rover = _score_rover(deformation_rmse, pgo_result.converged)
            records.append(
                {
                    "query_idx": candidate.query_idx,
                    "candidate_idx": candidate.candidate_idx,
                    "rank": candidate.rank,
                    "label": label,
                    "salad_score": candidate.score,
                    "trajectory_deformation_rmse": deformation_rmse,
                    "score_rover": score_rover,
                    "score_rover_source": (
                        "single_support_loop_factor" if score_rover is not None else None
                    ),
                    "pgo_converged": pgo_result.converged,
                    "pgo_error_before": pgo_result.error_before,
                    "pgo_error_after": pgo_result.error_after,
                    "pgo_failure_reason": pgo_result.failure_reason,
                }
            )
    return records


def _prepare_run_root(run_root: Path) -> None:
    if run_root.exists() and any(run_root.iterdir()):
        raise ValueError(f"run_root must be empty or absent, got non-empty directory: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)


def _mock_loop_factor(
    poses: list[np.ndarray],
    query_idx: int,
    candidate_idx: int,
    label: bool,
) -> np.ndarray:
    if label:
        return pose_between(poses[query_idx], poses[candidate_idx])
    return make_transform(np.eye(3), [-20.0, 0.0, 0.0])


def _is_positive_loop(query_idx: int, candidate_idx: int) -> bool:
    return query_idx == 7 and candidate_idx == 0


def _score_rover(deformation_rmse: float, pgo_converged: bool) -> float | None:
    if not pgo_converged:
        return None
    return -float(deformation_rmse)


def _new_candidate_record(
    config: RobustLoopVerifierConfig,
    candidate,
    label: bool,
    failure_reasons: list[str],
) -> dict[str, Any]:
    return {
        "query_idx": int(candidate.query_idx),
        "candidate_idx": int(candidate.candidate_idx),
        "rank": int(candidate.rank),
        "label": bool(label),
        "salad_score": float(candidate.score),
        "score_salad": float(candidate.score),
        "score_da3_sim3": None,
        "loop_factor": None,
        "score_rover": None,
        "score_rover_source": None,
        "support_idx": None,
        "support_rejection_reason": None,
        "support_baseline_m": None,
        "sim3_valid": False,
        "sim3_scale": None,
        "sim3_support_alignment_residual_m": None,
        "sim3_direction_error_deg": None,
        "sim3_rejection_reason": None,
        "trajectory_deformation_rmse": None,
        "pgo_converged": False,
        "pgo_error_before": None,
        "pgo_error_after": None,
        "pgo_failure_reason": None,
        "support_ensemble_enabled": bool(config.support_ensemble.enabled),
        "support_ensemble_support_count_requested": int(
            config.support_ensemble.support_count
        ),
        "support_ensemble_support_count_used": 0,
        "support_ensemble_supports": [],
        "support_ensemble_effective_support_count": None,
        "support_ensemble_loop_sigmas": None,
        "support_ensemble_sigma_rot": None,
        "support_ensemble_sigma_trans": None,
        "support_ensemble_uncertainty_logdet_penalty": None,
        "support_ensemble_loop_chi2_after": None,
        "support_ensemble_odom_strain_chi2_after": None,
        "support_ensemble_graph_evidence_nll": None,
        "score_support_ensemble": None,
        "failure_reasons": failure_reasons,
    }


def _score_candidate(
    config: RobustLoopVerifierConfig,
    candidate,
    label: bool,
    image_by_idx: Mapping[int, Path],
    odom_by_idx: Mapping[int, np.ndarray],
    cache_order: Sequence[int],
    da3_runner,
    timing: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    query_idx = int(candidate.query_idx)
    candidate_idx = int(candidate.candidate_idx)
    failure_reasons: list[str] = []
    candidate_start = _timing_start(timing)
    record = _new_candidate_record(config, candidate, label, failure_reasons)
    _init_record_timing(record, timing)

    if config.support_ensemble.enabled:
        record = _score_candidate_with_support_ensemble(
            config=config,
            query_idx=query_idx,
            candidate_idx=candidate_idx,
            image_by_idx=image_by_idx,
            odom_by_idx=odom_by_idx,
            cache_order=cache_order,
            da3_runner=da3_runner,
            record=record,
            failure_reasons=failure_reasons,
            timing=timing,
        )
        return _finish_candidate_timing(record, timing, candidate_start)

    support_start = _timing_start(timing)
    support = select_support(
        query_idx=query_idx,
        candidate_idx=candidate_idx,
        available_indices=cache_order,
        image_indices=list(image_by_idx),
        camera_poses=odom_by_idx,
        support_window=config.support_window,
        recent_exclusion_keyframes=config.recent_exclusion_keyframes,
        min_support_baseline_m=config.min_support_baseline_m,
    )
    _record_record_timing(record, timing, "support_selection", support_start)
    record["support_idx"] = support.support_idx
    record["support_baseline_m"] = support.support_baseline_m
    record["support_rejection_reason"] = support.rejection_reason
    if support.support_idx is None:
        reason = support.rejection_reason or "unknown"
        failure_reasons.append(f"support: {reason}")
        return _finish_candidate_timing(record, timing, candidate_start)

    try:
        da3_start = _timing_start(timing)
        triplet = build_da3_triplet(
            image_by_idx[query_idx],
            image_by_idx[candidate_idx],
            image_by_idx[support.support_idx],
            query_idx=query_idx,
            candidate_idx=candidate_idx,
            support_idx=support.support_idx,
        )
        da3_result = da3_runner.run_triplet(triplet)
        _record_record_timing(record, timing, "da3_triplet", da3_start)
    except Exception as exc:
        _record_record_timing(record, timing, "da3_triplet", da3_start)
        failure_reasons.append(f"da3: {type(exc).__name__}: {exc}")
        return _finish_candidate_timing(record, timing, candidate_start)

    return _score_single_support_candidate_after_da3(
        config=config,
        query_idx=query_idx,
        candidate_idx=candidate_idx,
        support_idx=support.support_idx,
        odom_by_idx=odom_by_idx,
        cache_order=cache_order,
        da3_result=da3_result,
        record=record,
        failure_reasons=failure_reasons,
        timing=timing,
        candidate_start=candidate_start,
    )


def _score_single_support_candidate_after_da3(
    config: RobustLoopVerifierConfig,
    query_idx: int,
    candidate_idx: int,
    support_idx: int,
    odom_by_idx: Mapping[int, np.ndarray],
    cache_order: Sequence[int],
    da3_result,
    record: dict[str, Any],
    failure_reasons: list[str],
    timing: dict[str, list[float]] | None,
    candidate_start: float | None,
    run_pgo: bool = True,
) -> dict[str, Any]:
    try:
        sim3_start = _timing_start(timing)
        sim3 = align_triplet_to_candidate_support(
            da3_result.predicted_c2w,
            odom_by_idx[candidate_idx],
            odom_by_idx[support_idx],
        )
        _record_record_timing(record, timing, "sim3_alignment", sim3_start)
    except Exception as exc:
        _record_record_timing(record, timing, "sim3_alignment", sim3_start)
        failure_reasons.append(f"sim3: {type(exc).__name__}: {exc}")
        return _finish_candidate_timing(record, timing, candidate_start)
    record["sim3_valid"] = bool(sim3.valid)
    record["sim3_scale"] = sim3.sim3_scale
    record["sim3_support_alignment_residual_m"] = sim3.support_alignment_residual_m
    record["sim3_direction_error_deg"] = sim3.direction_error_deg
    record["sim3_rejection_reason"] = sim3.rejection_reason
    if not sim3.valid or sim3.loop_factor is None:
        reason = sim3.rejection_reason or "unknown"
        failure_reasons.append(f"sim3: {reason}")
        return _finish_candidate_timing(record, timing, candidate_start)
    loop_factor, rejection_reason = _validated_loop_factor(sim3.loop_factor)
    if loop_factor is None:
        record["sim3_valid"] = False
        record["sim3_rejection_reason"] = rejection_reason
        failure_reasons.append(f"sim3: {rejection_reason}")
        return _finish_candidate_timing(record, timing, candidate_start)
    record["loop_factor"] = loop_factor.reshape(-1)
    record["score_da3_sim3"] = -float(sim3.support_alignment_residual_m)
    if not run_pgo:
        record["pgo_converged"] = None
        return _finish_candidate_timing(record, timing, candidate_start)

    prefix_indices = _prefix_indices_through_query(cache_order, query_idx)
    prefix_poses = [odom_by_idx[index] for index in prefix_indices]
    pgo_start = _timing_start(timing)
    pgo_result = run_full_prefix_pgo(
        prefix_indices=prefix_indices,
        odom_poses=prefix_poses,
        loop_from_idx=query_idx,
        loop_to_idx=candidate_idx,
        loop_factor=loop_factor,
        noise=_pgo_noise_from_config(config),
    )
    _record_record_timing(record, timing, "pgo", pgo_start)
    record["pgo_converged"] = pgo_result.converged
    record["pgo_error_before"] = pgo_result.error_before
    record["pgo_error_after"] = pgo_result.error_after
    record["pgo_failure_reason"] = pgo_result.failure_reason
    if not pgo_result.converged:
        reason = pgo_result.failure_reason or "unknown"
        failure_reasons.append(f"pgo: {reason}")
        return _finish_candidate_timing(record, timing, candidate_start)

    try:
        deformation_rmse = trajectory_deformation_rmse(prefix_poses, pgo_result.optimized_poses)
    except ValueError as exc:
        failure_reasons.append(f"trajectory: {exc}")
        return _finish_candidate_timing(record, timing, candidate_start)
    record["trajectory_deformation_rmse"] = float(deformation_rmse)
    record["score_rover"] = -float(deformation_rmse)
    record["score_rover_source"] = "single_support_loop_factor"
    return _finish_candidate_timing(record, timing, candidate_start)


def _score_candidates_with_batched_triplets(
    config: RobustLoopVerifierConfig,
    candidates: Sequence[Any],
    positive_indices: set[int],
    image_by_idx: Mapping[int, Path],
    odom_by_idx: Mapping[int, np.ndarray],
    cache_order: Sequence[int],
    da3_runner,
    timing: dict[str, list[float]] | None = None,
    run_pgo: bool = True,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any] | None] = [None for _ in candidates]
    pending: list[dict[str, Any]] = []

    for position, candidate in enumerate(candidates):
        query_idx = int(candidate.query_idx)
        candidate_idx = int(candidate.candidate_idx)
        failure_reasons: list[str] = []
        candidate_start = _timing_start(timing)
        record = _new_candidate_record(
            config,
            candidate,
            candidate_idx in positive_indices,
            failure_reasons,
        )
        _init_record_timing(record, timing)

        support_start = _timing_start(timing)
        support = select_support(
            query_idx=query_idx,
            candidate_idx=candidate_idx,
            available_indices=cache_order,
            image_indices=list(image_by_idx),
            camera_poses=odom_by_idx,
            support_window=config.support_window,
            recent_exclusion_keyframes=config.recent_exclusion_keyframes,
            min_support_baseline_m=config.min_support_baseline_m,
        )
        _record_record_timing(record, timing, "support_selection", support_start)
        record["support_idx"] = support.support_idx
        record["support_baseline_m"] = support.support_baseline_m
        record["support_rejection_reason"] = support.rejection_reason
        if support.support_idx is None:
            reason = support.rejection_reason or "unknown"
            failure_reasons.append(f"support: {reason}")
            records[position] = _finish_candidate_timing(record, timing, candidate_start)
            continue

        try:
            triplet = build_da3_triplet(
                image_by_idx[query_idx],
                image_by_idx[candidate_idx],
                image_by_idx[support.support_idx],
                query_idx=query_idx,
                candidate_idx=candidate_idx,
                support_idx=support.support_idx,
            )
        except Exception as exc:
            failure_reasons.append(f"triplet: {type(exc).__name__}: {exc}")
            records[position] = _finish_candidate_timing(record, timing, candidate_start)
            continue
        pending.append(
            {
                "position": position,
                "query_idx": query_idx,
                "candidate_idx": candidate_idx,
                "support_idx": support.support_idx,
                "triplet": triplet,
                "record": record,
                "failure_reasons": failure_reasons,
                "candidate_start": candidate_start,
            }
        )

    batch_size = max(1, int(config.da3.triplet_batch_size))
    for chunk_start in range(0, len(pending), batch_size):
        chunk = pending[chunk_start : chunk_start + batch_size]
        triplets = [item["triplet"] for item in chunk]
        da3_start = _timing_start(timing)
        try:
            da3_results = _run_da3_triplets(da3_runner, triplets)
            if len(da3_results) != len(triplets):
                raise ValueError("DA3 batched result count must match triplet count")
            _record_batch_record_timing(
                [item["record"] for item in chunk],
                timing,
                "da3_triplet",
                da3_start,
            )
        except Exception as exc:
            _record_batch_record_timing(
                [item["record"] for item in chunk],
                timing,
                "da3_triplet",
                da3_start,
            )
            for item in chunk:
                item["failure_reasons"].append(f"da3: {type(exc).__name__}: {exc}")
                records[item["position"]] = _finish_candidate_timing(
                    item["record"],
                    timing,
                    item["candidate_start"],
                )
            continue

        for item, da3_result in zip(chunk, da3_results):
            records[item["position"]] = _score_single_support_candidate_after_da3(
                config=config,
                query_idx=item["query_idx"],
                candidate_idx=item["candidate_idx"],
                support_idx=item["support_idx"],
                odom_by_idx=odom_by_idx,
                cache_order=cache_order,
                da3_result=da3_result,
                record=item["record"],
                failure_reasons=item["failure_reasons"],
                timing=timing,
                candidate_start=item["candidate_start"],
                run_pgo=run_pgo,
            )

    return [record for record in records if record is not None]


def _run_da3_triplets(da3_runner, triplets):
    if callable(getattr(da3_runner, "run_triplets", None)):
        return list(da3_runner.run_triplets(triplets))
    return [da3_runner.run_triplet(triplet) for triplet in triplets]


def _score_candidate_with_support_ensemble(
    config: RobustLoopVerifierConfig,
    query_idx: int,
    candidate_idx: int,
    image_by_idx: Mapping[int, Path],
    odom_by_idx: Mapping[int, np.ndarray],
    cache_order: Sequence[int],
    da3_runner,
    record: dict[str, Any],
    failure_reasons: list[str],
    timing: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    support_start = _timing_start(timing)
    support_selection = select_supports(
        query_idx=query_idx,
        candidate_idx=candidate_idx,
        available_indices=cache_order,
        image_indices=list(image_by_idx),
        camera_poses=odom_by_idx,
        support_window=config.support_window,
        recent_exclusion_keyframes=config.recent_exclusion_keyframes,
        min_support_baseline_m=config.min_support_baseline_m,
        support_count=config.support_ensemble.support_count,
    )
    _record_record_timing(record, timing, "support_selection", support_start)
    if not support_selection.supports:
        reason = support_selection.rejection_reason or "unknown"
        record["support_rejection_reason"] = reason
        failure_reasons.append(f"support_ensemble: {reason}")
        return record

    first_support = support_selection.supports[0]
    record["support_idx"] = first_support.support_idx
    record["support_baseline_m"] = first_support.support_baseline_m
    record["support_ensemble_support_count_used"] = len(support_selection.supports)
    support_records = [
        {
            "support_idx": support.support_idx,
            "support_baseline_m": support.support_baseline_m,
            "sim3_scale": None,
            "support_alignment_residual_m": None,
            "direction_error_deg": None,
        }
        for support in support_selection.supports
    ]
    support_record_by_idx = {
        support_record["support_idx"]: support_record for support_record in support_records
    }
    record["support_ensemble_supports"] = support_records

    valid_factors: list[SupportLoopFactor] = []
    valid_sim3_results = []
    for support in support_selection.supports:
        try:
            da3_start = _timing_start(timing)
            triplet = build_da3_triplet(
                image_by_idx[query_idx],
                image_by_idx[candidate_idx],
                image_by_idx[support.support_idx],
                query_idx=query_idx,
                candidate_idx=candidate_idx,
                support_idx=support.support_idx,
            )
            da3_result = da3_runner.run_triplet(triplet)
            _record_record_timing(record, timing, "da3_triplet", da3_start)
        except Exception as exc:
            _record_record_timing(record, timing, "da3_triplet", da3_start)
            failure_reasons.append(
                f"support_ensemble da3 support {support.support_idx}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue

        try:
            sim3_start = _timing_start(timing)
            sim3 = align_triplet_to_candidate_support(
                da3_result.predicted_c2w,
                odom_by_idx[candidate_idx],
                odom_by_idx[support.support_idx],
            )
            _record_record_timing(record, timing, "sim3_alignment", sim3_start)
        except Exception as exc:
            _record_record_timing(record, timing, "sim3_alignment", sim3_start)
            failure_reasons.append(
                f"support_ensemble sim3 support {support.support_idx}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue

        support_record = support_record_by_idx[support.support_idx]
        support_record["sim3_scale"] = sim3.sim3_scale
        support_record["support_alignment_residual_m"] = sim3.support_alignment_residual_m
        support_record["direction_error_deg"] = sim3.direction_error_deg
        if not sim3.valid or sim3.loop_factor is None:
            reason = sim3.rejection_reason or "unknown"
            failure_reasons.append(
                f"support_ensemble sim3 support {support.support_idx}: {reason}"
            )
            continue

        valid_factors.append(
            SupportLoopFactor(
                support_idx=support.support_idx,
                loop_factor=sim3.loop_factor,
                support_alignment_residual_m=float(sim3.support_alignment_residual_m),
                direction_error_deg=float(sim3.direction_error_deg),
                candidate_support_baseline_m=float(support.support_baseline_m),
            )
        )
        valid_sim3_results.append((support, sim3))

    if not valid_factors:
        failure_reasons.append("support_ensemble: no_valid_support_loop_factors")
        return record

    best_support, best_sim3 = min(
        valid_sim3_results,
        key=lambda item: float(item[1].support_alignment_residual_m),
    )
    record["support_idx"] = best_support.support_idx
    record["support_baseline_m"] = best_support.support_baseline_m
    record["sim3_valid"] = bool(best_sim3.valid)
    record["sim3_scale"] = best_sim3.sim3_scale
    record["sim3_support_alignment_residual_m"] = (
        best_sim3.support_alignment_residual_m
    )
    record["sim3_direction_error_deg"] = best_sim3.direction_error_deg
    record["sim3_rejection_reason"] = best_sim3.rejection_reason
    record["score_da3_sim3"] = -float(best_sim3.support_alignment_residual_m)

    ensemble = aggregate_support_loop_factors(
        valid_factors,
        _support_ensemble_config_from_settings(config),
    )
    for support_record in support_records:
        support_idx = support_record["support_idx"]
        support_record["weight"] = ensemble.support_weights.get(support_idx)
        support_record["residual_norm"] = ensemble.support_residual_norms.get(support_idx)
    if not ensemble.valid or ensemble.loop_factor_mean is None:
        reason = ensemble.rejection_reason or "unknown"
        failure_reasons.append(f"support_ensemble: {reason}")
        return record
    loop_factor, rejection_reason = _validated_loop_factor(ensemble.loop_factor_mean)
    if loop_factor is None:
        failure_reasons.append(f"support_ensemble: {rejection_reason}")
        return record
    record["loop_factor"] = loop_factor.reshape(-1)

    prefix_indices = _prefix_indices_through_query(cache_order, query_idx)
    prefix_poses = [odom_by_idx[index] for index in prefix_indices]
    pgo_start = _timing_start(timing)
    pgo_result = run_full_prefix_pgo(
        prefix_indices=prefix_indices,
        odom_poses=prefix_poses,
        loop_from_idx=query_idx,
        loop_to_idx=candidate_idx,
        loop_factor=loop_factor,
        noise=_pgo_noise_from_config(config),
        loop_sigmas_override=ensemble.loop_sigmas,
    )
    _record_record_timing(record, timing, "pgo", pgo_start)
    record["pgo_converged"] = pgo_result.converged
    record["pgo_error_before"] = pgo_result.error_before
    record["pgo_error_after"] = pgo_result.error_after
    record["pgo_failure_reason"] = pgo_result.failure_reason
    record["support_ensemble_loop_chi2_after"] = pgo_result.loop_chi2_after
    record["support_ensemble_odom_strain_chi2_after"] = (
        pgo_result.odom_strain_chi2_after
    )
    record["support_ensemble_effective_support_count"] = (
        ensemble.effective_support_count
    )
    record["support_ensemble_loop_sigmas"] = list(ensemble.loop_sigmas)
    record["support_ensemble_sigma_rot"] = ensemble.sigma_rot
    record["support_ensemble_sigma_trans"] = ensemble.sigma_trans
    record["support_ensemble_uncertainty_logdet_penalty"] = (
        ensemble.uncertainty_logdet_penalty
    )
    if not pgo_result.converged:
        reason = pgo_result.failure_reason or "unknown"
        failure_reasons.append(f"pgo: {reason}")
        return record

    try:
        deformation_rmse = trajectory_deformation_rmse(prefix_poses, pgo_result.optimized_poses)
    except ValueError as exc:
        failure_reasons.append(f"trajectory: {exc}")
        return record
    record["trajectory_deformation_rmse"] = float(deformation_rmse)
    record["score_rover"] = -float(deformation_rmse)
    record["score_rover_source"] = "support_ensemble_loop_factor"

    graph_nll = graph_evidence_nll(
        pgo_result.loop_chi2_after,
        pgo_result.odom_strain_chi2_after,
        ensemble.uncertainty_logdet_penalty,
    )
    record["support_ensemble_graph_evidence_nll"] = graph_nll
    record["score_support_ensemble"] = -graph_nll if np.isfinite(graph_nll) else None
    return record


TIMING_COMPONENTS = (
    "descriptor_compute",
    "retrieval_search",
    "support_selection",
    "da3_triplet",
    "sim3_alignment",
    "pgo",
    "metrics",
    "total_candidate",
)


def _new_timing_accumulator() -> dict[str, list[float]]:
    return {component: [] for component in TIMING_COMPONENTS}


def _timing_start(timing: dict[str, list[float]] | None) -> float | None:
    if timing is None:
        return None
    return time.perf_counter()


def _record_timing(
    timing: dict[str, list[float]] | None,
    component: str,
    start: float | None,
) -> float:
    if timing is None or start is None:
        return 0.0
    elapsed = max(0.0, time.perf_counter() - start)
    timing.setdefault(component, []).append(elapsed)
    return elapsed


def _init_record_timing(
    record: dict[str, Any],
    timing: dict[str, list[float]] | None,
) -> None:
    if timing is None:
        return
    for component in (
        "support_selection",
        "da3_triplet",
        "sim3_alignment",
        "pgo",
    ):
        record[f"timing_{component}_sec"] = 0.0
    record["timing_total_candidate_sec"] = 0.0


def _record_record_timing(
    record: dict[str, Any],
    timing: dict[str, list[float]] | None,
    component: str,
    start: float | None,
) -> None:
    elapsed = _record_timing(timing, component, start)
    if timing is None:
        return
    key = f"timing_{component}_sec"
    record[key] = float(record.get(key, 0.0)) + elapsed


def _record_batch_record_timing(
    records: Sequence[dict[str, Any]],
    timing: dict[str, list[float]] | None,
    component: str,
    start: float | None,
) -> None:
    if timing is None or start is None:
        return
    elapsed = max(0.0, time.perf_counter() - start)
    if not records:
        timing.setdefault(component, []).append(elapsed)
        return
    per_record_elapsed = elapsed / float(len(records))
    timing.setdefault(component, []).extend(per_record_elapsed for _ in records)
    key = f"timing_{component}_sec"
    for record in records:
        record[key] = float(record.get(key, 0.0)) + per_record_elapsed


def _finish_candidate_timing(
    record: dict[str, Any],
    timing: dict[str, list[float]] | None,
    start: float | None,
) -> dict[str, Any]:
    if timing is None or start is None:
        return record
    elapsed = max(0.0, time.perf_counter() - start)
    record["timing_total_candidate_sec"] = elapsed
    timing.setdefault("total_candidate", []).append(elapsed)
    return record


def _timing_summary(
    timing: dict[str, list[float]] | None,
    *,
    query_count: int,
    candidate_count: int,
) -> dict[str, Any] | None:
    if timing is None:
        return None
    component_totals = {
        component: float(sum(timing.get(component, [])))
        for component in TIMING_COMPONENTS
        if component != "total_candidate"
    }
    return {
        "query_count": int(query_count),
        "candidate_count": int(candidate_count),
        "component_totals_sec": component_totals,
        "per_component_sec": {
            component: _duration_stats(timing.get(component, []))
            for component in TIMING_COMPONENTS
            if component != "total_candidate"
        },
        "per_query_sec": {
            "retrieval_search": _duration_stats(timing.get("retrieval_search", [])),
        },
        "per_candidate_sec": {
            "total": _duration_stats(timing.get("total_candidate", [])),
            "support_selection": _duration_stats(timing.get("support_selection", [])),
            "da3_triplet": _duration_stats(timing.get("da3_triplet", [])),
            "sim3_alignment": _duration_stats(timing.get("sim3_alignment", [])),
            "pgo": _duration_stats(timing.get("pgo", [])),
        },
    }


def _duration_stats(values: Sequence[float]) -> dict[str, float | int]:
    finite = np.asarray(
        [float(value) for value in values if np.isfinite(float(value))],
        dtype=np.float64,
    )
    if finite.size == 0:
        return {
            "count": 0,
            "total": 0.0,
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(finite.size),
        "total": float(finite.sum()),
        "mean": float(finite.mean()),
        "p50": float(np.percentile(finite, 50)),
        "p90": float(np.percentile(finite, 90)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(finite.max()),
    }


def _compute_method_metrics(
    labels: Sequence[bool],
    method_scores: Mapping[str, Sequence[float | None]],
) -> dict[str, dict[str, float]]:
    metrics = {}
    for method, scores in method_scores.items():
        fixed_scores = assign_failure_worst_scores(scores)
        metrics[method] = {
            "AP": average_precision(labels, fixed_scores),
            "MR@100P": max_recall_at_100_precision(labels, fixed_scores),
        }
    return metrics


def _write_run_artifacts(
    output_root: Path,
    config: RobustLoopVerifierConfig,
    records: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Mapping[str, float]],
    timing_summary: Mapping[str, Any] | None = None,
) -> None:
    _write_jsonl(
        output_root / "candidate_records.jsonl",
        [_jsonable(record) for record in records],
    )
    write_json(output_root / "metrics.json", _jsonable(metrics))
    write_metrics_markdown(output_root / "metrics.md", metrics)
    if timing_summary is not None:
        write_json(output_root / "efficiency_timing.json", _jsonable(timing_summary))
    for dirname in ("pr_curves", "visual_records", "trajectory_plots"):
        (output_root / dirname).mkdir(parents=True, exist_ok=True)


def _write_zero_candidate_run(
    output_root: Path,
    config: RobustLoopVerifierConfig,
    timing: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    method_scores = {
        METHOD_SALAD: [],
        METHOD_DA3_SIM3: [],
        METHOD_ROVER: [],
    }
    if config.support_ensemble.enabled:
        method_scores[METHOD_SUPPORT_ENSEMBLE] = []
    metrics = _compute_method_metrics([], method_scores)
    timing_summary = _timing_summary(timing, query_count=0, candidate_count=0)
    _write_run_artifacts(output_root, config, [], metrics, timing_summary=timing_summary)
    result = {
        "candidate_count": 0,
        "metrics": metrics,
    }
    if timing_summary is not None:
        result["timing"] = timing_summary
    return result


def _has_legal_historical_image_candidate(
    query_indices: Sequence[int],
    image_indices: Sequence[int],
    recent_exclusion_keyframes: int,
) -> bool:
    image_index_set = set(image_indices)
    for query_idx in query_indices:
        if query_idx not in image_index_set:
            continue
        exclusion_threshold = query_idx - recent_exclusion_keyframes
        if any(candidate_idx < exclusion_threshold for candidate_idx in image_indices):
            return True
    return False


def _make_backends(config: RobustLoopVerifierConfig, backend: str):
    if backend == "mock":
        return _MockDescriptorBackend(), MockDa3Runner()
    if backend == "real":
        repo_root = Path(__file__).resolve().parents[2]
        salad_repo = repo_root / "da3_streaming" / "loop_utils" / "salad"
        salad_checkpoint = salad_repo / "weights" / "dino_salad.ckpt"
        return (
            SaladDescriptorBackend(
                SaladDescriptorBackendConfig(
                    salad_repo=salad_repo,
                    checkpoint_path=salad_checkpoint,
                )
            ),
            RealDa3Runner(
                RealDa3RunnerConfig(
                    process_res=config.da3.process_res,
                    ref_view_strategy=config.da3.ref_view_strategy,
                    triplet_batch_size=config.da3.triplet_batch_size,
                )
            ),
        )
    raise ValueError("backend must be one of {'mock', 'real'}")


def _is_da3_runner_injection(backend: Any) -> bool:
    return not isinstance(backend, str) and (
        callable(getattr(backend, "run_triplet", None))
        or callable(getattr(backend, "run_triplets", None))
    )


class _MockDescriptorBackend:
    def compute(
        self,
        image_paths: Sequence[str],
        keyframe_indices: Sequence[int],
    ) -> DescriptorSet:
        if len(image_paths) != len(keyframe_indices):
            raise ValueError("image_paths and keyframe_indices must have matching lengths")
        descriptors = []
        for keyframe_idx in keyframe_indices:
            angle = 2.0 * np.pi * (int(keyframe_idx) % 5) / 5.0
            descriptors.append([np.cos(angle), np.sin(angle), 0.1])
        return DescriptorSet(
            keyframe_indices=[int(index) for index in keyframe_indices],
            descriptors=np.asarray(descriptors, dtype=np.float64),
        )


def _parse_keyframe_row(row: Mapping[str, Any], sequence_cache: Path) -> dict[str, Any]:
    idx = int(row["idx"])
    image_path = row.get("image_path")
    return {
        "idx": idx,
        "timestamp": float(row["timestamp"]),
        "image_path": _resolve_cache_image_path(sequence_cache, image_path),
        "odom_pose": _parse_flat_pose(row["odom_pose"], "odom_pose"),
        "gt_pose": _parse_flat_pose(row["gt_pose"], "gt_pose"),
    }


def _read_positives(path: Path) -> dict[int, set[int]]:
    positives_by_query: dict[int, set[int]] = {}
    for row in read_jsonl(path):
        positives_by_query[int(row["query_idx"])] = {
            int(candidate_idx) for candidate_idx in row.get("positive_indices", [])
        }
    return positives_by_query


def _resolve_cache_image_path(sequence_cache: Path, image_path: Any) -> Path | None:
    if image_path is None:
        return None
    path = Path(str(image_path))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"image_path must be relative to sequence_cache: {image_path!r}")
    return sequence_cache / path


def _parse_flat_pose(value: Any, field_name: str) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape == (16,):
        pose = pose.reshape(4, 4)
    if pose.shape != (4, 4):
        raise ValueError(f"{field_name} must be a flattened 4x4 pose")
    if not np.all(np.isfinite(pose)):
        raise ValueError(f"{field_name} must contain only finite values")
    return pose


def _validated_loop_factor(value: Any) -> tuple[np.ndarray | None, str | None]:
    try:
        loop_factor = np.asarray(value, dtype=np.float64)
    except (OverflowError, TypeError, ValueError):
        return None, "invalid_loop_factor"
    if loop_factor.shape != (4, 4):
        return None, "invalid_loop_factor"
    if not np.all(np.isfinite(loop_factor)):
        return None, "invalid_loop_factor"
    return loop_factor, None


def _prefix_indices_through_query(cache_order: Sequence[int], query_idx: int) -> list[int]:
    try:
        query_position = list(cache_order).index(query_idx)
    except ValueError as exc:
        raise ValueError(f"query_idx {query_idx} is missing from cache order") from exc
    return list(cache_order[: query_position + 1])


def _pgo_noise_from_config(config: RobustLoopVerifierConfig) -> PgoNoise:
    return PgoNoise(
        prior_sigmas=tuple(config.pgo_noise.prior_sigmas),
        odom_sigmas=tuple(config.pgo_noise.odom_sigmas),
        loop_sigmas=tuple(config.pgo_noise.loop_sigmas),
    )


def _support_ensemble_config_from_settings(
    config: RobustLoopVerifierConfig,
) -> SupportEnsembleConfig:
    settings = config.support_ensemble
    return SupportEnsembleConfig(
        sigma_rot_floor=settings.sigma_rot_floor,
        sigma_trans_floor=settings.sigma_trans_floor,
        covariance_scale=settings.covariance_scale,
        c_align=settings.c_align,
        c_consensus=settings.c_consensus,
        lambda_dir=settings.lambda_dir,
        robust_iterations=settings.robust_iterations,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.floating):
        scalar = float(value)
        return scalar if math.isfinite(scalar) else None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(_jsonable(row), allow_nan=False, sort_keys=True) + "\n")
