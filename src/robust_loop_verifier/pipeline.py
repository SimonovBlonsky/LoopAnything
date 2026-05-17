"""End-to-end pipeline orchestration for robust loop verifier runs."""

from __future__ import annotations

import json
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
    SaladDescriptorBackend,
    SaladDescriptorBackendConfig,
    retrieve_historical_topk,
)
from robust_loop_verifier.schema import RobustLoopVerifierConfig
from robust_loop_verifier.sim3_factor import align_triplet_to_candidate_support
from robust_loop_verifier.support import select_support


METHOD_SALAD = "SALAD score only"
METHOD_DA3_SIM3 = "SALAD + DA3/Sim3 self-consistency score"
METHOD_ROVER = "SALAD + DA3-ROVER full-prefix trajectory score"


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
    backend: str = "real",
) -> dict[str, Any]:
    """Run the offline verifier against a preprocessed sequence cache."""

    if backend not in {"mock", "real"}:
        raise ValueError("backend must be one of {'mock', 'real'}")
    if query_limit < 0:
        raise ValueError("query_limit must be non-negative")

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
        return _write_zero_candidate_run(output_root, config)

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
        return _write_zero_candidate_run(output_root, config)

    descriptor_backend, da3_runner = _make_backends(config, backend)
    descriptor_set = descriptor_backend.compute(
        [str(keyframe["image_path"]) for keyframe in descriptor_rows],
        image_indices,
    )

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
        retrieval = retrieve_historical_topk(
            query_idx=query_idx,
            descriptors=descriptor_set,
            top_k=config.retrieval_top_k_main,
            recent_exclusion_keyframes=config.recent_exclusion_keyframes,
        )
        positive_indices = positives_by_query.get(query_idx, set())
        for candidate in retrieval.candidates:
            label = candidate.candidate_idx in positive_indices
            record = _score_candidate(
                config=config,
                candidate=candidate,
                label=label,
                image_by_idx=image_by_idx,
                odom_by_idx=odom_by_idx,
                cache_order=cache_order,
                da3_runner=da3_runner,
            )
            labels.append(label)
            records.append(record)

    metrics = _compute_method_metrics(
        labels,
        {
            METHOD_SALAD: [record["score_salad"] for record in records],
            METHOD_DA3_SIM3: [record["score_da3_sim3"] for record in records],
            METHOD_ROVER: [record["score_rover"] for record in records],
        },
    )
    _write_run_artifacts(output_root, config, records, metrics)

    return {
        "candidate_count": len(records),
        "metrics": metrics,
    }


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
            records.append(
                {
                    "query_idx": candidate.query_idx,
                    "candidate_idx": candidate.candidate_idx,
                    "rank": candidate.rank,
                    "label": label,
                    "salad_score": candidate.score,
                    "trajectory_deformation_rmse": deformation_rmse,
                    "score_rover": _score_rover(deformation_rmse, pgo_result.converged),
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


def _score_candidate(
    config: RobustLoopVerifierConfig,
    candidate,
    label: bool,
    image_by_idx: Mapping[int, Path],
    odom_by_idx: Mapping[int, np.ndarray],
    cache_order: Sequence[int],
    da3_runner,
) -> dict[str, Any]:
    query_idx = int(candidate.query_idx)
    candidate_idx = int(candidate.candidate_idx)
    failure_reasons: list[str] = []
    record = {
        "query_idx": query_idx,
        "candidate_idx": candidate_idx,
        "rank": int(candidate.rank),
        "label": bool(label),
        "salad_score": float(candidate.score),
        "score_salad": float(candidate.score),
        "score_da3_sim3": None,
        "score_rover": None,
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
        "failure_reasons": failure_reasons,
    }

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
    record["support_idx"] = support.support_idx
    record["support_baseline_m"] = support.support_baseline_m
    record["support_rejection_reason"] = support.rejection_reason
    if support.support_idx is None:
        reason = support.rejection_reason or "unknown"
        failure_reasons.append(f"support: {reason}")
        return record

    try:
        triplet = build_da3_triplet(
            image_by_idx[query_idx],
            image_by_idx[candidate_idx],
            image_by_idx[support.support_idx],
            query_idx=query_idx,
            candidate_idx=candidate_idx,
            support_idx=support.support_idx,
        )
        da3_result = da3_runner.run_triplet(triplet)
    except Exception as exc:
        failure_reasons.append(f"da3: {type(exc).__name__}: {exc}")
        return record

    try:
        sim3 = align_triplet_to_candidate_support(
            da3_result.predicted_c2w,
            odom_by_idx[candidate_idx],
            odom_by_idx[support.support_idx],
        )
    except Exception as exc:
        failure_reasons.append(f"sim3: {type(exc).__name__}: {exc}")
        return record
    record["sim3_valid"] = bool(sim3.valid)
    record["sim3_scale"] = sim3.sim3_scale
    record["sim3_support_alignment_residual_m"] = sim3.support_alignment_residual_m
    record["sim3_direction_error_deg"] = sim3.direction_error_deg
    record["sim3_rejection_reason"] = sim3.rejection_reason
    if not sim3.valid or sim3.loop_factor is None:
        reason = sim3.rejection_reason or "unknown"
        failure_reasons.append(f"sim3: {reason}")
        return record
    record["score_da3_sim3"] = -float(sim3.support_alignment_residual_m)

    prefix_indices = _prefix_indices_through_query(cache_order, query_idx)
    prefix_poses = [odom_by_idx[index] for index in prefix_indices]
    pgo_result = run_full_prefix_pgo(
        prefix_indices=prefix_indices,
        odom_poses=prefix_poses,
        loop_from_idx=query_idx,
        loop_to_idx=candidate_idx,
        loop_factor=sim3.loop_factor,
        noise=_pgo_noise_from_config(config),
    )
    record["pgo_converged"] = pgo_result.converged
    record["pgo_error_before"] = pgo_result.error_before
    record["pgo_error_after"] = pgo_result.error_after
    record["pgo_failure_reason"] = pgo_result.failure_reason
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
    return record


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
) -> None:
    _write_jsonl(
        output_root / "candidate_records.jsonl",
        [_jsonable(record) for record in records],
    )
    write_json(output_root / "metrics.json", _jsonable(metrics))
    write_metrics_markdown(output_root / "metrics.md", metrics)
    for dirname in ("pr_curves", "visual_records", "trajectory_plots"):
        (output_root / dirname).mkdir(parents=True, exist_ok=True)


def _write_zero_candidate_run(
    output_root: Path,
    config: RobustLoopVerifierConfig,
) -> dict[str, Any]:
    metrics = _compute_method_metrics(
        [],
        {
            METHOD_SALAD: [],
            METHOD_DA3_SIM3: [],
            METHOD_ROVER: [],
        },
    )
    _write_run_artifacts(output_root, config, [], metrics)
    return {
        "candidate_count": 0,
        "metrics": metrics,
    }


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
                )
            ),
        )
    raise ValueError("backend must be one of {'mock', 'real'}")


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


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")
