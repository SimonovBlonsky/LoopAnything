from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import is_dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from robust_loop_verifier.io import read_json, read_jsonl
from robust_loop_verifier.pipeline import score_frozen_query_candidates
from robust_loop_verifier.retrieval import (
    DescriptorSet,
    SaladDescriptorBackend,
    SaladDescriptorBackendConfig,
)
from robust_loop_verifier.rover_evaluation import verify_score_manifest
from robust_loop_verifier.schema import RobustLoopVerifierConfig
from robust_loop_verifier.score_sweep import compute_named_scores


def score_descriptor_pairs(
    pairs: Sequence[Mapping[str, object]],
    descriptors: DescriptorSet,
) -> list[dict[str, object]]:
    row_by_idx = {
        int(keyframe_idx): row
        for row, keyframe_idx in enumerate(descriptors.keyframe_indices)
    }
    descriptor_matrix = np.asarray(descriptors.descriptors, dtype=np.float64)
    output: list[dict[str, object]] = []
    for pair in pairs:
        pair_id = str(pair["pair_id"])
        query_row = row_by_idx.get(int(pair["query_idx"]))
        candidate_row = row_by_idx.get(int(pair["candidate_idx"]))
        if query_row is None or candidate_row is None:
            output.append({"pair_id": pair_id, "score": None, "status": "failed"})
            continue
        try:
            query_descriptor = _l2_normalize(descriptor_matrix[query_row])
            candidate_descriptor = _l2_normalize(descriptor_matrix[candidate_row])
        except ValueError:
            output.append({"pair_id": pair_id, "score": None, "status": "failed"})
            continue
        score = float(np.dot(query_descriptor, candidate_descriptor))
        output.append({"pair_id": pair_id, "score": score, "status": "ok"})
    return output


def candidate_records_to_score_rows(
    records: Sequence[Mapping[str, object]],
    score_field: str,
) -> list[dict[str, object]]:
    rows = []
    for record in records:
        score = _finite_float_or_none(record.get(score_field))
        rows.append(
            {
                "pair_id": str(record["pair_id"]),
                "score": score,
                "status": "ok" if score is not None else "failed",
            }
        )
    return rows


def load_frozen_pairs_by_sequence(
    benchmark_root: Path,
) -> dict[str, list[dict[str, object]]]:
    pairs = [dict(row) for row in read_jsonl(Path(benchmark_root) / "benchmark_pairs.jsonl")]
    grouped: dict[str, list[dict[str, object]]] = OrderedDict()
    for pair in pairs:
        sequence_key = _sequence_key(pair)
        grouped.setdefault(sequence_key, []).append(pair)
    return grouped


def compute_sequence_descriptors(
    pairs: Sequence[Mapping[str, object]],
    cache_root: Path,
    backend,
) -> DescriptorSet:
    needed_indices = _needed_keyframe_indices(pairs)
    keyframes = _read_keyframes(Path(cache_root), needed_indices=needed_indices)
    image_paths = [str(keyframes[index]["image_path"]) for index in needed_indices]
    return backend.compute(image_paths, needed_indices)


def score_netvlad_pairs(
    benchmark_root: Path,
    backend,
) -> list[dict[str, object]]:
    return _score_descriptor_backend_pairs(benchmark_root, backend)


def score_salad_pairs(
    benchmark_root: Path,
    backend,
) -> list[dict[str, object]]:
    return _score_descriptor_backend_pairs(benchmark_root, backend)


def score_verifier_pairs(
    benchmark_root: Path,
    config,
    da3_backend,
) -> list[dict[str, Any]]:
    benchmark_root = Path(benchmark_root)
    manifest_pairs = _read_manifest_pairs(benchmark_root)
    salad_scores = _load_frozen_salad_scores(benchmark_root, manifest_pairs)
    pairs_by_sequence = load_frozen_pairs_by_sequence(benchmark_root)
    cache_by_sequence = _cache_by_sequence(benchmark_root)
    recent_exclusion_by_sequence = _recent_exclusion_by_sequence(benchmark_root)
    records_by_pair_id: dict[str, dict[str, Any]] = {}
    for sequence_key, sequence_pairs in pairs_by_sequence.items():
        dataset_name = sequence_pairs[0]["dataset"]
        sequence_config = _config_for_sequence(config, dataset_name)
        sequence_config = _config_with_sequence_recent_exclusion(
            sequence_config,
            recent_exclusion_by_sequence.get(sequence_key),
        )
        sequence_da3_runner = _runner_for_sequence(da3_backend, dataset_name)
        sequence_cache = cache_by_sequence[sequence_key]
        pairs_by_pair_id = {str(pair["pair_id"]): pair for pair in sequence_pairs}
        keyframes = _read_keyframes(
            sequence_cache,
            needed_indices=None,
            require_odom=True,
        )
        image_by_idx = {
            index: Path(row["image_path"])
            for index, row in keyframes.items()
            if row["image_path"] is not None
        }
        odom_by_idx = {
            index: np.asarray(row["odom_pose"], dtype=np.float64)
            for index, row in keyframes.items()
        }
        cache_order = list(keyframes)
        for query_idx, query_pairs in _pairs_by_query(sequence_pairs).items():
            frozen_candidates = []
            for pair in query_pairs:
                pair_id = str(pair["pair_id"])
                salad_score = salad_scores[pair_id]
                if salad_score is None:
                    records_by_pair_id[pair_id] = _failed_frozen_salad_record(
                        pair,
                        sequence_config,
                    )
                    continue
                frozen_candidates.append(
                    {
                        "pair_id": pair_id,
                        "candidate_idx": int(pair["candidate_idx"]),
                        "rank": int(pair["rank"]),
                        "score": salad_score,
                    }
                )
            if not frozen_candidates:
                continue
            for record in score_frozen_query_candidates(
                config=sequence_config,
                query_idx=int(query_idx),
                frozen_candidates=frozen_candidates,
                image_by_idx=image_by_idx,
                odom_by_idx=odom_by_idx,
                cache_order=cache_order,
                da3_runner=sequence_da3_runner,
            ):
                pair = pairs_by_pair_id[str(record["pair_id"])]
                _attach_sequence_identity(record, pair)
                records_by_pair_id[str(record["pair_id"])] = record
    records = [records_by_pair_id[str(pair["pair_id"])] for pair in manifest_pairs]
    validate_score_row_order(
        manifest_pairs,
        records,
    )
    return records


def iter_geometry_factor_batches(
    benchmark_root: Path,
    config,
    da3_runner,
    *,
    completed_pair_ids: set[str] | None = None,
) -> Iterator[list[dict[str, Any]]]:
    benchmark_root = Path(benchmark_root)
    completed_pair_ids = {str(pair_id) for pair_id in (completed_pair_ids or set())}
    manifest_pairs = _read_manifest_pairs(benchmark_root)
    cache_by_sequence = _cache_by_sequence(benchmark_root)
    recent_exclusion_by_sequence = _recent_exclusion_by_sequence(benchmark_root)
    keyframes_by_sequence: dict[str, dict[int, dict[str, Any]]] = {}

    for query_pairs in _pending_query_batches(manifest_pairs, completed_pair_ids):
        first_pair = query_pairs[0]
        sequence_key = _sequence_key(first_pair)
        dataset_name = first_pair["dataset"]
        sequence_config = _config_for_sequence(config, dataset_name)
        sequence_config = _config_with_sequence_recent_exclusion(
            sequence_config,
            recent_exclusion_by_sequence.get(sequence_key),
        )
        sequence_runner = _runner_for_sequence(da3_runner, dataset_name)
        if sequence_key not in keyframes_by_sequence:
            keyframes_by_sequence[sequence_key] = _read_keyframes(
                cache_by_sequence[sequence_key],
                needed_indices=None,
                require_odom=True,
                require_gt=True,
            )
        keyframes = keyframes_by_sequence[sequence_key]
        _validate_query_batch_keyframes(keyframes, query_pairs, sequence_key)
        image_by_idx = {
            index: Path(row["image_path"])
            for index, row in keyframes.items()
            if row["image_path"] is not None
        }
        odom_by_idx = {
            index: np.asarray(row["odom_pose"], dtype=np.float64)
            for index, row in keyframes.items()
        }
        frozen_candidates = [
            {
                "pair_id": str(pair["pair_id"]),
                "candidate_idx": int(pair["candidate_idx"]),
                "rank": int(pair["rank"]),
                "score": float(pair.get("dbow2_score", 0.0)),
            }
            for pair in query_pairs
        ]
        scored_records = score_frozen_query_candidates(
            config=sequence_config,
            query_idx=int(first_pair["query_idx"]),
            frozen_candidates=frozen_candidates,
            image_by_idx=image_by_idx,
            odom_by_idx=odom_by_idx,
            cache_order=list(keyframes),
            da3_runner=sequence_runner,
            run_pgo=False,
        )
        record_by_pair_id = {str(record["pair_id"]): record for record in scored_records}
        batch = []
        for pair in query_pairs:
            pair_id = str(pair["pair_id"])
            record = record_by_pair_id[pair_id]
            query_keyframe = keyframes[int(pair["query_idx"])]
            candidate_keyframe = keyframes[int(pair["candidate_idx"])]
            record["gt_query_pose"] = (
                np.asarray(query_keyframe["gt_pose"], dtype=np.float64).reshape(-1).tolist()
            )
            record["gt_candidate_pose"] = (
                np.asarray(candidate_keyframe["gt_pose"], dtype=np.float64).reshape(-1).tolist()
            )
            _attach_sequence_identity(record, pair)
            batch.append(record)
        yield batch


def validate_score_row_order(
    pairs: Sequence[Mapping[str, object]],
    rows: Sequence[Mapping[str, object]],
) -> None:
    expected = [str(pair["pair_id"]) for pair in pairs]
    observed = [str(row["pair_id"]) for row in rows]
    duplicates = sorted(_duplicates(observed))
    if duplicates:
        raise ValueError(f"duplicate score rows: {duplicates[:5]}")
    unknown = sorted(set(observed) - set(expected))
    if unknown:
        raise ValueError(f"unknown pair_ids: {unknown[:5]}")
    if len(observed) != len(expected):
        raise ValueError(
            f"score row count mismatch: expected {len(expected)}, got {len(observed)}"
        )
    if observed != expected:
        raise ValueError("score rows must preserve benchmark pair order")


def records_to_named_score_rows(
    records: Sequence[Mapping[str, Any]],
    method_name: str,
) -> list[dict[str, object]]:
    scores = compute_named_scores(records, method_name)
    rows = []
    for record, score in zip(records, scores):
        valid = score is not None and math.isfinite(float(score))
        rows.append(
            {
                "pair_id": str(record["pair_id"]),
                "score": float(score) if valid else None,
                "status": "ok" if valid else "failed",
            }
        )
    return rows


def build_salad_backend(
    device: str = "cuda",
    *,
    salad_repo: Path | None = None,
    checkpoint_path: Path | None = None,
    backbone: str = "dinov2_vitb14",
    batch_size: int = 32,
):
    repo_root = Path(__file__).resolve().parents[2]
    salad_repo = Path(salad_repo) if salad_repo is not None else (
        repo_root / "da3_streaming" / "loop_utils" / "salad"
    )
    checkpoint_path = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else salad_repo / "weights" / "dino_salad.ckpt"
    )
    return SaladDescriptorBackend(
        SaladDescriptorBackendConfig(
            salad_repo=salad_repo,
            checkpoint_path=checkpoint_path,
            device=device,
            backbone=str(backbone),
            batch_size=int(batch_size),
        )
    )


def build_netvlad_backend(netvlad_root: Path, device: str = "cuda"):
    from baseline_scripts.netvlad_retrieval_pipeline import NetVladDescriptorBackend

    return NetVladDescriptorBackend(Path(netvlad_root), device)


def load_verifier_config_for_dataset(
    benchmark_root: Path,
    dataset_name: str,
) -> RobustLoopVerifierConfig:
    manifest = read_json(Path(benchmark_root) / "manifest.json")
    verifier_configs = manifest.get("verifier_configs", {})
    if not isinstance(verifier_configs, Mapping):
        raise ValueError("manifest verifier_configs must be a mapping")
    config_path = verifier_configs.get(str(dataset_name))
    if config_path is None:
        raise ValueError(f"missing verifier config for dataset: {dataset_name}")
    parsed_path = Path(str(config_path))
    if not parsed_path.is_absolute():
        parsed_path = Path(__file__).resolve().parents[2] / parsed_path
    return RobustLoopVerifierConfig.from_yaml(parsed_path)


def _score_descriptor_backend_pairs(benchmark_root: Path, backend) -> list[dict[str, object]]:
    benchmark_root = Path(benchmark_root)
    manifest_pairs = _read_manifest_pairs(benchmark_root)
    pairs_by_sequence = load_frozen_pairs_by_sequence(benchmark_root)
    cache_by_sequence = _cache_by_sequence(benchmark_root)
    rows_by_pair_id: dict[str, dict[str, object]] = {}
    for sequence_key, sequence_pairs in pairs_by_sequence.items():
        descriptors = compute_sequence_descriptors(
            sequence_pairs,
            cache_by_sequence[sequence_key],
            backend,
        )
        for row in score_descriptor_pairs(sequence_pairs, descriptors):
            rows_by_pair_id[str(row["pair_id"])] = row
    rows = [rows_by_pair_id[str(pair["pair_id"])] for pair in manifest_pairs]
    validate_score_row_order(manifest_pairs, rows)
    return rows


def _read_manifest_pairs(benchmark_root: Path) -> list[dict[str, object]]:
    return [dict(row) for row in read_jsonl(Path(benchmark_root) / "benchmark_pairs.jsonl")]


def _load_frozen_salad_scores(
    benchmark_root: Path,
    manifest_pairs: Sequence[Mapping[str, object]],
) -> dict[str, float | None]:
    pair_path = Path(benchmark_root) / "benchmark_pairs.jsonl"
    score_path = Path(benchmark_root) / "scores" / "salad.jsonl"
    score_manifest_path = score_path.with_suffix(".manifest.json")
    verify_score_manifest(
        pair_path,
        score_path,
        score_manifest_path,
        method_name="SALAD",
    )
    rows = [dict(row) for row in read_jsonl(score_path)]
    validate_score_row_order(manifest_pairs, rows)

    scores: dict[str, float | None] = {}
    for row in rows:
        pair_id = str(row["pair_id"])
        scores[pair_id] = _validated_frozen_salad_score(row)
    return scores


def _validated_frozen_salad_score(row: Mapping[str, object]) -> float | None:
    pair_id = str(row.get("pair_id", "<missing>"))
    status = row.get("status")
    if status not in ("ok", "failed"):
        raise ValueError(f"invalid frozen SALAD status for pair_id={pair_id}: {status!r}")

    raw_score = row.get("score")
    score = _finite_float_or_none(raw_score)
    if status == "ok":
        if score is None:
            raise ValueError(f"ok frozen SALAD row requires finite score for pair_id={pair_id}")
        return score

    if raw_score is not None:
        raise ValueError(f"failed frozen SALAD row must use score=None for pair_id={pair_id}")
    return None


def _cache_by_sequence(benchmark_root: Path) -> dict[str, Path]:
    manifest = read_json(Path(benchmark_root) / "manifest.json")
    rows = manifest.get("sequences")
    if not isinstance(rows, list):
        raise ValueError("manifest sequences must be a list")
    output = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("manifest sequence rows must be mappings")
        output[_sequence_key(row)] = Path(str(row["cache"]))
    return output


def _recent_exclusion_by_sequence(benchmark_root: Path) -> dict[str, int]:
    manifest = read_json(Path(benchmark_root) / "manifest.json")
    rows = manifest.get("sequences")
    if not isinstance(rows, list):
        raise ValueError("manifest sequences must be a list")
    output = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("manifest sequence rows must be mappings")
        if "recent_exclusion_keyframes" not in row:
            continue
        recent_exclusion = int(row["recent_exclusion_keyframes"])
        if recent_exclusion < 0:
            raise ValueError("manifest sequence recent_exclusion_keyframes must be non-negative")
        output[_sequence_key(row)] = recent_exclusion
    return output


def _config_with_sequence_recent_exclusion(config, recent_exclusion_keyframes: int | None):
    if recent_exclusion_keyframes is None:
        return config
    if not is_dataclass(config):
        return config
    if not hasattr(config, "recent_exclusion_keyframes"):
        return config
    return replace(config, recent_exclusion_keyframes=recent_exclusion_keyframes)


def _read_keyframes(
    cache_root: Path,
    *,
    needed_indices: Sequence[int] | None,
    require_odom: bool = False,
    require_gt: bool = False,
) -> dict[int, dict[str, Any]]:
    needed = None if needed_indices is None else set(int(index) for index in needed_indices)
    rows = {}
    for row in read_jsonl(Path(cache_root) / "keyframes.jsonl"):
        idx = int(row["idx"])
        if needed is not None and idx not in needed:
            continue
        image_path = row.get("image_path")
        if image_path is None and needed is not None:
            continue
        parsed = dict(row)
        parsed["image_path"] = (
            Path(cache_root) / str(image_path) if image_path is not None else None
        )
        if require_odom and "odom_pose" not in parsed:
            raise ValueError(f"missing odom_pose for keyframe {idx} in {cache_root}")
        if require_odom:
            parsed["odom_pose"] = _parse_required_keyframe_pose(
                parsed["odom_pose"],
                "odom_pose",
                idx,
                cache_root,
            )
        if require_gt and "gt_pose" not in parsed:
            raise ValueError(f"missing gt_pose for keyframe {idx} in {cache_root}")
        if require_gt:
            parsed["gt_pose"] = _parse_required_keyframe_pose(
                parsed["gt_pose"],
                "gt_pose",
                idx,
                cache_root,
            )
        rows[idx] = parsed
    if needed is not None:
        missing = sorted(needed - set(rows))
        if missing:
            raise ValueError(f"missing keyframe images for indices: {missing[:5]}")
    return rows


def _parse_required_keyframe_pose(
    value: object,
    field_name: str,
    keyframe_idx: int,
    cache_root: Path,
) -> np.ndarray:
    try:
        pose = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"keyframe {keyframe_idx} {field_name} in {cache_root} must be a finite 4x4 pose"
        ) from exc
    if pose.shape == (16,):
        pose = pose.reshape(4, 4)
    if pose.shape != (4, 4):
        raise ValueError(
            f"keyframe {keyframe_idx} {field_name} in {cache_root} must be a finite 4x4 pose"
        )
    if not np.all(np.isfinite(pose)):
        raise ValueError(
            f"keyframe {keyframe_idx} {field_name} in {cache_root} must be a finite 4x4 pose"
        )
    return pose


def _validate_query_batch_keyframes(
    keyframes: Mapping[int, Mapping[str, Any]],
    query_pairs: Sequence[Mapping[str, object]],
    sequence_key: str,
) -> None:
    for pair in query_pairs:
        pair_id = str(pair["pair_id"])
        for role, field_name in (("query", "query_idx"), ("candidate", "candidate_idx")):
            keyframe_idx = int(pair[field_name])
            if keyframe_idx not in keyframes:
                raise ValueError(
                    f"missing keyframe for sequence {sequence_key} pair_id={pair_id} "
                    f"{role} idx={keyframe_idx}"
                )


def _needed_keyframe_indices(pairs: Sequence[Mapping[str, object]]) -> list[int]:
    needed = {
        int(pair["query_idx"]) for pair in pairs
    } | {int(pair["candidate_idx"]) for pair in pairs}
    return sorted(needed)


def _pairs_by_query(
    pairs: Sequence[Mapping[str, object]],
) -> dict[int, list[Mapping[str, object]]]:
    output: dict[int, list[Mapping[str, object]]] = OrderedDict()
    for pair in pairs:
        output.setdefault(int(pair["query_idx"]), []).append(pair)
    return output


def _pending_query_batches(
    pairs: Sequence[Mapping[str, object]],
    completed_pair_ids: set[str],
) -> Iterator[list[Mapping[str, object]]]:
    batch: list[Mapping[str, object]] = []
    batch_key: tuple[str, int] | None = None
    for pair in pairs:
        if str(pair["pair_id"]) in completed_pair_ids:
            continue
        pair_key = (_sequence_key(pair), int(pair["query_idx"]))
        if batch and pair_key != batch_key:
            yield batch
            batch = []
        batch_key = pair_key
        batch.append(pair)
    if batch:
        yield batch


def _sequence_key(row: Mapping[str, object]) -> str:
    return f"{row['dataset']}/{row['platform']}/{row['sequence']}"


def _attach_sequence_identity(
    record: dict[str, Any],
    pair: Mapping[str, object],
) -> None:
    for field in ("dataset", "platform", "sequence"):
        if field in pair:
            record[field] = pair[field]
    if all(field in pair for field in ("dataset", "platform", "sequence")):
        record["sequence_key"] = _sequence_key(pair)


def _failed_frozen_salad_record(
    pair: Mapping[str, object],
    config,
) -> dict[str, Any]:
    support_ensemble = getattr(config, "support_ensemble", None)
    record = {
        "pair_id": str(pair["pair_id"]),
        "query_idx": int(pair["query_idx"]),
        "candidate_idx": int(pair["candidate_idx"]),
        "rank": int(pair["rank"]),
        "salad_score": None,
        "score_salad": None,
        "score_da3_sim3": None,
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
        "support_ensemble_enabled": bool(getattr(support_ensemble, "enabled", False)),
        "support_ensemble_support_count_requested": int(
            getattr(support_ensemble, "support_count", 0)
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
        "failure_reasons": ["frozen SALAD: failed or non-finite score"],
    }
    _attach_sequence_identity(record, pair)
    return record


def _config_for_sequence(config, dataset_name: object):
    if isinstance(config, Mapping):
        dataset_config = config.get(str(dataset_name))
        if dataset_config is None:
            raise ValueError(f"missing verifier config for dataset: {dataset_name}")
        return dataset_config
    return config


def _runner_for_sequence(da3_runner, dataset_name: object):
    if isinstance(da3_runner, Mapping):
        dataset_runner = da3_runner.get(str(dataset_name))
        if dataset_runner is None:
            raise ValueError(f"missing DA3 runner for dataset: {dataset_name}")
        return dataset_runner
    return da3_runner


def _l2_normalize(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("descriptor norm must be finite and positive")
    return vector / norm


def _finite_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _duplicates(values: Sequence[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates
