from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from loop_policy.da3_runner import (
    Da3Runner,
    DepthAnything3Runner,
    MockDa3Runner,
    build_da3_group,
)
from loop_policy.geometry import (
    camera_pose_from_lidar_pose,
    make_transform,
    pose_residual,
    quaternion_xyzw_to_matrix,
)
from loop_policy.io import load_aster_raw_sequence, write_jsonl
from loop_policy.labels import build_x_geom, compute_safe_loop_factor_v1
from loop_policy.retrieval import (
    DescriptorCache,
    DescriptorExtractor,
    DinoSaladDescriptorExtractor,
    normalize_descriptors,
    rank_causal_topk,
    save_descriptor_cache,
)
from loop_policy.schema import (
    CandidateFeatureRecord,
    KeyframeRecord,
    LoopPolicyDatasetConfig,
    dataclass_to_json_dict,
)
from loop_policy.sim3_prior import (
    Sim3PriorConfig,
    align_da3_poses_with_candidate_support_prior,
)
from loop_policy.support import select_supports


def platform_from_sequence(sequence: str) -> str:
    platform = sequence.split("_", 1)[0]
    if platform not in {"handheld", "legged", "ugv", "vehicle"}:
        raise ValueError(f"unsupported FusionPortable sequence platform in {sequence!r}")
    return platform


def _platform_summary_for_failed_sequence(sequence: str) -> str:
    return sequence.split("_", 1)[0] or "unknown"


def raw_dir_for_sequence(dataset_root: Path, sequence: str) -> Path:
    platform = platform_from_sequence(sequence)
    return Path(dataset_root) / platform / sequence / "raw"


def write_root_manifests(
    config: LoopPolicyDatasetConfig,
    summaries: List[Dict[str, object]],
) -> None:
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "loop_policy_dataset_v1",
        "causal": config.causal,
        "sequence_count": len(summaries),
        "settings": dataclass_to_json_dict(config),
    }
    (output_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    write_jsonl(output_root / "sequence_summaries.jsonl", summaries)


def audit_causal_leakage(records: List[Dict[str, object]]) -> bool:
    for record in records:
        query_timestamp = float(record["query_timestamp"])
        if float(record["candidate_timestamp"]) >= query_timestamp:
            return False
        for support_timestamp in record.get("selected_support_timestamps", []):
            if float(support_timestamp) >= query_timestamp:
                return False
    return True


def _camera_pose_from_pose_record(pose, t_camera_lidar: np.ndarray) -> np.ndarray:
    t_world_lidar = make_transform(
        quaternion_xyzw_to_matrix(pose.quaternion_xyzw),
        np.asarray(pose.position, dtype=np.float64),
    )
    return camera_pose_from_lidar_pose(t_world_lidar, t_camera_lidar)


def _pose_map(sequence) -> Dict[int, np.ndarray]:
    poses = {}
    has_trajectory_idx = any(
        keyframe.trajectory_idx is not None for keyframe in sequence.keyframes
    )
    if has_trajectory_idx:
        for keyframe in sequence.keyframes:
            if keyframe.trajectory_idx is None:
                raise ValueError(
                    "all keyframes must define trajectory_idx when any trajectory_idx is present"
                )
            if keyframe.trajectory_idx < 0 or keyframe.trajectory_idx >= len(sequence.trajectory):
                raise ValueError(
                    "trajectory_idx out of range for "
                    f"keyframe_idx={keyframe.keyframe_idx}: {keyframe.trajectory_idx}"
                )
            poses[keyframe.keyframe_idx] = _camera_pose_from_pose_record(
                sequence.trajectory[keyframe.trajectory_idx],
                sequence.t_camera_lidar,
            )
        return poses

    if len(sequence.keyframes) != len(sequence.trajectory):
        raise ValueError(
            "trajectory length must match keyframes when trajectory_idx is absent: "
            f"{len(sequence.trajectory)} poses for {len(sequence.keyframes)} keyframes"
        )
    for keyframe, pose in zip(sequence.keyframes, sequence.trajectory):
        poses[keyframe.keyframe_idx] = _camera_pose_from_pose_record(
            pose,
            sequence.t_camera_lidar,
        )
    return poses


def _descriptor_cache(sequence, extractor: DescriptorExtractor) -> DescriptorCache:
    image_keyframes = [
        keyframe for keyframe in sequence.keyframes if keyframe.image_path is not None
    ]
    image_paths = [Path(keyframe.image_path) for keyframe in image_keyframes]
    descriptors = normalize_descriptors(extractor.extract(image_paths))
    return DescriptorCache(
        keyframe_idx=np.asarray(
            [keyframe.keyframe_idx for keyframe in image_keyframes],
            dtype=np.int64,
        ),
        timestamps=np.asarray(
            [keyframe.timestamp for keyframe in image_keyframes],
            dtype=np.float64,
        ),
        descriptors=descriptors,
        normalized=True,
    )


def _keyframe_by_idx(keyframes: List[KeyframeRecord]) -> Dict[int, KeyframeRecord]:
    return {keyframe.keyframe_idx: keyframe for keyframe in keyframes}


def _resolve_keyframe_image_paths(
    keyframes: List[KeyframeRecord],
    raw_dir: Path,
) -> List[KeyframeRecord]:
    resolved_keyframes: List[KeyframeRecord] = []
    for keyframe in keyframes:
        if keyframe.image_path is None:
            resolved_keyframes.append(keyframe)
            continue

        image_path = Path(keyframe.image_path)
        if not image_path.is_absolute():
            image_path = Path(raw_dir) / image_path
        resolved_keyframes.append(replace(keyframe, image_path=str(image_path)))
    return resolved_keyframes


def _increment_negative_reason(negative_reasons: Dict[str, int], reason: Optional[str]) -> None:
    key = reason or "unknown_negative"
    negative_reasons[key] = negative_reasons.get(key, 0) + 1


def _finite_or_sentinel(value: float, sentinel: float) -> float:
    value = float(value)
    return value if math.isfinite(value) else float(sentinel)


def _sanitize_finite_values(
    values: Dict[str, float],
    config: LoopPolicyDatasetConfig,
) -> Dict[str, float]:
    sentinels = {
        "abs_log_sim3_scale": config.abs_log_sim3_scale_thr + 1.0,
        "support_align_rmse": config.support_align_rmse_thr + 1.0,
        "direction_error_deg": config.direction_error_thr_deg + 1.0,
        "aligned_vs_odom_rot_residual_deg": config.loose_rot_thr_deg + 1.0,
        "aligned_vs_odom_trans_residual_norm": config.loose_trans_thr_m + 1.0,
    }
    return {
        field: _finite_or_sentinel(value, sentinels.get(field, 0.0))
        for field, value in values.items()
    }


def _label_directory_parts(record: CandidateFeatureRecord) -> List[str]:
    return [
        "new_precondition_valid"
        if record.precondition_valid
        else "new_precondition_invalid",
        "sim3_quality_good"
        if record.labels["sim3_quality_good"]
        else "sim3_quality_bad",
        "odom_consistent_loose"
        if record.labels["odom_consistent_loose"]
        else "odom_consistent_not_loose",
        "safe_loop_factor_v1"
        if record.safe_loop_factor_v1
        else "safe_loop_factor_negative",
    ]


def _visualization_record_dir(output_dir: Path, record: CandidateFeatureRecord) -> Path:
    support_suffix = "_".join(f"{idx:06d}" for idx in record.selected_support_indices)
    record_name = f"q{record.query_idx:06d}_c{record.candidate_idx:06d}_s{support_suffix}"
    return output_dir / "visual_records" / Path(*_label_directory_parts(record)) / record_name


def _write_visualization_record(
    output_dir: Path,
    record: CandidateFeatureRecord,
    query: KeyframeRecord,
    candidate: KeyframeRecord,
    supports: List[KeyframeRecord],
) -> None:
    record_dir = _visualization_record_dir(output_dir, record)
    record_dir.mkdir(parents=True, exist_ok=True)

    if query.image_path is None or candidate.image_path is None:
        raise ValueError("visualization records require query and candidate images")
    shutil.copy2(query.image_path, record_dir / "query.png")
    shutil.copy2(candidate.image_path, record_dir / "candidate.png")
    support_paths = {}
    for support in supports:
        if support.image_path is None:
            raise ValueError("visualization records require support images")
        support_name = f"support_{support.keyframe_idx:06d}.png"
        shutil.copy2(support.image_path, record_dir / support_name)
        support_paths[str(support.keyframe_idx)] = str(support.image_path)

    payload = {
        "feature_record": dataclass_to_json_dict(record),
        "image_paths": {
            "query": str(query.image_path),
            "candidate": str(candidate.image_path),
            "supports": support_paths,
        },
    }
    (record_dir / "record.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def build_sequence_cache(
    raw_dir: Path,
    config: LoopPolicyDatasetConfig,
    descriptor_extractor: DescriptorExtractor,
    da3_runner: Optional[Da3Runner] = None,
) -> Dict[str, object]:
    sequence = load_aster_raw_sequence(raw_dir)
    keyframes = _resolve_keyframe_image_paths(sequence.keyframes, sequence.raw_dir)
    da3_runner = da3_runner or MockDa3Runner()
    output_dir = Path(config.output_root) / sequence.platform / sequence.sequence_name
    output_dir.mkdir(parents=True, exist_ok=True)

    descriptors = _descriptor_cache(replace(sequence, keyframes=keyframes), descriptor_extractor)
    save_descriptor_cache(output_dir / "descriptors.npz", descriptors)
    camera_poses = _pose_map(sequence)
    keyframes_by_idx = _keyframe_by_idx(keyframes)

    retrieval_records = []
    support_records = []
    feature_records = []
    negative_reasons: Dict[str, int] = {}

    query_keyframes = [
        keyframe for keyframe in keyframes if keyframe.image_path is not None
    ]
    if config.query_limit is not None:
        query_keyframes = query_keyframes[: config.query_limit]

    for query in query_keyframes:
        retrieval = rank_causal_topk(
            sequence=sequence.sequence_name,
            query=query,
            keyframes=keyframes,
            descriptors=descriptors,
            retrieval_pool_size=config.retrieval_pool_size,
            runtime_top_k=config.runtime_top_k,
            exclude_recent_keyframes=config.exclude_recent_keyframes,
        )
        retrieval_records.append(retrieval)

        for candidate in retrieval.candidates:
            candidate_kf = keyframes_by_idx[candidate.keyframe_idx]
            support = select_supports(
                sequence=sequence.sequence_name,
                query=query,
                candidate=candidate_kf,
                keyframes=keyframes,
                camera_poses_by_idx=camera_poses,
                support_window=config.support_window,
                support_count=config.support_count,
                exclude_recent_keyframes=config.exclude_recent_keyframes,
                min_support_baseline_m=config.min_support_baseline_m,
            )
            support_records.append(support)
            if support.rejected:
                _increment_negative_reason(negative_reasons, support.rejection_reason)
                continue

            supports = [keyframes_by_idx[idx] for idx in support.selected_support_indices]
            da3_result = da3_runner.run([build_da3_group(query, candidate_kf, supports)])[0]
            sim3 = align_da3_poses_with_candidate_support_prior(
                da3_query_pose=da3_result.camera_poses[0],
                da3_candidate_pose=da3_result.camera_poses[1],
                da3_support_poses=list(da3_result.camera_poses[2:]),
                odom_query_pose=camera_poses[query.keyframe_idx],
                odom_candidate_pose=camera_poses[candidate_kf.keyframe_idx],
                odom_support_poses=[
                    camera_poses[support_idx] for support_idx in support.selected_support_indices
                ],
                config=Sim3PriorConfig(
                    max_support_align_rmse_m=config.support_align_rmse_thr,
                    max_direction_error_deg=config.direction_error_thr_deg,
                ),
            )
            precondition_valid = sim3.accepted
            if not precondition_valid:
                _increment_negative_reason(negative_reasons, sim3.rejection_reason)
            odom_qc = pose_residual(
                camera_poses[candidate_kf.keyframe_idx],
                camera_poses[query.keyframe_idx],
            )
            odom_cs = pose_residual(
                camera_poses[candidate_kf.keyframe_idx],
                camera_poses[support.selected_support_indices[0]],
            )
            raw_metrics = {
                "abs_log_sim3_scale": sim3.abs_log_sim3_scale,
                "support_align_rmse": sim3.support_align_rmse,
                "direction_error_deg": sim3.direction_error_deg,
                "aligned_vs_odom_rot_residual_deg": sim3.aligned_vs_odom.rotation_deg,
                "aligned_vs_odom_trans_residual_norm": sim3.aligned_vs_odom.translation_norm,
            }
            labels = compute_safe_loop_factor_v1(precondition_valid, raw_metrics, config)
            metrics = _sanitize_finite_values(raw_metrics, config)
            x_geom_values = _sanitize_finite_values(
                {
                    "rank_norm": candidate.rank / max(1, config.retrieval_pool_size),
                    "salad_score_qc": candidate.score,
                    "salad_score_qs": 0.0,
                    "salad_score_cs": 0.0,
                    "salad_score_qc_minus_top1": candidate.score - retrieval.candidates[0].score,
                    "salad_score_qc_minus_topk": candidate.score - retrieval.candidates[-1].score,
                    "da3_rot_qc_deg": 0.0,
                    "da3_trans_qc_norm": float(
                        np.linalg.norm(
                            da3_result.camera_poses[0, :3, 3]
                            - da3_result.camera_poses[1, :3, 3]
                        )
                    ),
                    "da3_rot_cs_deg": 0.0,
                    "da3_trans_cs_norm": float(
                        np.linalg.norm(
                            da3_result.camera_poses[1, :3, 3]
                            - da3_result.camera_poses[2, :3, 3]
                        )
                    ),
                    "da3_rot_qs_deg": 0.0,
                    "da3_trans_qs_norm": float(
                        np.linalg.norm(
                            da3_result.camera_poses[0, :3, 3]
                            - da3_result.camera_poses[2, :3, 3]
                        )
                    ),
                    "odom_rot_qc_deg": odom_qc.rotation_deg,
                    "odom_trans_qc_norm": odom_qc.translation_norm,
                    "odom_rot_cs_deg": odom_cs.rotation_deg,
                    "odom_trans_cs_norm": odom_cs.translation_norm,
                    "support_baseline": support.selected_support_baselines[0],
                    "sim3_scale": sim3.sim3_scale,
                    "abs_log_sim3_scale": sim3.abs_log_sim3_scale,
                    "support_align_rmse": sim3.support_align_rmse,
                    "direction_error_deg": sim3.direction_error_deg,
                    "aligned_vs_odom_rot_residual_deg": sim3.aligned_vs_odom.rotation_deg,
                    "aligned_vs_odom_trans_residual_norm": sim3.aligned_vs_odom.translation_norm,
                    "aligned_loop_rot_deg": sim3.aligned_loop.rotation_deg,
                    "aligned_loop_trans_norm": sim3.aligned_loop.translation_norm,
                    "q_depth_conf_median": da3_result.depth_conf_medians[0],
                    "c_depth_conf_median": da3_result.depth_conf_medians[1],
                    "s_depth_conf_median": da3_result.depth_conf_medians[2],
                    "q_valid_depth_ratio": da3_result.valid_depth_ratios[0],
                    "c_valid_depth_ratio": da3_result.valid_depth_ratios[1],
                    "s_valid_depth_ratio": da3_result.valid_depth_ratios[2],
                    "min_depth_conf_median": min(da3_result.depth_conf_medians[:3]),
                },
                config,
            )
            x_geom = build_x_geom(x_geom_values)
            feature_record = CandidateFeatureRecord(
                sequence=sequence.sequence_name,
                query_idx=query.keyframe_idx,
                query_timestamp=query.timestamp,
                candidate_source="retrieval_topk",
                candidate_idx=candidate_kf.keyframe_idx,
                candidate_timestamp=candidate_kf.timestamp,
                causal=True,
                database_max_idx=retrieval.database_max_idx,
                database_max_timestamp=retrieval.database_max_timestamp,
                retrieval_db_size=retrieval.retrieval_db_size,
                support_snapshot_max_idx=support.support_snapshot_max_idx,
                support_snapshot_max_timestamp=support.support_snapshot_max_timestamp,
                selected_support_indices=support.selected_support_indices,
                selected_support_timestamps=support.selected_support_timestamps,
                support_count=support.support_count,
                precondition_valid=precondition_valid,
                negative_reason=sim3.rejection_reason,
                x_geom=x_geom,
                safe_loop_factor_v1=labels["safe_loop_factor_v1"],
                labels=labels,
                metrics=metrics,
            )
            feature_records.append(feature_record)
            if config.write_visualization_records:
                _write_visualization_record(
                    output_dir=output_dir,
                    record=feature_record,
                    query=query,
                    candidate=candidate_kf,
                    supports=supports,
                )

    sequence_index = {
        "schema_version": "loop_policy_dataset_v1",
        "sequence": sequence.sequence_name,
        "platform": sequence.platform,
        "causal": config.causal,
        "keyframe_count": len(sequence.keyframes),
        "image_keyframe_count": int(len(descriptors.keyframe_idx)),
        "settings": dataclass_to_json_dict(config),
    }
    feature_payloads = [dataclass_to_json_dict(record) for record in feature_records]
    summary = {
        "sequence": sequence.sequence_name,
        "platform": sequence.platform,
        "keyframe_count": len(sequence.keyframes),
        "image_keyframe_count": int(len(descriptors.keyframe_idx)),
        "query_count": len(query_keyframes),
        "retrieval_candidate_count": sum(len(record.candidates) for record in retrieval_records),
        "valid_support_count": sum(1 for record in support_records if not record.rejected),
        "da3_success_count": len(feature_records),
        "safe_loop_factor_positive_count": sum(
            1 for record in feature_records if record.safe_loop_factor_v1
        ),
        "negative_reasons": negative_reasons,
        "causal_leakage_audit_passed": audit_causal_leakage(feature_payloads),
    }

    (output_dir / "sequence_index.json").write_text(
        json.dumps(sequence_index, indent=2, sort_keys=True, allow_nan=False)
    )
    write_jsonl(output_dir / "retrieval_topk.jsonl", retrieval_records)
    write_jsonl(output_dir / "support_selection.jsonl", support_records)
    write_jsonl(output_dir / "candidate_features.jsonl", feature_records)
    (output_dir / "sequence_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build learned loop policy causal loop-policy dataset cache"
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--gt-root")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--retrieval-pool-size", type=int, default=50)
    parser.add_argument("--runtime-top-k", type=int, default=4)
    parser.add_argument("--write-visualization-records", action="store_true")
    parser.add_argument("--exclude-recent-keyframes", type=int, default=30)
    parser.add_argument("--support-window", type=int, default=20)
    parser.add_argument("--support-count", type=int, default=1)
    parser.add_argument("--min-support-baseline-m", type=float, default=0.3)
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--causal", action="store_true", default=True)
    parser.add_argument("--salad-checkpoint", required=True)
    parser.add_argument("--salad-device", default="cuda")
    parser.add_argument("--salad-image-size", type=int, nargs=2, default=[336, 336])
    parser.add_argument("--salad-batch-size", type=int, default=16)
    parser.add_argument("--da3-model", default="depth-anything/DA3-SMALL")
    parser.add_argument("--da3-device", default="cuda")
    parser.add_argument("--da3-process-res", type=int, default=504)
    parser.add_argument(
        "--da3-extrinsics-convention",
        choices=["c2w", "w2c"],
        default="w2c",
    )
    parser.add_argument(
        "--da3-ref-view-strategy",
        choices=["first", "middle", "saddle_balanced", "saddle_sim_range"],
        default="first",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = LoopPolicyDatasetConfig(
        dataset_root=args.dataset_root,
        gt_root=args.gt_root,
        output_root=args.output_root,
        sequences=tuple(args.sequences),
        retrieval_pool_size=args.retrieval_pool_size,
        runtime_top_k=args.runtime_top_k,
        write_visualization_records=args.write_visualization_records,
        exclude_recent_keyframes=args.exclude_recent_keyframes,
        support_window=args.support_window,
        support_count=args.support_count,
        min_support_baseline_m=args.min_support_baseline_m,
        query_limit=args.query_limit,
        causal=args.causal,
    )
    descriptor_extractor = DinoSaladDescriptorExtractor(
        checkpoint=Path(args.salad_checkpoint),
        device=args.salad_device,
        image_size=tuple(args.salad_image_size),
        batch_size=args.salad_batch_size,
    )
    da3_runner = DepthAnything3Runner(
        model_name=args.da3_model,
        device=args.da3_device,
        process_res=args.da3_process_res,
        extrinsics_are_c2w=args.da3_extrinsics_convention == "c2w",
        ref_view_strategy=args.da3_ref_view_strategy,
    )
    summaries: List[Dict[str, object]] = []
    failed = False
    for sequence in config.sequences:
        try:
            raw_dir = raw_dir_for_sequence(Path(config.dataset_root), sequence)
            summaries.append(
                build_sequence_cache(
                    raw_dir=raw_dir,
                    config=config,
                    descriptor_extractor=descriptor_extractor,
                    da3_runner=da3_runner,
                )
            )
        except Exception as exc:
            failed = True
            summaries.append(
                {
                    "sequence": sequence,
                    "platform": _platform_summary_for_failed_sequence(sequence),
                    "failed": True,
                    "error": repr(exc),
                    "causal_leakage_audit_passed": False,
                }
            )
    write_root_manifests(config, summaries)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
