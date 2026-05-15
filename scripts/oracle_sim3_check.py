from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from loop_policy.da3_runner import DepthAnything3Runner, build_da3_group
from loop_policy.dataset_builder import _pose_map, _resolve_keyframe_image_paths
from loop_policy.geometry import invert_transform, pose_residual
from loop_policy.io import load_aster_raw_sequence
from loop_policy.schema import KeyframeRecord
from loop_policy.sim3_prior import Sim3PriorConfig, align_da3_poses_with_candidate_support_prior
from loop_policy.support import select_supports


def _keyframes_by_idx(keyframes: list[KeyframeRecord]) -> dict[int, KeyframeRecord]:
    return {keyframe.keyframe_idx: keyframe for keyframe in keyframes}


def _translation(pose: np.ndarray) -> list[float]:
    return [float(value) for value in pose[:3, 3]]


def _rotation_det(pose: np.ndarray) -> float:
    return float(np.linalg.det(pose[:3, :3]))


def _sim3_payload(name: str, result) -> dict[str, object]:
    return {
        "name": name,
        "accepted": result.accepted,
        "rejection_reason": result.rejection_reason,
        "sim3_scale": result.sim3_scale,
        "abs_log_sim3_scale": result.abs_log_sim3_scale,
        "support_align_rmse": result.support_align_rmse,
        "direction_error_deg": result.direction_error_deg,
        "aligned_loop": {
            "rotation_deg": result.aligned_loop.rotation_deg,
            "translation_norm": result.aligned_loop.translation_norm,
        },
        "aligned_vs_odom": {
            "rotation_deg": result.aligned_vs_odom.rotation_deg,
            "translation_norm": result.aligned_vs_odom.translation_norm,
        },
        "aligned_query_translation": _translation(result.aligned_query_pose),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oracle DA3/Sim3 check for one loop pair.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(
            "/data/datasets/FusionPortable/fusionportable_loop_dataset/"
            "handheld/handheld_escalator00/raw"
        ),
    )
    parser.add_argument("--query-idx", type=int, default=137)
    parser.add_argument("--candidate-idx", type=int, default=63)
    parser.add_argument("--support-window", type=int, default=20)
    parser.add_argument("--support-count", type=int, default=1)
    parser.add_argument(
        "--support-idx",
        type=int,
        action="append",
        help="Override selected support indices. Can be passed multiple times.",
    )
    parser.add_argument("--exclude-recent-keyframes", type=int, default=30)
    parser.add_argument("--min-support-baseline-m", type=float, default=0.3)
    parser.add_argument("--da3-model", default="depth-anything/DA3-SMALL")
    parser.add_argument("--da3-device", default="cpu")
    parser.add_argument("--da3-process-res", type=int, default=504)
    parser.add_argument(
        "--da3-ref-view-strategy",
        choices=["first", "middle", "saddle_balanced", "saddle_sim_range"],
        default="first",
    )
    parser.add_argument("--support-align-rmse-thr", type=float, default=1.0)
    parser.add_argument("--direction-error-thr-deg", type=float, default=45.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequence = load_aster_raw_sequence(args.raw_dir)
    keyframes = _resolve_keyframe_image_paths(sequence.keyframes, sequence.raw_dir)
    by_idx = _keyframes_by_idx(keyframes)
    camera_poses = _pose_map(sequence)

    query = by_idx[args.query_idx]
    candidate = by_idx[args.candidate_idx]
    support = select_supports(
        sequence=sequence.sequence_name,
        query=query,
        candidate=candidate,
        keyframes=keyframes,
        camera_poses_by_idx=camera_poses,
        support_window=args.support_window,
        support_count=args.support_count,
        exclude_recent_keyframes=args.exclude_recent_keyframes,
        min_support_baseline_m=args.min_support_baseline_m,
    )
    if support.rejected:
        raise RuntimeError(f"support selection rejected: {support.rejection_reason}")

    support_indices = args.support_idx or support.selected_support_indices
    supports = [by_idx[idx] for idx in support_indices]
    group = build_da3_group(query, candidate, supports)
    da3_runner = DepthAnything3Runner(
        model_name=args.da3_model,
        device=args.da3_device,
        process_res=args.da3_process_res,
        ref_view_strategy=args.da3_ref_view_strategy,
    )
    da3_result = da3_runner.run([group])[0]

    odom_query_pose = camera_poses[query.keyframe_idx]
    odom_candidate_pose = camera_poses[candidate.keyframe_idx]
    odom_support_poses = [camera_poses[idx] for idx in support_indices]
    sim3_config = Sim3PriorConfig(
        max_support_align_rmse_m=args.support_align_rmse_thr,
        max_direction_error_deg=args.direction_error_thr_deg,
    )

    da3_c2w_result = align_da3_poses_with_candidate_support_prior(
        da3_query_pose=da3_result.camera_poses[0],
        da3_candidate_pose=da3_result.camera_poses[1],
        da3_support_poses=list(da3_result.camera_poses[2:]),
        odom_query_pose=odom_query_pose,
        odom_candidate_pose=odom_candidate_pose,
        odom_support_poses=odom_support_poses,
        config=sim3_config,
    )

    inverted_da3_poses = np.stack(
        [invert_transform(pose) for pose in da3_result.camera_poses],
        axis=0,
    )
    da3_w2c_result = align_da3_poses_with_candidate_support_prior(
        da3_query_pose=inverted_da3_poses[0],
        da3_candidate_pose=inverted_da3_poses[1],
        da3_support_poses=list(inverted_da3_poses[2:]),
        odom_query_pose=odom_query_pose,
        odom_candidate_pose=odom_candidate_pose,
        odom_support_poses=odom_support_poses,
        config=sim3_config,
    )

    output = {
        "sequence": sequence.sequence_name,
        "da3_config": {
            "process_res": args.da3_process_res,
            "ref_view_strategy": args.da3_ref_view_strategy,
            "extrinsics_convention": "w2c_input_inverted_to_c2w",
        },
        "query": {
            "idx": query.keyframe_idx,
            "timestamp": query.timestamp,
            "image_path": query.image_path,
            "odom_translation": _translation(odom_query_pose),
        },
        "candidate": {
            "idx": candidate.keyframe_idx,
            "timestamp": candidate.timestamp,
            "image_path": candidate.image_path,
            "odom_translation": _translation(odom_candidate_pose),
        },
        "support": {
            "auto_selected_support_indices": support.selected_support_indices,
            "used_support_indices": support_indices,
            "used_support_timestamps": [by_idx[idx].timestamp for idx in support_indices],
            "used_support_baselines": [
                float(np.linalg.norm(camera_poses[idx][:3, 3] - odom_candidate_pose[:3, 3]))
                for idx in support_indices
            ],
            "support_snapshot_max_idx": support.support_snapshot_max_idx,
        },
        "odom_loop_candidate_to_query": {
            "rotation_deg": pose_residual(odom_candidate_pose, odom_query_pose).rotation_deg,
            "translation_norm": pose_residual(odom_candidate_pose, odom_query_pose).translation_norm,
        },
        "da3_group": {
            "keyframe_indices": da3_result.keyframe_indices,
            "raw_pose_translations": [_translation(pose) for pose in da3_result.camera_poses],
            "raw_pose_rotation_dets": [_rotation_det(pose) for pose in da3_result.camera_poses],
            "depth_conf_medians": da3_result.depth_conf_medians,
            "valid_depth_ratios": da3_result.valid_depth_ratios,
        },
        "sim3_checks": [
            _sim3_payload("runner_c2w", da3_c2w_result),
            _sim3_payload("runner_c2w_inverted_debug", da3_w2c_result),
        ],
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
