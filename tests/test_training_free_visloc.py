import numpy as np
import pytest
import torch
from addict import Dict

from ablation.eval_training_free_visloc import (
    align_query_pose_multi_ref,
    align_query_pose_top1_anchor,
    parse_args,
    resolve_pose_output,
    select_topk_topm,
    validate_retrieval_backend,
)


def _pose(rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rot
    pose[:3, 3] = trans
    return pose


def _rotz(theta_deg: float) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def test_select_topk_and_topm_returns_ranked_indices():
    sims = torch.tensor([0.9, 0.1, 0.8, 0.3], dtype=torch.float32)
    topk, topm = select_topk_topm(sims, top_k=3, top_m=2)
    assert topk.tolist() == [0, 2, 3]
    assert topm.tolist() == [0, 2]


def test_select_topk_topm_rejects_invalid_limits():
    sims = torch.tensor([0.5, 0.4, 0.3], dtype=torch.float32)
    with pytest.raises(ValueError, match="top_m must be <= top_k"):
        select_topk_topm(sims, top_k=2, top_m=3)


def test_resolve_pose_output_routes_cam_dec_ray_and_both():
    cam_out = Dict(extrinsics=torch.ones(1, 2, 4, 4), intrinsics=torch.ones(1, 2, 3, 3))
    resolved_cam = resolve_pose_output(cam_out, pose_path="cam_dec")
    assert set(resolved_cam.keys()) == {"cam_dec"}
    assert resolved_cam["cam_dec"]["extrinsics"] is cam_out.extrinsics

    ray_out = Dict(extrinsics=torch.zeros(1, 2, 4, 4), intrinsics=torch.zeros(1, 2, 3, 3))
    resolved_ray = resolve_pose_output(ray_out, pose_path="ray")
    assert set(resolved_ray.keys()) == {"ray"}
    assert resolved_ray["ray"]["intrinsics"] is ray_out.intrinsics

    both_out = Dict(
        extrinsics=torch.ones(1, 2, 4, 4),
        intrinsics=torch.ones(1, 2, 3, 3),
        ray_extrinsics=torch.zeros(1, 2, 4, 4),
        ray_intrinsics=torch.zeros(1, 2, 3, 3),
    )
    resolved_both = resolve_pose_output(both_out, pose_path="both")
    assert set(resolved_both.keys()) == {"cam_dec", "ray"}
    assert resolved_both["ray"]["extrinsics"] is both_out.ray_extrinsics


def test_multi_ref_alignment_recovers_query_pose():
    gt_query = _pose(_rotz(30.0), np.array([2.0, -1.0, 0.5]))
    gt_ref1 = _pose(_rotz(10.0), np.array([0.0, 0.0, 0.0]))
    gt_ref2 = _pose(_rotz(-20.0), np.array([1.5, 0.3, 0.2]))
    gt_ref3 = _pose(_rotz(45.0), np.array([-0.5, 1.2, -0.1]))
    gt_group = np.stack([gt_query, gt_ref1, gt_ref2, gt_ref3], axis=0)

    sim_r = _rotz(25.0)
    sim_s = 1.7
    sim_t = np.array([1.2, -0.7, 0.4], dtype=np.float64)

    pred_group = []
    for pose in gt_group:
        r_gt, t_gt = pose[:3, :3], pose[:3, 3]
        r_pred = sim_r.T @ r_gt
        t_pred = (sim_r.T @ (t_gt - sim_t)) / sim_s
        pred_group.append(_pose(r_pred, t_pred))
    pred_group = np.stack(pred_group, axis=0)

    aligned_query = align_query_pose_multi_ref(pred_group, gt_group[1:])
    assert np.allclose(aligned_query[:3, :3], gt_query[:3, :3], atol=1e-6)
    assert np.allclose(aligned_query[:3, 3], gt_query[:3, 3], atol=1e-6)


def test_top1_anchor_alignment_recovers_query_pose():
    gt_query = _pose(_rotz(-15.0), np.array([0.5, 2.0, -0.3]))
    gt_ref1 = _pose(_rotz(20.0), np.array([1.2, -0.2, 0.4]))
    gt_ref2 = _pose(_rotz(5.0), np.array([-0.7, 0.3, 1.1]))
    gt_group = np.stack([gt_query, gt_ref1, gt_ref2], axis=0)

    anchor = _pose(_rotz(35.0), np.array([-1.0, 0.4, 0.2]))
    anchor_inv = np.linalg.inv(anchor)
    pred_group = np.stack([anchor_inv @ pose for pose in gt_group], axis=0)

    aligned_query = align_query_pose_top1_anchor(pred_group, gt_group[1:])
    assert np.allclose(aligned_query, gt_query, atol=1e-6)


def test_validate_retrieval_backend_accepts_supported_backends():
    assert validate_retrieval_backend("dino_salad") == "dino_salad"
    assert validate_retrieval_backend("da3_salad") == "da3_salad"
    with pytest.raises(ValueError, match="Unsupported retrieval backend"):
        validate_retrieval_backend("unknown")


def test_parse_args_parses_protocol_flags():
    args = parse_args(["--dataset", "7scenes", "--scene", "heads", "--backend", "dino_salad"])
    assert args.dataset == "7scenes"
    assert args.scene == "heads"
    assert args.backend == "dino_salad"
    assert args.pose_path == "cam_dec"
