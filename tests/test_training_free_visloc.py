import numpy as np
import pytest
import torch
from addict import Dict

from ablation.eval_training_free_visloc import (
    align_query_pose_multi_ref,
    align_query_pose_top1_anchor,
    build_da3_salad_retriever,
    evaluate_scene_training_free,
    parse_args,
    resolve_pose_output,
    select_topk_topm,
    validate_runtime_args,
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


def test_validate_runtime_args_requires_salad_checkpoint_for_dino():
    args = parse_args(
        [
            "--dataset",
            "7scenes",
            "--scene",
            "heads",
            "--retriever-backend",
            "dino_salad",
        ]
    )
    with pytest.raises(ValueError, match="salad-checkpoint"):
        validate_runtime_args(args)


def test_build_da3_salad_retriever_calls_pipeline_retrieval_only(monkeypatch):
    class FakePipeline:
        def __init__(self):
            self.seen = None

        def eval(self):
            return self

        def retrieval_only(self, images):
            self.seen = images
            return torch.ones(images.shape[0], 4)

        def load_state_dict(self, _state, strict=False):
            return {"strict": strict}

    fake_pipeline = FakePipeline()

    from ablation import eval_training_free_visloc as module

    monkeypatch.setattr(module, "load_config", lambda _p: {"model": {}})
    monkeypatch.setattr(module, "build_unified_pipeline", lambda _cfg, device: fake_pipeline)

    pipeline, retriever = build_da3_salad_retriever("cfg.yaml", None, device="cpu")
    assert pipeline is fake_pipeline
    out = retriever(torch.zeros(2, 1, 3, 8, 8))
    assert out.shape == (2, 4)
    assert fake_pipeline.seen is not None


def test_evaluate_scene_training_free_returns_audit_payload(monkeypatch):
    from ablation import eval_training_free_visloc as module

    # query + 2 refs
    pose_enc = torch.zeros(1, 3, 9)

    class FakePipeline:
        def pose_only(self, query_image, candidate_images, pose_path="cam_dec"):
            assert pose_path == "both"
            assert query_image.shape[1] == 1
            assert candidate_images.shape[1] == 2
            return Dict(
                pose_enc=pose_enc,
                extrinsics=torch.zeros(1, 3, 3, 4),  # cam_dec branch stores w2c in practice
                intrinsics=torch.eye(3)[None, None].repeat(1, 3, 1, 1),
                ray_extrinsics=torch.eye(4)[None, None].repeat(1, 3, 1, 1),
                ray_intrinsics=torch.eye(3)[None, None].repeat(1, 3, 1, 1),
            )

    monkeypatch.setattr(module, "preprocess_image", lambda _p, target_size: torch.zeros(3, *target_size))
    monkeypatch.setattr(module, "get_rot_err", lambda _a, _b: 0.0)

    # Force cam_dec c2w recovery through pose_enc (not output.extrinsics)
    gt_query = np.eye(4, dtype=np.float64)
    gt_ref1 = np.eye(4, dtype=np.float64)
    gt_ref1[:3, 3] = np.array([1.0, 0.0, 0.0])
    gt_ref2 = np.eye(4, dtype=np.float64)
    gt_ref2[:3, 3] = np.array([0.0, 1.0, 0.0])
    gt_group = np.stack([gt_query, gt_ref1, gt_ref2], axis=0)

    def fake_pose_encoding_to_extri_intri(_pose_enc, _hw):
        c2w = torch.from_numpy(gt_group[:, :3, :]).unsqueeze(0).float()
        ixt = torch.eye(3)[None, None].repeat(1, 3, 1, 1)
        return c2w, ixt

    monkeypatch.setattr(module, "pose_encoding_to_extri_intri", fake_pose_encoding_to_extri_intri)

    db_entries = [
        {"image_path": "db0.png", "pose": gt_ref1.astype(np.float32)},
        {"image_path": "db1.png", "pose": gt_ref2.astype(np.float32)},
    ]
    query_entries = [{"image_path": "q0.png", "pose": gt_query.astype(np.float32)}]

    def retriever(query_input):
        return torch.tensor([[1.0, 0.0]], dtype=torch.float32).repeat(query_input.shape[0], 1)

    db_descriptors = torch.tensor([[1.0, 0.0], [0.5, 0.0]], dtype=torch.float32)

    payload = evaluate_scene_training_free(
        pose_pipeline=FakePipeline(),
        retriever=retriever,
        db_entries=db_entries,
        query_entries=query_entries,
        db_descriptors=db_descriptors,
        device="cpu",
        top_k=2,
        top_m=2,
        pose_path="both",
        anchor_mode="multi_ref_alignment",
        target_size=(8, 8),
        config={"model": {"x": 1}},
        retriever_backend="da3_salad",
    )

    assert "rotation_errors" in payload
    assert "translation_errors" in payload
    assert "topk_indices" in payload
    assert "topm_indices" in payload
    assert "config" in payload
    assert "query_poses_cam_dec" in payload
    assert "query_poses_ray" in payload
