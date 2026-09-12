import os
import sys
from collections import OrderedDict
from types import ModuleType

import numpy as np
import pytest
import torch
from addict import Dict

from ablation.eval_training_free_visloc import (
    PROJECT_ROOT,
    REPO_ROOT,
    RELOC3R_ROOT,
    SALAD_ROOT,
    SRC_ROOT,
    _ensure_salad_path,
    align_query_pose_multi_ref,
    align_query_pose_top1_anchor,
    _bootstrap_import_paths,
    default_data_root,
    build_output_path,
    build_da3_salad_retriever,
    evaluate_scene_training_free,
    estimate_query_pose_motion_averaging,
    group_to_query_to_db_relative_pose,
    group_to_all_query_to_db_relative_poses,
    load_dino_salad_retriever,
    _purge_salad_modules,
    parse_args,
    resolve_pose_output,
    select_topk_topm,
    summarize_result_medians,
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


def test_group_to_query_to_db_relative_pose_matches_reloc3r_convention():
    query_pose = _pose(_rotz(15.0), np.array([0.3, -0.2, 1.1]))
    db_pose = _pose(_rotz(-25.0), np.array([1.7, 0.4, -0.6]))
    pred_group = np.stack([query_pose, db_pose], axis=0)

    rel_pose = group_to_query_to_db_relative_pose(pred_group)
    expected = np.linalg.inv(db_pose) @ query_pose

    assert np.allclose(rel_pose, expected, atol=1e-6)


def test_motion_averaging_recovers_query_pose_from_pairwise_relposes():
    query_pose = _pose(_rotz(10.0), np.array([0.4, 0.7, -0.2]))
    ref1_pose = _pose(_rotz(-5.0), np.array([1.0, 0.0, 0.1]))
    ref2_pose = _pose(_rotz(20.0), np.array([-0.4, 1.2, 0.3]))

    relposes_q2d = np.stack(
        [
            np.linalg.inv(ref1_pose) @ query_pose,
            np.linalg.inv(ref2_pose) @ query_pose,
        ],
        axis=0,
    )
    ref_gt = np.stack([ref1_pose, ref2_pose], axis=0)

    pred_query = estimate_query_pose_motion_averaging(relposes_q2d, ref_gt)
    assert np.allclose(pred_query, query_pose, atol=1e-6)


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
    assert args.anchor_mode == "reloc3r_motion_averaging"


def test_parse_args_parses_cpu_fallback_bounds():
    args = parse_args(
        [
            "--dataset",
            "7scenes",
            "--scene",
            "heads",
            "--cpu-fallback-max-queries",
            "4",
            "--cpu-fallback-max-db-entries",
            "64",
        ]
    )
    assert args.cpu_fallback_max_queries == 4
    assert args.cpu_fallback_max_db_entries == 64


def test_default_data_root_matches_workspace_layout():
    root_7 = default_data_root("7scenes")
    root_c = default_data_root("cambridge")
    assert root_7.endswith("/reloc3r/data/7scenes")
    assert root_c.endswith("/reloc3r/data/cambridge")


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


def test_bootstrap_import_paths_covers_runtime_dependencies():
    paths = ["/tmp/placeholder"]
    bootstrapped = _bootstrap_import_paths(paths)

    assert str(SRC_ROOT) in bootstrapped
    assert str(PROJECT_ROOT) in bootstrapped
    assert str(REPO_ROOT) in bootstrapped
    assert str(RELOC3R_ROOT) in bootstrapped
    assert str(SALAD_ROOT) not in bootstrapped


def test_ensure_salad_path_adds_salad_root_without_dropping_repo_paths():
    paths = ["/tmp/placeholder"]
    bootstrapped = _ensure_salad_path(paths)

    assert str(REPO_ROOT) in bootstrapped
    assert str(SALAD_ROOT) in bootstrapped
    assert bootstrapped.index(str(SALAD_ROOT)) < bootstrapped.index(str(REPO_ROOT))


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


def test_load_dino_salad_retriever_uses_local_vprmodel_recipe(monkeypatch, tmp_path):
    captured = {}

    class FakeModel:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def load_state_dict(self, state_dict, strict=True):
            captured["state_dict_keys"] = list(state_dict.keys())
            captured["strict"] = strict

        def to(self, device):
            captured["device"] = device
            return self

        def eval(self):
            captured["eval"] = True
            return self

        def __call__(self, images):
            captured["shape"] = tuple(images.shape)
            return torch.ones(images.shape[0], 8)

    fake_module = ModuleType("vpr_model")
    fake_module.VPRModel = FakeModel
    monkeypatch.setitem(sys.modules, "vpr_model", fake_module)
    monkeypatch.delenv("XFORMERS_DISABLED", raising=False)
    monkeypatch.setattr(
        torch,
        "load",
        lambda _path, map_location=None: OrderedDict(
            [("backbone.model.cls_token", torch.tensor(1.0))]
        ),
    )

    checkpoint = tmp_path / "dino_salad_512_32.ckpt"
    checkpoint.write_text("placeholder")

    retriever = load_dino_salad_retriever(str(checkpoint), device="cpu")
    descriptors = retriever(torch.zeros(2, 1, 3, 8, 8))

    assert descriptors.shape == (2, 8)
    assert captured["init"]["agg_config"]["num_clusters"] == 16
    assert captured["init"]["agg_config"]["cluster_dim"] == 32
    assert captured["strict"] is True
    assert captured["device"] == "cpu"
    # DINOv2 patch alignment: 8x8 input is resized up to 14x14 (patch_size=14).
    assert captured["shape"] == (2, 3, 14, 14)
    assert os.environ["XFORMERS_DISABLED"] == "1"


def test_purge_salad_modules_removes_shadowing_utils_and_models():
    fake_registry = {
        "utils": ModuleType("utils"),
        "utils.misc": ModuleType("utils.misc"),
        "models": ModuleType("models"),
        "models.helper": ModuleType("models.helper"),
        "depth_anything_3": ModuleType("depth_anything_3"),
    }
    fake_registry["utils"].__file__ = str(SALAD_ROOT / "utils" / "__init__.py")
    fake_registry["utils.misc"].__file__ = str(SALAD_ROOT / "utils" / "misc.py")
    fake_registry["models"].__file__ = str(SALAD_ROOT / "models" / "__init__.py")
    fake_registry["models.helper"].__file__ = str(SALAD_ROOT / "models" / "helper.py")
    fake_registry["depth_anything_3"].__file__ = str(PROJECT_ROOT / "src" / "depth_anything_3" / "__init__.py")

    _purge_salad_modules(fake_registry)

    assert "utils" not in fake_registry
    assert "utils.misc" not in fake_registry
    assert "models" not in fake_registry
    assert "models.helper" not in fake_registry
    assert "depth_anything_3" in fake_registry


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

    monkeypatch.setattr(module, "preprocess_image", lambda _p, target_size=None: torch.zeros(3, 8, 8))
    monkeypatch.setattr(module, "preprocess_image_for_pose", lambda _p, _k, **kw: torch.zeros(3, 8, 8))
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
    assert "rotation_errors_cam_dec" in payload
    assert "translation_errors_cam_dec" in payload
    assert "rotation_errors_ray" in payload
    assert "translation_errors_ray" in payload
    assert "effective_anchor_modes_cam_dec" in payload
    assert "effective_anchor_modes_ray" in payload


def test_pairwise_motion_averaging_uses_full_topk_for_pose(monkeypatch):
    from ablation import eval_training_free_visloc as module

    class FakePipeline:
        def __init__(self):
            self.pose_only_calls = 0

        def pose_only(self, query_image, candidate_images, pose_path="cam_dec"):
            self.pose_only_calls += 1
            assert pose_path == "cam_dec"
            assert query_image.shape[1] == 1
            assert candidate_images.shape[1] == 1
            return Dict(
                extrinsics=torch.eye(4)[None, None].repeat(1, 2, 1, 1),
                intrinsics=torch.eye(3)[None, None].repeat(1, 2, 1, 1),
            )

    fake_pipeline = FakePipeline()
    gt_query = np.eye(4, dtype=np.float64)
    gt_ref0 = np.eye(4, dtype=np.float64)
    gt_ref0[:3, 3] = np.array([1.0, 0.0, 0.0])
    gt_ref1 = np.eye(4, dtype=np.float64)
    gt_ref1[:3, 3] = np.array([0.0, 1.0, 0.0])
    gt_ref2 = np.eye(4, dtype=np.float64)
    gt_ref2[:3, 3] = np.array([0.0, 0.0, 1.0])

    db_entries = [
        {"image_path": "db0.png", "pose": gt_ref0.astype(np.float32)},
        {"image_path": "db1.png", "pose": gt_ref1.astype(np.float32)},
        {"image_path": "db2.png", "pose": gt_ref2.astype(np.float32)},
    ]
    query_entries = [{"image_path": "q0.png", "pose": gt_query.astype(np.float32)}]

    monkeypatch.setattr(module, "preprocess_image", lambda _p, target_size=None: torch.zeros(3, 8, 8))
    monkeypatch.setattr(module, "preprocess_image_for_pose", lambda _p, _k, **kw: torch.zeros(3, 8, 8))
    monkeypatch.setattr(module, "get_rot_err", lambda _a, _b: 0.0)
    monkeypatch.setattr(
        module,
        "_extract_group_c2w",
        lambda _resolved, _branch, _hw: np.stack(
            [
                np.stack([gt_query, gt_ref0], axis=0),
            ],
            axis=0,
        ),
    )

    captured = {}

    def fake_motion_averaging(relposes_q2d, ref_gt):
        captured["num_relposes"] = relposes_q2d.shape[0]
        captured["num_refs"] = ref_gt.shape[0]
        return gt_query

    monkeypatch.setattr(module, "estimate_query_pose_motion_averaging", fake_motion_averaging)

    def retriever(query_input):
        return torch.tensor([[1.0, 0.0]], dtype=torch.float32).repeat(query_input.shape[0], 1)

    db_descriptors = torch.tensor(
        [[1.0, 0.0], [0.8, 0.0], [0.7, 0.0]],
        dtype=torch.float32,
    )

    payload = evaluate_scene_training_free(
        pose_pipeline=fake_pipeline,
        retriever=retriever,
        db_entries=db_entries,
        query_entries=query_entries,
        db_descriptors=db_descriptors,
        device="cpu",
        top_k=3,
        top_m=2,
        pose_path="cam_dec",
        anchor_mode="reloc3r_motion_averaging",
        target_size=(8, 8),
        config={"model": {"x": 1}},
        retriever_backend="da3_salad",
    )

    assert fake_pipeline.pose_only_calls == 3
    assert captured["num_relposes"] == 3
    assert captured["num_refs"] == 3
    assert payload["topm_indices"][0].tolist() == [0, 1, 2]


def test_build_output_path_includes_pose_path_and_anchor_mode(tmp_path):
    output_path = build_output_path(
        output_dir=tmp_path,
        retriever_backend="da3_salad",
        dataset="7scenes",
        scene="heads",
        pose_path="both",
        anchor_mode="multi_ref_alignment",
    )

    assert output_path.parent == tmp_path
    assert output_path.name == (
        "training_free_da3_salad_7scenes_heads_both_multi_ref_alignment.npz"
    )


def test_summarize_result_medians_reports_all_available_branches():
    payload = {
        "rotation_errors_cam_dec": np.array([1.0, 3.0], dtype=np.float32),
        "translation_errors_cam_dec": np.array([0.2, 0.4], dtype=np.float32),
        "rotation_errors_ray": np.array([5.0, 7.0], dtype=np.float32),
        "translation_errors_ray": np.array([1.0, 3.0], dtype=np.float32),
        "primary_pose_branch": "cam_dec",
    }

    summaries = summarize_result_medians(payload)
    assert [summary["branch"] for summary in summaries] == ["cam_dec", "ray"]
    assert summaries[0]["median_translation"] == pytest.approx(0.3)
    assert summaries[0]["median_rotation"] == pytest.approx(2.0)
    assert summaries[1]["median_translation"] == pytest.approx(2.0)
    assert summaries[1]["median_rotation"] == pytest.approx(6.0)


def test_group_to_all_query_to_db_relative_poses_matches_reloc3r_convention():
    query_pose = _pose(_rotz(15.0), np.array([0.3, -0.2, 1.1]))
    db1_pose = _pose(_rotz(-25.0), np.array([1.7, 0.4, -0.6]))
    db2_pose = _pose(_rotz(40.0), np.array([-0.3, 0.8, 0.2]))
    pred_group = np.stack([query_pose, db1_pose, db2_pose], axis=0)

    relposes = group_to_all_query_to_db_relative_poses(pred_group)

    assert relposes.shape == (2, 4, 4)
    assert np.allclose(relposes[0], np.linalg.inv(db1_pose) @ query_pose, atol=1e-6)
    assert np.allclose(relposes[1], np.linalg.inv(db2_pose) @ query_pose, atol=1e-6)


def test_multiview_motion_averaging_uses_single_pose_only_forward(monkeypatch):
    from ablation import eval_training_free_visloc as module

    class FakePipeline:
        def __init__(self):
            self.pose_only_calls = 0
            self.last_candidate_count = None

        def pose_only(self, query_image, candidate_images, pose_path="cam_dec"):
            self.pose_only_calls += 1
            assert pose_path == "cam_dec"
            assert query_image.shape[1] == 1
            self.last_candidate_count = int(candidate_images.shape[1])
            return Dict(
                extrinsics=torch.eye(4)[None, None].repeat(1, 1 + self.last_candidate_count, 1, 1),
                intrinsics=torch.eye(3)[None, None].repeat(1, 1 + self.last_candidate_count, 1, 1),
            )

    fake_pipeline = FakePipeline()

    gt_query = np.eye(4, dtype=np.float64)
    gt_ref0 = _pose(_rotz(10.0), np.array([1.0, 0.0, 0.0]))
    gt_ref1 = _pose(_rotz(-5.0), np.array([0.0, 1.0, 0.0]))
    gt_ref2 = _pose(_rotz(20.0), np.array([0.0, 0.0, 1.0]))

    db_entries = [
        {"image_path": "db0.png", "pose": gt_ref0.astype(np.float32)},
        {"image_path": "db1.png", "pose": gt_ref1.astype(np.float32)},
        {"image_path": "db2.png", "pose": gt_ref2.astype(np.float32)},
    ]
    query_entries = [{"image_path": "q0.png", "pose": gt_query.astype(np.float32)}]

    monkeypatch.setattr(module, "preprocess_image", lambda _p, target_size=None: torch.zeros(3, 8, 8))
    monkeypatch.setattr(module, "preprocess_image_for_pose", lambda _p, _k, **kw: torch.zeros(3, 8, 8))
    monkeypatch.setattr(module, "get_rot_err", lambda _a, _b: 0.0)
    monkeypatch.setattr(
        module,
        "_extract_group_c2w",
        lambda _resolved, _branch, _hw: np.stack(
            [np.stack([gt_query, gt_ref0, gt_ref1, gt_ref2], axis=0)], axis=0,
        ),
    )

    captured = {}

    def fake_motion_averaging(relposes_q2d, ref_gt):
        captured["num_relposes"] = relposes_q2d.shape[0]
        captured["num_refs"] = ref_gt.shape[0]
        return gt_query

    monkeypatch.setattr(module, "estimate_query_pose_motion_averaging", fake_motion_averaging)

    def retriever(query_input):
        return torch.tensor([[1.0, 0.0]], dtype=torch.float32).repeat(query_input.shape[0], 1)

    db_descriptors = torch.tensor(
        [[1.0, 0.0], [0.8, 0.0], [0.7, 0.0]],
        dtype=torch.float32,
    )

    payload = evaluate_scene_training_free(
        pose_pipeline=fake_pipeline,
        retriever=retriever,
        db_entries=db_entries,
        query_entries=query_entries,
        db_descriptors=db_descriptors,
        device="cpu",
        top_k=3,
        top_m=2,  # top_m is ignored by this mode; top_k drives the multi-view forward.
        pose_path="cam_dec",
        anchor_mode="multiview_motion_averaging",
        target_size=(8, 8),
        config={"model": {"x": 1}},
        retriever_backend="da3_salad",
    )

    assert fake_pipeline.pose_only_calls == 1  # single multi-view forward, not K
    assert fake_pipeline.last_candidate_count == 3
    assert captured["num_relposes"] == 3
    assert captured["num_refs"] == 3
    assert payload["topm_indices"][0].tolist() == [0, 1, 2]
    assert payload["effective_anchor_modes_cam_dec"][0] == "multiview_motion_averaging"


def test_parse_args_accepts_multiview_motion_averaging():
    args = parse_args(
        [
            "--dataset", "7scenes",
            "--scene", "heads",
            "--anchor-mode", "multiview_motion_averaging",
        ]
    )
    assert args.anchor_mode == "multiview_motion_averaging"
