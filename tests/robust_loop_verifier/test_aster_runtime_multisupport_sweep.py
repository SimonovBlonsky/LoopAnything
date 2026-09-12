from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _load_script():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "robust_loop_verification_scripts"
        / "sweep_aster_runtime_multisupport_sim3.py"
    )
    spec = importlib.util.spec_from_file_location(
        "sweep_aster_runtime_multisupport_sim3",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _pose_at_x(x: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = x
    return pose


def test_candidate_nearest_supports_excludes_query_candidate_and_recent_frames():
    module = _load_script()
    poses = {idx: _pose_at_x(float(idx)) for idx in range(8)}

    supports = module.select_candidate_nearest_supports(
        candidate_idx=3,
        query_idx=6,
        odom_by_idx=poses,
        max_supports=4,
        exclude_recent_from_query=1,
        min_baseline_m=0.5,
    )

    assert [support.idx for support in supports] == [2, 4, 1, 0]
    assert all(support.idx not in {3, 6, 7} for support in supports)


def test_candidate_temporal_supports_follow_candidate_index_neighborhood():
    module = _load_script()
    poses = {idx: _pose_at_x(float(idx)) for idx in range(8)}

    supports = module.select_candidate_temporal_supports(
        candidate_idx=3,
        query_idx=6,
        odom_by_idx=poses,
        max_supports=4,
        exclude_recent_from_query=1,
        min_baseline_m=0.5,
    )

    assert [support.idx for support in supports] == [2, 4, 1, 0]
    assert all(support.idx not in {3, 6, 7} for support in supports)


def test_median_scale_alignment_resists_single_scale_outlier():
    module = _load_script()
    da3_candidate = _pose_at_x(0.0)
    da3_query = _pose_at_x(0.5)
    odom_candidate = _pose_at_x(10.0)
    da3_supports = [_pose_at_x(1.0), _pose_at_x(2.0), _pose_at_x(4.0)]
    odom_supports = [_pose_at_x(13.0), _pose_at_x(16.0), _pose_at_x(100.0)]

    result = module.align_with_median_support_scale(
        da3_query_c2w=da3_query,
        da3_candidate_c2w=da3_candidate,
        da3_support_c2w=da3_supports,
        odom_candidate_c2w=odom_candidate,
        odom_support_c2w=odom_supports,
    )

    assert result.valid is True
    np.testing.assert_allclose(result.scale, 3.0)
    np.testing.assert_allclose(result.aligned_query_c2w[:3, 3], [11.5, 0.0, 0.0])
    assert result.support_rmse_m > 1.0
    assert max(item["scale"] for item in result.per_support) > 10.0


def test_run_triplets_in_chunks_respects_requested_batch_size():
    module = _load_script()

    class FakeRunner:
        def __init__(self):
            self.batch_sizes = []

        def run_triplets(self, triplets):
            self.batch_sizes.append(len(triplets))
            return list(triplets)

    runner = FakeRunner()

    results = module.run_triplets_in_chunks(runner, list(range(5)), batch_size=2)

    assert results == [0, 1, 2, 3, 4]
    assert runner.batch_sizes == [2, 2, 1]


def test_build_da3_view_group_uses_query_candidate_and_all_selected_supports():
    module = _load_script()
    image_by_idx = {idx: Path(f"{idx:06d}.png") for idx in [52, 2, 1, 3, 0]}
    supports = [
        module.SupportCandidate(idx=1, baseline_m=1.0),
        module.SupportCandidate(idx=3, baseline_m=1.2),
        module.SupportCandidate(idx=0, baseline_m=2.0),
    ]

    group = module.build_da3_view_group(
        query_idx=52,
        candidate_idx=2,
        supports=supports,
        image_by_idx=image_by_idx,
    )

    assert group.keyframe_indices == (52, 2, 1, 3, 0)
    assert group.view_roles == ("query", "candidate", "support_1", "support_2", "support_3")
    assert group.image_paths == tuple(str(image_by_idx[idx]) for idx in [52, 2, 1, 3, 0])


def test_run_view_group_returns_variable_length_c2w(tmp_path):
    module = _load_script()
    from PIL import Image

    image_paths = []
    for idx in range(4):
        path = tmp_path / f"{idx}.png"
        Image.new("RGB", (8, 8), color=(idx, idx, idx)).save(path)
        image_paths.append(str(path))

    class FakeModel:
        def input_processor(self, images, **kwargs):
            import torch

            return torch.zeros(len(images), 3, 8, 8), None, None

        def forward(self, image_batch, **kwargs):
            import torch

            batch, views = image_batch.shape[:2]
            extrinsics = torch.eye(4).repeat(batch, views, 1, 1)
            extrinsics[:, :, 0, 3] = torch.arange(views).float()
            return {"extrinsics": extrinsics}

    class FakeRunner:
        config = module.RealDa3RunnerConfig(device="cpu", extrinsics_are_c2w=True)

        def _load_model(self):
            return FakeModel()

    group = module.Da3ViewGroup(
        image_paths=tuple(image_paths),
        keyframe_indices=(10, 2, 1, 3),
        view_roles=("query", "candidate", "support_1", "support_2"),
    )

    result = module.run_da3_view_group(FakeRunner(), group)

    assert result.predicted_c2w.shape == (4, 4, 4)
    assert result.keyframe_indices == group.keyframe_indices


def test_candidate_temporal_stride_supports_expand_around_candidate_with_gaps():
    module = _load_script()
    poses = {idx: _pose_at_x(float(idx)) for idx in range(12)}

    supports = module.select_candidate_temporal_stride_supports(
        candidate_idx=5,
        query_idx=10,
        odom_by_idx=poses,
        max_supports=6,
        exclude_recent_from_query=1,
        min_baseline_m=0.5,
        stride_pattern=(1, 3, 5),
    )

    assert [support.idx for support in supports] == [4, 6, 2, 8, 0]
    assert all(support.idx not in {5, 9, 10, 11} for support in supports)


def test_select_supports_accepts_temporal_stride_mode():
    module = _load_script()
    poses = {idx: _pose_at_x(float(idx)) for idx in range(12)}

    supports = module.select_supports(
        mode="temporal_stride",
        candidate_idx=5,
        query_idx=10,
        odom_by_idx=poses,
        max_supports=4,
        exclude_recent_from_query=1,
        min_baseline_m=0.5,
    )

    assert [support.idx for support in supports] == [4, 6, 3, 7]


def test_consistency_selection_prefers_low_residual_and_stable_scale_supports():
    module = _load_script()
    supports = [
        module.SupportCandidate(idx=1, baseline_m=1.0),
        module.SupportCandidate(idx=2, baseline_m=1.0),
        module.SupportCandidate(idx=3, baseline_m=1.0),
        module.SupportCandidate(idx=4, baseline_m=1.0),
    ]
    predicted = np.stack(
        [
            _pose_at_x(0.0),
            _pose_at_x(0.0),
            _pose_at_x(1.0),
            _pose_at_x(2.0),
            _pose_at_x(3.0),
            _pose_at_x(1.0),
        ],
        axis=0,
    )
    odom_camera_by_idx = {
        2: _pose_at_x(10.0),
        1: _pose_at_x(13.0),
        2 + 100: _pose_at_x(16.0),
        3: _pose_at_x(19.0),
        4: _pose_at_x(30.0),
    }
    # Use candidate idx 2; support idx 2 would collide with candidate, so remap it after creation.
    supports = [
        module.SupportCandidate(idx=1, baseline_m=1.0),
        module.SupportCandidate(idx=102, baseline_m=1.0),
        module.SupportCandidate(idx=3, baseline_m=1.0),
        module.SupportCandidate(idx=4, baseline_m=1.0),
    ]
    odom_camera_by_idx = {
        2: _pose_at_x(10.0),
        1: _pose_at_x(13.0),
        102: _pose_at_x(16.0),
        3: _pose_at_x(19.0),
        4: _pose_at_x(30.0),
    }

    selected, metrics = module.select_supports_by_da3_consistency(
        supports=supports,
        predicted_c2w=predicted,
        candidate_idx=2,
        odom_camera_by_idx=odom_camera_by_idx,
        max_supports=3,
    )

    assert [support.idx for support in selected] == [1, 102, 3]
    assert metrics["selection_strategy"] == "da3_consistency_topk"
    assert len(metrics["ranked_support_indices"]) == 4
    assert metrics["ranked_support_indices"][-1] == 4


def test_make_support_visualization_writes_montage_and_manifest(tmp_path):
    module = _load_script()
    image_by_idx = {}
    for idx, color in [(52, (255, 0, 0)), (2, (0, 255, 0)), (1, (0, 0, 255))]:
        path = tmp_path / f"{idx}.png"
        Image.new("RGB", (16, 12), color=color).save(path)
        image_by_idx[idx] = path

    path = module.write_support_visualization(
        output_dir=tmp_path / "viz",
        query_idx=52,
        candidate_idx=2,
        support_count=1,
        strategy="temporal",
        image_by_idx=image_by_idx,
        selected_supports=[module.SupportCandidate(idx=1, baseline_m=1.5)],
        pool_supports=[module.SupportCandidate(idx=1, baseline_m=1.5)],
        extra_metrics={"scale_mad": 0.0},
    )

    assert path.exists()
    assert path.with_suffix(".json").exists()
    assert "temporal_q000052_c000002_k01" in path.name
