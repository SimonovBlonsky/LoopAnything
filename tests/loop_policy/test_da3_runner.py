from types import SimpleNamespace

import numpy as np
import pytest

from loop_policy.da3_runner import (
    Da3Group,
    Da3Runner,
    DepthAnything3Runner,
    MockDa3Runner,
    build_da3_group,
)
from loop_policy.schema import KeyframeRecord


def test_build_da3_group_is_candidate_local():
    query = KeyframeRecord(10, 110.0, "query.jpg")
    candidate = KeyframeRecord(4, 104.0, "candidate.jpg")
    supports = [KeyframeRecord(2, 102.0, "support.jpg")]

    group = build_da3_group(query, candidate, supports)

    assert group.keyframe_indices == [10, 4, 2]
    assert group.image_paths == ["query.jpg", "candidate.jpg", "support.jpg"]


def test_mock_da3_runner_returns_one_result_per_group():
    groups = [
        Da3Group(keyframe_indices=[10, 4, 2], image_paths=["q.jpg", "c.jpg", "s.jpg"]),
        Da3Group(keyframe_indices=[10, 3, 1], image_paths=["q.jpg", "c2.jpg", "s2.jpg"]),
    ]

    results = MockDa3Runner().run(groups)

    assert len(results) == 2
    assert results[0].camera_poses.shape == (3, 4, 4)
    assert results[0].depth_conf_medians == [1.0, 1.0, 1.0]
    assert results[0].valid_depth_ratios == [1.0, 1.0, 1.0]
    np.testing.assert_allclose(results[0].camera_poses[0], np.eye(4), atol=1e-9)


def test_build_da3_group_requires_image_paths():
    query = KeyframeRecord(10, 110.0, "query.jpg")
    candidate = KeyframeRecord(4, 104.0, None)

    with pytest.raises(
        ValueError, match="DA3 group requires image_path for query, candidate, and supports"
    ):
        build_da3_group(query, candidate, [])


def test_da3_runner_base_run_is_abstract_boundary():
    with pytest.raises(NotImplementedError):
        Da3Runner().run([])


def test_depth_anything3_runner_converts_3x4_extrinsics_and_confidence_stats():
    extrinsics = np.array(
        [
            [[1.0, 0.0, 0.0, 2.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0, 4.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        ],
        dtype=np.float64,
    )
    conf = np.array([[[1.0, 2.0], [np.inf, -1.0]], [[0.0, np.nan], [5.0, 7.0]]])
    model = _FakeDa3Model(SimpleNamespace(extrinsics=extrinsics, conf=conf))
    runner = DepthAnything3Runner(
        "unused",
        device="cpu",
        process_res=256,
        extrinsics_are_c2w=True,
    )
    runner._model = model
    group = Da3Group(keyframe_indices=[10, 4], image_paths=["q.jpg", "c.jpg"])

    result = runner.run([group])[0]

    assert model.calls == [
        {
            "image_paths": ["q.jpg", "c.jpg"],
            "align_to_input_ext_scale": False,
            "process_res": 256,
            "process_res_method": "upper_bound_resize",
            "export_format": "mini_npz",
            "export_dir": None,
            "ref_view_strategy": "first",
        }
    ]
    assert result.keyframe_indices == [10, 4]
    assert result.camera_poses.shape == (2, 4, 4)
    np.testing.assert_allclose(result.camera_poses[0, :3, :4], extrinsics[0])
    np.testing.assert_allclose(result.camera_poses[0, 3], [0.0, 0.0, 0.0, 1.0])
    assert result.depth_conf_medians == [1.5, 6.0]
    np.testing.assert_allclose(result.valid_depth_ratios, [0.5, 0.5])


def test_depth_anything3_runner_defaults_to_da3_w2c_extrinsics():
    extrinsics = np.repeat(np.eye(4, dtype=np.float64)[None], 1, axis=0)
    extrinsics[0, 0, 3] = 3.0
    model = _FakeDa3Model(SimpleNamespace(extrinsics=extrinsics, conf=None))
    runner = DepthAnything3Runner("unused", device="cpu")
    runner._model = model
    group = Da3Group(keyframe_indices=[10], image_paths=["q.jpg"])

    result = runner.run([group])[0]

    np.testing.assert_allclose(result.camera_poses[0, :3, 3], [-3.0, 0.0, 0.0])
    assert result.depth_conf_medians == [1.0]
    assert result.valid_depth_ratios == [1.0]


def test_depth_anything3_runner_forwards_configured_reference_view_strategy():
    extrinsics = np.repeat(np.eye(4, dtype=np.float64)[None], 1, axis=0)
    model = _FakeDa3Model(SimpleNamespace(extrinsics=extrinsics, conf=None))
    runner = DepthAnything3Runner(
        "unused",
        device="cpu",
        ref_view_strategy="middle",
    )
    runner._model = model
    group = Da3Group(keyframe_indices=[10], image_paths=["q.jpg"])

    runner.run([group])

    assert model.calls[0]["ref_view_strategy"] == "middle"


def test_depth_anything3_runner_rejects_pose_count_mismatch():
    extrinsics = np.repeat(np.eye(4, dtype=np.float64)[None], 1, axis=0)
    model = _FakeDa3Model(SimpleNamespace(extrinsics=extrinsics, conf=None))
    runner = DepthAnything3Runner("unused", device="cpu")
    runner._model = model
    group = Da3Group(keyframe_indices=[10, 4], image_paths=["q.jpg", "c.jpg"])

    with pytest.raises(ValueError, match="DA3 prediction extrinsics count mismatch"):
        runner.run([group])


def test_depth_anything3_runner_rejects_nonfinite_extrinsics():
    extrinsics = np.repeat(np.eye(4, dtype=np.float64)[None], 1, axis=0)
    extrinsics[0, 0, 3] = np.nan
    model = _FakeDa3Model(SimpleNamespace(extrinsics=extrinsics, conf=None))
    runner = DepthAnything3Runner("unused", device="cpu")
    runner._model = model
    group = Da3Group(keyframe_indices=[10], image_paths=["q.jpg"])

    with pytest.raises(ValueError, match="DA3 prediction extrinsics contain non-finite values"):
        runner.run([group])


def test_depth_anything3_runner_rejects_confidence_count_mismatch():
    extrinsics = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
    conf = np.ones((1, 2, 2), dtype=np.float64)
    model = _FakeDa3Model(SimpleNamespace(extrinsics=extrinsics, conf=conf))
    runner = DepthAnything3Runner("unused", device="cpu")
    runner._model = model
    group = Da3Group(keyframe_indices=[10, 4], image_paths=["q.jpg", "c.jpg"])

    with pytest.raises(ValueError, match="DA3 prediction confidence count mismatch"):
        runner.run([group])


class _FakeDa3Model:
    def __init__(self, prediction):
        self.prediction = prediction
        self.calls = []

    def inference(self, image_paths, **kwargs):
        self.calls.append({"image_paths": image_paths, **kwargs})
        return self.prediction
