import sys
from types import SimpleNamespace

import numpy as np
import pytest

from robust_loop_verifier.geometry import make_transform
from robust_loop_verifier.pgo import PgoNoise, run_full_prefix_pgo, trajectory_deformation_rmse


def _require_gtsam():
    return pytest.importorskip("gtsam")


def _poses(count):
    return [make_transform(np.eye(3), [float(i), 0.0, 0.0]) for i in range(count)]


def _bent_poses():
    translations = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [2.0, 2.0, 0.0],
        [3.0, 2.0, 0.0],
    ]
    return [make_transform(np.eye(3), translation) for translation in translations]


def _assert_structured_failure(result, original_poses):
    assert not result.converged
    assert result.error_before == np.inf
    assert result.error_after == np.inf
    assert result.failure_reason
    assert len(result.optimized_poses) == len(original_poses)
    for original, returned in zip(original_poses, result.optimized_poses):
        assert returned is not original
        np.testing.assert_allclose(returned, original)


def test_full_prefix_pgo_true_loop_has_small_deformation():
    _require_gtsam()
    poses = _poses(6)
    loop_factor = np.linalg.inv(poses[5]) @ poses[0]
    result = run_full_prefix_pgo(
        prefix_indices=list(range(6)),
        odom_poses=poses,
        loop_from_idx=5,
        loop_to_idx=0,
        loop_factor=loop_factor,
        noise=PgoNoise.default_for_tests(),
    )
    assert result.converged
    assert trajectory_deformation_rmse(poses, result.optimized_poses) < 1e-4


def test_full_prefix_pgo_false_loop_has_larger_deformation():
    _require_gtsam()
    poses = _bent_poses()
    bad_loop = make_transform(np.eye(3), [-20.0, 0.0, 0.0])
    result = run_full_prefix_pgo(
        prefix_indices=list(range(6)),
        odom_poses=poses,
        loop_from_idx=5,
        loop_to_idx=0,
        loop_factor=bad_loop,
        noise=PgoNoise.default_for_tests(),
    )
    assert result.converged
    assert trajectory_deformation_rmse(poses, result.optimized_poses) > 0.1


def test_full_prefix_pgo_rejects_prefix_pose_length_mismatch():
    poses = _poses(2)
    result = run_full_prefix_pgo(
        prefix_indices=[0, 1, 2],
        odom_poses=poses,
        loop_from_idx=2,
        loop_to_idx=0,
        loop_factor=make_transform(np.eye(3), [-2.0, 0.0, 0.0]),
        noise=PgoNoise.default_for_tests(),
    )

    _assert_structured_failure(result, poses)
    assert "length mismatch" in result.failure_reason


def test_full_prefix_pgo_rejects_nan_loop_factor_with_copied_failure_poses():
    poses = _poses(3)
    loop_factor = make_transform(np.eye(3), [-2.0, 0.0, 0.0])
    loop_factor[0, 3] = np.nan

    result = run_full_prefix_pgo(
        prefix_indices=[0, 1, 2],
        odom_poses=poses,
        loop_from_idx=2,
        loop_to_idx=0,
        loop_factor=loop_factor,
        noise=PgoNoise.default_for_tests(),
    )

    _assert_structured_failure(result, poses)
    assert "loop_factor" in result.failure_reason


def test_full_prefix_pgo_rejects_nan_noise_sigma():
    poses = _poses(3)
    noise = PgoNoise(
        prior_sigmas=(1e-6, 1e-6, np.nan, 1e-6, 1e-6, 1e-6),
        odom_sigmas=(0.05, 0.05, 0.05, 0.05, 0.05, 0.05),
        loop_sigmas=(0.05, 0.05, 0.05, 0.05, 0.05, 0.05),
    )

    result = run_full_prefix_pgo(
        prefix_indices=[0, 1, 2],
        odom_poses=poses,
        loop_from_idx=2,
        loop_to_idx=0,
        loop_factor=np.linalg.inv(poses[2]) @ poses[0],
        noise=noise,
    )

    _assert_structured_failure(result, poses)
    assert "prior_sigmas" in result.failure_reason


@pytest.mark.parametrize(
    "sigmas",
    [
        (0.05, 0.05, 0.05, 0.05, 0.05),
        (0.05, 0.05, 0.05, 0.05, 0.05, 0.0),
        (0.05, 0.05, 0.05, 0.05, 0.05, -0.1),
    ],
)
def test_full_prefix_pgo_rejects_bad_noise_sigma_shapes_and_values(sigmas):
    poses = _poses(3)
    noise = PgoNoise(
        prior_sigmas=(1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6),
        odom_sigmas=sigmas,
        loop_sigmas=(0.05, 0.05, 0.05, 0.05, 0.05, 0.05),
    )

    result = run_full_prefix_pgo(
        prefix_indices=[0, 1, 2],
        odom_poses=poses,
        loop_from_idx=2,
        loop_to_idx=0,
        loop_factor=np.linalg.inv(poses[2]) @ poses[0],
        noise=noise,
    )

    _assert_structured_failure(result, poses)
    assert "odom_sigmas" in result.failure_reason


@pytest.mark.parametrize(
    "loop_factor",
    [
        np.array(
            [
                [1.0, 0.0, 0.0, -2.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 2.0],
            ]
        ),
        np.array(
            [
                [2.0, 0.0, 0.0, -2.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    ],
)
def test_full_prefix_pgo_rejects_malformed_loop_factor(loop_factor):
    poses = _poses(3)

    result = run_full_prefix_pgo(
        prefix_indices=[0, 1, 2],
        odom_poses=poses,
        loop_from_idx=2,
        loop_to_idx=0,
        loop_factor=loop_factor,
        noise=PgoNoise.default_for_tests(),
    )

    _assert_structured_failure(result, poses)
    assert "loop_factor" in result.failure_reason


def test_full_prefix_pgo_rejects_malformed_odom_pose():
    poses = _poses(3)
    poses[1] = poses[1].copy()
    poses[1][3, 3] = 2.0

    result = run_full_prefix_pgo(
        prefix_indices=[0, 1, 2],
        odom_poses=poses,
        loop_from_idx=2,
        loop_to_idx=0,
        loop_factor=make_transform(np.eye(3), [-2.0, 0.0, 0.0]),
        noise=PgoNoise.default_for_tests(),
    )

    _assert_structured_failure(result, poses)
    assert "odom_poses[1]" in result.failure_reason


def test_full_prefix_pgo_rejects_same_loop_endpoint():
    poses = _poses(3)

    result = run_full_prefix_pgo(
        prefix_indices=[0, 1, 2],
        odom_poses=poses,
        loop_from_idx=2,
        loop_to_idx=2,
        loop_factor=make_transform(np.eye(3), [0.0, 0.0, 0.0]),
        noise=PgoNoise.default_for_tests(),
    )

    _assert_structured_failure(result, poses)
    assert "loop_from_idx and loop_to_idx must differ" in result.failure_reason


def test_full_prefix_pgo_rejects_nonfinite_error_after_optimization(monkeypatch):
    poses = _poses(2)

    class FakeGraph:
        def __init__(self):
            self.error_calls = 0

        def add(self, _factor):
            pass

        def error(self, _values):
            self.error_calls += 1
            if self.error_calls == 1:
                return 0.0
            return np.nan

    class FakeValues:
        def insert(self, _key, _pose):
            pass

        def atPose3(self, _key):
            return FakePose3()

    class FakePose3:
        def __init__(self, *_args):
            pass

        def matrix(self):
            return np.eye(4)

    fake_gtsam = SimpleNamespace(
        NonlinearFactorGraph=FakeGraph,
        Values=FakeValues,
        symbol=lambda _prefix, index: index,
        noiseModel=SimpleNamespace(
            Diagonal=SimpleNamespace(Sigmas=lambda _sigmas: object())
        ),
        Pose3=FakePose3,
        Rot3=lambda _rotation: object(),
        PriorFactorPose3=lambda *_args: object(),
        BetweenFactorPose3=lambda *_args: object(),
        LevenbergMarquardtParams=lambda: object(),
        LevenbergMarquardtOptimizer=lambda _graph, _initial, _params: SimpleNamespace(
            optimize=lambda: FakeValues()
        ),
    )
    monkeypatch.setitem(sys.modules, "gtsam", fake_gtsam)

    result = run_full_prefix_pgo(
        prefix_indices=[0, 1],
        odom_poses=poses,
        loop_from_idx=1,
        loop_to_idx=0,
        loop_factor=make_transform(np.eye(3), [-1.0, 0.0, 0.0]),
        noise=PgoNoise.default_for_tests(),
    )

    _assert_structured_failure(result, poses)
    assert "non-finite" in result.failure_reason


def test_full_prefix_pgo_rejects_invalid_optimized_pose(monkeypatch):
    poses = _poses(2)

    class FakeGraph:
        def add(self, _factor):
            pass

        def error(self, _values):
            return 0.0

    class FakeValues:
        def insert(self, _key, _pose):
            pass

        def atPose3(self, _key):
            return FakePose3()

    class FakePose3:
        def __init__(self, *_args):
            pass

        def matrix(self):
            pose = np.eye(4)
            pose[3, 3] = 2.0
            return pose

    fake_gtsam = SimpleNamespace(
        NonlinearFactorGraph=FakeGraph,
        Values=FakeValues,
        symbol=lambda _prefix, index: index,
        noiseModel=SimpleNamespace(
            Diagonal=SimpleNamespace(Sigmas=lambda _sigmas: object())
        ),
        Pose3=FakePose3,
        Rot3=lambda _rotation: object(),
        PriorFactorPose3=lambda *_args: object(),
        BetweenFactorPose3=lambda *_args: object(),
        LevenbergMarquardtParams=lambda: object(),
        LevenbergMarquardtOptimizer=lambda _graph, _initial, _params: SimpleNamespace(
            optimize=lambda: FakeValues()
        ),
    )
    monkeypatch.setitem(sys.modules, "gtsam", fake_gtsam)

    result = run_full_prefix_pgo(
        prefix_indices=[0, 1],
        odom_poses=poses,
        loop_from_idx=1,
        loop_to_idx=0,
        loop_factor=make_transform(np.eye(3), [-1.0, 0.0, 0.0]),
        noise=PgoNoise.default_for_tests(),
    )

    _assert_structured_failure(result, poses)
    assert "optimized_poses[0]" in result.failure_reason


def test_trajectory_deformation_rmse_handles_short_and_collinear_trajectories():
    assert trajectory_deformation_rmse(_poses(1), _poses(1)) == 0.0

    original = _poses(6)
    optimized = [make_transform(np.eye(3), [float(i) * 2.0 + 5.0, 0.0, 0.0]) for i in range(6)]

    assert np.isfinite(trajectory_deformation_rmse(original, optimized))


def test_trajectory_deformation_rmse_sim3_aligns_collinear_scale_and_translation():
    original = _poses(4)
    optimized = [
        make_transform(np.eye(3), [10.0 + 2.0 * float(i), 0.0, 0.0]) for i in range(4)
    ]

    assert trajectory_deformation_rmse(original, optimized) < 1e-9


def test_full_prefix_pgo_returns_poses_in_prefix_order():
    _require_gtsam()
    prefix_indices = [10, 20, 30, 40]
    poses = _poses(4)
    loop_factor = np.linalg.inv(poses[3]) @ poses[0]

    result = run_full_prefix_pgo(
        prefix_indices=prefix_indices,
        odom_poses=poses,
        loop_from_idx=40,
        loop_to_idx=10,
        loop_factor=loop_factor,
        noise=PgoNoise.default_for_tests(),
    )

    assert result.converged
    assert len(result.optimized_poses) == len(prefix_indices)
    np.testing.assert_allclose(result.optimized_poses[0], poses[0], atol=1e-6)
    np.testing.assert_allclose(result.optimized_poses[-1], poses[-1], atol=1e-6)
