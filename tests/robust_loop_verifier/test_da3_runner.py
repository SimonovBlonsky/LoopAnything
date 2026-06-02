import numpy as np
from PIL import Image


def test_build_da3_triplet_preserves_query_candidate_support_order():
    from robust_loop_verifier.da3_runner import build_da3_triplet

    triplet = build_da3_triplet(
        "query.png",
        "candidate.png",
        "support.png",
        query_idx=7,
        candidate_idx=3,
        support_idx=4,
    )

    assert triplet.image_paths == ("query.png", "candidate.png", "support.png")
    assert triplet.keyframe_indices == (7, 3, 4)
    assert triplet.view_roles == ("query", "candidate", "support")


def test_mock_da3_runner_returns_c2w_triplet_result():
    from robust_loop_verifier.da3_runner import MockDa3Runner, build_da3_triplet

    triplet = build_da3_triplet(
        "query.png",
        "candidate.png",
        "support.png",
        query_idx=7,
        candidate_idx=3,
        support_idx=4,
    )

    result = MockDa3Runner().run_triplet(triplet)

    assert result.predicted_c2w.shape == (3, 4, 4)
    np.testing.assert_allclose(result.predicted_c2w[0], np.eye(4))
    assert result.keyframe_indices == triplet.keyframe_indices
    assert result.view_roles == triplet.view_roles


def test_mock_da3_runner_returns_nonzero_candidate_support_baseline():
    from robust_loop_verifier.da3_runner import MockDa3Runner, build_da3_triplet

    triplet = build_da3_triplet(
        "query.png",
        "candidate.png",
        "support.png",
        query_idx=7,
        candidate_idx=3,
        support_idx=4,
    )

    result = MockDa3Runner().run_triplet(triplet)

    np.testing.assert_allclose(result.predicted_c2w[0], np.eye(4))
    candidate_support = result.predicted_c2w[2, :3, 3] - result.predicted_c2w[1, :3, 3]
    assert np.linalg.norm(candidate_support) > 0.0


def test_mock_da3_runner_batches_independent_triplets():
    from robust_loop_verifier.da3_runner import MockDa3Runner, build_da3_triplet

    triplets = [
        build_da3_triplet(
            f"query-{idx}.png",
            f"candidate-{idx}.png",
            f"support-{idx}.png",
            query_idx=10,
            candidate_idx=idx,
            support_idx=idx + 1,
        )
        for idx in range(4)
    ]

    results = MockDa3Runner().run_triplets(triplets)

    assert len(results) == len(triplets)
    for triplet, result in zip(triplets, results):
        assert result.predicted_c2w.shape == (3, 4, 4)
        assert result.keyframe_indices == triplet.keyframe_indices
        assert result.view_roles == ("query", "candidate", "support")


def test_real_da3_runner_batches_triplets_on_batch_dimension(tmp_path):
    import torch

    from robust_loop_verifier.da3_runner import (
        RealDa3Runner,
        RealDa3RunnerConfig,
        build_da3_triplet,
    )

    image_paths = []
    for idx in range(6):
        image_path = tmp_path / f"{idx}.png"
        Image.new("RGB", (32, 24), color=(idx, idx, idx)).save(image_path)
        image_paths.append(image_path)

    class FakeDa3Model:
        def __init__(self):
            self.forward_shapes = []

        def input_processor(self, images, **kwargs):
            assert len(images) == 3
            return torch.zeros((3, 3, 28, 28), dtype=torch.float32), None, None

        def forward(self, image, **kwargs):
            self.forward_shapes.append(tuple(image.shape))
            batch, views = image.shape[:2]
            extrinsics = torch.eye(4).repeat(batch, views, 1, 1)
            return {"extrinsics": extrinsics}

    fake_model = FakeDa3Model()
    runner = RealDa3Runner(
        RealDa3RunnerConfig(device="cpu", extrinsics_are_c2w=True),
        model=fake_model,
    )
    triplets = [
        build_da3_triplet(
            image_paths[0],
            image_paths[1],
            image_paths[2],
            query_idx=10,
            candidate_idx=1,
            support_idx=2,
        ),
        build_da3_triplet(
            image_paths[3],
            image_paths[4],
            image_paths[5],
            query_idx=10,
            candidate_idx=4,
            support_idx=5,
        ),
    ]

    results = runner.run_triplets(triplets)

    assert fake_model.forward_shapes == [(2, 3, 3, 28, 28)]
    assert len(results) == 2
    assert all(result.predicted_c2w.shape == (3, 4, 4) for result in results)


def test_convert_da3_extrinsics_projects_rotation_to_so3():
    from robust_loop_verifier.da3_runner import convert_da3_extrinsics_to_c2w

    extrinsics = np.eye(4, dtype=np.float64)[None]
    extrinsics[0, :3, :3] = np.array(
        [
            [1.0, 0.02, 0.0],
            [0.0, 0.98, 0.01],
            [0.0, 0.0, 1.03],
        ],
        dtype=np.float64,
    )

    c2w = convert_da3_extrinsics_to_c2w(extrinsics, extrinsics_are_c2w=True)

    rotation = c2w[0, :3, :3]
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-8)
