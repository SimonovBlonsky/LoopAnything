import numpy as np


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
