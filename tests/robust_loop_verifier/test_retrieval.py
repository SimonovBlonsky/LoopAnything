import numpy as np
import pytest


def _descriptor_set():
    from robust_loop_verifier.retrieval import DescriptorSet

    return DescriptorSet(
        keyframe_indices=[0, 1, 2],
        descriptors=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=np.float64,
        ),
    )


def test_retrieval_excludes_future_and_recent_candidates():
    from robust_loop_verifier.retrieval import DescriptorSet, retrieve_historical_topk

    descriptors = DescriptorSet(
        keyframe_indices=[0, 1, 2, 3, 4],
        descriptors=np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.8, 0.2],
                [1.0, 0.0],
            ],
            dtype=np.float64,
        ),
    )

    record = retrieve_historical_topk(
        query_idx=4,
        descriptors=descriptors,
        top_k=2,
        recent_exclusion_keyframes=1,
    )

    assert record.query_idx == 4
    assert [candidate.candidate_idx for candidate in record.candidates] == [0, 1]


def test_retrieval_uses_larger_similarity_as_better():
    from robust_loop_verifier.retrieval import DescriptorSet, retrieve_historical_topk

    descriptors = DescriptorSet(
        keyframe_indices=[0, 1, 3],
        descriptors=np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=np.float64,
        ),
    )

    record = retrieve_historical_topk(
        query_idx=3,
        descriptors=descriptors,
        top_k=2,
        recent_exclusion_keyframes=0,
    )

    assert record.candidates[0].candidate_idx == 1
    assert record.candidates[0].score > record.candidates[1].score


def test_retrieval_includes_zero_score_candidates_when_top_k_requests_them():
    from robust_loop_verifier.retrieval import DescriptorSet, retrieve_historical_topk

    descriptors = DescriptorSet(
        keyframe_indices=[0, 1, 2, 3, 4],
        descriptors=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.8, 0.2],
                [1.0, 0.0],
            ],
            dtype=np.float64,
        ),
    )

    record = retrieve_historical_topk(
        query_idx=4,
        descriptors=descriptors,
        top_k=3,
        recent_exclusion_keyframes=1,
    )

    assert [candidate.candidate_idx for candidate in record.candidates] == [0, 1, 2]
    assert record.candidates[1].score == 0.0
    assert record.candidates[2].score == 0.0


def test_retrieval_rejects_negative_top_k():
    from robust_loop_verifier.retrieval import retrieve_historical_topk

    with pytest.raises(ValueError, match="top_k"):
        retrieve_historical_topk(
            query_idx=2,
            descriptors=_descriptor_set(),
            top_k=-1,
            recent_exclusion_keyframes=0,
        )


@pytest.mark.parametrize("top_k", [1.5, True])
def test_retrieval_rejects_non_integer_top_k(top_k):
    from robust_loop_verifier.retrieval import retrieve_historical_topk

    with pytest.raises(ValueError, match="top_k"):
        retrieve_historical_topk(
            query_idx=2,
            descriptors=_descriptor_set(),
            top_k=top_k,
            recent_exclusion_keyframes=0,
        )


def test_retrieval_rejects_negative_recent_exclusion():
    from robust_loop_verifier.retrieval import retrieve_historical_topk

    with pytest.raises(ValueError, match="recent_exclusion_keyframes"):
        retrieve_historical_topk(
            query_idx=2,
            descriptors=_descriptor_set(),
            top_k=1,
            recent_exclusion_keyframes=-1,
        )


@pytest.mark.parametrize("recent_exclusion_keyframes", [1.5, False])
def test_retrieval_rejects_non_integer_recent_exclusion(recent_exclusion_keyframes):
    from robust_loop_verifier.retrieval import retrieve_historical_topk

    with pytest.raises(ValueError, match="recent_exclusion_keyframes"):
        retrieve_historical_topk(
            query_idx=2,
            descriptors=_descriptor_set(),
            top_k=1,
            recent_exclusion_keyframes=recent_exclusion_keyframes,
        )


def test_retrieval_rejects_descriptor_row_count_mismatch():
    from robust_loop_verifier.retrieval import DescriptorSet, retrieve_historical_topk

    descriptors = DescriptorSet(
        keyframe_indices=[0, 1, 2],
        descriptors=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float64,
        ),
    )

    with pytest.raises(ValueError, match="row count"):
        retrieve_historical_topk(
            query_idx=2,
            descriptors=descriptors,
            top_k=1,
            recent_exclusion_keyframes=0,
        )


def test_retrieval_rejects_missing_query():
    from robust_loop_verifier.retrieval import retrieve_historical_topk

    with pytest.raises(ValueError, match="missing"):
        retrieve_historical_topk(
            query_idx=4,
            descriptors=_descriptor_set(),
            top_k=1,
            recent_exclusion_keyframes=0,
        )


def test_retrieval_rejects_zero_norm_descriptor():
    from robust_loop_verifier.retrieval import DescriptorSet, retrieve_historical_topk

    descriptors = DescriptorSet(
        keyframe_indices=[0, 1, 2],
        descriptors=np.array(
            [
                [1.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=np.float64,
        ),
    )

    with pytest.raises(ValueError, match="norm"):
        retrieve_historical_topk(
            query_idx=2,
            descriptors=descriptors,
            top_k=2,
            recent_exclusion_keyframes=0,
        )
