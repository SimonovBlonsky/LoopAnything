from pathlib import Path

import numpy as np
import pytest

from loop_policy.retrieval import (
    DescriptorCache,
    DescriptorExtractor,
    PrecomputedDescriptorExtractor,
    causal_retrieval_database,
    load_descriptor_cache,
    normalize_descriptors,
    rank_causal_topk,
    save_descriptor_cache,
)
from loop_policy.schema import KeyframeRecord


def _keyframes():
    return [
        KeyframeRecord(keyframe_idx=i, timestamp=100.0 + i, image_path=f"{i}.jpg")
        for i in range(6)
    ]


def test_normalize_descriptors_l2_normalizes_rows():
    descriptors = np.array([[3.0, 4.0], [0.0, 0.0]])

    normalized = normalize_descriptors(descriptors)

    assert normalized.dtype == np.float64
    np.testing.assert_allclose(normalized, [[0.6, 0.8], [0.0, 0.0]])


def test_descriptor_extractor_requires_subclass_implementation():
    with pytest.raises(NotImplementedError):
        DescriptorExtractor().extract([Path("frames/000001.jpg")])


def test_precomputed_descriptor_extractor_stacks_descriptors_by_path_string():
    extractor = PrecomputedDescriptorExtractor(
        {
            "frames/000001.jpg": np.array([1.0, 2.0]),
            "frames/000002.jpg": np.array([3.0, 4.0]),
        }
    )

    descriptors = extractor.extract([Path("frames/000001.jpg"), Path("frames/000002.jpg")])

    np.testing.assert_allclose(descriptors, [[1.0, 2.0], [3.0, 4.0]])


def test_causal_retrieval_database_excludes_future_recent_and_missing_images():
    keyframes = _keyframes()
    keyframes[1] = KeyframeRecord(keyframe_idx=1, timestamp=101.0, image_path=None)

    database = causal_retrieval_database(
        keyframes=keyframes,
        query=keyframes[5],
        exclude_recent_keyframes=2,
    )

    assert [record.keyframe_idx for record in database] == [0, 2]


def test_rank_causal_topk_returns_scores_sorted_by_score_then_index():
    keyframes = _keyframes()
    descriptors = DescriptorCache(
        keyframe_idx=np.arange(6),
        timestamps=np.array([record.timestamp for record in keyframes]),
        descriptors=normalize_descriptors(
            np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.8, 0.2],
                    [0.8, 0.2],
                    [0.3, 0.7],
                    [1.0, 0.0],
                ]
            )
        ),
        normalized=True,
    )

    record = rank_causal_topk(
        sequence="handheld_room01",
        query=keyframes[5],
        keyframes=keyframes,
        descriptors=descriptors,
        retrieval_pool_size=3,
        runtime_top_k=2,
        exclude_recent_keyframes=1,
    )

    assert record.sequence == "handheld_room01"
    assert record.query_idx == 5
    assert record.query_timestamp == 105.0
    assert record.causal is True
    assert record.database_max_idx == 3
    assert record.database_max_timestamp == 103.0
    assert record.retrieval_db_size == 4
    assert [candidate.keyframe_idx for candidate in record.candidates] == [0, 2, 3]
    assert [candidate.rank for candidate in record.candidates] == [1, 2, 3]
    assert [candidate.runtime_topk for candidate in record.candidates] == [True, True, False]


def test_rank_causal_topk_skips_candidates_missing_descriptors_but_keeps_database_audit():
    keyframes = _keyframes()
    descriptors = DescriptorCache(
        keyframe_idx=np.array([0, 2, 5]),
        timestamps=np.array([100.0, 102.0, 105.0]),
        descriptors=normalize_descriptors(np.array([[1.0, 0.0], [0.5, 0.5], [1.0, 0.0]])),
        normalized=True,
    )

    record = rank_causal_topk(
        sequence="handheld_room01",
        query=keyframes[5],
        keyframes=keyframes,
        descriptors=descriptors,
        retrieval_pool_size=5,
        runtime_top_k=5,
        exclude_recent_keyframes=1,
    )

    assert record.database_max_idx == 3
    assert record.database_max_timestamp == 103.0
    assert record.retrieval_db_size == 4
    assert [candidate.keyframe_idx for candidate in record.candidates] == [0, 2]


def test_rank_causal_topk_rejects_unnormalized_descriptor_cache():
    keyframes = _keyframes()
    descriptors = DescriptorCache(
        keyframe_idx=np.arange(6),
        timestamps=np.array([record.timestamp for record in keyframes]),
        descriptors=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.8, 0.2],
                [0.8, 0.2],
                [0.3, 0.7],
                [1.0, 0.0],
            ]
        ),
        normalized=False,
    )

    with pytest.raises(ValueError, match="descriptor cache must be normalized"):
        rank_causal_topk(
            sequence="handheld_room01",
            query=keyframes[5],
            keyframes=keyframes,
            descriptors=descriptors,
            retrieval_pool_size=3,
            runtime_top_k=2,
            exclude_recent_keyframes=1,
        )


def test_descriptor_cache_rejects_duplicate_keyframe_indices():
    with pytest.raises(ValueError, match="keyframe_idx must be unique"):
        DescriptorCache(
            keyframe_idx=np.array([1, 1]),
            timestamps=np.array([101.0, 102.0]),
            descriptors=np.array([[1.0, 0.0], [0.0, 1.0]]),
            normalized=True,
        )


def test_descriptor_cache_rejects_row_count_mismatch():
    with pytest.raises(ValueError, match="descriptor cache row count mismatch"):
        DescriptorCache(
            keyframe_idx=np.array([1, 2]),
            timestamps=np.array([101.0]),
            descriptors=np.array([[1.0, 0.0], [0.0, 1.0]]),
            normalized=True,
        )


def test_descriptor_cache_rejects_malformed_dimensions():
    with pytest.raises(ValueError, match="descriptors must be a 2D array"):
        DescriptorCache(
            keyframe_idx=np.array([1, 2]),
            timestamps=np.array([101.0, 102.0]),
            descriptors=np.array([1.0, 0.0]),
            normalized=True,
        )


def test_descriptor_cache_roundtrip_preserves_arrays_and_normalized_flag(tmp_path):
    cache = DescriptorCache(
        keyframe_idx=np.array([1, 3]),
        timestamps=np.array([101.5, 103.5]),
        descriptors=np.array([[0.5, 0.5], [0.0, 1.0]]),
        normalized=True,
    )
    path = tmp_path / "descriptors" / "cache.npz"

    save_descriptor_cache(path, cache)
    loaded = load_descriptor_cache(path)

    np.testing.assert_array_equal(loaded.keyframe_idx, cache.keyframe_idx)
    np.testing.assert_allclose(loaded.timestamps, cache.timestamps)
    np.testing.assert_allclose(loaded.descriptors, cache.descriptors)
    assert loaded.normalized is True
    assert loaded.index_map() == {1: 0, 3: 1}


def test_descriptor_cache_roundtrip_preserves_false_normalized_flag_as_bool_array(tmp_path):
    cache = DescriptorCache(
        keyframe_idx=np.array([1]),
        timestamps=np.array([101.5]),
        descriptors=np.array([[0.5, 0.5]]),
        normalized=False,
    )
    path = tmp_path / "descriptors" / "cache.npz"

    save_descriptor_cache(path, cache)

    with np.load(path) as data:
        assert data["normalized"].dtype == np.bool_
        assert data["normalized"].shape == (1,)

    loaded = load_descriptor_cache(path)

    assert loaded.normalized is False


def test_rank_causal_topk_rejects_missing_query_descriptor():
    keyframes = _keyframes()
    descriptors = DescriptorCache(
        keyframe_idx=np.array([0, 1, 2]),
        timestamps=np.array([100.0, 101.0, 102.0]),
        descriptors=np.eye(3),
        normalized=True,
    )

    with pytest.raises(ValueError, match="missing query descriptor"):
        rank_causal_topk(
            sequence="handheld_room01",
            query=keyframes[5],
            keyframes=keyframes,
            descriptors=descriptors,
            retrieval_pool_size=3,
            runtime_top_k=2,
            exclude_recent_keyframes=1,
        )
