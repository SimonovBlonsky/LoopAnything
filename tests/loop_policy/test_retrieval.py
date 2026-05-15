import sys
import types
from pathlib import Path

import numpy as np
import pytest

from loop_policy.retrieval import (
    DescriptorCache,
    DescriptorExtractor,
    DinoSaladDescriptorExtractor,
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


def test_dino_salad_descriptor_extractor_batches_and_returns_numpy(monkeypatch, tmp_path):
    import torch

    class FakeImage:
        def __init__(self, path):
            self.path = Path(path)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def convert(self, mode):
            assert mode == "RGB"
            return self

    class FakeImageModule:
        @staticmethod
        def open(path):
            return FakeImage(path)

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_sizes = []

        def forward(self, batch):
            self.batch_sizes.append(batch.shape[0])
            return torch.cat([batch, batch + 10.0], dim=1)

    fake_model = FakeModel()
    extractor = DinoSaladDescriptorExtractor(
        checkpoint=tmp_path / "unused.ckpt",
        device="cpu",
        batch_size=2,
    )
    monkeypatch.setitem(sys.modules, "PIL", type("FakePIL", (), {"Image": FakeImageModule}))
    monkeypatch.setattr(extractor, "_load_model", lambda: fake_model)
    monkeypatch.setattr(
        extractor,
        "_transform",
        lambda: lambda image: torch.tensor([float(image.path.stem)], dtype=torch.float32),
    )

    descriptors = extractor.extract([Path("1.jpg"), Path("2.jpg"), Path("3.jpg")])

    assert fake_model.batch_sizes == [2, 1]
    assert descriptors.shape == (3, 2)
    np.testing.assert_allclose(descriptors, [[1.0, 11.0], [2.0, 12.0], [3.0, 13.0]])


def test_dino_salad_load_model_strips_common_checkpoint_prefix(tmp_path):
    import torch

    checkpoint = tmp_path / "prefixed.ckpt"
    torch.save({"state_dict": {"model.weight": torch.tensor([[2.0, 3.0]])}}, checkpoint)
    extractor = DinoSaladDescriptorExtractor(checkpoint=checkpoint, device="cpu")
    extractor._build_model = lambda: torch.nn.Linear(2, 1, bias=False)

    model = extractor._load_model()

    np.testing.assert_allclose(model.weight.detach().numpy(), [[2.0, 3.0]])


def test_dino_salad_load_model_rejects_checkpoint_with_no_matching_keys(tmp_path):
    import torch

    checkpoint = tmp_path / "unmatched.ckpt"
    torch.save({"state_dict": {"unrelated.weight": torch.tensor([[2.0, 3.0]])}}, checkpoint)
    extractor = DinoSaladDescriptorExtractor(checkpoint=checkpoint, device="cpu")
    extractor._build_model = lambda: torch.nn.Linear(2, 1, bias=False)

    with pytest.raises(ValueError, match="did not match any model parameters"):
        extractor._load_model()


def test_dino_salad_load_model_rejects_partial_one_key_checkpoint(tmp_path):
    import torch

    class MultiParameterModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2)
            self.aggregator = torch.nn.Linear(2, 2)
            self.projection = torch.nn.Linear(2, 1)

    checkpoint = tmp_path / "partial.ckpt"
    torch.save({"state_dict": {"backbone.weight": torch.ones(2, 2)}}, checkpoint)
    extractor = DinoSaladDescriptorExtractor(checkpoint=checkpoint, device="cpu")
    extractor._build_model = MultiParameterModel

    with pytest.raises(ValueError, match="matched 1/6"):
        extractor._load_model()


def test_dino_salad_load_model_prefers_best_prefixed_candidate_over_partial_unprefixed(
    tmp_path,
):
    import torch

    checkpoint = tmp_path / "mixed.ckpt"
    torch.save(
        {
            "state_dict": {
                "weight": torch.tensor([[99.0, 99.0]]),
                "model.weight": torch.tensor([[2.0, 3.0]]),
                "model.bias": torch.tensor([4.0]),
            }
        },
        checkpoint,
    )
    extractor = DinoSaladDescriptorExtractor(checkpoint=checkpoint, device="cpu")
    extractor._build_model = lambda: torch.nn.Linear(2, 1)

    model = extractor._load_model()

    np.testing.assert_allclose(model.weight.detach().numpy(), [[2.0, 3.0]])
    np.testing.assert_allclose(model.bias.detach().numpy(), [4.0])


def test_dino_salad_build_model_fallback_does_not_import_vpr_model(monkeypatch, tmp_path):
    import torch

    helper = types.ModuleType("models.helper")

    def get_backbone(name, config):
        assert name == "dinov2_vitb14"
        assert config["return_token"] is True
        return torch.nn.Identity()

    def get_aggregator(name, config):
        assert name == "SALAD"
        assert config["num_clusters"] == 64
        return torch.nn.Identity()

    helper.get_backbone = get_backbone
    helper.get_aggregator = get_aggregator
    models = types.ModuleType("models")
    models.__path__ = []
    monkeypatch.setitem(sys.modules, "models", models)
    monkeypatch.setitem(sys.modules, "models.helper", helper)
    monkeypatch.delitem(sys.modules, "vpr_model", raising=False)

    extractor = DinoSaladDescriptorExtractor(checkpoint=tmp_path / "unused.ckpt", device="cpu")

    model = extractor._build_model()

    assert "vpr_model" not in sys.modules
    assert torch.equal(model(torch.tensor([1.0])), torch.tensor([1.0]))
