"""Tests for the retrieval strategy module."""

import pytest
import torch

from depth_anything_3.model.retrieval_strategy import (
    BaseRetrievalStrategy,
    GumbelTopKRetrieval,
    HardTopKRetrieval,
    SoftAttentionRetrieval,
    SoftTopKRetrieval,
    build_retrieval_strategy,
)

D = 128
N = 20
TOP_K = 5


@pytest.fixture
def query():
    return torch.randn(D)


@pytest.fixture
def candidates():
    return torch.randn(N, D)


# ---------- SoftAttentionRetrieval ----------


class TestSoftAttentionRetrieval:
    def test_train_mode_shape_and_properties(self, query, candidates):
        strategy = SoftAttentionRetrieval(temperature=1.0, top_k=TOP_K)
        strategy.train()
        weights = strategy(query, candidates)

        assert weights.shape == (N,)
        assert torch.all(weights > 0), "All weights should be positive in train mode"
        assert torch.allclose(
            weights.sum(), torch.tensor(1.0), atol=1e-5
        ), "Weights should sum to 1"

    def test_eval_mode_hard_mask(self, query, candidates):
        strategy = SoftAttentionRetrieval(temperature=1.0, top_k=TOP_K)
        strategy.eval()
        weights = strategy(query, candidates)

        assert weights.shape == (N,)
        assert torch.all(
            (weights == 0.0) | (weights == 1.0)
        ), "Eval weights should be 0 or 1"
        assert weights.sum().item() == TOP_K, f"Should have exactly {TOP_K} ones"

    def test_gradient_flow_train_mode(self, candidates):
        query = torch.randn(D, requires_grad=True)
        strategy = SoftAttentionRetrieval(temperature=1.0, top_k=TOP_K)
        strategy.train()

        weights = strategy(query, candidates)
        # Use a weighted sum so gradients are non-trivial (plain sum of softmax is always 1)
        target = torch.arange(N, dtype=torch.float32)
        loss = (weights * target).sum()
        loss.backward()

        assert query.grad is not None, "Gradient should flow through in train mode"
        assert not torch.all(query.grad == 0), "Gradient should be non-zero"


# ---------- HardTopKRetrieval ----------


class TestHardTopKRetrieval:
    def test_always_hard_mask_train(self, query, candidates):
        strategy = HardTopKRetrieval(top_k=TOP_K)
        strategy.train()
        weights = strategy(query, candidates)

        assert weights.shape == (N,)
        assert torch.all((weights == 0.0) | (weights == 1.0))
        assert weights.sum().item() == TOP_K

    def test_always_hard_mask_eval(self, query, candidates):
        strategy = HardTopKRetrieval(top_k=TOP_K)
        strategy.eval()
        weights = strategy(query, candidates)

        assert weights.shape == (N,)
        assert torch.all((weights == 0.0) | (weights == 1.0))
        assert weights.sum().item() == TOP_K


# ---------- SoftTopKRetrieval ----------


class TestSoftTopKRetrieval:
    def test_train_mode_topk_nonzero_sum_to_one(self, query, candidates):
        strategy = SoftTopKRetrieval(top_k=TOP_K, temperature=1.0)
        strategy.train()
        weights = strategy(query, candidates)

        assert weights.shape == (N,)
        nonzero_count = (weights > 0).sum().item()
        assert nonzero_count == TOP_K, f"Exactly {TOP_K} should be non-zero"
        assert torch.allclose(
            weights.sum(), torch.tensor(1.0), atol=1e-5
        ), "Non-zero weights should sum to 1"

    def test_eval_mode_hard_mask(self, query, candidates):
        strategy = SoftTopKRetrieval(top_k=TOP_K, temperature=1.0)
        strategy.eval()
        weights = strategy(query, candidates)

        assert torch.all((weights == 0.0) | (weights == 1.0))
        assert weights.sum().item() == TOP_K


# ---------- GumbelTopKRetrieval ----------


class TestGumbelTopKRetrieval:
    def test_gradient_flow_train_mode(self, candidates):
        query = torch.randn(D, requires_grad=True)
        strategy = GumbelTopKRetrieval(top_k=TOP_K, tau=1.0)
        strategy.train()

        weights = strategy(query, candidates)
        loss = weights.sum()
        loss.backward()

        assert query.grad is not None, "Gradient should flow through in train mode"

    def test_eval_mode_hard_mask(self, query, candidates):
        strategy = GumbelTopKRetrieval(top_k=TOP_K, tau=1.0)
        strategy.eval()
        weights = strategy(query, candidates)

        assert weights.shape == (N,)
        assert torch.all((weights == 0.0) | (weights == 1.0))
        assert weights.sum().item() == TOP_K


# ---------- build_retrieval_strategy ----------


class TestBuildRetrievalStrategy:
    @pytest.mark.parametrize(
        "config,expected_cls",
        [
            (
                {"strategy": "soft_attention", "temperature": 0.5, "top_k": 3},
                SoftAttentionRetrieval,
            ),
            ({"strategy": "hard_topk", "top_k": 3}, HardTopKRetrieval),
            (
                {"strategy": "soft_topk", "top_k": 3, "temperature": 0.5},
                SoftTopKRetrieval,
            ),
            ({"strategy": "gumbel_topk", "top_k": 3, "tau": 0.5}, GumbelTopKRetrieval),
        ],
    )
    def test_instantiates_correct_class(self, config, expected_cls):
        strategy = build_retrieval_strategy(config)
        assert isinstance(strategy, expected_cls)
        assert isinstance(strategy, BaseRetrievalStrategy)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            build_retrieval_strategy({"strategy": "nonexistent"})
