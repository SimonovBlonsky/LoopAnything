"""Pluggable retrieval strategies for the unified visual localization pipeline."""

from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseRetrievalStrategy(nn.Module):
    """Abstract base class for retrieval strategies.

    All strategies take a query descriptor [D] and candidate descriptors [N, D]
    and return weights [N].
    """

    @abstractmethod
    def forward(
        self, query_descriptor: torch.Tensor, candidate_descriptors: torch.Tensor
    ) -> torch.Tensor:
        """Compute retrieval weights.

        Args:
            query_descriptor: shape [D]
            candidate_descriptors: shape [N, D]

        Returns:
            weights: shape [N]
        """
        ...


class SoftAttentionRetrieval(BaseRetrievalStrategy):
    """Train: softmax over all cosine similarities. Eval: hard 0/1 top-K mask."""

    def __init__(self, temperature: float = 1.0, top_k: int = 5):
        super().__init__()
        self.temperature = temperature
        self.top_k = top_k

    def forward(
        self, query_descriptor: torch.Tensor, candidate_descriptors: torch.Tensor
    ) -> torch.Tensor:
        sims = F.cosine_similarity(
            query_descriptor.unsqueeze(0), candidate_descriptors, dim=-1
        )  # [N]

        if self.training:
            return F.softmax(sims / self.temperature, dim=-1)
        else:
            k = min(self.top_k, sims.shape[0])
            _, topk_idx = sims.topk(k)
            mask = torch.zeros_like(sims)
            mask[topk_idx] = 1.0
            return mask


class HardTopKRetrieval(BaseRetrievalStrategy):
    """Always returns a hard 0/1 mask with exactly top_k ones."""

    def __init__(self, top_k: int = 5):
        super().__init__()
        self.top_k = top_k

    def forward(
        self, query_descriptor: torch.Tensor, candidate_descriptors: torch.Tensor
    ) -> torch.Tensor:
        sims = F.cosine_similarity(
            query_descriptor.unsqueeze(0), candidate_descriptors, dim=-1
        )
        k = min(self.top_k, sims.shape[0])
        _, topk_idx = sims.topk(k)
        mask = torch.zeros_like(sims)
        mask[topk_idx] = 1.0
        return mask


class SoftTopKRetrieval(BaseRetrievalStrategy):
    """Train: top-K then softmax within K (zeros elsewhere). Eval: hard mask."""

    def __init__(self, top_k: int = 5, temperature: float = 1.0):
        super().__init__()
        self.top_k = top_k
        self.temperature = temperature

    def forward(
        self, query_descriptor: torch.Tensor, candidate_descriptors: torch.Tensor
    ) -> torch.Tensor:
        sims = F.cosine_similarity(
            query_descriptor.unsqueeze(0), candidate_descriptors, dim=-1
        )
        k = min(self.top_k, sims.shape[0])
        topk_vals, topk_idx = sims.topk(k)

        if self.training:
            soft_weights = F.softmax(topk_vals / self.temperature, dim=-1)
            weights = torch.zeros_like(sims)
            weights[topk_idx] = soft_weights
            return weights
        else:
            mask = torch.zeros_like(sims)
            mask[topk_idx] = 1.0
            return mask


class GumbelTopKRetrieval(BaseRetrievalStrategy):
    """Train: gumbel_softmax for differentiable discrete selection. Eval: hard mask."""

    def __init__(self, top_k: int = 5, tau: float = 1.0):
        super().__init__()
        self.top_k = top_k
        self.tau = tau

    def forward(
        self, query_descriptor: torch.Tensor, candidate_descriptors: torch.Tensor
    ) -> torch.Tensor:
        sims = F.cosine_similarity(
            query_descriptor.unsqueeze(0), candidate_descriptors, dim=-1
        )

        if self.training:
            # Use gumbel_softmax with hard=True for straight-through estimator
            logits = sims / self.tau
            # Sample top_k by repeating gumbel softmax and accumulating
            # Use a single gumbel_softmax call - produces a soft one-hot,
            # then we take the top_k from the resulting distribution
            gumbel_weights = F.gumbel_softmax(logits, tau=self.tau, hard=False)
            k = min(self.top_k, sims.shape[0])
            _, topk_idx = gumbel_weights.topk(k)
            # Create mask and use straight-through estimator
            hard_mask = torch.zeros_like(sims)
            hard_mask[topk_idx] = 1.0
            # Straight-through: forward uses hard mask, backward uses soft weights
            weights = hard_mask - gumbel_weights.detach() + gumbel_weights
            return weights
        else:
            k = min(self.top_k, sims.shape[0])
            _, topk_idx = sims.topk(k)
            mask = torch.zeros_like(sims)
            mask[topk_idx] = 1.0
            return mask


def build_retrieval_strategy(config: dict) -> BaseRetrievalStrategy:
    """Factory function to instantiate a retrieval strategy from a config dict.

    Args:
        config: dict with key "strategy" (one of "soft_attention", "hard_topk",
                "soft_topk", "gumbel_topk") and optional keys "top_k",
                "temperature", "tau".

    Returns:
        An instance of the requested retrieval strategy.
    """
    strategy = config["strategy"]
    kwargs = {k: v for k, v in config.items() if k != "strategy"}

    registry = {
        "soft_attention": SoftAttentionRetrieval,
        "hard_topk": HardTopKRetrieval,
        "soft_topk": SoftTopKRetrieval,
        "gumbel_topk": GumbelTopKRetrieval,
    }

    if strategy not in registry:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose from {list(registry.keys())}"
        )

    return registry[strategy](**kwargs)
