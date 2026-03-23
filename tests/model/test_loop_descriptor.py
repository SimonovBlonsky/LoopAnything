import torch

from depth_anything_3.model.loop_descriptor import (
    build_loop_descriptor,
    confidence_weighted_pool,
)


def test_build_loop_descriptor_l2_normalizes_concat_descriptor():
    camera = torch.tensor([[3.0, 4.0]])
    patches = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    descriptor = build_loop_descriptor(camera, patches)
    assert descriptor.shape == (1, 4)
    assert torch.allclose(torch.linalg.norm(descriptor, dim=-1), torch.ones(1), atol=1e-6)


def test_confidence_weighted_pool_suppresses_low_conf_tokens():
    patches = torch.tensor([[[10.0, 0.0], [1.0, 0.0]]])
    weights = torch.tensor([[0.0, 1.0]])
    pooled = confidence_weighted_pool(patches, weights)
    assert torch.allclose(pooled, torch.tensor([[1.0, 0.0]]), atol=1e-5)
