import torch

from depth_anything_3.model.loop_descriptor import (
    build_loop_descriptor,
    confidence_map_to_token_weights,
    confidence_weighted_pool,
)


def test_build_loop_descriptor_matches_expected_concat_content():
    camera = torch.tensor([[3.0, 4.0]])
    patches = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

    descriptor = build_loop_descriptor(camera, patches)

    expected = torch.tensor([[0.42426407, 0.56568545, 0.5, 0.5]])
    assert descriptor.shape == (1, 4)
    assert torch.allclose(descriptor, expected, atol=1e-6)


def test_build_loop_descriptor_supports_view_dimension_and_preserves_order():
    camera = torch.tensor([[[3.0, 4.0], [0.0, 5.0]]])
    patches = torch.tensor([
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 0.0], [0.0, 2.0]],
        ]
    ])

    descriptor = build_loop_descriptor(camera, patches)

    expected = torch.tensor([
        [
            [0.42426407, 0.56568545, 0.5, 0.5],
            [0.0, 0.70710677, 0.5, 0.5],
        ]
    ])
    assert descriptor.shape == (1, 2, 4)
    assert torch.allclose(descriptor, expected, atol=1e-6)


def test_build_loop_descriptor_uses_signed_gem_for_negative_tokens():
    camera = torch.tensor([[3.0, 4.0]])
    patches = torch.tensor([[[-1.0, 0.0], [0.0, -1.0]]])
    weights = torch.tensor([[1.0, 0.0]])

    descriptor = build_loop_descriptor(camera, patches, weights=weights, pooling="gem")

    expected = torch.tensor([[0.42426407, 0.56568545, -0.70710677, 0.0]])
    assert torch.allclose(descriptor, expected, atol=1e-6)


def test_confidence_weighted_pool_suppresses_low_conf_tokens():
    patches = torch.tensor([[[10.0, 0.0], [1.0, 0.0]]])
    weights = torch.tensor([[0.0, 1.0]])
    pooled = confidence_weighted_pool(patches, weights)
    assert torch.allclose(pooled, torch.tensor([[1.0, 0.0]]), atol=1e-5)


def test_confidence_weighted_pool_supports_weighted_view_dimension():
    patches = torch.tensor([
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 0.0], [0.0, 2.0]],
        ]
    ])
    weights = torch.tensor([
        [
            [[0.0], [1.0]],
            [[1.0], [0.0]],
        ]
    ])

    pooled = confidence_weighted_pool(patches, weights)

    expected = torch.tensor([[[0.0, 1.0], [2.0, 0.0]]])
    assert pooled.shape == (1, 2, 2)
    assert torch.allclose(pooled, expected, atol=1e-6)


def test_build_loop_descriptor_supports_weighted_view_dimension():
    camera = torch.tensor([[[3.0, 4.0], [0.0, 5.0]]])
    patches = torch.tensor([
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 0.0], [0.0, 2.0]],
        ]
    ])
    weights = torch.tensor([
        [
            [[0.0], [1.0]],
            [[1.0], [0.0]],
        ]
    ])

    descriptor = build_loop_descriptor(camera, patches, weights=weights)

    expected = torch.tensor([
        [
            [0.42426407, 0.56568545, 0.0, 0.70710677],
            [0.0, 0.70710677, 0.70710677, 0.0],
        ]
    ])
    assert descriptor.shape == (1, 2, 4)
    assert torch.allclose(descriptor, expected, atol=1e-6)


def test_confidence_map_to_token_weights_supports_view_dimensions():
    confidence_map = torch.tensor([
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    ])
    confidence_map_5d = confidence_map.unsqueeze(2)

    expected = torch.tensor([
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ]
    ])

    weights_4d = confidence_map_to_token_weights(confidence_map)
    weights_5d = confidence_map_to_token_weights(confidence_map_5d)

    assert weights_4d.shape == (1, 2, 4)
    assert weights_5d.shape == (1, 2, 4)
    assert torch.allclose(weights_4d, expected, atol=1e-6)
    assert torch.allclose(weights_5d, expected, atol=1e-6)
