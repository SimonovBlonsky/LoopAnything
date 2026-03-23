import json

import numpy as np
import torch
from torch import nn

import da3_streaming.loop_utils.da3_loop_detector as da3_loop_detector_module
from da3_streaming.loop_utils.da3_loop_detector import DA3LoopDetector
from depth_anything_3.model.loop_descriptor import build_loop_descriptor


class _StubBackbone:
    def __init__(self, patch_tokens, camera_tokens):
        self.patch_tokens = patch_tokens
        self.camera_tokens = camera_tokens
        self.calls = []

    def __call__(self, batch_tensor, ref_view_strategy=None):
        self.calls.append((batch_tensor.clone(), ref_view_strategy))
        return [(self.patch_tokens, self.camera_tokens)], None


class _StubDA3(nn.Module):
    def __init__(self, patch_tokens, camera_tokens):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.backbone = _StubBackbone(patch_tokens, camera_tokens)
        self.model = type("_StubModel", (), {"backbone": self.backbone})()
        self.input_calls = []

    def input_processor(self, image_paths, process_res=504, process_res_method="upper_bound_resize"):
        self.input_calls.append(
            {
                "image_paths": list(image_paths),
                "process_res": process_res,
                "process_res_method": process_res_method,
            }
        )
        batch = torch.stack(
            [torch.full((3, 14, 14), float(idx + 1)) for idx, _ in enumerate(image_paths)]
        )
        return batch, None, None


class _AutocastRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, device_type, dtype, enabled=True):
        self.calls.append((device_type, dtype, enabled))

        class _Context:
            def __enter__(self_inner):
                return None

            def __exit__(self_inner, exc_type, exc, tb):
                return False

        return _Context()


class _LoadableStubDA3(nn.Module):
    def __init__(self, **model_config):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.model_config = model_config
        self.loaded_state = None
        self.eval_called = False
        self.moved_to = None

    def load_state_dict(self, state_dict, strict=False):
        self.loaded_state = (state_dict, strict)

    def eval(self):
        self.eval_called = True
        return self

    def to(self, device):
        self.moved_to = torch.device(device)
        return self


def test_find_loop_closures_applies_threshold_gap_and_topk(tmp_path):
    detector = DA3LoopDetector(
        image_dir=str(tmp_path),
        config={
            "Loop": {
                "DA3": {
                    "top_k": 2,
                    "similarity_threshold": 0.8,
                    "use_nms": False,
                    "nms_threshold": 0,
                }
            },
            "Model": {"loop_temporal_exclusion": 1},
        },
    )
    detector.image_paths = [tmp_path / f"{idx}.png" for idx in range(4)]
    detector.descriptors = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.0],
            [1.0, 0.0],
            [0.9, 0.0],
        ]
    )

    loops = detector.find_loop_closures()

    assert (0, 1) not in [(a, b) for a, b, _ in loops]
    assert any({a, b} == {0, 2} or {a, b} == {1, 3} for a, b, _ in loops)


def test_search_descriptors_uses_faiss_index(monkeypatch, tmp_path):
    calls = {}

    class StubFaissIndex:
        def __init__(self, dim):
            calls['dim'] = dim

        def add(self, descriptors):
            calls['added'] = descriptors.copy()

        def search(self, descriptors, k):
            calls['searched'] = (descriptors.copy(), k)
            return (
                torch.tensor([[1.0, 0.8], [1.0, 0.8]], dtype=torch.float32).numpy(),
                torch.tensor([[0, 1], [1, 0]], dtype=torch.int64).numpy(),
            )

    monkeypatch.setattr(da3_loop_detector_module.faiss, 'IndexFlatIP', StubFaissIndex)

    detector = DA3LoopDetector(image_dir=str(tmp_path), config={})
    descriptors = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32).numpy()

    similarities, indices = detector._search_descriptors(descriptors, 2)

    assert calls['dim'] == 2
    assert calls['added'].shape == (2, 2)
    assert calls['added'].dtype == np.float32
    assert descriptors.dtype == np.float32
    assert (calls['added'] == descriptors).all()
    searched_descriptors, searched_k = calls['searched']
    assert searched_descriptors.shape == (2, 2)
    assert searched_descriptors.dtype == descriptors.dtype
    assert (searched_descriptors == descriptors).all()
    assert searched_k == 2
    assert similarities.shape == (2, 2)
    assert similarities.dtype == descriptors.dtype
    assert indices.tolist() == [[0, 1], [1, 0]]


def test_extract_descriptors_uses_injected_da3_components_and_populates_timing(tmp_path, monkeypatch):
    autocast_recorder = _AutocastRecorder()
    monkeypatch.setattr(da3_loop_detector_module.torch, "autocast", autocast_recorder)

    patch_tokens = torch.tensor(
        [
            [[[1.0, 0.0], [0.0, 1.0]]],
            [[[2.0, 0.0], [0.0, 2.0]]],
        ]
    )
    camera_tokens = torch.tensor(
        [
            [[3.0, 4.0]],
            [[0.0, 5.0]],
        ]
    )
    stub_da3 = _StubDA3(patch_tokens=patch_tokens, camera_tokens=camera_tokens)
    detector = DA3LoopDetector(
        image_dir=str(tmp_path),
        config={
            "Loop": {
                "DA3": {
                    "batch_size": 2,
                    "process_res": 224,
                    "process_res_method": "upper_bound_resize",
                }
            },
            "Model": {"ref_view_strategy_loop": "middle"},
        },
        da3_model=stub_da3,
    )
    detector.image_paths = [tmp_path / "0.png", tmp_path / "1.png"]

    model, device = detector.load_model()
    descriptors = detector.extract_descriptors()

    expected = build_loop_descriptor(camera_tokens, patch_tokens)[:, 0, :]

    assert model is stub_da3
    assert device == stub_da3.anchor.device
    assert stub_da3.input_calls == [
        {
            "image_paths": [str(tmp_path / "0.png"), str(tmp_path / "1.png")],
            "process_res": 224,
            "process_res_method": "upper_bound_resize",
        }
    ]
    assert len(stub_da3.backbone.calls) == 1
    backbone_input, ref_view_strategy = stub_da3.backbone.calls[0]
    assert backbone_input.shape == (2, 1, 3, 14, 14)
    assert ref_view_strategy == "middle"
    assert torch.allclose(descriptors, expected, atol=1e-6)
    assert autocast_recorder.calls == [("cpu", torch.float16, False)]
    assert detector.extract_time_s >= 0.0
    assert detector.extract_images_per_sec > 0.0
    assert detector.extract_ms_per_image >= 0.0


def test_load_model_builds_da3_from_config_when_not_injected(tmp_path, monkeypatch):
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps({'encoder': 'stub'}), encoding='utf-8')

    monkeypatch.setattr(da3_loop_detector_module, 'DepthAnything3', _LoadableStubDA3)
    monkeypatch.setattr(
        da3_loop_detector_module,
        'load_file',
        lambda path: {'weight': torch.tensor([1.0]), 'path': str(path)},
    )

    detector = DA3LoopDetector(
        image_dir=str(tmp_path),
        config={'Weights': {'DA3_CONFIG': str(config_path), 'DA3': str(tmp_path / 'model.safetensors')}},
    )

    model, device = detector.load_model()

    assert isinstance(model, _LoadableStubDA3)
    assert model.model_config == {'encoder': 'stub'}
    assert model.eval_called is True
    assert model.loaded_state[1] is False
    assert torch.equal(model.loaded_state[0]['weight'], torch.tensor([1.0]))
    assert model.loaded_state[0]['path'] == str(tmp_path / 'model.safetensors')
    assert model.moved_to == device
    assert detector.da3_model is model
    assert detector.device == device
