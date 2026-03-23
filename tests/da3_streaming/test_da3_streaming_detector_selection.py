import importlib
import json


def _build_base_config(tmp_path, backend):
    return {
        "Weights": {
            "DA3_CONFIG": str(tmp_path / "config.json"),
            "DA3": str(tmp_path / "model.safetensors"),
            "SALAD": str(tmp_path / "salad.ckpt"),
        },
        "Model": {
            "chunk_size": 4,
            "overlap": 2,
            "loop_enable": True,
            "delete_temp_files": False,
            "ref_view_strategy": "saddle_balanced",
            "ref_view_strategy_loop": "saddle_balanced",
            "save_depth_conf_result": False,
        },
        "Loop": {
            "backend": backend,
            "DA3": {},
            "SALAD": {
                "image_size": [336, 336],
                "batch_size": 32,
                "similarity_threshold": 0.85,
                "top_k": 5,
                "use_nms": True,
                "nms_threshold": 25,
            },
            "SIM3_Optimizer": {},
        },
    }


def test_da3_streaming_selects_da3_detector(monkeypatch, tmp_path):
    module = importlib.import_module("da3_streaming.da3_streaming")

    class DummyModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.loaded = None
            self.device = None
            self.eval_called = False

        def load_state_dict(self, state_dict, strict=False):
            self.loaded = (state_dict, strict)

        def eval(self):
            self.eval_called = True
            return self

        def to(self, device):
            self.device = device
            return self

    class DummyDetector:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs
            self.load_model_called = False

        def load_model(self):
            self.load_model_called = True

    class DummyOptimizer:
        def __init__(self, config):
            self.config = config

    monkeypatch.setattr(module, "DepthAnything3", DummyModel)
    monkeypatch.setattr(module, "load_file", lambda path: {"weights": path})
    monkeypatch.setattr(module, "Sim3LoopOptimizer", DummyOptimizer)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(module.torch.cuda, "get_device_capability", lambda *args, **kwargs: (0, 0))
    monkeypatch.setattr(module, "DA3LoopDetector", DummyDetector, raising=False)
    monkeypatch.setattr(module, "LoopDetector", DummyDetector, raising=False)

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({}), encoding="utf-8")

    config = _build_base_config(tmp_path, backend='da3')
    config['Weights']['DA3_CONFIG'] = str(config_path)

    streaming = module.DA3_Streaming(
        image_dir=str(tmp_path / "images"),
        save_dir=str(tmp_path / "workspace"),
        config=config,
    )

    assert isinstance(streaming.loop_detector, DummyDetector)
    assert streaming.loop_detector.kwargs["da3_model"] is streaming.model
    assert streaming.loop_detector.load_model_called is False


def test_da3_streaming_selects_salad_detector(monkeypatch, tmp_path):
    module = importlib.import_module('da3_streaming.da3_streaming')

    class DummyModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def load_state_dict(self, state_dict, strict=False):
            self.loaded = (state_dict, strict)

        def eval(self):
            return self

        def to(self, device):
            self.device = device
            return self

    class DummyDA3Detector:
        def __init__(self, *args, **kwargs):
            raise AssertionError('DA3 detector should not be selected for SALAD backend')

    class DummySaladDetector:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs
            self.load_model_called = False

        def load_model(self):
            self.load_model_called = True

    class DummyOptimizer:
        def __init__(self, config):
            self.config = config

    monkeypatch.setattr(module, 'DepthAnything3', DummyModel)
    monkeypatch.setattr(module, 'load_file', lambda path: {'weights': path})
    monkeypatch.setattr(module, 'Sim3LoopOptimizer', DummyOptimizer)
    monkeypatch.setattr(module.torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(module.torch.cuda, 'get_device_capability', lambda *args, **kwargs: (0, 0))
    monkeypatch.setattr(module, 'DA3LoopDetector', DummyDA3Detector, raising=False)
    monkeypatch.setattr(module, 'LoopDetector', DummySaladDetector, raising=False)

    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps({}), encoding='utf-8')

    config = _build_base_config(tmp_path, backend='salad')
    config['Weights']['DA3_CONFIG'] = str(config_path)

    streaming = module.DA3_Streaming(
        image_dir=str(tmp_path / 'images'),
        save_dir=str(tmp_path / 'workspace'),
        config=config,
    )

    assert isinstance(streaming.loop_detector, DummySaladDetector)
    assert streaming.loop_detector.load_model_called is True
    assert streaming.loop_detector.kwargs['image_dir'] == str(tmp_path / 'images')
