import numpy as np
import pytest

from robust_loop_verifier.da3_runner import (
    RealDa3Runner,
    RealDa3RunnerConfig,
    build_da3_triplet,
    convert_da3_extrinsics_to_c2w,
)
from robust_loop_verifier.retrieval import (
    SaladDescriptorBackend,
    SaladDescriptorBackendConfig,
    _strip_checkpoint_prefixes,
)


def test_real_da3_config_enforces_reference_strategy_and_resolution():
    config = RealDa3RunnerConfig(process_res=504)
    assert config.model_name == "depth-anything/DA3-LARGE-1.1"
    assert config.ref_view_strategy == "first"
    assert not config.extrinsics_are_c2w
    assert config.local_files_only


def test_real_da3_runner_loads_default_model_from_local_snapshot(tmp_path, monkeypatch):
    import depth_anything_3.api as da3_api

    snapshot = (
        tmp_path
        / "models--depth-anything--DA3-LARGE-1.1"
        / "snapshots"
        / "abc123"
    )
    snapshot.mkdir(parents=True)
    refs = tmp_path / "models--depth-anything--DA3-LARGE-1.1" / "refs"
    refs.mkdir()
    (refs / "main").write_text("abc123", encoding="utf-8")
    captured = {}

    class FakeModel:
        def to(self, device):
            captured["device"] = device
            return self

        def eval(self):
            captured["eval"] = True
            return self

    def fake_from_pretrained(model_name_or_path, **kwargs):
        captured["model_name_or_path"] = model_name_or_path
        captured["kwargs"] = kwargs
        return FakeModel()

    monkeypatch.setattr(da3_api.DepthAnything3, "from_pretrained", fake_from_pretrained)

    runner = RealDa3Runner(
        RealDa3RunnerConfig(
            cache_dir=tmp_path,
            device="cpu",
        )
    )

    model = runner._load_model()

    assert isinstance(model, FakeModel)
    assert captured["model_name_or_path"] == str(snapshot)
    assert captured["kwargs"] == {"local_files_only": True}
    assert captured["device"] == "cpu"
    assert captured["eval"] is True


def test_convert_da3_3x4_w2c_to_c2w():
    extrinsics = np.eye(4, dtype=np.float64)[None, :3, :]
    extrinsics[0, 0, 3] = 3.0
    c2w = convert_da3_extrinsics_to_c2w(extrinsics, extrinsics_are_c2w=False)
    assert c2w.shape == (1, 4, 4)
    np.testing.assert_allclose(c2w[0, :3, 3], [-3.0, 0.0, 0.0])


def test_salad_backend_config_preserves_local_repo_and_checkpoint_paths(tmp_path):
    config = SaladDescriptorBackendConfig(
        salad_repo=tmp_path / "salad",
        checkpoint_path=tmp_path / "dino_salad.ckpt",
        device="cuda",
    )
    assert config.backbone == "dinov2_vitb14"


def test_strip_checkpoint_prefixes_removes_repeated_common_prefixes():
    state_dict = {
        "module.net.model.encoder.weight": np.array([1.0]),
        "net.module.bias": np.array([2.0]),
    }

    cleaned = _strip_checkpoint_prefixes(state_dict)

    assert sorted(cleaned) == ["bias", "encoder.weight"]


def test_salad_backend_empty_input_fails_before_model_load(tmp_path, monkeypatch):
    backend = SaladDescriptorBackend(
        SaladDescriptorBackendConfig(
            salad_repo=tmp_path / "salad",
            checkpoint_path=tmp_path / "dino_salad.ckpt",
            device="cpu",
        )
    )
    monkeypatch.setattr(
        backend,
        "_load_model",
        lambda: (_ for _ in ()).throw(AssertionError("model loaded")),
    )

    with pytest.raises(ValueError, match="image_paths must not be empty"):
        backend.compute([], [])


def test_salad_backend_loads_local_model_without_torch_hub_load(tmp_path, monkeypatch):
    salad_repo = tmp_path / "salad"
    (salad_repo / "models" / "backbones").mkdir(parents=True)
    (salad_repo / "vpr_model.py").write_text(
        """
class VPRModel:
    init_kwargs = None

    def __init__(self, **kwargs):
        type(self).init_kwargs = kwargs
        self.loaded_keys = None
        self.device = None

    def load_state_dict(self, state_dict, strict=False):
        self.loaded_keys = sorted(state_dict)
        return [], []

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self
""",
        encoding="utf-8",
    )
    (salad_repo / "models" / "__init__.py").write_text("", encoding="utf-8")
    (salad_repo / "models" / "backbones" / "__init__.py").write_text("", encoding="utf-8")
    (salad_repo / "models" / "backbones" / "dinov2.py").write_text(
        "DINOV2_ARCHS = {'dinov2_vitb14': 768}\n",
        encoding="utf-8",
    )

    import torch

    monkeypatch.setattr(
        torch.hub,
        "load",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("torch.hub.load called")),
    )
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: {"state_dict": {"module.net.model.weight": torch.ones(1)}},
    )

    backend = SaladDescriptorBackend(
        SaladDescriptorBackendConfig(
            salad_repo=salad_repo,
            checkpoint_path=tmp_path / "dino_salad.ckpt",
            device="cpu",
        )
    )

    model = backend._load_model()

    assert model.loaded_keys == ["weight"]
    assert model.device == "cpu"


def test_real_da3_runner_accepts_injected_model_and_returns_c2w(tmp_path):
    import torch
    from PIL import Image

    class FakeDa3Model:
        def input_processor(self, images, **kwargs):
            assert len(images) == 3
            return torch.zeros((3, 3, 224, 224)), None, None

        def forward(self, image, **kwargs):
            assert image.shape == (1, 3, 3, 224, 224)
            extrinsics = np.repeat(np.eye(4, dtype=np.float64)[None, None], 3, axis=1)
            extrinsics[0, 1, 0, 3] = 3.0
            extrinsics[0, 2, 1, 3] = 4.0
            return {"extrinsics": torch.from_numpy(extrinsics)}

    image_paths = []
    for name in ("query.png", "candidate.png", "support.png"):
        path = tmp_path / name
        Image.new("RGB", (8, 8)).save(path)
        image_paths.append(path)

    triplet = build_da3_triplet(*image_paths, query_idx=7, candidate_idx=3, support_idx=4)
    runner = RealDa3Runner(
        RealDa3RunnerConfig(model_name="depth-anything/DA3-SMALL", device="cpu"),
        model=FakeDa3Model(),
    )

    result = runner.run_triplet(triplet)

    assert result.predicted_c2w.shape == (3, 4, 4)
    np.testing.assert_allclose(result.predicted_c2w[1, :3, 3], [-3.0, 0.0, 0.0])
    np.testing.assert_allclose(result.predicted_c2w[2, :3, 3], [0.0, -4.0, 0.0])
