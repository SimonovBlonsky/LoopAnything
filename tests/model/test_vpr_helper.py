import json

import pytest
import torch

from depth_anything_3.model import vpr_helper


def test_get_aggregator_builds_salad_from_local_package():
    aggregator = vpr_helper.build_aggregator(
        "SALAD",
        {"num_channels": 1536, "num_clusters": 64, "cluster_dim": 128, "token_dim": 256},
    )
    assert aggregator.__class__.__name__ == "SALAD"


def test_extract_prefixed_state_dict_handles_common_prefixes():
    state_dict = {
        "model.aggregator.score.weight": torch.ones(1),
        "model.aggregator.score.bias": torch.zeros(1),
        "model.backbone.weight": torch.randn(1),
    }
    extracted = vpr_helper.extract_prefixed_state_dict(
        state_dict,
        ["aggregator.", "model.aggregator.", "module.aggregator."],
    )
    assert set(extracted) == {"score.weight", "score.bias"}


def test_extract_prefixed_state_dict_handles_module_prefix():
    state_dict = {
        "module.aggregator.dust_bin": torch.tensor(1.0),
        "module.aggregator.score.weight": torch.ones(1),
    }
    extracted = vpr_helper.extract_prefixed_state_dict(
        state_dict,
        ["aggregator.", "model.aggregator.", "module.aggregator."],
    )
    assert set(extracted) == {"dust_bin", "score.weight"}


def test_load_aggregator_weights_unwraps_state_dict(tmp_path):
    ckpt_path = tmp_path / "salad.ckpt"
    torch.save({"state_dict": {"aggregator.dust_bin": torch.tensor(1.5)}}, ckpt_path)

    class StubAggregator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dust_bin = torch.nn.Parameter(torch.tensor(0.0))

    aggregator = StubAggregator()
    vpr_helper.load_aggregator_weights_from_salad_ckpt(aggregator, ckpt_path)
    assert aggregator.dust_bin.item() == pytest.approx(1.5)


def test_load_aggregator_weights_unwraps_model_field(tmp_path):
    ckpt_path = tmp_path / "salad_model.ckpt"
    torch.save({"model": {"model.aggregator.dust_bin": torch.tensor(2.5)}}, ckpt_path)

    class StubAggregator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dust_bin = torch.nn.Parameter(torch.tensor(0.0))

    aggregator = StubAggregator()
    vpr_helper.load_aggregator_weights_from_salad_ckpt(aggregator, ckpt_path)
    assert aggregator.dust_bin.item() == pytest.approx(2.5)


def test_load_aggregator_weights_unwraps_raw_flat_dict(tmp_path):
    ckpt_path = tmp_path / "salad_flat.ckpt"
    torch.save({"module.aggregator.dust_bin": torch.tensor(3.5)}, ckpt_path)

    class StubAggregator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dust_bin = torch.nn.Parameter(torch.tensor(0.0))

    aggregator = StubAggregator()
    vpr_helper.load_aggregator_weights_from_salad_ckpt(aggregator, ckpt_path)
    assert aggregator.dust_bin.item() == pytest.approx(3.5)


def test_load_aggregator_weights_raises_when_no_prefixed_keys_exist(tmp_path):
    ckpt_path = tmp_path / "bad.ckpt"
    torch.save({"state_dict": {"backbone.weight": torch.tensor(1.0)}}, ckpt_path)

    class StubAggregator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dust_bin = torch.nn.Parameter(torch.tensor(0.0))

    with pytest.raises(ValueError, match="aggregator-prefixed"):
        vpr_helper.load_aggregator_weights_from_salad_ckpt(StubAggregator(), ckpt_path)


def test_build_vpr_model_rejects_missing_da3_source():
    with pytest.raises(ValueError, match="exactly one DA3 source"):
        vpr_helper.build_vpr_model(
            agg_arch="GeM",
            agg_config={"p": 3},
        )


def test_build_da3_model_rejects_partial_config_weight_pair():
    with pytest.raises(ValueError, match="provided together"):
        vpr_helper.build_da3_model(da3_config_path="/tmp/fake.yaml")


def test_build_da3_model_from_json_and_safetensors_paths(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    weight_path = tmp_path / "model.safetensors"
    config = {"model_name": "da3-large", "image_size": 1024}
    config_path.write_text(json.dumps(config))
    weight_path.write_text("stub")

    calls = {}

    class StubModel(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            calls["config"] = kwargs

        def load_state_dict(self, state_dict, strict=True):
            calls["state_dict"] = state_dict
            calls["strict"] = strict

        def eval(self):
            calls["eval_called"] = True
            return self

    monkeypatch.setattr(vpr_helper, "DepthAnything3", StubModel)
    monkeypatch.setattr(vpr_helper, "load_file", lambda path: {"dummy": torch.tensor(1.0), "path": str(path)})

    model = vpr_helper.build_da3_model(da3_config_path=config_path, da3_weight_path=weight_path)

    assert isinstance(model, StubModel)
    assert calls["config"] == config
    assert calls["state_dict"] == {"dummy": torch.tensor(1.0), "path": str(weight_path)}
    assert calls["strict"] is False
    assert calls["eval_called"] is True


def test_build_da3_model_from_model_name_or_path(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(vpr_helper.DepthAnything3, "from_pretrained", lambda model_name: sentinel)
    result = vpr_helper.build_da3_model(da3_model_name_or_path="depth-anything/DA3-LARGE-1.1")
    assert result is sentinel


def test_build_vpr_model_rejects_mixed_da3_source_modes():
    with pytest.raises(ValueError, match="exactly one DA3 source"):
        vpr_helper.build_vpr_model(
            da3_model=object(),
            da3_model_name_or_path="depth-anything/DA3-LARGE-1.1",
            agg_arch="GeM",
            agg_config={"p": 3},
        )


def test_build_da3_encoder_adapter_threads_arguments(monkeypatch):
    calls = {}

    class StubDA3(torch.nn.Module):
        pass

    class RecordingAdapter(torch.nn.Module):
        def __init__(self, model, feat_layer=-1, ref_view_strategy="saddle_balanced", patch_size=14):
            calls["model"] = model
            calls["feat_layer"] = feat_layer
            calls["ref_view_strategy"] = ref_view_strategy
            calls["patch_size"] = patch_size

    monkeypatch.setattr(vpr_helper, "DA3EncoderAdapter", RecordingAdapter)
    stub = StubDA3()
    vpr_helper.build_da3_encoder_adapter(
        stub,
        feat_layer=2,
        ref_view_strategy="first",
        patch_size=16,
    )
    assert calls == {
        "model": stub,
        "feat_layer": 2,
        "ref_view_strategy": "first",
        "patch_size": 16,
    }


def test_build_vpr_model_with_existing_da3_model(monkeypatch):
    class StubDA3(torch.nn.Module):
        pass

    class RecordingVPRModel(torch.nn.Module):
        def __init__(self, encoder, aggregator, agg_arch):
            super().__init__()
            self.encoder = encoder
            self.aggregator = aggregator
            self.agg_arch = agg_arch

    monkeypatch.setattr(vpr_helper, "build_aggregator", lambda *args, **kwargs: torch.nn.Identity())
    monkeypatch.setattr(vpr_helper, "build_da3_encoder_adapter", lambda model, **kwargs: torch.nn.Identity())
    monkeypatch.setattr(vpr_helper, "VPRModel", RecordingVPRModel)

    result = vpr_helper.build_vpr_model(
        da3_model=StubDA3(),
        agg_arch="GeM",
        agg_config={"p": 3},
    )
    assert result.agg_arch == "GeM"
    assert result.training is False


def test_build_da3_model_from_config_and_weight_paths(monkeypatch):
    calls = {}

    class StubModel(torch.nn.Module):
        def load_state_dict(self, state_dict, strict=True):
            calls["state_dict"] = state_dict
            calls["strict"] = strict

        def eval(self):
            calls["eval_called"] = True
            return self

    monkeypatch.setattr(vpr_helper, "load_config", lambda path: {"config_path": path})
    monkeypatch.setattr(vpr_helper, "create_object", lambda cfg: StubModel())
    monkeypatch.setattr(
        vpr_helper.torch,
        "load",
        lambda path, map_location=None: {"state_dict": {"dummy": torch.tensor(1.0)}},
    )

    model = vpr_helper.build_da3_model(
        da3_config_path="/tmp/fake.yaml",
        da3_weight_path="/tmp/fake.pt",
    )
    assert isinstance(model, StubModel)
    assert calls["state_dict"] == {"dummy": torch.tensor(1.0)}
    assert calls["strict"] is True
    assert calls["eval_called"] is True


def test_build_vpr_model_threads_aggregator_ckpt_and_strict(monkeypatch):
    calls = {}

    class StubDA3(torch.nn.Module):
        pass

    class RecordingVPRModel(torch.nn.Module):
        def __init__(self, encoder, aggregator, agg_arch):
            super().__init__()
            self.encoder = encoder
            self.aggregator = aggregator
            self.agg_arch = agg_arch

    monkeypatch.setattr(vpr_helper, "build_aggregator", lambda *args, **kwargs: torch.nn.Identity())
    monkeypatch.setattr(vpr_helper, "build_da3_encoder_adapter", lambda model, **kwargs: torch.nn.Identity())
    monkeypatch.setattr(vpr_helper, "VPRModel", RecordingVPRModel)

    def fake_load_aggregator_weights(aggregator, ckpt_path, strict=True):
        calls["ckpt_path"] = ckpt_path
        calls["strict"] = strict

    monkeypatch.setattr(vpr_helper, "load_aggregator_weights_from_salad_ckpt", fake_load_aggregator_weights)

    vpr_helper.build_vpr_model(
        da3_model=StubDA3(),
        agg_arch="GeM",
        agg_config={"p": 3},
        aggregator_ckpt_path="/tmp/agg.ckpt",
        strict=False,
    )
    assert calls == {"ckpt_path": "/tmp/agg.ckpt", "strict": False}


def test_build_vpr_model_returns_eval_mode_model(monkeypatch):
    class StubDA3(torch.nn.Module):
        pass

    class DropoutAggregator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = torch.nn.Dropout(p=0.5)

    monkeypatch.setattr(vpr_helper, "build_da3_encoder_adapter", lambda model, **kwargs: torch.nn.Identity())
    monkeypatch.setattr(vpr_helper, "build_aggregator", lambda *args, **kwargs: DropoutAggregator())

    model = vpr_helper.build_vpr_model(
        da3_model=StubDA3(),
        agg_arch="SALAD",
        agg_config={"num_channels": 4, "num_clusters": 2, "cluster_dim": 2, "token_dim": 2},
    )

    assert model.training is False
    assert model.aggregator.training is False
    assert model.aggregator.dropout.training is False
