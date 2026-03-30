import importlib
import sys
from types import SimpleNamespace

import pytest
import pytorch_lightning as pl
import torch


def import_trainer_module():
    sys.modules.pop("train.train_da3_adapter_salad", None)
    importlib.invalidate_caches()
    return importlib.import_module("train.train_da3_adapter_salad")


class TinyEncoder(torch.nn.Module):
    def __init__(self, requires_grad=True):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=requires_grad)


class TinyAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([2.0]))


class TinyAggregator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([3.0]))


class TinyVPRModel(torch.nn.Module):
    def __init__(self, encoder_requires_grad=True):
        super().__init__()
        self.encoder = TinyEncoder(requires_grad=encoder_requires_grad)
        self.feature_adapter = TinyAdapter()
        self.aggregator = TinyAggregator()


def patch_model_construction(monkeypatch, trainer_module, *, encoder_requires_grad=True):
    calls = {}
    vpr_model = TinyVPRModel(encoder_requires_grad=encoder_requires_grad)

    def fake_build_vpr_model(**kwargs):
        calls.update(kwargs)
        return vpr_model

    monkeypatch.setattr(trainer_module, "build_vpr_model", fake_build_vpr_model)
    monkeypatch.setattr(trainer_module.salad_utils, "get_loss", lambda *args, **kwargs: object())
    monkeypatch.setattr(trainer_module.salad_utils, "get_miner", lambda *args, **kwargs: object())
    return calls, vpr_model


def test_parse_args_defaults_match_dual_branch_protocol():
    trainer_module = import_trainer_module()

    args = trainer_module.parse_args([])

    assert args.da3_model_name_or_path == "depth-anything/DA3-BASE"
    assert args.feature_source == "aux"
    assert args.aux_layer == 5
    assert args.aux_global_token_mode == "cls_token"
    assert args.feature_adapter_arch == "dual_branch"
    assert args.adapter_local_bottleneck == 256
    assert args.adapter_global_hidden_dim == 256
    assert args.agg_arch == "salad"
    assert args.aggregator_ckpt_path == str(trainer_module.DEFAULT_AGGREGATOR_CKPT_PATH)
    assert trainer_module.parse_args(["--seed", "17"]).seed == 17


def test_configure_optimizers_uses_only_adapter_and_aggregator_param_groups(monkeypatch):
    trainer_module = import_trainer_module()
    calls, vpr_model = patch_model_construction(monkeypatch, trainer_module)

    module = trainer_module.DA3AdapterSALADLightningModule()
    optim_config = module.configure_optimizers()
    optimizer = optim_config["optimizer"]

    assert calls["aggregator_ckpt_path"] == str(trainer_module.DEFAULT_AGGREGATOR_CKPT_PATH)
    assert len(optimizer.param_groups) == 2
    assert [group["lr"] for group in optimizer.param_groups] == [1e-4, 6e-5]

    encoder_param_ids = {id(parameter) for parameter in vpr_model.encoder.parameters()}
    adapter_param_ids = {id(parameter) for parameter in vpr_model.feature_adapter.parameters()}
    aggregator_param_ids = {id(parameter) for parameter in vpr_model.aggregator.parameters()}
    optimized_param_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    assert optimized_param_ids == adapter_param_ids | aggregator_param_ids
    assert optimized_param_ids.isdisjoint(encoder_param_ids)
    assert all(not parameter.requires_grad for parameter in vpr_model.encoder.parameters())


def test_parse_args_supports_disabling_aggregator_warm_start(monkeypatch):
    trainer_module = import_trainer_module()
    calls, _ = patch_model_construction(monkeypatch, trainer_module)

    module = trainer_module.DA3AdapterSALADLightningModule(
        args=trainer_module.parse_args(["--aggregator-ckpt-path", "none"])
    )
    assert module is not None
    assert calls["aggregator_ckpt_path"] is None

    calls.clear()
    trainer_module.DA3AdapterSALADLightningModule(
        args=trainer_module.parse_args(["--aggregator-ckpt-path", ""])
    )
    assert calls["aggregator_ckpt_path"] is None


def test_startup_validation_rejects_trainable_encoder_params():
    trainer_module = import_trainer_module()

    module = trainer_module.DA3AdapterSALADLightningModule.__new__(
        trainer_module.DA3AdapterSALADLightningModule
    )
    pl.LightningModule.__init__(module)
    module.vpr_model = TinyVPRModel(encoder_requires_grad=True)

    with pytest.raises(ValueError, match="encoder"):
        module._validate_startup_state()


def test_checkpoint_monitor_is_pitts30k_val_r1(monkeypatch):
    trainer_module = import_trainer_module()

    class FakeModelCheckpoint:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.callbacks = kwargs["callbacks"]
            self.precision = kwargs["precision"]
            self.max_epochs = kwargs["max_epochs"]
            self.check_val_every_n_epoch = kwargs["check_val_every_n_epoch"]
            self.num_sanity_val_steps = kwargs["num_sanity_val_steps"]

    monkeypatch.setattr(trainer_module.pl.callbacks, "ModelCheckpoint", FakeModelCheckpoint)
    monkeypatch.setattr(trainer_module.pl, "Trainer", FakeTrainer)

    trainer = trainer_module.build_trainer(
        trainer_module.parse_args([]),
        SimpleNamespace(encoder_arch="DA3EncoderAdapter"),
    )

    assert trainer.callbacks[0].monitor == "pitts30k_val/R1"


def test_validation_sets_keep_pitts30k_test_enabled(monkeypatch):
    trainer_module = import_trainer_module()
    captured = {}

    class FakeDataModule:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.val_set_names = kwargs["val_set_names"]

    monkeypatch.setattr(trainer_module, "GSVCitiesDataModule", FakeDataModule)

    trainer_module.build_datamodule(trainer_module.parse_args([]))

    assert "pitts30k_test" in captured["val_set_names"]


def test_scheduler_defaults_match_salad_baseline(monkeypatch):
    trainer_module = import_trainer_module()
    patch_model_construction(monkeypatch, trainer_module)

    module = trainer_module.DA3AdapterSALADLightningModule()
    optim_config = module.configure_optimizers()
    scheduler = optim_config["lr_scheduler"]["scheduler"]

    assert module.optimizer_name == "adamw"
    assert module.weight_decay == pytest.approx(9.5e-9)
    assert module.momentum == pytest.approx(0.9)
    assert module.lr_sched == "linear"
    assert module.lr_sched_args == {
        "start_factor": 1.0,
        "end_factor": 0.2,
        "total_iters": 4000,
    }
    assert isinstance(scheduler, torch.optim.lr_scheduler.LinearLR)


def test_datamodule_defaults_match_dual_branch_protocol(monkeypatch):
    trainer_module = import_trainer_module()
    captured = {}

    class FakeDataModule:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.val_set_names = kwargs["val_set_names"]

    monkeypatch.setattr(trainer_module, "GSVCitiesDataModule", FakeDataModule)

    trainer_module.build_datamodule(trainer_module.parse_args([]))

    assert captured["batch_size"] == 60
    assert captured["img_per_place"] == 4
    assert captured["min_img_per_place"] == 4
    assert captured["image_size"] == (224, 224)
    assert captured["val_set_names"] == ["pitts30k_val", "pitts30k_test"]


def test_trainer_defaults_match_dual_branch_protocol(monkeypatch):
    trainer_module = import_trainer_module()

    class FakeModelCheckpoint:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.callbacks = kwargs["callbacks"]
            self.precision = kwargs["precision"]
            self.max_epochs = kwargs["max_epochs"]
            self.check_val_every_n_epoch = kwargs["check_val_every_n_epoch"]
            self.num_sanity_val_steps = kwargs["num_sanity_val_steps"]

    monkeypatch.setattr(trainer_module.pl.callbacks, "ModelCheckpoint", FakeModelCheckpoint)
    monkeypatch.setattr(trainer_module.pl, "Trainer", FakeTrainer)

    trainer = trainer_module.build_trainer(
        trainer_module.parse_args([]),
        SimpleNamespace(encoder_arch="DA3EncoderAdapter"),
    )
    checkpoint = trainer.callbacks[0]

    assert trainer.precision == "16-mixed"
    assert trainer.max_epochs == 4
    assert trainer.check_val_every_n_epoch == 1
    assert trainer.num_sanity_val_steps == 0
    assert checkpoint.save_top_k == 3
    assert checkpoint.save_last is True
