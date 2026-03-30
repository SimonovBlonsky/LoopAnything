import builtins
import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytorch_lightning as pl
import torch


def import_trainer_module():
    sys.modules.pop("train.train_da3_adapter_salad", None)
    importlib.invalidate_caches()
    return importlib.import_module("train.train_da3_adapter_salad")


def test_import_adds_salad_root_to_sys_path_for_legacy_salad_absolute_imports():
    project_root = Path(__file__).resolve().parents[1]
    salad_root = project_root / "da3_streaming" / "loop_utils" / "salad"
    sys.path[:] = [path for path in sys.path if path != str(salad_root)]

    trainer_module = import_trainer_module()

    assert str(trainer_module.SALAD_ROOT) in sys.path


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
    def __init__(self, encoder_requires_grad=True, feature_adapter=None):
        super().__init__()
        self.encoder = TinyEncoder(requires_grad=encoder_requires_grad)
        self.feature_adapter = feature_adapter if feature_adapter is not None else TinyAdapter()
        self.aggregator = TinyAggregator()


def patch_model_construction(
    monkeypatch,
    trainer_module,
    *,
    encoder_requires_grad=True,
    feature_adapter=None,
):
    calls = {}
    vpr_model = TinyVPRModel(
        encoder_requires_grad=encoder_requires_grad,
        feature_adapter=feature_adapter,
    )

    def fake_build_vpr_model(**kwargs):
        calls.update(kwargs)
        return vpr_model

    salad_utils = SimpleNamespace(
        get_loss=lambda *args, **kwargs: object(),
        get_miner=lambda *args, **kwargs: object(),
        get_validation_recalls=lambda *args, **kwargs: {1: 0.0, 5: 0.0, 10: 0.0},
    )
    monkeypatch.setattr(trainer_module, "build_vpr_model", fake_build_vpr_model)
    if hasattr(trainer_module, "salad_utils"):
        monkeypatch.setattr(trainer_module, "salad_utils", salad_utils)
    monkeypatch.setattr(trainer_module, "_get_salad_utils", lambda: salad_utils, raising=False)
    return calls, vpr_model


def set_existing_default_checkpoint(monkeypatch, trainer_module, tmp_path):
    checkpoint_path = tmp_path / "dino_salad_512_32.ckpt"
    checkpoint_path.write_bytes(b"stub")
    monkeypatch.setattr(trainer_module, "DEFAULT_AGGREGATOR_CKPT_PATH", checkpoint_path)
    return checkpoint_path


def test_parse_args_defaults_match_dual_branch_protocol():
    trainer_module = import_trainer_module()

    args = trainer_module.parse_args([])

    assert args.seed == 0
    assert args.da3_model_name_or_path == "depth-anything/DA3-BASE"
    assert args.feature_source == "aux"
    assert args.aux_layer == 5
    assert args.aux_global_token_mode == "cls_token"
    assert args.feature_adapter_arch == "dual_branch"
    assert args.adapter_local_bottleneck == 256
    assert args.adapter_global_hidden_dim == 256
    assert args.agg_arch == "salad"
    assert args.agg_num_clusters == 16
    assert args.agg_cluster_dim == 32
    assert args.agg_token_dim == 32
    assert args.aggregator_ckpt_path == str(trainer_module.DEFAULT_AGGREGATOR_CKPT_PATH)


def test_parse_args_rejects_seed_outside_three_seed_protocol():
    trainer_module = import_trainer_module()

    with pytest.raises(SystemExit):
        trainer_module.parse_args(["--seed", "7"])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_parse_args_accepts_only_protocol_seed_values(seed):
    trainer_module = import_trainer_module()

    assert trainer_module.parse_args(["--seed", str(seed)]).seed == seed


@pytest.mark.parametrize("feature_adapter_arch", ["identity", "patch_only", "dual_branch"])
def test_parse_args_supports_all_feature_adapter_arch_ablation_modes(feature_adapter_arch):
    trainer_module = import_trainer_module()

    args = trainer_module.parse_args(["--feature-adapter-arch", feature_adapter_arch])

    assert args.feature_adapter_arch == feature_adapter_arch


@pytest.mark.parametrize(
    "argv",
    [
        ["--adapter-lr", "2e-4"],
        ["--lr", "1e-4"],
        ["--batch-size", "32"],
        ["--img-per-place", "2"],
        ["--min-img-per-place", "2"],
        ["--image-size", "256", "256"],
        ["--precision", "32"],
        ["--max-epochs", "5"],
        ["--check-val-every-n-epoch", "2"],
        ["--num-sanity-val-steps", "1"],
    ],
)
def test_parse_args_rejects_mutating_pinned_protocol_flags(argv):
    trainer_module = import_trainer_module()

    with pytest.raises(SystemExit):
        trainer_module.parse_args(argv)


def test_configure_optimizers_uses_only_adapter_and_aggregator_param_groups(
    monkeypatch, tmp_path
):
    trainer_module = import_trainer_module()
    expected_ckpt_path = set_existing_default_checkpoint(monkeypatch, trainer_module, tmp_path)
    calls, vpr_model = patch_model_construction(monkeypatch, trainer_module)

    protocol_breaking_args = SimpleNamespace(
        **vars(trainer_module.parse_args([])),
        adapter_lr=9.9,
        lr=8.8,
    )
    module = trainer_module.DA3AdapterSALADLightningModule(args=protocol_breaking_args)
    optim_config = module.configure_optimizers()
    optimizer = optim_config["optimizer"]

    assert calls["aggregator_ckpt_path"] == str(expected_ckpt_path)
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


@pytest.mark.parametrize(
    ("feature_adapter_arch", "feature_adapter", "expected_group_count", "expected_adapter_config"),
    [
        ("identity", torch.nn.Identity(), 1, None),
        ("patch_only", TinyAdapter(), 2, {"bottleneck": 640}),
        (
            "dual_branch",
            TinyAdapter(),
            2,
            {"local_bottleneck": 256, "global_hidden_dim": 256},
        ),
    ],
)
def test_supported_adapter_modes_instantiate_and_build_expected_optimizer_groups(
    monkeypatch,
    tmp_path,
    feature_adapter_arch,
    feature_adapter,
    expected_group_count,
    expected_adapter_config,
):
    trainer_module = import_trainer_module()
    set_existing_default_checkpoint(monkeypatch, trainer_module, tmp_path)
    calls, vpr_model = patch_model_construction(
        monkeypatch,
        trainer_module,
        feature_adapter=feature_adapter,
    )

    module = trainer_module.DA3AdapterSALADLightningModule(
        args=trainer_module.parse_args(["--feature-adapter-arch", feature_adapter_arch])
    )
    optim_config = module.configure_optimizers()
    optimizer = optim_config["optimizer"]

    assert calls["feature_adapter_arch"] == feature_adapter_arch
    assert calls["feature_adapter_config"] == expected_adapter_config
    assert len(optimizer.param_groups) == expected_group_count

    optimized_param_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    aggregator_param_ids = {id(parameter) for parameter in vpr_model.aggregator.parameters()}
    adapter_param_ids = {id(parameter) for parameter in vpr_model.feature_adapter.parameters()}

    assert aggregator_param_ids <= optimized_param_ids
    if feature_adapter_arch == "identity":
        assert optimizer.param_groups[0]["lr"] == pytest.approx(6e-5)
        assert optimized_param_ids == aggregator_param_ids
        assert not adapter_param_ids
    else:
        assert [group["lr"] for group in optimizer.param_groups] == [1e-4, 6e-5]
        assert optimized_param_ids == adapter_param_ids | aggregator_param_ids


def test_train_keeps_encoder_in_eval_mode_while_adapter_and_aggregator_remain_trainable(
    monkeypatch, tmp_path
):
    trainer_module = import_trainer_module()
    set_existing_default_checkpoint(monkeypatch, trainer_module, tmp_path)
    _, vpr_model = patch_model_construction(monkeypatch, trainer_module)

    module = trainer_module.DA3AdapterSALADLightningModule()
    module.train()

    assert module.vpr_model is vpr_model
    assert module.vpr_model.encoder.training is False
    assert any(parameter.requires_grad for parameter in module.vpr_model.feature_adapter.parameters())
    assert any(parameter.requires_grad for parameter in module.vpr_model.aggregator.parameters())


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


def test_default_aggregator_shape_keeps_default_warm_start_enabled(monkeypatch, tmp_path):
    trainer_module = import_trainer_module()
    expected_ckpt_path = set_existing_default_checkpoint(monkeypatch, trainer_module, tmp_path)
    calls, _ = patch_model_construction(monkeypatch, trainer_module)

    module = trainer_module.DA3AdapterSALADLightningModule(args=trainer_module.parse_args([]))

    assert module is not None
    assert calls["aggregator_ckpt_path"] == str(expected_ckpt_path)


def test_non_default_aggregator_shape_rejects_default_warm_start(monkeypatch, tmp_path):
    trainer_module = import_trainer_module()
    set_existing_default_checkpoint(monkeypatch, trainer_module, tmp_path)
    patch_model_construction(monkeypatch, trainer_module)

    with pytest.raises(ValueError, match="compatible checkpoint"):
        trainer_module.DA3AdapterSALADLightningModule(
            args=trainer_module.parse_args(
                [
                    "--agg-num-clusters",
                    "24",
                    "--agg-cluster-dim",
                    "48",
                    "--agg-token-dim",
                    "64",
                ]
            )
        )


def test_default_local_warm_start_requires_explicit_local_checkpoint(monkeypatch, tmp_path):
    trainer_module = import_trainer_module()
    missing_local_ckpt = tmp_path / "missing_dino_salad_512_32.ckpt"
    monkeypatch.setattr(trainer_module, "DEFAULT_AGGREGATOR_CKPT_PATH", missing_local_ckpt)
    patch_model_construction(monkeypatch, trainer_module)

    with pytest.raises(FileNotFoundError, match="local SALAD warm-start checkpoint"):
        trainer_module.DA3AdapterSALADLightningModule(args=trainer_module.parse_args([]))


def test_default_aggregator_checkpoint_path_stays_rooted_in_active_worktree():
    trainer_module = import_trainer_module()

    assert trainer_module.DEFAULT_AGGREGATOR_CKPT_PATH == (
        trainer_module.SALAD_ROOT / "weights" / "dino_salad_512_32.ckpt"
    )
    assert "LoopAnything/.worktrees" in str(trainer_module.DEFAULT_AGGREGATOR_CKPT_PATH)


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
        SimpleNamespace(
            precision="32",
            max_epochs=99,
            check_val_every_n_epoch=7,
            num_sanity_val_steps=5,
            save_top_k=1,
            save_last=False,
        ),
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

    trainer_module.build_datamodule(
        SimpleNamespace(
            batch_size=1,
            img_per_place=1,
            min_img_per_place=1,
            image_size=(64, 64),
            num_workers=0,
        )
    )

    assert "pitts30k_test" in captured["val_set_names"]


def test_scheduler_defaults_match_salad_baseline(monkeypatch, tmp_path):
    trainer_module = import_trainer_module()
    set_existing_default_checkpoint(monkeypatch, trainer_module, tmp_path)
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


def test_patch_only_default_build_vpr_model_config_uses_pinned_640_bottleneck(monkeypatch, tmp_path):
    trainer_module = import_trainer_module()
    set_existing_default_checkpoint(monkeypatch, trainer_module, tmp_path)
    calls, _ = patch_model_construction(monkeypatch, trainer_module)

    trainer_module.DA3AdapterSALADLightningModule(
        args=trainer_module.parse_args(["--feature-adapter-arch", "patch_only"])
    )

    assert calls["feature_adapter_arch"] == "patch_only"
    assert calls["feature_adapter_config"] == {"bottleneck": 640}


def test_build_vpr_model_receives_overridden_salad_shape_config_without_warm_start(
    monkeypatch, tmp_path
):
    trainer_module = import_trainer_module()
    set_existing_default_checkpoint(monkeypatch, trainer_module, tmp_path)
    calls, _ = patch_model_construction(monkeypatch, trainer_module)

    trainer_module.DA3AdapterSALADLightningModule(
        args=trainer_module.parse_args(
            [
                "--agg-num-clusters",
                "24",
                "--agg-cluster-dim",
                "48",
                "--agg-token-dim",
                "64",
                "--aggregator-ckpt-path",
                "none",
            ]
        )
    )

    assert calls["agg_config"]["num_channels"] == 768
    assert calls["agg_config"]["num_clusters"] == 24
    assert calls["agg_config"]["cluster_dim"] == 48
    assert calls["agg_config"]["token_dim"] == 64
    assert calls["aggregator_ckpt_path"] is None


def test_build_datamodule_uses_package_qualified_import_path(monkeypatch):
    trainer_module = import_trainer_module()
    monkeypatch.setattr(trainer_module, "GSVCitiesDataModule", None)

    class FakeDataModule:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.val_set_names = kwargs["val_set_names"]

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "dataloaders.GSVCitiesDataloader":
            raise AssertionError("unqualified dataloaders import should not be used")
        if name == "da3_streaming.loop_utils.salad.dataloaders.GSVCitiesDataloader":
            module = types.ModuleType(name)
            module.GSVCitiesDataModule = FakeDataModule
            return module
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    datamodule = trainer_module.build_datamodule(SimpleNamespace())

    assert isinstance(datamodule, FakeDataModule)
    assert datamodule.kwargs["val_set_names"] == ["pitts30k_val", "pitts30k_test"]


def test_datamodule_defaults_match_dual_branch_protocol(monkeypatch):
    trainer_module = import_trainer_module()
    captured = {}

    class FakeDataModule:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.val_set_names = kwargs["val_set_names"]

    monkeypatch.setattr(trainer_module, "GSVCitiesDataModule", FakeDataModule)

    trainer_module.build_datamodule(
        SimpleNamespace(
            batch_size=1,
            img_per_place=1,
            min_img_per_place=1,
            image_size=(64, 64),
            num_workers=0,
        )
    )

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
        SimpleNamespace(
            precision="32",
            max_epochs=99,
            check_val_every_n_epoch=7,
            num_sanity_val_steps=5,
            save_top_k=1,
            save_last=False,
        ),
        SimpleNamespace(encoder_arch="DA3EncoderAdapter"),
    )
    checkpoint = trainer.callbacks[0]

    assert trainer.precision == "16-mixed"
    assert trainer.max_epochs == 4
    assert trainer.check_val_every_n_epoch == 1
    assert trainer.num_sanity_val_steps == 0
    assert checkpoint.save_top_k == 3
    assert checkpoint.save_last is True
