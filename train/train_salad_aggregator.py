from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - import smoke fallback
    torch = None

    class _Module:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__()

    class nn:  # type: ignore[no-redef]
        Module = _Module

try:
    import pytorch_lightning as pl
except Exception:  # pragma: no cover - import smoke fallback
    class _LightningModule(nn.Module):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__()

    class _Trainer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

        def fit(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("Trainer is a placeholder in this scaffold.")

    class pl:  # type: ignore[no-redef]
        LightningModule = _LightningModule
        Trainer = _Trainer

try:
    from depth_anything_3.model.da3 import DA3
except Exception:  # pragma: no cover - optional import placeholder
    DA3 = None

try:
    from depth_anything_3.data import salad_utils
except Exception:  # pragma: no cover - optional import placeholder
    salad_utils = None

try:
    from depth_anything_3.data.salad_datamodule import SaladDataModule
except Exception:  # pragma: no cover - optional import placeholder
    SaladDataModule = None

try:
    from depth_anything_3.model.vpr_model import VPRModel
except Exception:  # pragma: no cover - optional import placeholder
    class VPRModel(nn.Module):  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__()


class DA3Layer5CamTokenEncoder(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Encoder scaffold only; implement training contract later.")


class DA3SALADLightningModule(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        self.backbone = DA3
        self.vpr_model = VPRModel
        self.salad_utils = salad_utils

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Lightning scaffold only; implement model logic later.")

    def training_step(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Lightning scaffold only; implement training later.")

    def configure_optimizers(self) -> Any:
        raise NotImplementedError("Lightning scaffold only; implement optimization later.")


def build_datamodule() -> Any:
    if SaladDataModule is None:
        return None
    return SaladDataModule()


def build_trainer(model: pl.LightningModule) -> pl.Trainer:
    return pl.Trainer(
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )


def main() -> None:
    datamodule = build_datamodule()
    model = DA3SALADLightningModule()
    trainer = build_trainer(model)
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
