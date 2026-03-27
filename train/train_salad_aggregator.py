from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from depth_anything_3.data import salad_utils
from depth_anything_3.data.salad_datamodule import SaladDataModule
from depth_anything_3.model.da3 import DA3
from depth_anything_3.model.vpr_model import VPRModel


class DA3Layer5CamTokenEncoder(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Task 1 scaffold only; encoder logic is not implemented yet.")


class DA3SALADLightningModule(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Task 1 scaffold only; model logic is not implemented yet.")

    def training_step(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Task 1 scaffold only; training logic is not implemented yet.")

    def configure_optimizers(self) -> Any:
        raise NotImplementedError("Task 1 scaffold only; optimization is not implemented yet.")


def build_datamodule() -> SaladDataModule:
    return SaladDataModule()


def build_trainer(model: pl.LightningModule) -> pl.Trainer:
    raise NotImplementedError("Task 1 scaffold only; trainer wiring is not implemented yet.")


def main() -> None:
    raise SystemExit(
        "Task 1 scaffold only: train/train_salad_aggregator.py is import-safe, but execution is not implemented yet."
    )


if __name__ == "__main__":
    main()
