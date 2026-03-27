from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pytorch_lightning as pl
import torch.nn as nn

from da3_streaming.loop_utils.salad import utils as salad_utils
from da3_streaming.loop_utils.salad.dataloaders.GSVCitiesDataloader import GSVCitiesDataModule
from depth_anything_3.api import DepthAnything3
from depth_anything_3.model.vpr_model import VPRModel


class DA3Layer5CamTokenEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Task 1 scaffold only; encoder logic is not implemented yet.")


class DA3SALADLightningModule(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Task 1 scaffold only; model logic is not implemented yet.")

    def training_step(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Task 1 scaffold only; training logic is not implemented yet.")

    def configure_optimizers(self) -> Any:
        raise NotImplementedError("Task 1 scaffold only; optimization is not implemented yet.")


def build_datamodule() -> GSVCitiesDataModule:
    return GSVCitiesDataModule()


def build_trainer(model: pl.LightningModule) -> pl.Trainer:
    raise NotImplementedError("Task 1 scaffold only; trainer wiring is not implemented yet.")


def main() -> None:
    raise SystemExit(
        "Task 1 scaffold only: train/train_salad_aggregator.py is import-safe, but execution is not implemented yet."
    )


if __name__ == "__main__":
    main()
