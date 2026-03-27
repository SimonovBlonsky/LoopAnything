from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SALAD_ROOT = PROJECT_ROOT / "da3_streaming" / "loop_utils" / "salad"
for path in (PROJECT_ROOT, SRC_ROOT, SALAD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pytorch_lightning as pl
import torch
import torch.nn as nn

from da3_streaming.loop_utils.salad import utils as salad_utils
from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule
from depth_anything_3.api import DepthAnything3
from depth_anything_3.model.vpr_model import VPRModel


class DA3Layer5CamTokenEncoder(nn.Module):
    PATCH_SIZE = 14
    AUX_LAYER = 5
    AUX_DIM = 768

    def __init__(self) -> None:
        super().__init__()
        self.da3 = DepthAnything3.from_pretrained("depth-anything/DA3-BASE")
        for parameter in self.da3.parameters():
            parameter.requires_grad = False
        self.da3.eval()

    def forward(self, images: torch.Tensor) -> dict[str, Any]:
        if images.ndim != 4:
            raise ValueError(f"Expected images with shape [B, 3, H, W], got {tuple(images.shape)}")
        if images.shape[1] != 3:
            raise ValueError(f"Expected 3-channel images, got {images.shape[1]}")

        batch_size, _, height, width = images.shape
        if height % self.PATCH_SIZE != 0 or width % self.PATCH_SIZE != 0:
            raise ValueError(
                f"Input spatial size must be divisible by {self.PATCH_SIZE}, got {(height, width)}"
            )

        hp = height // self.PATCH_SIZE
        wp = width // self.PATCH_SIZE
        if next(self.da3.parameters()).device != images.device:
            self.da3.to(images.device)

        transformer = self.da3.model.backbone.pretrained
        views = images.unsqueeze(1)
        with torch.inference_mode():
            _, aux_outputs = transformer._get_intermediate_layers_not_chunked(
                views,
                n=[],
                export_feat_layers=[self.AUX_LAYER],
            )

        assert len(aux_outputs) == 1, (
            f"Expected one aux output for layer {self.AUX_LAYER}, got {len(aux_outputs)}"
        )
        tokens = transformer.norm(aux_outputs[0])[:, 0]
        assert tokens.ndim == 3, (
            f"Expected normalized tokens with shape [B, T, C], got {tuple(tokens.shape)}"
        )
        assert tokens.shape[0] == batch_size, f"Expected batch size {batch_size}, got {tokens.shape[0]}"
        assert tokens.shape[-1] == self.AUX_DIM, (
            f"Expected aux token dim {self.AUX_DIM}, got {tokens.shape[-1]}"
        )

        global_token = tokens[:, 0]
        patch_tokens = tokens[:, 1:]
        assert patch_tokens.shape[1] == hp * wp, (
            f"Expected {hp * wp} patch tokens from spatial shape {(hp, wp)}, got {patch_tokens.shape[1]}"
        )

        feature_map = patch_tokens.transpose(1, 2).reshape(batch_size, self.AUX_DIM, hp, wp)
        assert feature_map.shape == (batch_size, self.AUX_DIM, hp, wp), (
            f"Unexpected feature map shape {tuple(feature_map.shape)}"
        )
        assert global_token.shape == (batch_size, self.AUX_DIM), (
            f"Unexpected global token shape {tuple(global_token.shape)}"
        )

        if not torch.isfinite(tokens).all():
            raise ValueError("Non-finite normalized DA3 aux tokens")
        if not torch.isfinite(global_token).all():
            raise ValueError("Non-finite DA3 global token")
        if not torch.isfinite(patch_tokens).all():
            raise ValueError("Non-finite DA3 patch tokens")
        if not torch.isfinite(feature_map).all():
            raise ValueError("Non-finite DA3 feature map")

        return {
            "patch_tokens": patch_tokens,
            "feature_map": feature_map,
            "global_token": global_token,
            "spatial_shape": (hp, wp),
        }


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
    raise NotImplementedError("Task 1 scaffold only; datamodule wiring is not implemented yet.")


def build_trainer(model: pl.LightningModule) -> pl.Trainer:
    raise NotImplementedError("Task 1 scaffold only; trainer wiring is not implemented yet.")


def main() -> None:
    raise SystemExit(
        "Task 1 scaffold only: train/train_salad_aggregator.py is import-safe, but execution is not implemented yet."
    )


if __name__ == "__main__":
    main()
