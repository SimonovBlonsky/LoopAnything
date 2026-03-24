from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import torch

from depth_anything_3.api import DepthAnything3
from depth_anything_3.cfg import create_object, load_config
from depth_anything_3.model import VPRaggregators
from depth_anything_3.model.vpr_encoder_adapter import DA3EncoderAdapter
from depth_anything_3.model.vpr_model import VPRModel


AGGREGATOR_STATE_PREFIXES = (
    "aggregator.",
    "model.aggregator.",
    "module.aggregator.",
)


def build_aggregator(agg_arch, agg_config=None):
    agg_config = {} if agg_config is None else dict(agg_config)
    name = agg_arch.lower()
    if name == "cosplace":
        return VPRaggregators.CosPlace(**agg_config)
    if name == "gem":
        agg_config.setdefault("p", 3)
        return VPRaggregators.GeMPool(**agg_config)
    if name == "convap":
        return VPRaggregators.ConvAP(**agg_config)
    if name == "mixvpr":
        return VPRaggregators.MixVPR(**agg_config)
    if name == "salad":
        return VPRaggregators.SALAD(**agg_config)
    raise ValueError(f"Unsupported aggregator: {agg_arch}")


def extract_prefixed_state_dict(state_dict: Mapping[str, torch.Tensor], prefixes: Iterable[str]):
    extracted = {}
    for key, value in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                extracted[key[len(prefix) :]] = value
                break
    return extracted


def _unwrap_checkpoint_state_dict(checkpoint):
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Expected checkpoint to contain a mapping of parameters")
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], Mapping):
        return checkpoint["state_dict"]
    if "model" in checkpoint and isinstance(checkpoint["model"], Mapping):
        return checkpoint["model"]
    return checkpoint


def load_aggregator_weights_from_salad_ckpt(aggregator, ckpt_path, strict=True):
    checkpoint = torch.load(Path(ckpt_path), map_location="cpu")
    state_dict = _unwrap_checkpoint_state_dict(checkpoint)
    aggregator_state_dict = extract_prefixed_state_dict(state_dict, AGGREGATOR_STATE_PREFIXES)
    if not aggregator_state_dict:
        raise ValueError("No aggregator-prefixed keys found in SALAD checkpoint")
    aggregator.load_state_dict(aggregator_state_dict, strict=strict)
    return aggregator


def build_da3_model(*, da3_config_path=None, da3_weight_path=None, da3_model_name_or_path=None):
    has_config_mode = da3_config_path is not None or da3_weight_path is not None
    has_name_mode = da3_model_name_or_path is not None
    if has_config_mode and has_name_mode:
        raise ValueError("Provide exactly one DA3 source mode")
    if has_config_mode:
        if not (da3_config_path and da3_weight_path):
            raise ValueError("da3_config_path and da3_weight_path must be provided together")
        cfg = load_config(da3_config_path)
        model = create_object(cfg)
        checkpoint = torch.load(da3_weight_path, map_location="cpu")
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        model.load_state_dict(checkpoint, strict=True)
        model.eval()
        return model
    if has_name_mode:
        return DepthAnything3.from_pretrained(da3_model_name_or_path)
    raise ValueError("No DA3 source provided")


def build_da3_encoder_adapter(da3_model, *, feat_layer=-1, ref_view_strategy="saddle_balanced", patch_size=14):
    return DA3EncoderAdapter(
        da3_model,
        feat_layer=feat_layer,
        ref_view_strategy=ref_view_strategy,
        patch_size=patch_size,
    )


def build_vpr_model(
    *,
    da3_model=None,
    da3_config_path=None,
    da3_weight_path=None,
    da3_model_name_or_path=None,
    feat_layer=-1,
    ref_view_strategy="saddle_balanced",
    patch_size=14,
    agg_arch,
    agg_config=None,
    aggregator_ckpt_path=None,
    strict=True,
):
    has_existing_model = da3_model is not None
    has_config_mode = da3_config_path is not None or da3_weight_path is not None
    has_name_mode = da3_model_name_or_path is not None
    if sum((has_existing_model, has_config_mode, has_name_mode)) != 1:
        raise ValueError("Provide exactly one DA3 source")

    if da3_model is None:
        da3_model = build_da3_model(
            da3_config_path=da3_config_path,
            da3_weight_path=da3_weight_path,
            da3_model_name_or_path=da3_model_name_or_path,
        )

    encoder = build_da3_encoder_adapter(
        da3_model,
        feat_layer=feat_layer,
        ref_view_strategy=ref_view_strategy,
        patch_size=patch_size,
    )
    aggregator = build_aggregator(agg_arch, agg_config=agg_config)
    if aggregator_ckpt_path is not None:
        load_aggregator_weights_from_salad_ckpt(aggregator, aggregator_ckpt_path, strict=strict)
    return VPRModel(encoder=encoder, aggregator=aggregator, agg_arch=agg_arch)


get_aggregator = build_aggregator
