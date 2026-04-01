from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import torch
import yaml

from depth_anything_3.api import DepthAnything3
from depth_anything_3.model import VPRaggregators
from depth_anything_3.model.cross_view_fusion import build_cross_view_fusion
from depth_anything_3.model.retrieval_strategy import build_retrieval_strategy
from depth_anything_3.model.unified_pipeline import UnifiedPipeline
from depth_anything_3.model.vpr_feature_adapter import (
    DualBranchFeatureAdapter,
    IdentityFeatureAdapter,
    PatchOnlyFeatureAdapter,
)


# ---------------------------------------------------------------------------
# VPR component builders (migrated from vpr_helper.py)
# ---------------------------------------------------------------------------


def build_aggregator(agg_arch: str, agg_config: dict | None = None):
    """Build a VPR aggregator by name."""
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


def build_feature_adapter(adapter_arch: str | None = None, adapter_config: dict | None = None):
    """Build a feature adapter by name."""
    adapter_config = {} if adapter_config is None else dict(adapter_config)
    if adapter_arch is None:
        return IdentityFeatureAdapter()
    name = str(adapter_arch).lower()
    if name == "identity":
        return IdentityFeatureAdapter()
    if name == "patch_only":
        return PatchOnlyFeatureAdapter(**adapter_config)
    if name == "dual_branch":
        return DualBranchFeatureAdapter(**adapter_config)
    raise ValueError(f"Unsupported feature adapter: {adapter_arch}")


def extract_prefixed_state_dict(
    state_dict: Mapping[str, torch.Tensor], prefixes: Iterable[str],
) -> dict[str, torch.Tensor]:
    """Extract keys matching any prefix, stripping the prefix."""
    extracted = {}
    for key, value in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                extracted[key[len(prefix):]] = value
                break
    return extracted


def _unwrap_checkpoint_state_dict(checkpoint):
    """Unwrap a Lightning/PyTorch checkpoint to its raw state_dict."""
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Expected checkpoint to contain a mapping of parameters")
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], Mapping):
        return checkpoint["state_dict"]
    if "model" in checkpoint and isinstance(checkpoint["model"], Mapping):
        return checkpoint["model"]
    return checkpoint


VPR_ADAPTER_PREFIXES = (
    "feature_adapter.",
    "vpr_model.feature_adapter.",
    "module.vpr_model.feature_adapter.",
)
VPR_AGGREGATOR_PREFIXES = (
    "aggregator.",
    "vpr_model.aggregator.",
    "module.vpr_model.aggregator.",
    "model.aggregator.",
    "module.aggregator.",
)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _load_vpr_weights(feature_adapter, aggregator, vpr_checkpoint_path):
    """Load VPR checkpoint into feature_adapter and aggregator."""
    checkpoint = torch.load(Path(vpr_checkpoint_path), map_location="cpu")
    state_dict = _unwrap_checkpoint_state_dict(checkpoint)

    adapter_sd = extract_prefixed_state_dict(state_dict, VPR_ADAPTER_PREFIXES)
    if adapter_sd:
        feature_adapter.load_state_dict(adapter_sd, strict=True)

    agg_sd = extract_prefixed_state_dict(state_dict, VPR_AGGREGATOR_PREFIXES)
    if agg_sd:
        aggregator.load_state_dict(agg_sd, strict=True)


def _apply_freeze(pipeline: UnifiedPipeline, freeze_config: dict):
    """Freeze modules according to config."""
    if freeze_config.get("backbone", True):
        pipeline.da3_backbone.requires_grad_(False)
        pipeline.da3_backbone.eval()

    if freeze_config.get("vpr", True):
        pipeline.feature_adapter.requires_grad_(False)
        pipeline.feature_adapter.eval()
        pipeline.aggregator.requires_grad_(False)
        pipeline.aggregator.eval()

    if freeze_config.get("fusion", False):
        pipeline.cross_view_fusion.requires_grad_(False)
        pipeline.cross_view_fusion.eval()

    if freeze_config.get("head", False):
        pipeline.da3_head.requires_grad_(False)
        pipeline.da3_head.eval()
        pipeline.cam_dec.requires_grad_(False)
        pipeline.cam_dec.eval()


def build_unified_pipeline(config: dict, device: str = "cpu") -> UnifiedPipeline:
    """Build UnifiedPipeline from config dict.

    Args:
        config: dict with keys: da3_model_name_or_path, vpr_checkpoint,
                aux_layer, freeze, retrieval, cross_view_fusion,
                feature_adapter_arch, feature_adapter_config,
                agg_arch, agg_config.
    """
    model_config = config.get("model", config)

    # 1. Load DA3 model and extract backbone + head + cam_dec
    da3_model_name = model_config.get("da3_model_name_or_path", "depth-anything/DA3-BASE")
    da3_wrapper = DepthAnything3.from_pretrained(da3_model_name)
    da3_net = da3_wrapper.model  # DepthAnything3Net

    backbone = da3_net.backbone
    da3_head = da3_net.head
    cam_dec = da3_net.cam_dec

    # 2. Build VPR components
    agg_arch = model_config.get("agg_arch", "salad")
    agg_config = model_config.get("agg_config", {"num_channels": 768, "num_clusters": 16, "cluster_dim": 32, "token_dim": 32})
    aggregator = build_aggregator(agg_arch, agg_config=agg_config)

    adapter_arch = model_config.get("feature_adapter_arch", "patch_only")
    adapter_config = model_config.get("feature_adapter_config", {"channels": 768})
    feature_adapter = build_feature_adapter(adapter_arch, adapter_config=adapter_config)

    # Load VPR checkpoint if provided
    vpr_ckpt = model_config.get("vpr_checkpoint")
    if vpr_ckpt and Path(vpr_ckpt).is_file():
        _load_vpr_weights(feature_adapter, aggregator, vpr_ckpt)

    # 3. Build new components
    retrieval_config = model_config.get("retrieval", {"strategy": "soft_attention", "top_k": 10, "temperature": 1.0})
    retrieval_strategy = build_retrieval_strategy(retrieval_config)

    fusion_config = model_config.get("cross_view_fusion", {"embed_dim": 1536, "num_heads": 8, "num_layers": 2})
    cross_view_fusion = build_cross_view_fusion(fusion_config)

    aux_layer = model_config.get("aux_layer", 5)

    # 4. Assemble pipeline
    pipeline = UnifiedPipeline(
        da3_backbone=backbone,
        feature_adapter=feature_adapter,
        aggregator=aggregator,
        retrieval_strategy=retrieval_strategy,
        cross_view_fusion=cross_view_fusion,
        da3_head=da3_head,
        cam_dec=cam_dec,
        aux_layer=aux_layer,
    )

    # 5. Apply freeze
    freeze_config = model_config.get("freeze", {"backbone": True, "vpr": True, "fusion": False, "head": False})
    _apply_freeze(pipeline, freeze_config)

    pipeline.to(device)
    return pipeline
