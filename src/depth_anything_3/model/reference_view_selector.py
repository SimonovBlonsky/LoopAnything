# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Reference View Selection Strategies

This module provides different strategies for selecting a reference view
from multiple input views in multi-view depth estimation.
"""

import torch
from typing import Literal


RefViewStrategy = Literal["first", "middle", "saddle_balanced", "saddle_sim_range"]


def select_reference_view(
    x: torch.Tensor,
    strategy: RefViewStrategy = "saddle_balanced",
) -> torch.Tensor:
    """
    Select a reference view from multiple views using the specified strategy.
    
    Args:
        x: Input tensor of shape (B, S, N, C) where
           B = batch size
           S = number of views
           N = number of tokens
           C = channel dimension
        strategy: Selection strategy, one of:
            - "first": Always select the first view
            - "middle": Select the middle view
            - "saddle_balanced": Select view with balanced features across multiple metrics
            - "saddle_sim_range": Select view with largest similarity range
    
    Returns:
        b_idx: Tensor of shape (B,) containing the selected view index for each batch
    """
    B, S, N, C = x.shape

    # For single view, no reordering needed
    if S <= 1:
        return torch.zeros(B, dtype=torch.long, device=x.device)
    
    # Simple position-based strategies
    if strategy == "first":
        return torch.zeros(B, dtype=torch.long, device=x.device)
    
    elif strategy == "middle":
        return torch.full((B,), S // 2, dtype=torch.long, device=x.device)
    
    # 这里默认使用每个视角的第 0 个 token（通常是 cls / camera token 混合语义）
    # 作为整张图的全局摘要，用它来判断谁更适合作“参考视角”。
    img_class_feat = x[:, :, 0] / x[:, :, 0].norm(dim=-1, keepdim=True)  # B S C

    if strategy == "saddle_balanced":
        # “saddle_balanced” 不是挑最中心或最极端的视角，
        # 而是同时考虑相似度、特征范数、特征方差，选一个“信息量适中”的参考视角。
        sim = torch.matmul(img_class_feat, img_class_feat.transpose(1, 2))  # B S S
        sim_no_diag = sim - torch.eye(S, device=sim.device).unsqueeze(0)
        sim_score = sim_no_diag.sum(dim=-1) / (S - 1)  # B S
        
        feat_norm = x[:, :, 0].norm(dim=-1)  # B S
        feat_var = img_class_feat.var(dim=-1)  # B S
        
        # Normalize all metrics to [0, 1]
        def normalize_metric(metric):
            min_val = metric.min(dim=1, keepdim=True).values
            max_val = metric.max(dim=1, keepdim=True).values
            return (metric - min_val) / (max_val - min_val + 1e-8)
        
        sim_score_norm = normalize_metric(sim_score)
        norm_norm = normalize_metric(feat_norm)
        var_norm = normalize_metric(feat_var)
        
        # 每个指标都归一化后，选最接近“中间态”的视角。
        balance_score = (
            (sim_score_norm - 0.5).abs() +
            (norm_norm - 0.5).abs() +
            (var_norm - 0.5).abs()
        )
        b_idx = balance_score.argmin(dim=1)
        
    elif strategy == "saddle_sim_range":
        # 选“和某些视角很像、和另一些视角差异也很大”的视角，
        # 作为跨视角建模时的锚点。
        sim = torch.matmul(img_class_feat, img_class_feat.transpose(1, 2))  # B S S
        sim_no_diag = sim - torch.eye(S, device=sim.device).unsqueeze(0)
        
        sim_max = sim_no_diag.max(dim=-1).values  # B S
        sim_min = sim_no_diag.min(dim=-1).values  # B S
        sim_range = sim_max - sim_min
        b_idx = sim_range.argmax(dim=1)
    
    else:
        raise ValueError(
            f"Unknown reference view selection strategy: {strategy}. "
            f"Must be one of: 'first', 'middle', 'saddle_balanced', 'saddle_sim_range'"
        )
    
    return b_idx


def reorder_by_reference(
    x: torch.Tensor,
    b_idx: torch.Tensor,
) -> torch.Tensor:
    """
    Reorder views to place the selected reference view first.
    
    Args:
        x: Input tensor of shape (B, S, N, C)
        b_idx: Reference view indices of shape (B,)
    
    Returns:
        Reordered tensor with reference view at position 0
    
    Example:
        If b_idx = [2] and S = 5 (views [0,1,2,3,4]),
        result order is [2,0,1,3,4] (ref_idx first, then others in order)
    """
    B, S = x.shape[0], x.shape[1]

    # For single view, no reordering needed
    if S <= 1:
        return x
    
    # Create position indices: (B, S) where each row is [0, 1, 2, ..., S-1]
    positions = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)  # B S
    
    # 重排规则非常简单：
    # 把参考视角挪到位置 0，其余视角保持相对顺序不变。
    
    b_idx_expanded = b_idx.unsqueeze(1)  # B 1
    
    # Create the reordering indices
    # For positions 1 to ref_idx: map to indices 0 to ref_idx-1 (shift by -1)
    # For positions > ref_idx: keep the same
    reorder_indices = positions.clone()
    reorder_indices = torch.where(
        (positions > 0) & (positions <= b_idx_expanded),
        positions - 1,
        positions
    )
    # Set position 0 to ref_idx
    reorder_indices[:, 0] = b_idx
    
    # Gather using advanced indexing
    batch_indices = torch.arange(B, device=x.device).unsqueeze(1)  # B 1
    x_reordered = x[batch_indices, reorder_indices]
    
    return x_reordered


def restore_original_order(
    x: torch.Tensor,
    b_idx: torch.Tensor,
) -> torch.Tensor:
    """
    Restore original view order after processing.
    
    Args:
        x: Reordered tensor of shape (B, S, ...)
        b_idx: Original reference view indices of shape (B,)
    
    Returns:
        Tensor with original view order restored
    
    Example:
        If original order was [0, 1, 2, 3, 4] and b_idx=2,
        reordered becomes [2, 0, 1, 3, 4] (reference at position 0),
        restore should return [0, 1, 2, 3, 4] (original order).
    """
    B, S = x.shape[0], x.shape[1]

    # For single view, no restoration needed
    if S <= 1:
        return x
    
    # Create target position indices: (B, S) where each row is [0, 1, 2, ..., S-1]
    target_positions = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)  # B S
    
    # 这是 reorder 的逆操作：
    # 把“位置 0 的参考视角”放回原来的索引位置。
    
    b_idx_expanded = b_idx.unsqueeze(1)  # B 1
    
    # Create the restore indices
    restore_indices = torch.where(
        target_positions < b_idx_expanded,
        target_positions + 1,  # Positions before ref_idx come from current position + 1
        target_positions        # Positions after ref_idx stay the same
    )
    # Target position = ref_idx comes from current position 0
    # Use scatter to set specific positions
    restore_indices = torch.scatter(
        restore_indices,
        dim=1,
        index=b_idx_expanded,
        src=torch.zeros_like(b_idx_expanded)
    )
    
    # Gather using advanced indexing
    batch_indices = torch.arange(B, device=x.device).unsqueeze(1)  # B 1
    x_restored = x[batch_indices, restore_indices]
    
    return x_restored
