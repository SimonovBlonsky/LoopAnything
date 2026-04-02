from __future__ import annotations

import torch
import torch.nn as nn


class _CrossAttentionLayer(nn.Module):
    """Single cross-attention layer: Q attends to K/V with pre-norm and FFN."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: [B, Nq, C]
            kv: [B, Nkv, C]
        Returns:
            [B, Nq, C]
        """
        q_normed = self.norm_q(query)
        kv_normed = self.norm_kv(kv)
        attn_out, _ = self.cross_attn(q_normed, kv_normed, kv_normed)
        query = query + attn_out
        query = query + self.ffn(self.norm_ffn(query))
        return query


class CrossViewFusion(nn.Module):
    """Cross-view fusion via cross-attention between query and candidate features.

    Uses zero-initialized residual gates so fusion output starts as identity
    (the original query tokens pass through unchanged). This ensures:
    - Pretrained downstream modules (da3_head, cam_dec) receive valid inputs from the start
    - The fusion contribution is learned gradually during training
    - No NaN from randomly initialized attention producing garbage outputs

    Default: unidirectional (query attends to candidates).
    Optional: bidirectional (candidates also attend to query).
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 2,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.q2c_layers = nn.ModuleList([
            _CrossAttentionLayer(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        if bidirectional:
            self.c2q_layers = nn.ModuleList([
                _CrossAttentionLayer(embed_dim, num_heads, dropout) for _ in range(num_layers)
            ])

        self.cam_cross_attn = _CrossAttentionLayer(embed_dim, num_heads, dropout)

        # Zero-initialized gates: at init, fusion output = original query tokens
        self.patch_gate = nn.Parameter(torch.zeros(1))
        self.cam_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        query_patch_tokens: torch.Tensor,
        query_camera_token: torch.Tensor,
        candidate_patch_tokens: torch.Tensor,
        candidate_camera_tokens: torch.Tensor,
        candidate_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_patch_tokens: [B, N_tokens, C]
            query_camera_token: [B, C]
            candidate_patch_tokens: [B, K, N_tokens, C]
            candidate_camera_tokens: [B, K, C]
            candidate_weights: [B, K] retrieval weights

        Returns:
            enhanced_patch_tokens: [B, N_tokens, C]
            enhanced_camera_token: [B, C]
        """
        B, K, N, C = candidate_patch_tokens.shape

        # Weight candidate features by retrieval scores: [B, K, 1, 1] * [B, K, N, C]
        w = candidate_weights.unsqueeze(-1).unsqueeze(-1)  # [B, K, 1, 1]
        weighted_cand_patches = candidate_patch_tokens * w  # [B, K, N, C]

        # Flatten candidates into single KV sequence: [B, K*N, C]
        kv_patches = weighted_cand_patches.reshape(B, K * N, C)

        # Query-to-candidate cross attention on patch tokens
        fused_patches = query_patch_tokens
        for layer in self.q2c_layers:
            fused_patches = layer(fused_patches, kv_patches)

        if self.bidirectional:
            enhanced_kv = kv_patches
            for layer in self.c2q_layers:
                enhanced_kv = layer(enhanced_kv, fused_patches)

        # Gated residual: start as identity, gradually blend in fusion
        patch_delta = fused_patches - query_patch_tokens
        enhanced_patches = query_patch_tokens + self.patch_gate * patch_delta

        # Camera token fusion in FP32 to avoid NaN with small KV sequence (K=10)
        with torch.cuda.amp.autocast(enabled=False):
            w_cam = candidate_weights.float().unsqueeze(-1)  # [B, K, 1]
            weighted_cand_cams = candidate_camera_tokens.float() * w_cam  # [B, K, C]
            query_cam_seq = query_camera_token.float().unsqueeze(1)  # [B, 1, C]
            fused_cam_seq = self.cam_cross_attn(query_cam_seq, weighted_cand_cams)  # [B, 1, C]
            fused_cam = fused_cam_seq.squeeze(1)  # [B, C]

        # Gated residual for camera token
        cam_delta = fused_cam.to(query_camera_token.dtype) - query_camera_token
        enhanced_camera_token = query_camera_token + self.cam_gate * cam_delta

        return enhanced_patches, enhanced_camera_token


def build_cross_view_fusion(config: dict) -> CrossViewFusion:
    """Build CrossViewFusion from config dict."""
    return CrossViewFusion(
        embed_dim=config.get("embed_dim", 768),
        num_heads=config.get("num_heads", 8),
        num_layers=config.get("num_layers", 2),
        bidirectional=config.get("bidirectional", False),
        dropout=config.get("dropout", 0.0),
    )
