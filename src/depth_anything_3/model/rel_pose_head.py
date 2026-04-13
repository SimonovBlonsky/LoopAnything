"""Pairwise relative pose regression head for DA3 camera tokens.

Takes two camera tokens (one per view) from the DA3 backbone's last layer
and predicts the 4x4 relative pose matrix between them.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelPoseHead(nn.Module):
    """Predict pairwise relative pose from two DA3 camera tokens.

    Architecture:
        concat(cam_token_a, cam_token_b) -> MLP -> fc_rot(9D SVD) + fc_t(3)

    The 9D rotation representation is projected to SO(3) via SVD
    orthogonalization (same method as Reloc3r's PoseHead).
    """

    def __init__(self, token_dim: int = 2048):
        super().__init__()
        in_dim = token_dim * 2  # concatenation of two camera tokens

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
        )
        self.fc_rot = nn.Linear(1024, 9)
        self.fc_t = nn.Linear(1024, 3)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def svd_orthogonalize(m: torch.Tensor) -> torch.Tensor:
        """Project a batch of 3x3 matrices onto SO(3) via SVD.

        Adapted from Reloc3r (reloc3r/pose_head.py:63-82).

        Args:
            m: [B, 3, 3] matrices.

        Returns:
            [B, 3, 3] rotation matrices in SO(3).
        """
        if m.dim() < 3:
            m = m.reshape(-1, 3, 3)
        m_t = torch.transpose(F.normalize(m, p=2, dim=-1), dim0=-1, dim1=-2)
        u, _s, v = torch.svd(m_t)
        det = torch.det(torch.matmul(v, u.transpose(-2, -1)))
        r = torch.matmul(
            torch.cat([v[:, :, :-1], v[:, :, -1:] * det.view(-1, 1, 1)], dim=2),
            u.transpose(-2, -1),
        )
        return r

    def forward(
        self,
        cam_token_a: torch.Tensor,
        cam_token_b: torch.Tensor,
    ) -> torch.Tensor:
        """Predict relative pose from view A to view B.

        The output P satisfies: point_in_B = P @ point_in_A, i.e.
        P = inv(c2w_B) @ c2w_A  (the "A-to-B" transform).

        Args:
            cam_token_a: [B, C] camera token for view A.
            cam_token_b: [B, C] camera token for view B.

        Returns:
            [B, 4, 4] relative pose matrix.
        """
        feat = torch.cat([cam_token_a, cam_token_b], dim=-1)  # [B, 2C]
        feat = self.mlp(feat)  # [B, 1024]

        out_t = self.fc_t(feat)  # [B, 3]
        out_r = self.fc_rot(feat)  # [B, 9]

        # SVD orthogonalization must run in FP32 for numerical stability.
        with torch.cuda.amp.autocast(enabled=False):
            rot = self.svd_orthogonalize(out_r.float().reshape(-1, 3, 3))

        B = cam_token_a.shape[0]
        pose = torch.zeros(B, 4, 4, device=feat.device, dtype=feat.dtype)
        pose[:, :3, :3] = rot.to(feat.dtype)
        pose[:, :3, 3] = out_t
        pose[:, 3, 3] = 1.0
        return pose
