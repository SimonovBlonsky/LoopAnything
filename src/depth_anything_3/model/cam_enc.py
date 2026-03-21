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

import torch.nn as nn

from depth_anything_3.model.utils.attention import Mlp
from depth_anything_3.model.utils.block import Block
from depth_anything_3.model.utils.transform import extri_intri_to_pose_encoding
from depth_anything_3.utils.geometry import affine_inverse


class CameraEnc(nn.Module):
    """
    CameraHead predicts camera parameters from token representations using iterative refinement.

    It applies a series of transformer blocks (the "trunk") to dedicated camera tokens.
    """

    def __init__(
        self,
        dim_out: int = 1024,
        dim_in: int = 9,
        trunk_depth: int = 4,
        target_dim: int = 9,
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        **kwargs,
    ):
        super().__init__()
        self.target_dim = target_dim
        self.trunk_depth = trunk_depth
        # 论文 3.2 中的 camera condition injection：
        # 先把 (t, q, fov) 编成 token，再用一个很轻量的 transformer
        # 做少量上下文建模，使它的分布更接近视觉 token。
        self.trunk = nn.Sequential(
            *[
                Block(
                    dim=dim_out,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                )
                for _ in range(trunk_depth)
            ]
        )
        self.token_norm = nn.LayerNorm(dim_out)
        self.trunk_norm = nn.LayerNorm(dim_out)
        self.pose_branch = Mlp(
            in_features=dim_in,
            hidden_features=dim_out // 2,
            out_features=dim_out,
            drop=0,
        )

    def forward(
        self,
        ext,
        ixt,
        image_size,
    ) -> tuple:
        # 输入 ext 是 w2c，而 pose encoding 统一使用更直观的 c2w 表示。
        c2ws = affine_inverse(ext)
        # 把外参/内参压成 9 维向量：(t[3], quat[4], fov_h, fov_w)。
        # 这正对应论文里的 camera token 条件注入格式。
        pose_encoding = extri_intri_to_pose_encoding(
            c2ws,
            ixt,
            image_size,
        )
        # MLP 先把几何参数投到 backbone 的 embedding 维度，
        # 后续主干会把它当作每个 view 的第一个 token 使用。
        pose_tokens = self.pose_branch(pose_encoding)
        pose_tokens = self.token_norm(pose_tokens)
        pose_tokens = self.trunk(pose_tokens)
        pose_tokens = self.trunk_norm(pose_tokens)
        return pose_tokens
