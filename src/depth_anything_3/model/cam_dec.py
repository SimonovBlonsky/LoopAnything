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

import torch
import torch.nn as nn


class CameraDec(nn.Module):
    def __init__(self, dim_in=1536):
        super().__init__()
        output_dim = dim_in
        # 论文 3.1 末尾的轻量 camera head：
        # 这里只回归每个视角 1 个 token 对应的相机参数，计算量很小。
        self.backbone = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
        )
        self.fc_t = nn.Linear(output_dim, 3)
        self.fc_qvec = nn.Linear(output_dim, 4)
        self.fc_fov = nn.Sequential(nn.Linear(output_dim, 2), nn.ReLU())

    def forward(self, feat, camera_encoding=None, *args, **kwargs):
        B, N = feat.shape[:2]
        # feat 一般来自最后一层的 camera token，形状是 [B, S, C]。
        feat = feat.reshape(B * N, -1)
        feat = self.backbone(feat)

        # 平移始终由当前 token 回归。
        out_t = self.fc_t(feat.float()).reshape(B, N, 3)
        if camera_encoding is None:
            # 默认情况：旋转和视场角都由 camera head 自己预测。
            out_qvec = self.fc_qvec(feat.float()).reshape(B, N, 4)
            out_fov = self.fc_fov(feat.float()).reshape(B, N, 2)
        else:
            # 如果外部已经提供了 camera encoding，可直接复用其中的旋转/FOV，
            # 仅保留当前 head 对平移的更新。
            out_qvec = camera_encoding[..., 3:7]
            out_fov = camera_encoding[..., -2:]

        # 输出仍保持 DA3 内部统一的 9 维 pose encoding 格式。
        pose_enc = torch.cat([out_t, out_qvec, out_fov], dim=-1)
        return pose_enc
