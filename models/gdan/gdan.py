"""
GDANet: Learning Geometry-Disentangled Representation for Complementary
        Understanding of 3D Object Point Cloud

Paper: Xu et al., AAAI 2021 — https://arxiv.org/abs/2012.10921
Source: https://github.com/mutianxu/GDANet (no license)

Clean implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import (
    extract_local_features,
    geometry_disentangle,
    CrossAttentionModule,
)

from ..build import MODELS
from ..base_model import BasePointCloudModel

# Shared building blocks

class ConvBN2d(nn.Module):
    """Conv2d(kernel=1) + BatchNorm2d."""

    def __init__(self, in_ch: int, out_ch: int, momentum: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=True),
            nn.BatchNorm2d(out_ch, momentum=momentum),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ConvBN1d(nn.Module):
    """Conv1d(kernel=1) + BatchNorm1d."""

    def __init__(self, in_ch: int, out_ch: int, momentum: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, bias=True),
            nn.BatchNorm1d(out_ch, momentum=momentum),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# Stage blocks

class LocalEdgeConv(nn.Module):
    """
    Local feature extraction:
      k-NN edge features  →  two Conv2d layers  →  max-pool over neighbours.

    Args:
        in_ch:  input channels C
        out_ch: output channels
        k:      neighbourhood size
    """

    def __init__(self, in_ch: int, out_ch: int, k: int):
        super().__init__()
        self.k = k
        self.conv1 = ConvBN2d(in_ch * 2, out_ch)
        self.conv2 = ConvBN2d(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, N]
        Returns:
            [B, out_ch, N]
        """
        edge = extract_local_features(x, self.k)   # [B, 2C, N, k]
        h = F.relu(self.conv1(edge))
        h = F.relu(self.conv2(h))
        return h.max(dim=-1)[0]                    # [B, out_ch, N]

class SharpGentleBlock(nn.Module):
    """
    Geometry disentanglement + dual cross-attention + channel fusion.

    1. Split features into sharp (high-variation) and gentle (low-variation).
    2. Apply cross-attention from the full feature set to each component.
    3. Concatenate the two attended outputs and fuse with a 1D conv.

    Args:
        channels:   feature channels C
        num_select: M — points in each component
    """

    def __init__(self, channels: int, num_select: int):
        super().__init__()
        self.num_select = num_select
        self.sharp_attn = CrossAttentionModule(channels)
        self.gentle_attn = CrossAttentionModule(channels)
        self.fuse = nn.Sequential(
            ConvBN1d(channels * 2, channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, N]
        Returns:
            [B, C, N]
        """
        sharp, gentle = geometry_disentangle(x, self.num_select)
        y_s = self.sharp_attn(x, sharp.transpose(1, 2))    # [B, C, N]
        y_g = self.gentle_attn(x, gentle.transpose(1, 2))   # [B, C, N]
        return self.fuse(torch.cat([y_s, y_g], dim=1))       # [B, C, N]

# Main network

@MODELS.register_module()
class GDAN(BasePointCloudModel):
    """
    Geometry-Disentangled classifier for 3-D point cloud recognition.

    Architecture (three hierarchical stages):
      Stage 1  local edge conv → sharp/gentle attention → z1
      Stage 2  local edge conv (skip: pts + z1) → sharp/gentle attention → z2
      Stage 3  local edge conv (skip: pts + z1 + z2) → projection → z3
      Global   concat(z1, z2, z3) → 1D conv → max+mean pool → MLP
    """

    def __init__(self, config, **kwargs):
        super().__init__(config)

        input_channels = config.get('channels_input', 3)
        k = config.get('k', 30)
        num_select = config.get('num_select', 256)
        dropout = config.get('dropout', 0.4)

        self.emb_dims = 512

        # Stage 1
        self.stage1_local = LocalEdgeConv(input_channels, 64, k)
        self.stage1_geo = SharpGentleBlock(64, num_select)

        # Stage 2 — receives original points + stage-1 output
        self.stage2_local = LocalEdgeConv(input_channels + 64, 64, k)
        self.stage2_geo = SharpGentleBlock(64, num_select)

        # Stage 3 — receives all previous, no geometry disentanglement
        self.stage3_local = LocalEdgeConv(input_channels + 64 + 64, 128, k)
        self.stage3_proj = nn.Sequential(
            ConvBN1d(128, 128),
            nn.ReLU(inplace=True),
        )

        # Global aggregation
        self.global_conv = nn.Sequential(
            ConvBN1d(64 + 64 + 128, 512),
            nn.ReLU(inplace=True),
        )

        # MLP head  (1024 = 512 max-pool + 512 mean-pool)
        self.head = nn.Sequential(
            nn.Linear(1024, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, self.num_classes, bias=True),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pre-classifier global feature (B, 1024)."""
        if x.dim() == 3 and x.shape[1] > x.shape[2]:
            x = x.transpose(1, 2).contiguous()
        pts = x
        f1 = self.stage1_local(pts)
        z1 = self.stage1_geo(f1)
        f2 = self.stage2_local(torch.cat([pts, z1], 1))
        z2 = self.stage2_geo(f2)
        f3 = self.stage3_local(torch.cat([pts, z1, z2], dim=1))
        z3 = self.stage3_proj(f3)
        g = self.global_conv(torch.cat([z1, z2, z3], dim=1))
        return torch.cat([g.max(dim=-1)[0], g.mean(dim=-1)], dim=1)

    def cls_head(self, feat: torch.Tensor) -> torch.Tensor:
        return self.head(feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cls_head(self.forward_features(x))
