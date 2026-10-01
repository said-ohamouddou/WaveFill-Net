"""
GDANet: Learning Geometry-Disentangled Representation for Complementary
        Understanding of 3D Object Point Cloud

Paper: Xu et al., AAAI 2021 — https://arxiv.org/abs/2012.10921
Source: https://github.com/mutianxu/GDANet (no license)

Clean implementation.
"""

import torch
import torch.nn as nn
from typing import Tuple

from pointnet2_ops.pointnet2_utils import grouping_operation


# KNN + local feature helpers

def _knn_indices(points: torch.Tensor, k: int) -> torch.Tensor:
    """
    Return k-nearest-neighbour indices for each point.

    Args:
        points: [B, C, N]
        k:      number of neighbours

    Returns:
        [B, N, k] int32 indices
    """
    pts_t = points.transpose(1, 2)                          # [B, N, C]
    dists = torch.cdist(pts_t, pts_t)                       # [B, N, N]
    return dists.topk(k, dim=-1, largest=False)[1].int()    # [B, N, k]


def extract_local_features(points: torch.Tensor, k: int) -> torch.Tensor:
    """
    Build edge features for each point from its k nearest neighbours.

    Each edge is represented as (neighbour − centre, neighbour),
    yielding a [B, 2C, N, k] output ready for Conv2d.

    Args:
        points: [B, C, N]
        k:      neighbourhood size

    Returns:
        [B, 2C, N, k]
    """
    idx = _knn_indices(points, k)                               # [B, N, k]
    grouped = grouping_operation(points.contiguous(), idx)      # [B, C, N, k]
    centre = points.unsqueeze(-1).expand_as(grouped)
    return torch.cat([grouped - centre, grouped], dim=1)        # [B, 2C, N, k]


def extract_local_features_with_normals(
    points: torch.Tensor,
    normals: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """
    Like extract_local_features but also appends surface normals.

    Returns [B, 3C, N, k].
    """
    idx = _knn_indices(points, k)
    grouped_pts = grouping_operation(points.contiguous(), idx)   # [B, C, N, k]
    grouped_nrm = grouping_operation(normals.contiguous(), idx)   # [B, C, N, k]
    centre = points.unsqueeze(-1).expand_as(grouped_pts)
    return torch.cat([grouped_pts - centre, grouped_pts, grouped_nrm], dim=1)


# Geometry disentanglement

def geometry_disentangle(
    features: torch.Tensor,
    num_select: int,
    tau: float = 0.2,
    sigma: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split a feature set into high-variation (sharp) and low-variation (gentle)
    subsets following the Geometry Disentangle Module in GDANet.

    Paper equations implemented here:
      Eq. 2 — Gaussian adjacency with proximity mask:
                  W_ij = exp(−‖xᵢ − xⱼ‖² / σ²)  if  ‖xᵢ − xⱼ‖ < τ  else 0
      Row-normalise → stochastic adjacency A  (each row sums to 1)
      Eq. 4 — Smoothed feature:  nᵢ = Σⱼ Aᵢⱼ xⱼ   (matrix form: N = A X)
      Eq. 5 — Variation score:   πᵢ = ‖xᵢ − nᵢ‖²

    Args:
        features:   [B, C, N]
        num_select: M — points to keep in each component
        tau:        proximity threshold (distance mask)
        sigma:      Gaussian bandwidth

    Returns:
        sharp:  [B, M, C]   highest-variation points
        gentle: [B, M, C]   lowest-variation points
    """
    B, C, N = features.shape
    pts = features.transpose(1, 2).contiguous()       # [B, N, C]

    # Pairwise L2 distances
    dists = torch.cdist(pts, pts)                     # [B, N, N]

    # Gaussian adjacency + proximity mask (Eq. 2)
    mask = (dists < tau).float()
    W = torch.exp(-(dists ** 2) / (sigma ** 2)) * mask

    # Row-normalise → stochastic adjacency
    A = W / W.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    # Smoothed features (Eq. 4)
    smoothed = torch.bmm(A, pts)                      # [B, N, C]

    # Variation score (Eq. 5)
    scores = (pts - smoothed).pow(2).sum(dim=-1)      # [B, N]

    # Select top-M (sharp) and bottom-M (gentle)
    b = torch.arange(B, device=features.device).unsqueeze(1)
    sharp = pts[b, scores.topk(num_select, largest=True)[1]]   # [B, M, C]
    gentle = pts[b, scores.topk(num_select, largest=False)[1]]   # [B, M, C]

    return sharp, gentle


# Cross-attention

class CrossAttentionModule(nn.Module):
    """
    Cross-attention between a query feature set and a context set.

    Uses scaled dot-product attention (Vaswani et al., 2017):
      Attention(Q, K, V) = softmax(Q Kᵀ / √d) V

    The output projection is zero-initialised so the module starts as
    an identity (residual passes through unchanged), which improves
    training stability.

    Args:
        channels:  number of input/output channels C
        reduction: channel reduction factor for the inner dimension
    """

    def __init__(self, channels: int, reduction: int = 2):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.scale = mid ** -0.5

        self.to_q = nn.Conv1d(channels, mid, 1, bias=False)
        self.to_k = nn.Conv1d(channels, mid, 1, bias=False)
        self.to_v = nn.Conv1d(channels, mid, 1, bias=False)
        self.out_proj = nn.Sequential(
            nn.Conv1d(mid, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )
        # Zero-init output projection → identity at init
        nn.init.zeros_(self.out_proj[0].weight)
        nn.init.zeros_(self.out_proj[1].weight)
        nn.init.zeros_(self.out_proj[1].bias)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query:   [B, C, N_q]
            context: [B, C, N_c]

        Returns:
            [B, C, N_q]
        """
        q = self.to_q(query)    # [B, mid, N_q]
        k = self.to_k(context)  # [B, mid, N_c]
        v = self.to_v(context)  # [B, mid, N_c]

        # Scaled dot-product attention with softmax
        attn = torch.bmm(q.transpose(1, 2), k) * self.scale   # [B, N_q, N_c]
        attn = attn.softmax(dim=-1)

        out = torch.bmm(attn, v.transpose(1, 2))   # [B, N_q, mid]
        out = self.out_proj(out.transpose(1, 2))    # [B, C,   N_q]

        return out + query                          # residual
