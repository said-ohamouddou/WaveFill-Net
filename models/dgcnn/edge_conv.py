"""
Dynamic Graph CNN for Learning on Point Clouds

Paper: ACM Transactions on Graphics 2019
Authors: Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E. Sarma, Michael M. Bronstein, Justin M. Solomon
Source: Implementation adapted from: https://github.com/WangYueFt/dgcnn
License: MIT
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    """
    Find k nearest neighbors

    Args:
        x: Input features [B, C, N]
        k: Number of neighbors

    Returns:
        idx: Indices of k nearest neighbors [B, N, k]
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # [B, N, N]
    xx = torch.sum(x**2, dim=1, keepdim=True)  # [B, 1, N]
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # [B, N, N]

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # [B, N, k]
    return idx


def get_graph_feature(x, k=20, idx=None):
    """
    Construct graph features using k-NN

    Args:
        x: Input point features [B, C, N]
        k: Number of neighbors
        idx: Pre-computed neighbor indices (optional)

    Returns:
        Graph features [B, 2*C, N, k]
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    if idx is None:
        idx = knn(x, k=k)  # [B, N, k]

    device = torch.device('cuda' if x.is_cuda else 'cpu')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # [B, N, C]
    feature = x.view(batch_size * num_points, -1)[idx, :]  # [B*N*k, C]
    feature = feature.view(batch_size, num_points, k, num_dims)  # [B, N, k, C]
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # [B, N, k, C]

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()  # [B, 2*C, N, k]

    return feature


class EdgeConv(nn.Module):
    """
    EdgeConv layer for dynamic graph construction
    """

    def __init__(self, in_channels, out_channels, k=20, aggr='max'):
        super(EdgeConv, self).__init__()
        self.k = k
        self.aggr = aggr

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input features [B, C, N]

        Returns:
            Output features [B, out_channels, N]
        """
        x = get_graph_feature(x, k=self.k)  # [B, 2*C, N, k]
        x = self.conv(x)  # [B, out_channels, N, k]

        if self.aggr == 'max':
            x = x.max(dim=-1, keepdim=False)[0]  # [B, out_channels, N]
        elif self.aggr == 'mean':
            x = x.mean(dim=-1, keepdim=False)
        else:
            raise ValueError(f"Aggregation {self.aggr} not supported")

        return x
