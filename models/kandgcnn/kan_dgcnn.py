"""
Introducing the Short-Time Fourier Kolmogorov Arnold Network: A Dynamic Graph CNN Approach for Tree Species Classification in 3D Point Clouds

Paper: Pattern Recognition 2026
Authors: Said Ohamouddou, Abdellatif El Afia, Mohamed Ohamouddou, Rafik Lasri, Hanaa El Afia, Raddouane Chiheb
Source: Implementation adapted from: https://github.com/said-ohamouddou/STFT-KAN-liteDGCNN
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kan_conv import KANConv2DLayer, KANConv1DLayer
from .kan import KAN
from ..build import MODELS
from ..base_model import BasePointCloudModel, print_model_info

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    if idx is None:
        idx = knn(x, k=k)

    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature

@MODELS.register_module()
class KANDGCNN(BasePointCloudModel):
    def __init__(self, config, num_classes=40, channels=3, **kwargs):
        super(KANDGCNN, self).__init__(config, num_classes)

        channels = config.channels_input
        self.in_channels = channels
        self.k = config.k
        self.emb_dims = config.emb_dims

        # get_graph_feature emits 2*channels; first conv must match.
        self.conv1 = nn.Sequential(KANConv2DLayer(2 * channels, 128, kernel_size=1))
        self.conv2 = nn.Sequential(KANConv2DLayer(64 * 2, 64, kernel_size=1))

        self.conv5 = nn.Sequential(KANConv1DLayer(128, self.emb_dims, kernel_size=1))

        self.linear1 = KAN([self.emb_dims * 2, self.num_classes])

    def forward_features(self, x):
        """Return the pre-classifier global feature (B, emb_dims*2)."""
        if x.dim() == 3 and x.shape[1] != self.in_channels and x.shape[-1] == self.in_channels:
            x = x.permute(0, 2, 1)
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        x = self.conv5(x1)
        m = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        a = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        return torch.cat((m, a), 1)

    def cls_head(self, feat):
        return self.linear1(feat)

    def forward(self, x):
        return self.cls_head(self.forward_features(x))
