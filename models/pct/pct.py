"""
PCT: Point Cloud Transformer

Paper: Computational Visual Media 2021
Authors: Meng-Hao Guo, Jun-Xiong Cai, Zheng-Ning Liu, Tai-Jiang Mu, Ralph R. Martin, Shi-Min Hu
Source: Implementation adapted from: https://github.com/MenghaoGuo/PCT
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pct_layers import Point_Transformer_Last, Local_op
from .pct_utils import sample_and_group
from ..build import MODELS
from ..base_model import BasePointCloudModel, print_model_info

@MODELS.register_module()
class PCT(BasePointCloudModel):
    def __init__(self, config, num_classes=7, channels=3, **kwargs):
        super(PCT, self).__init__(config, num_classes)

        channels = config.channels_input
        dropout = config.dropout
        npoint1 = config.npoint1
        npoint2 = config.npoint2
        radius1 = config.radius1
        radius2 = config.radius2
        nsample = config.nsample

        self.channels = channels
        self.npoint1 = npoint1
        self.npoint2 = npoint2
        self.radius1 = radius1
        self.radius2 = radius2
        self.nsample = nsample

        self.conv1 = nn.Conv1d(channels, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.pt_last = Point_Transformer_Last()

        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(1024),
                                       nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, self.num_classes)

    def forward_features(self, x):
        """Return the pre-classifier global feature (B, 1024)."""
        if x.dim() == 3 and x.shape[1] != self.channels and x.shape[-1] == self.channels:
            x = x.permute(0, 2, 1)

        xyz = x[:, :3, :].permute(0, 2, 1).contiguous()
        batch_size, _, _ = x.size()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=self.npoint1, radius=self.radius1, nsample=self.nsample, xyz=xyz, points=x)
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=self.npoint2, radius=self.radius2, nsample=self.nsample, xyz=new_xyz, points=feature)
        feature_1 = self.gather_local_1(new_feature)

        x = self.pt_last(feature_1)
        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)
        return F.adaptive_max_pool1d(x, 1).view(batch_size, -1)

    def cls_head(self, feat):
        x = F.leaky_relu(self.bn6(self.linear1(feat)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        return self.linear3(x)

    def forward(self, x):
        return self.cls_head(self.forward_features(x))
