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

from .edge_conv import EdgeConv
from ..build import MODELS
from ..base_model import BasePointCloudModel, print_model_info

@MODELS.register_module()
class DGCNN(BasePointCloudModel):
    def __init__(self, config, num_classes=7, channels=3, **kwargs):
        super(DGCNN, self).__init__(config, num_classes)

        channels = config.channels_input
        self.k = config.k
        self.emb_dims = config.emb_dims
        dropout = config.dropout

        self.conv1 = EdgeConv(channels, 64, k=self.k)
        self.conv2 = EdgeConv(64, 64, k=self.k)
        self.conv3 = EdgeConv(64, 128, k=self.k)
        self.conv4 = EdgeConv(128, 256, k=self.k)

        self.conv5 = nn.Sequential(
            nn.Conv1d(512, self.emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.emb_dims),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.linear1 = nn.Linear(self.emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout)

        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout)

        self.linear3 = nn.Linear(256, self.num_classes)

    def forward_features(self, x):
        """Return the pre-classifier global feature (B, emb_dims*2)."""
        if x.dim() == 3 and x.shape[-1] == 3:
            x = x.permute(0, 2, 1)
        elif x.dim() == 3 and x.shape[1] > 10:
            x = x.permute(0, 2, 1)

        batch_size = x.size(0)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        m = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        a = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        return torch.cat((m, a), 1)

    def cls_head(self, feat):
        x = F.leaky_relu(self.bn6(self.linear1(feat)), negative_slope=0.2)
        x = self.dropout1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dropout2(x)
        return self.linear3(x)

    def forward(self, x):
        return self.cls_head(self.forward_features(x))
