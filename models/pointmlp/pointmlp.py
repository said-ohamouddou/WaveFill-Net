"""
Rethinking Network Design and Local Geometry in Point Cloud: A Simple Residual MLP Framework

Paper: ICLR 2022
Authors: Xu Ma, Can Qin, Haoxuan You, Haoxi Ran, Yun Fu
Source: Implementation adapted from: https://github.com/ma-xu/pointMLP-pytorch
License: Apache-2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointmlp_utils import (
    get_activation,
    LocalGrouper,
    ConvBNReLU1D,
    PreExtraction,
    PosExtraction
)
from ..build import MODELS
from ..base_model import BasePointCloudModel, print_model_info

class Model(nn.Module):
    def __init__(self, points=1024, class_num=40, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="center",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[2, 2, 2, 2], dropout=0.5,
                 in_channels=3, **kwargs):
        super(Model, self).__init__()
        self.stages = len(pre_blocks)
        self.class_num = class_num
        self.points = points
        self.in_channels = in_channels
        self.embedding = ConvBNReLU1D(in_channels, embed_dim, bias=bias, activation=activation)
        assert len(pre_blocks) == len(k_neighbors) == len(reducers) == len(pos_blocks) == len(dim_expansion), \
            "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."
        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        last_channel = embed_dim
        anchor_points = self.points
        for i in range(len(pre_blocks)):
            out_channel = last_channel * dim_expansion[i]
            pre_block_num = pre_blocks[i]
            pos_block_num = pos_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce
            local_grouper = LocalGrouper(last_channel, anchor_points, kneighbor, use_xyz, normalize)
            self.local_grouper_list.append(local_grouper)
            pre_block_module = PreExtraction(last_channel, out_channel, pre_block_num, groups=groups,
                                             res_expansion=res_expansion,
                                             bias=bias, activation=activation, use_xyz=use_xyz)
            self.pre_blocks_list.append(pre_block_module)
            pos_block_module = PosExtraction(out_channel, pos_block_num, groups=groups,
                                             res_expansion=res_expansion, bias=bias, activation=activation)
            self.pos_blocks_list.append(pos_block_module)
            last_channel = out_channel
        self.act = get_activation(activation)
        self.classifier = nn.Sequential(
            nn.Linear(last_channel, 512),
            nn.BatchNorm1d(512),
            self.act,
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            self.act,
            nn.Dropout(dropout),
            nn.Linear(256, self.class_num)
        )

    def forward_features(self, x):
        """Return the pre-classifier global feature (B, last_channel)."""
        xyz = x[:, :3, :].permute(0, 2, 1).contiguous()
        x = self.embedding(x)
        for i in range(self.stages):
            xyz, x = self.local_grouper_list[i](xyz, x.permute(0, 2, 1))
            x = self.pre_blocks_list[i](x)
            x = self.pos_blocks_list[i](x)
        return F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)

    def forward(self, x):
        return self.classifier(self.forward_features(x))

@MODELS.register_module()
class PointMLP(BasePointCloudModel):
    def __init__(self, config, num_classes=7, channels=3, **kwargs):
        num_classes = config.num_classes
        super(PointMLP, self).__init__(config, num_classes)

        points = config.points
        embed_dim = config.embed_dim
        groups = config.groups
        res_expansion = config.res_expansion
        activation = config.activation
        bias = config.bias
        use_xyz = config.use_xyz
        normalize = config.normalize
        dim_expansion = config.dim_expansion
        pre_blocks = config.pre_blocks
        pos_blocks = config.pos_blocks
        k_neighbors = config.k_neighbors
        reducers = config.reducers
        dropout = config.dropout
        in_channels = config.channels_input if hasattr(config, 'channels_input') else channels

        self.channels = in_channels

        self.model = Model(
            points=points,
            class_num=num_classes,
            embed_dim=embed_dim,
            groups=groups,
            res_expansion=res_expansion,
            activation=activation,
            bias=bias,
            use_xyz=use_xyz,
            normalize=normalize,
            dim_expansion=dim_expansion,
            pre_blocks=pre_blocks,
            pos_blocks=pos_blocks,
            k_neighbors=k_neighbors,
            reducers=reducers,
            dropout=dropout,
            in_channels=in_channels,
        )

    def _to_bcn(self, x):
        if x.dim() == 3 and x.shape[1] != self.channels and x.shape[-1] == self.channels:
            x = x.permute(0, 2, 1)
        return x

    def forward_features(self, x):
        return self.model.forward_features(self._to_bcn(x))

    def cls_head(self, feat):
        return self.model.classifier(feat)

    def forward(self, x):
        return self.model(self._to_bcn(x))

