"""
DeepGCNs: Can GCNs Go as Deep as CNNs?

Paper: ICCV 2019
Authors: Guohao Li, Matthias Muller, Ali Thabet, Bernard Ghanem
Source: Implementation adapted from: https://github.com/lightaime/deep_gcns_torch
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..build import MODELS
from ..base_model import BasePointCloudModel, print_model_info
from .torch_nn import BasicConv, Seq
from .torch_edge import DilatedKnnGraph, DenseDilatedKnnGraph
from .torch_vertex import GraphConv2d, ResDynBlock2d, DenseDynBlock2d, PlainDynBlock2d


@MODELS.register_module()
class DeepGCN(BasePointCloudModel):
    """
    DeepGCN: Can GCNs Go as Deep as CNNs?

    Supports three block types:
    - 'res': Residual blocks with optional dilation
    - 'dense': Dense connections between blocks
    - 'plain': Plain GCN without residual connections
    """

    def __init__(self, config, num_classes=7, channels=3, **kwargs):
        super(DeepGCN, self).__init__(config, num_classes)

        # Hidden GCN width is `config.channels`; input feature dim comes from `channels_input`
        in_channels = config.channels_input if hasattr(config, 'channels_input') else channels
        self.in_channels = in_channels
        channels = in_channels  # used by self.head below
        gcn_channels = config.channels
        k = config.k
        act = config.act
        norm = config.norm
        bias = config.bias
        knn = config.knn
        epsilon = config.epsilon
        stochastic = config.stochastic
        conv = config.conv
        c_growth = config.c_growth
        emb_dims = config.emb_dims
        dropout = config.dropout

        self.n_blocks = config.n_blocks
        block_type = config.block_type
        use_dilation = config.use_dilation

        if knn == 'matrix':
            self.knn = DenseDilatedKnnGraph(k, 1, stochastic, epsilon)
        else:
            self.knn = DilatedKnnGraph(k, 1, stochastic, epsilon)

        self.head = GraphConv2d(channels, gcn_channels, conv, act, norm, bias=False)

        if block_type.lower() == 'dense':
            self.backbone = Seq(*[
                DenseDynBlock2d(
                    gcn_channels + c_growth * i,
                    c_growth,
                    k,
                    1 + i,
                    conv,
                    act,
                    norm,
                    bias,
                    stochastic,
                    epsilon,
                    knn
                )
                for i in range(self.n_blocks - 1)
            ])
            fusion_dims = int(
                (gcn_channels + gcn_channels + c_growth * (self.n_blocks - 1)) * self.n_blocks // 2
            )

        elif block_type.lower() == 'res':
            if use_dilation:
                self.backbone = Seq(*[
                    ResDynBlock2d(
                        gcn_channels,
                        k,
                        i + 1,
                        conv,
                        act,
                        norm,
                        bias,
                        stochastic,
                        epsilon,
                        knn
                    )
                    for i in range(self.n_blocks - 1)
                ])
            else:
                self.backbone = Seq(*[
                    ResDynBlock2d(
                        gcn_channels,
                        k,
                        1,
                        conv,
                        act,
                        norm,
                        bias,
                        stochastic,
                        epsilon,
                        knn
                    )
                    for _ in range(self.n_blocks - 1)
                ])
            fusion_dims = int(gcn_channels + c_growth * (self.n_blocks - 1))

        else:
            stochastic = False
            self.backbone = Seq(*[
                PlainDynBlock2d(
                    gcn_channels,
                    k,
                    1,
                    conv,
                    act,
                    norm,
                    bias,
                    stochastic,
                    epsilon,
                    knn
                )
                for i in range(self.n_blocks - 1)
            ])
            fusion_dims = int(gcn_channels + c_growth * (self.n_blocks - 1))

        self.fusion_block = BasicConv(
            [fusion_dims, emb_dims],
            'leakyrelu',
            norm,
            bias=False
        )

        self.prediction = Seq(*[
            BasicConv([emb_dims * 2, 512], 'leakyrelu', norm, drop=dropout),
            BasicConv([512, 256], 'leakyrelu', norm, drop=dropout),
            BasicConv([256, self.num_classes], None, None)
        ])

        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward_features(self, inputs):
        """Return the pre-classifier global feature (B, 2*emb_dims)."""
        if inputs.dim() == 3:
            if inputs.shape[1] != self.in_channels and inputs.shape[-1] == self.in_channels:
                inputs = inputs.permute(0, 2, 1).contiguous()
            inputs = inputs.unsqueeze(-1)

        feats = [self.head(inputs, self.knn(inputs[:, 0:3]))]
        for i in range(self.n_blocks - 1):
            feats.append(self.backbone[i](feats[-1]))
        feats = torch.cat(feats, dim=1)
        fusion = self.fusion_block(feats)

        x1 = F.adaptive_max_pool2d(fusion, 1)
        x2 = F.adaptive_avg_pool2d(fusion, 1)
        return torch.cat((x1, x2), dim=1).squeeze(-1).squeeze(-1)

    def cls_head(self, feat):
        """Classify from pre-classifier feature. Accepts (B, D) or (B, D, 1, 1)."""
        if feat.dim() == 2:
            feat = feat.unsqueeze(-1).unsqueeze(-1)
        return self.prediction(feat).squeeze(-1).squeeze(-1)

    def forward(self, inputs):
        return self.cls_head(self.forward_features(inputs))
