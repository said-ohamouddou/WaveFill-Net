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


class Point_Transformer_Last(nn.Module):
    def __init__(self, channels=256):
        super(Point_Transformer_Last, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

    def forward(self, x):
        #
        # b, 3, npoint, nsample
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample
        # permute reshape
        batch_size, _, N = x.size()

        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        batch_size, _, N = x.size()
        x = F.relu(self.bn1(self.conv1(x)))  # B, D, N
        x = F.relu(self.bn2(self.conv2(x)))  # B, D, N
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x
