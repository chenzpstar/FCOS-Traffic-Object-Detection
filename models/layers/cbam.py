# -*- coding: utf-8 -*-
"""
# @file name  : cbam.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-04-08
# @brief      : CBAM模块类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size,
                 stride=1,
                 padding=0,
                 relu=True):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_planes,
                              out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=1e-5,
                                 momentum=0.01,
                                 affine=True)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self,
                 gate_channels,
                 reduction_ratio=16,
                 pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )
        self.pool_types = pool_types

    def forward(self, x):
        _, _, h, w = x.size()
        channel_att_sum = None

        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (h, w), stride=(h, w))
                channel_att_raw = self.fc(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (h, w), stride=(h, w))
                channel_att_raw = self.fc(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum += channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(
            x)
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat([
            torch.max(x, 1)[0].unsqueeze(1),
            torch.mean(x, 1).unsqueeze(1),
        ],
                         dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2,
                                 1,
                                 kernel_size=7,
                                 stride=1,
                                 padding=3,
                                 relu=False)

    def forward(self, x):
        x_out = self.compress(x)
        x_out = self.spatial(x_out)
        scale = F.sigmoid(x_out)
        return x * scale


class CBAM(nn.Module):
    def __init__(self,
                 gate_channels,
                 reduction_ratio=16,
                 pool_types=['avg', 'max'],
                 no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio,
                                       pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
