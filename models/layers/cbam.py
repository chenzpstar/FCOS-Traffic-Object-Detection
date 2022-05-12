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


class ChannelGate(nn.Module):
    def __init__(self,
                 gate_channels,
                 reduction_ratio=16,
                 pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.fc = nn.Sequential(
            nn.Conv2d(gate_channels,
                      gate_channels // reduction_ratio,
                      kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_channels // reduction_ratio,
                      gate_channels,
                      kernel_size=1),
        )
        self.pool_types = pool_types

    def forward(self, x):
        h, w = x.shape[2:]
        channel_att = 0

        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avgpool = F.avg_pool2d(x, (h, w), stride=(h, w))
                channel_att += self.fc(avgpool)
            elif pool_type == 'max':
                maxpool = F.max_pool2d(x, (h, w), stride=(h, w))
                channel_att += self.fc(maxpool)

        scale = F.sigmoid(channel_att)
        return x * scale


class SpatialGate(nn.Module):
    def __init__(self, pool_types=['avg', 'max']):
        super(SpatialGate, self).__init__()
        self.in_channels = len(pool_types)
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels,
                      1,
                      kernel_size=7,
                      stride=1,
                      padding=3,
                      bias=False),
            nn.BatchNorm2d(1, momentum=0.01),
        )
        self.pool_types = pool_types

    def forward(self, x):
        spatial_att = []

        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avgpool = torch.mean(x, 1, keepdim=True)
                spatial_att.append(avgpool)
            elif pool_type == 'max':
                maxpool = torch.max(x, 1, keepdim=True)[0]
                spatial_att.append(maxpool)

        x_out = torch.cat(spatial_att, dim=1)
        x_out = self.conv(x_out)
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
            self.SpatialGate = SpatialGate(pool_types)

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
