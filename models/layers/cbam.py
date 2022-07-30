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

try:
    from .conv import conv7x7
except:
    from conv import conv7x7


class ChannelGate(nn.Module):
    def __init__(self,
                 channels,
                 reduction_ratio=16,
                 pool_types=("avg", "max")):
        super(ChannelGate, self).__init__()
        self.pool_types = pool_types
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1),
        )

    def forward(self, x):
        h, w = x.shape[-2:]  # bchw
        channel_att = 0.0

        for type in self.pool_types:
            if type == 'avg':
                pool_feat = F.avg_pool2d(x, (h, w), stride=(h, w))
            elif type == 'max':
                pool_feat = F.max_pool2d(x, (h, w), stride=(h, w))
            channel_att += self.fc(pool_feat)

        return x * F.sigmoid(channel_att)


class SpatialGate(nn.Module):
    def __init__(self, pool_types=("avg", "max")):
        super(SpatialGate, self).__init__()
        self.pool_types = pool_types
        self.conv = conv7x7(len(pool_types), 1, act=None)

    def forward(self, x):
        spatial_att = []

        for type in self.pool_types:
            if type == 'avg':
                pool_feat = x.mean(dim=1)
            elif type == 'max':
                pool_feat = x.max(dim=1)[0]
            spatial_att.append(pool_feat)

        spatial_att = torch.stack(spatial_att, dim=1)

        return x * F.sigmoid(self.conv(spatial_att))


class CBAM(nn.Module):
    def __init__(self,
                 channels,
                 reduction_ratio=16,
                 pool_types=("avg", "max"),
                 no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(channels, reduction_ratio, pool_types)
        if not no_spatial:
            self.SpatialGate = SpatialGate(pool_types)
        self.no_spatial = no_spatial

    def forward(self, x):
        x = self.ChannelGate(x)
        if not self.no_spatial:
            x = self.SpatialGate(x)

        return x
