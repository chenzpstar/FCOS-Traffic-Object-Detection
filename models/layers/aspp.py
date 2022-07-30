# -*- coding: utf-8 -*-
"""
# @file name  : aspp.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-04-06
# @brief      : ASPP模块类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .conv import conv1x1, conv3x3
except:
    from conv import conv1x1, conv3x3


class ASPP(nn.Module):
    def __init__(self, in_channels, num_channels, rate=(6, 12, 18)):
        super(ASPP, self).__init__()
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv1x1(in_channels, num_channels),
        )
        self.conv = conv1x1(in_channels, num_channels)
        self.atrous = nn.ModuleList([
            conv3x3(in_channels, num_channels, padding=r, dilation=r)
            for r in rate
        ])
        self.proj = conv1x1(num_channels * 5, num_channels)

    def forward(self, x):
        pool_feat = self.avgpool(x)
        pool_feat = F.interpolate(pool_feat,
                                  size=x.shape[-2:],
                                  mode="bilinear")
        conv_feat = self.conv(x)

        return self.proj(
            torch.cat((pool_feat, conv_feat) + (a(x) for a in self.atrous),
                      dim=1))
