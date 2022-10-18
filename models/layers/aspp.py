# -*- coding: utf-8 -*-
"""
# @file name  : aspp.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-04-06
# @brief      : ASPP模块类
# @reference  : https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/deeplabv3.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .conv import conv1x1, conv3x3
except:
    from conv import conv1x1, conv3x3


class ASPP(nn.Module):
    def __init__(self, in_channels, num_channels, atrous_rate=(6, 12, 18)):
        super(ASPP, self).__init__()
        self.conv = conv1x1(in_channels, num_channels)
        self.atrous = nn.ModuleList([
            conv3x3(in_channels, num_channels, padding=r, dilation=r)
            for r in atrous_rate
        ])
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv1x1(in_channels, num_channels),
        )
        self.proj = conv1x1(num_channels * 5, num_channels)

    def forward(self, x):
        conv_feat = self.conv(x)
        pool_feat = self.avgpool(x)
        pool_feat = F.interpolate(pool_feat,
                                  size=x.shape[-2:],
                                  mode="bilinear")

        return self.proj(
            torch.cat((conv_feat) + (a(x) for a in self.atrous) + (pool_feat),
                      dim=1))
