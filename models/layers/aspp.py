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


def conv(in_channels,
         out_channels,
         kernel_size=1,
         stride=1,
         padding=0,
         dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels,
                  out_channels,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=padding,
                  dilation=dilation,
                  bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class ASPP(nn.Module):
    def __init__(self, in_channel, num_channel, rate=(6, 12, 18)):
        super(ASPP, self).__init__()
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv(in_channel, num_channel, kernel_size=1),
        )
        self.conv = conv(in_channel, num_channel, kernel_size=1)
        self.atrous = nn.ModuleList([
            conv(in_channel,
                 num_channel,
                 kernel_size=3,
                 stride=1,
                 padding=r,
                 dilation=r) for r in rate
        ])
        self.proj = conv(num_channel * 5, num_channel, kernel_size=1)

    def forward(self, x):
        pool_feat = self.avgpool(x)
        pool_feat = F.interpolate(pool_feat, size=x.shape[2:], mode="bilinear")
        conv_feat = self.conv(x)

        return self.proj(
            torch.cat([pool_feat, conv_feat] + [a(x) for a in self.atrous],
                      dim=1))
