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
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=padding,
                     dilation=dilation)


# without bn version
class ASPP(nn.Module):
    def __init__(self, in_channel=512, num_channel=256, r=(6, 12, 18)):
        super(ASPP, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv_1x1 = conv(in_channel, num_channel)
        self.atrous_1x1 = conv(in_channel, num_channel)
        self.atrous_3x3 = nn.ModuleList([
            conv(in_channel,
                 num_channel,
                 kernel_size=3,
                 stride=1,
                 padding=x,
                 dilation=x) for x in r
        ])
        self.conv_out = conv(num_channel * 5, num_channel)

    def forward(self, x):
        conv_feat = self.conv_1x1(self.avgpool(x))
        conv_feat = F.interpolate(conv_feat, size=x.shape[2:], mode="bilinear")

        atrous_feat = self.atrous_1x1(x)

        return self.conv_out(
            torch.cat([conv_feat, atrous_feat] +
                      [a(x) for a in self.atrous_3x3],
                      dim=1))
