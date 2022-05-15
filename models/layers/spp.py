# -*- coding: utf-8 -*-
"""
# @file name  : spp.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-04-05
# @brief      : SPP模块类
"""

import torch
import torch.nn as nn


def conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels,
                  out_channels,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=padding,
                  bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class SPP(nn.Module):
    # Spatial Pyramid Pooling layer used in YOLOv3-SPP
    def __init__(self, in_channel, out_channel, kernel_size=(5, 9, 13)):
        super(SPP, self).__init__()
        num_channel = in_channel // 2  # hidden channels
        self.conv = conv(in_channel, num_channel, kernel_size=1)
        self.maxpool = nn.ModuleList([
            nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
            for k in kernel_size
        ])
        self.proj = conv(num_channel * 4, out_channel, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)

        return self.proj(torch.cat([x] + [m(x) for m in self.maxpool], dim=1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, in_channel, out_channel, kernel_size=5):
        # equivalent to SPP(kernel_size=(5, 9, 13))
        super(SPPF, self).__init__()
        num_channel = in_channel // 2  # hidden channels
        self.conv = conv(in_channel, num_channel, kernel_size=1)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size,
                                    stride=1,
                                    padding=kernel_size // 2)
        self.proj = conv(num_channel * 4, out_channel, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)

        return self.proj(torch.cat([x, y1, y2, y3], dim=1))
