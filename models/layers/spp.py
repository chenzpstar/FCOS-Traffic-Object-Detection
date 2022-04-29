# -*- coding: utf-8 -*-
"""
# @file name  : spp.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-04-05
# @brief      : SPP模块类
"""

import torch
import torch.nn as nn


def autopad(k, p=None):
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1,
                              c2,
                              kernel_size=k,
                              stride=s,
                              padding=autopad(k, p),
                              bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class SPP(nn.Module):
    # Spatial Pyramid Pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        ch = c1 // 2  # hidden channels
        self.conv1 = Conv(c1, ch, 1, 1)
        self.conv2 = Conv(ch * (len(k) + 1), c2, 1, 1)
        self.maxpool = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [m(x) for m in self.maxpool], dim=1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super(SPPF, self).__init__()
        ch = c1 // 2  # hidden channels
        self.conv1 = Conv(c1, ch, 1, 1)
        self.conv2 = Conv(ch * 4, c2, 1, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], dim=1))
