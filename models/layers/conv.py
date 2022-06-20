# -*- coding: utf-8 -*-
"""
# @file name  : cbam.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-06-18
# @brief      : Conv模块类
"""

import torch.nn as nn

__all__ = ['Conv', 'conv1x1', 'conv3x3', 'conv7x7']


class Conv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 norm="bn",
                 act="relu"):
        super(Conv, self).__init__()
        if norm is None:
            self.conv = nn.Conv2d(in_channels,
                                  out_channels,
                                  kernel_size,
                                  stride,
                                  padding,
                                  dilation,
                                  groups,
                                  bias=True)
            self.norm = None
        elif norm == "bn":
            self.conv = nn.Conv2d(in_channels,
                                  out_channels,
                                  kernel_size,
                                  stride,
                                  padding,
                                  dilation,
                                  groups,
                                  bias=False)
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == "gn":
            self.conv = nn.Conv2d(in_channels,
                                  out_channels,
                                  kernel_size,
                                  stride,
                                  padding,
                                  dilation,
                                  groups,
                                  bias=True)
            self.norm = nn.GroupNorm(32, out_channels)
        else:
            raise NotImplementedError(
                "norm layer only implemented ['bn', 'gn']")

        if act is None:
            self.act = None
        elif act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "relu6":
            self.act = nn.ReLU6(inplace=True)
        elif act == "lrelu":
            self.act = nn.LeakyReLU(inplace=True)
        elif act == "silu":
            self.act = nn.SiLU(inplace=True)
        else:
            raise NotImplementedError(
                "act func only implemented ['relu', 'relu6', 'lrelu', 'silu']")

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)

        return x


def conv1x1(in_channels,
            out_channels,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            norm="bn",
            act="relu"):
    return Conv(in_channels, out_channels, 1, stride, padding, dilation,
                groups, norm, act)


def conv3x3(in_channels,
            out_channels,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            norm="bn",
            act="relu"):
    return Conv(in_channels, out_channels, 3, stride, padding, dilation,
                groups, norm, act)


def conv7x7(in_channels,
            out_channels,
            stride=1,
            padding=3,
            dilation=1,
            groups=1,
            norm="bn",
            act="relu"):
    return Conv(in_channels, out_channels, 7, stride, padding, dilation,
                groups, norm, act)
