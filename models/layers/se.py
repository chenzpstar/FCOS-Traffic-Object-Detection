# -*- coding: utf-8 -*-
"""
# @file name  : se.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-04-07
# @brief      : SE模块类
"""

import torch.nn as nn


class SELayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SELayer, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.avgpool(x))
