# -*- coding: utf-8 -*-
"""
# @file name  : fpn.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-04
# @brief      : FPN模型类
"""

import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_channels, out_channels, stride=1, bias=True):
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=bias)


def conv1x1(in_channels, out_channels, stride=1, bias=True):
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size=1,
                     stride=stride,
                     bias=bias)


class FPN(nn.Module):
    def __init__(self,
                 in_channels,
                 num_channels=256,
                 use_p5=True,
                 init_weights=True):
        super(FPN, self).__init__()
        self.proj5 = conv1x1(in_channels[0], num_channels)
        self.proj4 = conv1x1(in_channels[1], num_channels)
        self.proj3 = conv1x1(in_channels[2], num_channels)
        # self.proj2 = conv1x1(in_channels[3], num_channels)

        self.conv5 = conv3x3(num_channels, num_channels)
        self.conv4 = conv3x3(num_channels, num_channels)
        self.conv3 = conv3x3(num_channels, num_channels)
        # self.conv2 = conv3x3(num_channels, num_channels)

        in_channel = num_channels if use_p5 else in_channels[0]
        self.conv6 = conv3x3(in_channel, num_channels, stride=2)
        self.conv7 = conv3x3(num_channels, num_channels, stride=2)
        self.use_p5 = use_p5

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight,
                                         mode='fan_out',
                                         nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def upsample(self, src_feat, tar_feat):
        return F.interpolate(src_feat,
                             size=tar_feat.shape[-2:],
                             mode="nearest")

    def forward(self, feats):
        _, c3, c4, c5 = feats

        p5 = self.proj5(c5)
        p4 = self.proj4(c4) + self.upsample(p5, c4)
        p3 = self.proj3(c3) + self.upsample(p4, c3)
        # p2 = self.proj2(c2) + self.upsample(p3, c2)

        p5 = self.conv5(p5)
        p4 = self.conv4(p4)
        p3 = self.conv3(p3)
        # p2 = self.conv2(p2)

        in_feat = p5 if self.use_p5 else c5
        p6 = self.conv6(in_feat)
        p7 = self.conv7(F.relu(p6))

        return [p3, p4, p5, p6, p7]


if __name__ == "__main__":

    import torch

    model = FPN(backbone="darknet19")

    c5 = torch.rand(2, 1024, 7, 7)
    c4 = torch.rand(2, 512, 14, 14)
    c3 = torch.rand(2, 256, 28, 28)
    c2 = torch.rand(2, 128, 56, 56)

    out = model([c2, c3, c4, c5])
    [print(stage_out.shape) for stage_out in out]
