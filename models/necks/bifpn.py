# -*- coding: utf-8 -*-
"""
# @file name  : bifpn.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-04-23
# @brief      : BiFPN模型类
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


class BiFPN(nn.Module):
    def __init__(self,
                 in_channels,
                 num_channels=256,
                 use_p5=True,
                 init_weights=True):
        super(BiFPN, self).__init__()
        self.proj5_1 = conv1x1(in_channels[0], num_channels)
        self.proj4_1 = conv1x1(in_channels[1], num_channels)
        self.proj3_1 = conv1x1(in_channels[2], num_channels)
        self.proj2_1 = conv1x1(in_channels[3], num_channels)

        self.proj5_2 = conv1x1(in_channels[0], num_channels)
        self.proj4_2 = conv1x1(in_channels[1], num_channels)
        self.proj3_2 = conv1x1(in_channels[2], num_channels)

        # self.conv_p5 = conv3x3(num_channels, num_channels)
        # self.conv_p4 = conv3x3(num_channels, num_channels)
        # self.conv_p3 = conv3x3(num_channels, num_channels)

        self.new3 = conv3x3(num_channels, num_channels, stride=2)
        self.new4 = conv3x3(num_channels, num_channels, stride=2)
        self.new5 = conv3x3(num_channels, num_channels, stride=2)

        self.conv_n2 = conv3x3(num_channels, num_channels)
        self.conv_n3 = conv3x3(num_channels, num_channels)
        self.conv_n4 = conv3x3(num_channels, num_channels)
        self.conv_n5 = conv3x3(num_channels, num_channels)

        self.relu = nn.ReLU(inplace=True)

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

    def upsample(self, src_feat, dst_feat):
        return F.interpolate(src_feat,
                             size=dst_feat.shape[-2:],
                             mode="nearest")

    def forward(self, feats):
        c2, c3, c4, c5 = feats

        p5_1 = self.relu(self.proj5_1(c5))
        p4_1 = self.relu(self.proj4_1(c4)) + self.upsample(p5_1, c4)
        p3_1 = self.relu(self.proj3_1(c3)) + self.upsample(p4_1, c3)
        p2 = self.relu(self.proj2_1(c2)) + self.upsample(p3_1, c2)

        # p5_1 = self.relu(self.conv_p5(p5_1))
        # p4_1 = self.relu(self.conv_p4(p4_1))
        # p3_1 = self.relu(self.conv_p3(p3_1))

        p5_2 = self.relu(self.proj5_2(c5))
        p4_2 = self.relu(self.proj4_2(c4))
        p3_2 = self.relu(self.proj3_2(c3))

        n2 = p2
        n3 = p3_1 + p3_2 + self.relu(self.new3(n2))
        n4 = p4_1 + p4_2 + self.relu(self.new4(n3))
        n5 = p5_2 + self.relu(self.new5(n4))

        n2 = self.relu(self.conv_n2(n2))
        n3 = self.relu(self.conv_n3(n3))
        n4 = self.relu(self.conv_n4(n4))
        n5 = self.relu(self.conv_n5(n5))

        return [n2, n3, n4, n5]


if __name__ == "__main__":

    import torch

    model = BiFPN([2048, 1024, 512, 256])

    c5 = torch.rand(2, 2048, 7, 7)
    c4 = torch.rand(2, 1024, 14, 14)
    c3 = torch.rand(2, 512, 28, 28)
    c2 = torch.rand(2, 256, 56, 56)

    out = model([c2, c3, c4, c5])
    [print(stage_out.shape) for stage_out in out]
