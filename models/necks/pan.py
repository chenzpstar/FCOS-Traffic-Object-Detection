# -*- coding: utf-8 -*-
"""
# @file name  : pan.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-04
# @brief      : PAN模型类
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


class PAN(nn.Module):
    def __init__(self,
                 in_channels,
                 num_channels=256,
                 use_p5=True,
                 init_weights=True):
        super(PAN, self).__init__()
        num_layers = len(in_channels)
        self.projs = nn.ModuleList(
            [conv1x1(in_channels[i], num_channels) for i in range(num_layers)])
        self.news = nn.ModuleList([
            conv3x3(num_channels, num_channels, stride=2)
            for _ in range(num_layers - 1)
        ])
        self.convs = nn.ModuleList(
            [conv3x3(num_channels, num_channels) for _ in range(num_layers)])
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
        projs, outs = [], []
        last_feat = None

        for feat, proj in zip(feats[::-1], self.projs):
            if last_feat is None:
                last_feat = self.relu(proj(feat))
            else:
                last_feat = self.relu(proj(feat)) + self.upsample(
                    last_feat, feat)
            projs.append(last_feat)

        outs.append(self.relu(self.convs[0](last_feat)))

        for feat, new, conv in zip(projs[::-1][1:], self.news, self.convs[1:]):
            last_feat = feat + self.relu(new(last_feat))
            outs.append(self.relu(conv(last_feat)))

        return outs


if __name__ == "__main__":

    import torch

    model = PAN([2048, 1024, 512, 256])
    print(model)

    c5 = torch.rand(2, 2048, 7, 7)
    c4 = torch.rand(2, 1024, 14, 14)
    c3 = torch.rand(2, 512, 28, 28)
    c2 = torch.rand(2, 256, 56, 56)

    outs = model([c2, c3, c4, c5])
    [print(stage_outs.shape) for stage_outs in outs]
