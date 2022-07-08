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
                 out_channels=256,
                 use_p5=True,
                 init_weights=True):
        super(BiFPN, self).__init__()
        num_layers = len(in_channels)
        self.projs_1 = nn.ModuleList(
            [conv1x1(in_channels[i], out_channels) for i in range(num_layers)])
        self.projs_2 = nn.ModuleList([
            conv1x1(in_channels[i], out_channels)
            for i in range(num_layers - 1)
        ])
        self.news = nn.ModuleList([
            conv3x3(out_channels, out_channels, stride=2)
            for _ in range(num_layers - 1)
        ])
        self.convs = nn.ModuleList(
            [conv3x3(out_channels, out_channels) for _ in range(num_layers)])
        self.relu = nn.ReLU(inplace=True)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def upsample(self, src_feat, dst_feat):
        return F.interpolate(src_feat,
                             size=dst_feat.shape[-2:],
                             mode="nearest")

    def forward(self, feats):
        projs_1, projs_2, outs = [], [], []
        last_feat = None

        for feat, proj_1, proj_2 in zip(feats[::-1][:-1], self.projs_1[:-1],
                                        self.projs_2):
            if last_feat is None:
                last_feat = self.relu(proj_1(feat))
            else:
                last_feat = self.relu(proj_1(feat)) + self.upsample(
                    last_feat, feat)
            projs_1.insert(0, last_feat)
            projs_2.insert(0, self.relu(proj_2(feat)))

        last_feat = self.relu(self.projs_1[-1](feats[0])) + self.upsample(
            last_feat, feats[0])
        projs_1.insert(0, last_feat)
        outs.append(self.relu(self.convs[0](last_feat)))

        for feat_1, feat_2, new, conv in zip(projs_1[1:-1], projs_2[:-1],
                                             self.news[:-1], self.convs[1:-1]):
            last_feat = feat_1 + feat_2 + self.relu(new(last_feat))
            outs.append(self.relu(conv(last_feat)))

        last_feat = projs_2[-1] + self.relu(self.news[-1](last_feat))
        outs.append(self.relu(self.convs[-1](last_feat)))

        return outs


if __name__ == "__main__":

    import torch

    model = BiFPN([2048, 1024, 512, 256])
    print(model)

    c5 = torch.rand(2, 2048, 7, 7)
    c4 = torch.rand(2, 1024, 14, 14)
    c3 = torch.rand(2, 512, 28, 28)
    c2 = torch.rand(2, 256, 56, 56)

    outs = model([c2, c3, c4, c5])
    [print(stage_outs.shape) for stage_outs in outs]
