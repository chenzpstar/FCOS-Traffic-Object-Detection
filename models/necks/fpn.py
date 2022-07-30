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
                 out_channels=256,
                 use_p5=True,
                 init_weights=True):
        super(FPN, self).__init__()
        num_layers = len(in_channels) - 1
        self.projs = nn.ModuleList(
            [conv1x1(in_channels[i], out_channels) for i in range(num_layers)])
        self.convs = nn.ModuleList(
            [conv3x3(out_channels, out_channels) for _ in range(num_layers)])

        in_channel = out_channels if use_p5 else in_channels[0]
        self.conv6 = conv3x3(in_channel, out_channels, stride=2)
        self.conv7 = conv3x3(out_channels, out_channels, stride=2)
        self.use_p5 = use_p5

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def upsample(self, src_feat, dst_feat):
        return F.interpolate(src_feat,
                             size=dst_feat.shape[-2:],
                             mode="nearest")

    def forward(self, feats):
        outs = []
        last_feat = None

        for feat, proj, conv in zip(feats[::-1], self.projs, self.convs):
            if last_feat is None:
                last_feat = proj(feat)
            else:
                last_feat = proj(feat) + self.upsample(last_feat, feat)
            outs.insert(0, conv(last_feat))

        in_feat = outs[-1] if self.use_p5 else feats[-1]
        outs.append(self.conv6(in_feat))
        outs.append(self.conv7(F.relu(outs[-1])))

        return outs


if __name__ == "__main__":

    import torch

    model = FPN((2048, 1024, 512, 256))
    print(model)

    c5 = torch.rand(2, 2048, 7, 7)
    c4 = torch.rand(2, 1024, 14, 14)
    c3 = torch.rand(2, 512, 28, 28)
    c2 = torch.rand(2, 256, 56, 56)

    outs = model((c2, c3, c4, c5))
    [print(stage_outs.shape) for stage_outs in outs]
