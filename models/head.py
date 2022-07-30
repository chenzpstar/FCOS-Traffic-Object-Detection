# -*- coding: utf-8 -*-
"""
# @file name  : head.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-05
# @brief      : FCOS检测网络类
"""

from math import log

import torch
import torch.nn as nn

try:
    from .layers import conv3x3
    from .utils import decode_coords, reshape_feats
except:
    from layers import conv3x3
    from utils import decode_coords, reshape_feats


class FCOSHead(nn.Module):
    def __init__(self,
                 in_channels=256,
                 num_convs=4,
                 num_classes=3,
                 prior=0.01,
                 use_gn=True,
                 ctr_on_reg=True,
                 strides=[8, 16, 32, 64, 128],
                 init_weights=True):
        super(FCOSHead, self).__init__()
        self.in_channels = in_channels
        self.num_convs = num_convs
        self.num_classes = num_classes
        self.prior = prior
        self.use_gn = use_gn
        self.ctr_on_reg = ctr_on_reg
        self.strides = strides

        norm = "gn" if use_gn else "bn"
        cls_branch = [
            conv3x3(in_channels, in_channels, norm=norm)
            for _ in range(num_convs)
        ]
        reg_branch = [
            conv3x3(in_channels, in_channels, norm=norm)
            for _ in range(num_convs)
        ]
        self.cls_conv = nn.Sequential(*cls_branch)
        self.reg_conv = nn.Sequential(*reg_branch)

        self.cls_logits = nn.Conv2d(in_channels, num_classes, 3, 1, 1)
        self.reg_preds = nn.Conv2d(in_channels, 4, 3, 1, 1)
        self.ctr_logits = nn.Conv2d(in_channels, 1, 3, 1, 1)

        self.scales = nn.ModuleList(
            [ScaleExp(1.0) for _ in range(len(strides))])

        if init_weights:
            self._initialize_weights()

        # cls bias init
        nn.init.constant_(self.cls_logits.bias, -log((1.0 - prior) / prior))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, feats):
        assert len(feats) == len(self.strides)

        cls_logits = []
        reg_preds = []
        ctr_logits = []
        coords = []

        for feat, scale, stride in zip(feats, self.scales, self.strides):
            cls_conv_out = self.cls_conv(feat)
            reg_conv_out = self.reg_conv(feat)

            cls_logits.append(self.cls_logits(cls_conv_out))
            reg_preds.append(scale(self.reg_preds(reg_conv_out)))

            if not self.ctr_on_reg:
                ctr_logits.append(self.ctr_logits(cls_conv_out))
            else:
                ctr_logits.append(self.ctr_logits(reg_conv_out))

            coords.append(
                decode_coords(feat, stride).to(feat.device, non_blocking=True))

        cls_logits = reshape_feats(cls_logits)  # bchw -> b(hw)c
        reg_preds = reshape_feats(reg_preds)  # bchw -> b(hw)c
        ctr_logits = reshape_feats(ctr_logits)  # bchw -> b(hw)c

        return cls_logits, reg_preds, ctr_logits, coords


class ScaleExp(nn.Module):
    def __init__(self, init_value=1.0):
        super(ScaleExp, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, x):
        return torch.exp(x * self.scale)


if __name__ == "__main__":

    import torch
    torch.manual_seed(0)

    model = FCOSHead(num_classes=3)
    print(model)

    p7 = torch.rand(2, 256, 2, 2)
    p6 = torch.rand(2, 256, 4, 4)
    p5 = torch.rand(2, 256, 7, 7)
    p4 = torch.rand(2, 256, 14, 14)
    p3 = torch.rand(2, 256, 28, 28)

    outs = model((p3, p4, p5, p6, p7))
    [
        print(stage_outs.shape) for branch_outs in outs
        for stage_outs in branch_outs
    ]
