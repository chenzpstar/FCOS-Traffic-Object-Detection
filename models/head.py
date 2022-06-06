# -*- coding: utf-8 -*-
"""
# @file name  : head.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-05
# @brief      : FCOS检测网络类
"""

import math

import torch
import torch.nn as nn


def conv3x3(in_channels, out_channels, stride=1, bias=True):
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=bias)


class FCOSHead(nn.Module):
    def __init__(self,
                 in_channels=256,
                 num_convs=4,
                 num_classes=3,
                 use_gn=True,
                 ctr_on_reg=True,
                 prior=0.01,
                 init_weights=True):
        super(FCOSHead, self).__init__()
        self.in_channels = in_channels
        self.num_convs = num_convs
        self.num_classes = num_classes
        self.use_gn = use_gn
        self.ctr_on_reg = ctr_on_reg
        self.prior = prior

        cls_branch = []
        reg_branch = []

        for _ in range(num_convs):
            cls_branch.append(conv3x3(in_channels, in_channels))
            if use_gn:
                cls_branch.append(nn.GroupNorm(32, in_channels))
            else:
                cls_branch.append(nn.BatchNorm2d(32, in_channels))
            cls_branch.append(nn.ReLU(inplace=True))

            reg_branch.append(conv3x3(in_channels, in_channels))
            if use_gn:
                reg_branch.append(nn.GroupNorm(32, in_channels))
            else:
                reg_branch.append(nn.BatchNorm2d(32, in_channels))
            reg_branch.append(nn.ReLU(inplace=True))

        self.cls_conv = nn.Sequential(*cls_branch)
        self.reg_conv = nn.Sequential(*reg_branch)

        self.cls_logits = conv3x3(in_channels, num_classes)
        self.reg_preds = conv3x3(in_channels, 4)
        self.ctr_logits = conv3x3(in_channels, 1)

        self.scales = nn.ModuleList([ScaleExp(1.0) for _ in range(5)])

        if init_weights:
            self._initialize_weights()

        # cls bias init
        nn.init.constant_(self.cls_logits.bias, -math.log(
            (1.0 - prior) / prior))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, feats):
        cls_logits = []
        reg_preds = []
        ctr_logits = []

        for feat, scale in zip(feats, self.scales):
            cls_conv_out = self.cls_conv(feat)
            reg_conv_out = self.reg_conv(feat)

            cls_logits.append(self.cls_logits(cls_conv_out))
            reg_preds.append(scale(self.reg_preds(reg_conv_out)))

            if not self.ctr_on_reg:
                ctr_logits.append(self.ctr_logits(cls_conv_out))
            else:
                ctr_logits.append(self.ctr_logits(reg_conv_out))

        return cls_logits, reg_preds, ctr_logits


class ScaleExp(nn.Module):
    def __init__(self, init_value=1.0):
        super(ScaleExp, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, x):
        return torch.exp(x * self.scale)


if __name__ == "__main__":

    import torch

    model = FCOSHead(num_classes=3)

    p7 = torch.rand(2, 256, 2, 2)
    p6 = torch.rand(2, 256, 4, 4)
    p5 = torch.rand(2, 256, 7, 7)
    p4 = torch.rand(2, 256, 14, 14)
    p3 = torch.rand(2, 256, 28, 28)

    out = model([p3, p4, p5, p6, p7])
    [print(stage_out.shape) for branch_out in out for stage_out in branch_out]
