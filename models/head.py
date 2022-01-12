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


class FCOSHead(nn.Module):
    def __init__(self,
                 num_feat=256,
                 num_cls=3,
                 use_gn=True,
                 ctr_on_reg=True,
                 prior=0.01,
                 init_weights=True):
        super(FCOSHead, self).__init__()
        self.num_cls = num_cls
        self.use_gn = use_gn
        self.ctr_on_reg = ctr_on_reg
        self.prior = prior

        cls_branch = []
        reg_branch = []

        for _ in range(4):
            cls_branch.append(
                nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1))
            if use_gn:
                cls_branch.append(nn.GroupNorm(32, num_feat))
            cls_branch.append(nn.ReLU(inplace=True))

            reg_branch.append(
                nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1))
            if use_gn:
                reg_branch.append(nn.GroupNorm(32, num_feat))
            reg_branch.append(nn.ReLU(inplace=True))

        self.cls_conv = nn.Sequential(*cls_branch)
        self.reg_conv = nn.Sequential(*reg_branch)

        self.cls_logits = nn.Conv2d(num_feat,
                                    num_cls,
                                    kernel_size=3,
                                    padding=1)
        self.reg_preds = nn.Conv2d(num_feat, 4, kernel_size=3, padding=1)
        self.ctr_logits = nn.Conv2d(num_feat, 1, kernel_size=3, padding=1)

        if init_weights:
            self._initialize_weights()

        nn.init.constant_(self.cls_logits.bias, -math.log(
            (1 - prior) / prior))  # cls bias init
        self.scale_exp = nn.ModuleList([ScaleExp(1.0) for _ in range(5)])

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, feats):
        cls_logits = []
        reg_preds = []
        ctr_logits = []

        for i, feat in enumerate(feats):
            cls_conv_out = self.cls_conv(feat)
            reg_conv_out = self.reg_conv(feat)

            cls_logits.append(self.cls_logits(cls_conv_out))
            reg_preds.append(self.scale_exp[i](self.reg_preds(reg_conv_out)))

            if not self.ctr_on_reg:
                ctr_logits.append(self.ctr_logits(cls_conv_out))
            else:
                ctr_logits.append(self.ctr_logits(reg_conv_out))

        return cls_logits, reg_preds, ctr_logits


class ScaleExp(nn.Module):
    def __init__(self, init_value=1.0):
        super(ScaleExp, self).__init__()
        self.scale = nn.Parameter(torch.tensor([init_value],
                                               dtype=torch.float))

    def forward(self, x):
        return torch.exp(x * self.scale)


if __name__ == "__main__":

    import torch

    model = FCOSHead(num_cls=3)

    p7 = torch.rand(2, 256, 2, 2)
    p6 = torch.rand(2, 256, 4, 4)
    p5 = torch.rand(2, 256, 7, 7)
    p4 = torch.rand(2, 256, 14, 14)
    p3 = torch.rand(2, 256, 28, 28)

    out = model([p3, p4, p5, p6, p7])
    [print(stage_out.shape) for branch_out in out for stage_out in branch_out]
