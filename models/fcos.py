# -*- coding: utf-8 -*-
"""
# @file name  : fcos.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-05
# @brief      : FCOS模型
"""

import torch.nn as nn

from backbones import darknet19_bn
from head import FCOSHead
from necks import darknet19_fpn

from config import FCOSConfig


class FCOS(nn.Module):
    def __init__(self, backbone, neck, head, cfg=None):
        super(FCOS, self).__init__()
        if cfg is None:
            self.cfg = FCOSConfig
        else:
            self.cfg = cfg

        self.backbone = backbone(pretrained=self.cfg.pretrained)
        self.neck = neck(num_feat=self.cfg.num_feat, use_p5=self.cfg.use_p5)
        self.head = head(
            num_feat=self.cfg.num_feat,
            num_cls=self.cfg.num_cls,
            use_gn=self.cfg.use_gn,
            ctr_on_reg=self.cfg.ctr_on_reg,
            prior=self.cfg.prior,
        )

    def forward(self, x):
        conv_out = self.backbone(x)
        proj_out = self.neck(conv_out)
        cls_logits, reg_preds, ctr_logits = self.head(proj_out)

        return cls_logits, reg_preds, ctr_logits


if __name__ == "__main__":

    import torch

    model = FCOS(darknet19_bn, darknet19_fpn, FCOSHead)

    x = torch.rand(2, 3, 224, 224)
    out = model(x)
    [print(stage_out.shape) for branch_out in out for stage_out in branch_out]
