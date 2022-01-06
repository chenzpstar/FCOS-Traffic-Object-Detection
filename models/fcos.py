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
            cfg = FCOSConfig
        self.backbone = backbone(pretrained=cfg.pretrained)
        self.neck = neck(num_feat=cfg.num_feat, use_p5=cfg.use_p5)
        self.head = head(
            num_feat=cfg.num_feat,
            num_cls=cfg.num_cls,
            use_gn=cfg.use_gn,
            ctr_on_reg=cfg.ctr_on_reg,
            prior=cfg.prior,
        )

    def forward(self, x):
        conv_out = self.backbone(x)
        proj_out = self.neck(conv_out[1:])
        cls_logits, reg_preds, ctr_logits = self.head(proj_out)

        return cls_logits, reg_preds, ctr_logits


if __name__ == "__main__":

    import torch

    model = FCOS(darknet19_bn, darknet19_fpn, FCOSHead)

    x = torch.rand(2, 3, 224, 224)
    out = model(x)
    [print(stage_out.shape) for branch_out in out for stage_out in branch_out]
