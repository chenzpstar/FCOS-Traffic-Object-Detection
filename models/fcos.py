# -*- coding: utf-8 -*-
"""
# @file name  : fcos.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-05
# @brief      : FCOS模型类
"""

import torch.nn as nn

from backbones import darknet19_bn
from config import FCOSConfig
from detect import FCOSDetect
from head import FCOSHead
from loss import FCOSLoss
from necks import darknet19_fpn
from target import FCOSTarget


class FCOS(nn.Module):
    def __init__(self, backbone, neck, mode="train", cfg=None):
        super(FCOS, self).__init__()
        if cfg is None:
            self.cfg = FCOSConfig
        else:
            self.cfg = cfg

        self.backbone = backbone(pretrained=self.cfg.pretrained)
        self.neck = neck(num_feat=self.cfg.num_feat, use_p5=self.cfg.use_p5)
        self.head = FCOSHead(
            num_feat=self.cfg.num_feat,
            num_cls=self.cfg.num_cls,
            use_gn=self.cfg.use_gn,
            ctr_on_reg=self.cfg.ctr_on_reg,
            prior=self.cfg.prior,
        )

        self.mode = mode
        if mode == "train":
            self.target_layer = FCOSTarget(self.cfg)
            self.loss_layer = FCOSLoss(self.cfg)
        elif mode == "inference":
            self.detect_layer = FCOSDetect(self.cfg)
            self.clip_boxes = ClipBoxes()

    def forward(self, inputs):
        if self.mode == "train":
            imgs, cls_ids, boxes = inputs

            backbone_out = self.backbone(imgs)
            neck_out = self.neck(backbone_out)
            preds = self.head(neck_out)

            targets = self.target_layer(preds, cls_ids, boxes)
            losses = self.loss_layer(preds, targets)

            return losses

        elif self.mode == "inference":
            imgs = inputs

            backbone_out = self.backbone(imgs)
            neck_out = self.neck(backbone_out)
            preds = self.head(neck_out)

            cls_scores, cls_ids, boxes = self.detect_layer(preds)
            boxes = [
                self.clip_boxes(img, box) for img, box in zip(imgs, boxes)
            ]

            return cls_scores, cls_ids, boxes


class ClipBoxes(nn.Module):
    def __init__(self):
        super(ClipBoxes, self).__init__()

    def forward(self, img, boxes):
        h, w = img.shape[-2:]  # chw
        boxes = boxes.clamp(min=0)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(max=w - 1)
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(max=h - 1)

        return boxes


if __name__ == "__main__":

    import torch
    torch.manual_seed(0)

    flag = 2

    if flag == 1:
        model = FCOS(darknet19_bn, darknet19_fpn, mode="train")

        imgs = torch.rand(2, 3, 224, 224)
        cls_ids = torch.rand(2, 3)
        boxes = torch.rand(2, 3, 4)

        out = model((imgs, cls_ids, boxes))
        [print(branch_out.item()) for branch_out in out]

    if flag == 2:
        model = FCOS(darknet19_bn, darknet19_fpn, mode="inference")

        imgs = torch.rand(2, 3, 224, 224)
        out = model(imgs)
        [
            print(batch_out.shape) for result_out in out
            for batch_out in result_out
        ]
