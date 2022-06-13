# -*- coding: utf-8 -*-
"""
# @file name  : fcos.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-05
# @brief      : FCOS模型类
"""

import torch.nn as nn

try:
    from .backbones import *
    from .config import FCOSConfig
    from .detect import FCOSDetect
    from .head import FCOSHead
    from .loss import FCOSLoss
    from .necks import FPN, PAN, BiFPN
    from .target import FCOSTarget
except:
    from backbones import *
    from config import FCOSConfig
    from detect import FCOSDetect
    from head import FCOSHead
    from loss import FCOSLoss
    from necks import FPN, PAN, BiFPN
    from target import FCOSTarget


class FCOS(nn.Module):
    def __init__(self, cfg=None):
        super(FCOS, self).__init__()
        self.cfg = FCOSConfig if cfg is None else cfg

        # 1. backbone
        if self.cfg.backbone == "vgg16":
            self.backbone = vgg16_bn(pretrained=self.cfg.pretrained)
            self.stage_channels = [512, 512, 256, 128]
        elif self.cfg.backbone == "resnet50":
            self.backbone = resnet50(pretrained=self.cfg.pretrained)
            self.stage_channels = [2048, 1024, 512, 256]
        elif self.cfg.backbone == "darknet19":
            self.backbone = darknet19(pretrained=self.cfg.pretrained)
            self.stage_channels = [1024, 512, 256, 128]
        elif self.cfg.backbone == "mobilenet":
            self.backbone = mobilenetv2(pretrained=self.cfg.pretrained)
            self.stage_channels = [320, 96, 32, 24]
        elif self.cfg.backbone == "shufflenet":
            self.backbone = shufflenetv2_x1_0(pretrained=self.cfg.pretrained)
            self.stage_channels = [464, 232, 116, 24]
        elif self.cfg.backbone == "efficientnet":
            self.backbone = efficientnetv2_s(pretrained=self.cfg.pretrained)
            self.stage_channels = [256, 160, 64, 48]

        # 2. neck
        if self.cfg.neck == "fpn":
            self.neck = FPN(in_channels=self.stage_channels,
                            num_channels=self.cfg.num_channels,
                            use_p5=self.cfg.use_p5)
        elif self.cfg.neck == "pan":
            self.neck = PAN(in_channels=self.stage_channels,
                            num_channels=self.cfg.num_channels)
        elif self.cfg.neck == "bifpn":
            self.neck = BiFPN(in_channels=self.stage_channels,
                              num_channels=self.cfg.num_channels)

        # 3. head
        self.head = FCOSHead(in_channels=self.cfg.num_channels,
                             num_convs=self.cfg.num_convs,
                             num_classes=self.cfg.num_classes,
                             use_gn=self.cfg.use_gn,
                             ctr_on_reg=self.cfg.ctr_on_reg,
                             prior=self.cfg.prior)

    def forward(self, imgs):
        backbone_feats = self.backbone(imgs)
        neck_feats = self.neck(backbone_feats)
        preds = self.head(neck_feats)

        return preds


class FCOSDetector(nn.Module):
    def __init__(self, mode="train", cfg=None):
        super(FCOSDetector, self).__init__()
        self.cfg = FCOSConfig if cfg is None else cfg
        self.fcos = FCOS(self.cfg)
        self.mode = mode
        if mode == "train":
            self.target_layer = FCOSTarget(self.cfg)
            self.loss_layer = FCOSLoss(self.cfg)
        elif mode == "inference":
            self.detect_layer = FCOSDetect(self.cfg)

    def forward(self, inputs):
        if self.mode == "train":
            imgs, labels, boxes = inputs
            preds = self.fcos(imgs)
            targets = self.target_layer(preds[0], labels, boxes)
            losses = self.loss_layer(preds, targets)

            return losses

        elif self.mode == "inference":
            imgs = inputs
            preds = self.fcos(imgs)
            outs = self.detect_layer(preds, imgs)

            return outs


if __name__ == "__main__":

    import torch
    torch.manual_seed(0)

    # flag = 1
    flag = 2

    if flag == 1:
        model = FCOSDetector(mode="train")

        imgs = torch.rand(2, 3, 224, 224)
        labels = torch.rand(2, 3)
        boxes = torch.rand(2, 3, 4)

        out = model((imgs, labels, boxes))
        [print(branch_out.item()) for branch_out in out]

    if flag == 2:
        model = FCOSDetector(mode="inference")

        imgs = torch.rand(2, 3, 224, 224)
        out = model(imgs)
        [
            print(batch_out.shape) for result_out in out
            for batch_out in result_out
        ]
