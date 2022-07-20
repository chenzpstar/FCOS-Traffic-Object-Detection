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


def build_backbone(name, pretrained=False):
    if name == "vgg16":
        backbone, stage_channels = vgg16_bn(pretrained)
    elif name == "resnet50":
        backbone, stage_channels = resnet50(pretrained)
    elif name == "darknet19":
        backbone, stage_channels = darknet19(pretrained)
    elif name == "darknet53":
        backbone, stage_channels = darknet53(pretrained)
    elif name == "mobilenet":
        backbone, stage_channels = mobilenetv2(pretrained)
    elif name == "shufflenet":
        backbone, stage_channels = shufflenetv2_x1_0(pretrained)
    elif name == "efficientnet":
        backbone, stage_channels = efficientnetv2_s(pretrained)
    else:
        raise NotImplementedError(
            "backbone only implemented ['vgg16', 'resnet50', 'darknet19', 'darknet53', 'mobilenet', 'shufflenet', 'efficientnet']"
        )

    return backbone, stage_channels


def build_neck(name, in_channels, out_channels, use_p5):
    if name == "fpn":
        neck = FPN(in_channels, out_channels, use_p5)
    elif name == "pan":
        neck = PAN(in_channels, out_channels)
    elif name == "bifpn":
        neck = BiFPN(in_channels, out_channels)
    else:
        raise NotImplementedError(
            "neck only implemented ['fpn', 'pan', 'bifpn']")

    return neck


class FCOS(nn.Module):
    def __init__(self, cfg=None):
        super(FCOS, self).__init__()
        self.cfg = FCOSConfig if cfg is None else cfg

        # 1. backbone
        self.backbone, self.stage_channels = build_backbone(
            self.cfg.backbone, self.cfg.pretrained)

        # 2. neck
        self.neck = build_neck(self.cfg.neck, self.stage_channels,
                               self.cfg.num_channels, self.cfg.use_p5)

        # 3. head
        self.head = FCOSHead(self.cfg.num_channels, self.cfg.num_convs,
                             self.cfg.num_classes, self.cfg.prior,
                             self.cfg.use_gn, self.cfg.ctr_on_reg,
                             self.cfg.strides)

    def train(self, mode=True):
        super(FCOS, self).train(mode)
        if self.cfg.freeze_bn:
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.cfg.freeze_bn_affine:
                        for param in m.parameters():
                            param.requires_grad = False

        if self.cfg.freeze_backbone:
            for m in self.backbone.modules():
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def forward(self, imgs):
        backbone_feats = self.backbone(imgs)
        neck_feats = self.neck(backbone_feats)
        preds = self.head(neck_feats)

        return preds


class FCOSDetector(nn.Module):
    def __init__(self, cfg=None):
        super(FCOSDetector, self).__init__()
        self.cfg = FCOSConfig if cfg is None else cfg
        self.fcos = FCOS(self.cfg)
        self.target_layer = FCOSTarget(self.cfg)
        self.loss_layer = FCOSLoss(self.cfg)
        self.detect_layer = FCOSDetect(self.cfg)

    def forward(self, imgs, annos=None, mode="train"):
        preds = self.fcos(imgs)

        if mode == "train":
            assert annos is not None
            labels, boxes = annos
            targets = self.target_layer(labels, boxes, preds[-1])

            return self.loss_layer(preds, targets)

        elif mode == "infer":
            return self.detect_layer(preds, imgs)


if __name__ == "__main__":

    import torch
    torch.manual_seed(0)

    # flag = 1
    flag = 2

    model = FCOSDetector()

    imgs = torch.rand(2, 3, 224, 224)
    labels = torch.randint(1, 4, (2, 3))
    boxes = torch.rand(2, 3, 4)

    if flag == 1:
        outs = model(imgs, (labels, boxes))
        [print(branch_outs.item()) for branch_outs in outs]

    if flag == 2:
        outs = model(imgs, mode="infer")
        [
            print(batch_outs.shape) for result_outs in outs
            for batch_outs in result_outs
        ]
