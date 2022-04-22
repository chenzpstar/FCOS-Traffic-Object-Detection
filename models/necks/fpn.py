# -*- coding: utf-8 -*-
"""
# @file name  : fpn.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-04
# @brief      : FPN模型类
"""

import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, backbone, num_feat=256, use_p5=True, init_weights=True):
        super(FPN, self).__init__()
        if backbone == "vgg16":
            in_feats = [512, 512, 256]
        elif backbone == "resnet50":
            in_feats = [2048, 1024, 512]
        elif backbone == "darknet19":
            in_feats = [1024, 512, 256]
        elif backbone == "mobilenet":
            in_feats = [320, 96, 32]
        elif backbone == "shufflenet":
            in_feats = [464, 232, 116]

        self.proj5 = nn.Conv2d(in_feats[0], num_feat, kernel_size=1, padding=0)
        self.proj4 = nn.Conv2d(in_feats[1], num_feat, kernel_size=1, padding=0)
        self.proj3 = nn.Conv2d(in_feats[2], num_feat, kernel_size=1, padding=0)

        self.conv5 = nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1)

        in_feat = num_feat if use_p5 else in_feats[0]
        self.conv6 = nn.Conv2d(in_feat,
                               num_feat,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.conv7 = nn.Conv2d(num_feat,
                               num_feat,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.use_p5 = use_p5

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight,
                                         mode='fan_out',
                                         nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def upsample(self, src_feat, tar_feat):
        return F.interpolate(src_feat, tar_feat.shape[2:], mode="nearest")

    def forward(self, feats):
        _, c3, c4, c5 = feats

        p5 = self.proj5(c5)
        p4 = self.proj4(c4) + self.upsample(p5, c4)
        p3 = self.proj3(c3) + self.upsample(p4, c3)

        p5 = self.conv5(p5)
        p4 = self.conv4(p4)
        p3 = self.conv3(p3)

        in_feat = p5 if self.use_p5 else c5
        p6 = self.conv6(in_feat)
        p7 = self.conv7(F.relu(p6))

        return [p3, p4, p5, p6, p7]


def vgg16_fpn(**kwargs):
    model = FPN(backbone="vgg16", **kwargs)
    return model


def resnet50_fpn(**kwargs):
    model = FPN(backbone="resnet50", **kwargs)
    return model


def darknet19_fpn(**kwargs):
    model = FPN(backbone="darknet19", **kwargs)
    return model


def mobilenet_fpn(**kwargs):
    model = FPN(backbone="mobilenet", **kwargs)
    return model


def shufflenet_fpn(**kwargs):
    model = FPN(backbone="shufflenet", **kwargs)
    return model


if __name__ == "__main__":

    import torch

    model = darknet19_fpn()

    c5 = torch.rand(2, 1024, 7, 7)
    c4 = torch.rand(2, 512, 14, 14)
    c3 = torch.rand(2, 256, 28, 28)
    c2 = torch.rand(2, 128, 56, 56)

    out = model([c2, c3, c4, c5])
    [print(stage_out.shape) for stage_out in out]
