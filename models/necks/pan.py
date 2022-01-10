# -*- coding: utf-8 -*-
"""
# @file name  : pan.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-04
# @brief      : PAN模型
"""

import torch.nn as nn
import torch.nn.functional as F


class PAN(nn.Module):
    def __init__(self, in_feats, num_feat=256, use_p5=True, init_weights=True):
        super(PAN, self).__init__()
        self.proj5 = nn.Conv2d(in_feats[0], num_feat, kernel_size=1, padding=0)
        self.proj4 = nn.Conv2d(in_feats[1], num_feat, kernel_size=1, padding=0)
        self.proj3 = nn.Conv2d(in_feats[2], num_feat, kernel_size=1, padding=0)
        self.proj2 = nn.Conv2d(in_feats[3], num_feat, kernel_size=1, padding=0)

        self.conv_p5 = nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1)
        self.conv_p4 = nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1)
        self.conv_p3 = nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1)
        self.conv_p2 = nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1)

        self.new3 = nn.Conv2d(num_feat,
                              num_feat,
                              kernel_size=3,
                              stride=2,
                              padding=1)
        self.new4 = nn.Conv2d(num_feat,
                              num_feat,
                              kernel_size=3,
                              stride=2,
                              padding=1)
        self.new5 = nn.Conv2d(num_feat,
                              num_feat,
                              kernel_size=3,
                              stride=2,
                              padding=1)

        self.conv_n3 = nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1)
        self.conv_n4 = nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1)
        self.conv_n5 = nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)

        if init_weights:
            self._initialize_weights()

    def upsample(self, src_feat, tar_feat):
        return F.interpolate(src_feat, tar_feat.shape[2:], mode="nearest")

    def forward(self, feats):
        c2, c3, c4, c5 = feats

        p5 = self.relu(self.proj5(c5))
        p4 = self.relu(self.proj4(c4)) + self.upsample(p5, c4)
        p3 = self.relu(self.proj3(c3)) + self.upsample(p4, c3)
        p2 = self.relu(self.proj2(c2)) + self.upsample(p3, c2)

        p5 = self.relu(self.conv_p5(p5))
        p4 = self.relu(self.conv_p4(p4))
        p3 = self.relu(self.conv_p3(p3))
        p2 = self.relu(self.conv_p2(p2))

        n2 = p2
        n3 = p3 + self.relu(self.new3(n2))
        n4 = p4 + self.relu(self.new3(n3))
        n5 = p5 + self.relu(self.new3(n4))

        n3 = self.relu(self.conv_n3(n3))
        n4 = self.relu(self.conv_n4(n4))
        n5 = self.relu(self.conv_n5(n5))

        return [n2, n3, n4, n5]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def vgg16_pan(**kwargs):
    model = PAN([512, 512, 256, 128], **kwargs)
    return model


def resnet50_pan(**kwargs):
    model = PAN([2048, 1024, 512, 256], **kwargs)
    return model


def darknet19_pan(**kwargs):
    model = PAN([1024, 512, 256, 128], **kwargs)
    return model


if __name__ == "__main__":

    import torch

    model = darknet19_pan()

    c5 = torch.rand(2, 1024, 7, 7)
    c4 = torch.rand(2, 512, 14, 14)
    c3 = torch.rand(2, 256, 28, 28)
    c2 = torch.rand(2, 128, 56, 56)

    out = model([c2, c3, c4, c5])
    [print(stage_out.shape) for stage_out in out]
