# -*- coding: utf-8 -*-
"""
# @file name  : darknet.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-04
# @brief      : DarkNet模型类
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, ".."))

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from layers import conv1x1, conv3x3

__all__ = ['DarkNet', 'darknet19', 'darknet53']

model_urls = {
    'darknet19':
    'https://github.com/gau-nernst/vision-toolbox/releases/download/v0.0.1/darknet19-da4bd7c9.pth',
    'darknet53':
    'https://github.com/gau-nernst/vision-toolbox/releases/download/v0.0.1/darknet53-7433abc1.pth',
}


class BasicBlock(nn.Module):
    expansion = 2

    def __init__(self, in_channels, num_channels, shortcut=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv1x1(in_channels, num_channels)
        self.conv2 = conv3x3(num_channels, num_channels * self.expansion)
        self.shortcut = shortcut

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.shortcut:
            out += residual

        return out


class DarkNet(nn.Module):
    def __init__(self, block, layers, shortcut=False, init_weights=True):
        super(DarkNet, self).__init__()
        self.in_channels = 32
        self.shortcut = shortcut
        self.conv0 = conv3x3(3, 32)
        self.conv1 = self._make_layer(block, 32, layers[0])
        self.conv2 = self._make_layer(block, 64, layers[1])
        self.conv3 = self._make_layer(block, 128, layers[2])
        self.conv4 = self._make_layer(block, 256, layers[3])
        self.conv5 = self._make_layer(block, 512, layers[4])

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        c1 = self.conv1(self.conv0(x))
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)

        return c2, c3, c4, c5

    def _make_layer(self, block, num_channels, num_blocks):
        if not self.shortcut:
            layers = [
                nn.MaxPool2d(kernel_size=2, stride=2),
                conv3x3(self.in_channels, num_channels * block.expansion),
            ]
        else:
            layers = [
                conv3x3(self.in_channels,
                        num_channels * block.expansion,
                        stride=2)
            ]
        self.in_channels = num_channels * block.expansion
        for _ in range(num_blocks):
            layers.append(block(self.in_channels, num_channels, self.shortcut))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


def _darknet(block, layers, shortcut, name, pretrained=False):
    if pretrained:
        model = DarkNet(block, layers, shortcut, init_weights=False)
        model_weights = model_zoo.load_url(model_urls[name])
        model_dict = dict(
            zip(model.state_dict().keys(), model_weights.values()))
        model.load_state_dict(model_dict)
    else:
        model = DarkNet(block, layers, shortcut)

    return model


def darknet19(pretrained=False):
    backbone = _darknet(BasicBlock, (0, 1, 1, 2, 2), False, 'darknet19',
                        pretrained)
    out_channels = (1024, 512, 256, 128)

    return backbone, out_channels


def darknet53(pretrained=False):
    backbone = _darknet(BasicBlock, (1, 2, 8, 8, 4), True, 'darknet53',
                        pretrained)
    out_channels = (1024, 512, 256, 128)

    return backbone, out_channels


if __name__ == "__main__":

    import torch
    from torchsummary import summary

    model = darknet19()[0]
    summary(model, (3, 224, 224), 2, device="cpu")

    x = torch.rand(2, 3, 224, 224)
    outs = model(x)
    [print(stage_outs.shape) for stage_outs in outs]
