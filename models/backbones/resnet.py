# -*- coding: utf-8 -*-
"""
# @file name  : resnet.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-04
# @brief      : ResNet模型类
# @reference  : https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, ".."))

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from layers import conv1x1, conv3x3, conv7x7

__all__ = [
    'ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
]

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet50_v2': 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet101_v2':
    'https://download.pytorch.org/models/resnet101-cd907fc2.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnet152_v2':
    'https://download.pytorch.org/models/resnet152-f82ba261.pth',
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, num_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, num_channels, stride=stride)
        self.conv2 = conv3x3(num_channels, num_channels, act=None)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, num_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(in_channels, num_channels)
        self.conv2 = conv3x3(num_channels, num_channels, stride=stride)
        self.conv3 = conv1x1(num_channels,
                             num_channels * self.expansion,
                             act=None)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 init_weights=True,
                 zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = conv7x7(3, 64, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = self._make_layer(block, 64, layers[0])
        self.conv3 = self._make_layer(block, 128, layers[1], stride=2)
        self.conv4 = self._make_layer(block, 256, layers[2], stride=2)
        self.conv5 = self._make_layer(block, 512, layers[3], stride=2)

        if init_weights:
            self._initialize_weights()

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.zeros_(m.conv3.norm.weight)
                elif isinstance(m, BasicBlock):
                    nn.init.zeros_(m.conv2.norm.weight)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(self.maxpool(c1))
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)

        return c2, c3, c4, c5

    def _make_layer(self, block, num_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != num_channels * block.expansion:
            downsample = conv1x1(self.in_channels,
                                 num_channels * block.expansion,
                                 stride=stride,
                                 act=None)

        layers = [block(self.in_channels, num_channels, stride, downsample)]
        self.in_channels = num_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, num_channels))

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


def _resnet(block, layers, name, pretrained=False):
    if pretrained:
        model = ResNet(block, layers, init_weights=False)
        model_weights = model_zoo.load_url(model_urls[name])
        if not any('num_batches_tracked' in k for k in model_weights.keys()):
            model_keys = (k for k in model.state_dict().keys()
                          if 'num_batches_tracked' not in k)
            model_dict = dict(zip(model_keys, model_weights.values()))
        else:
            model_dict = dict(
                zip(model.state_dict().keys(), model_weights.values()))
        model.load_state_dict(model_dict)
    else:
        model = ResNet(block, layers)

    return model


def resnet18(pretrained=False):
    backbone = _resnet(BasicBlock, (2, 2, 2, 2), 'resnet18', pretrained)
    out_channels = (512, 256, 128, 64)

    return backbone, out_channels


def resnet34(pretrained=False):
    backbone = _resnet(BasicBlock, (3, 4, 6, 3), 'resnet34', pretrained)
    out_channels = (512, 256, 128, 64)

    return backbone, out_channels


def resnet50(pretrained=False):
    backbone = _resnet(Bottleneck, (3, 4, 6, 3), 'resnet50_v2', pretrained)
    out_channels = (2048, 1024, 512, 256)

    return backbone, out_channels


def resnet101(pretrained=False):
    backbone = _resnet(Bottleneck, (3, 4, 23, 3), 'resnet101_v2', pretrained)
    out_channels = (2048, 1024, 512, 256)

    return backbone, out_channels


def resnet152(pretrained=False):
    backbone = _resnet(Bottleneck, (3, 8, 36, 3), 'resnet152_v2', pretrained)
    out_channels = (2048, 1024, 512, 256)

    return backbone, out_channels


if __name__ == "__main__":

    import torch
    from torchsummary import summary

    model = resnet50()[0]
    summary(model, (3, 224, 224), 2, device="cpu")

    x = torch.rand(2, 3, 224, 224)
    outs = model(x)
    [print(stage_outs.shape) for stage_outs in outs]
