# -*- coding: utf-8 -*-
"""
# @file name  : resnet.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-04
# @brief      : ResNet模型类
# @reference  : https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
"""

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = [
    'ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
]

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-cd907fc2.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-f82ba261.pth',
}


def conv3x3(in_planes, out_planes, stride=1, act=True):
    return nn.Sequential(
        nn.Conv2d(in_planes,
                  out_planes,
                  kernel_size=3,
                  stride=stride,
                  padding=1,
                  bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True) if act else nn.Identity(),
    )


def conv1x1(in_planes, out_planes, stride=1, act=True):
    return nn.Sequential(
        nn.Conv2d(in_planes,
                  out_planes,
                  kernel_size=1,
                  stride=stride,
                  bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True) if act else nn.Identity(),
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.conv2 = conv3x3(planes, planes, act=False)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.conv3 = conv1x1(planes, planes * self.expansion, act=False)
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
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
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
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(self.maxpool(c1))
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)

        return c2, c3, c4, c5

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = conv1x1(self.inplanes,
                                 planes * block.expansion,
                                 stride=stride,
                                 act=False)

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def _resnet(block, layers, name, pretrained=False):
    if pretrained:
        model = ResNet(block, layers, init_weights=False)
        model_weights = model_zoo.load_url(model_urls[name])
        state_dict = {
            k:
            model_weights[k] if k in model_weights else model.state_dict()[k]
            for k in model.state_dict()
        }
        model.load_state_dict(state_dict)
    else:
        model = ResNet(block, layers)

    return model


def resnet18(pretrained=False):
    return _resnet(BasicBlock, [2, 2, 2, 2], 'resnet18', pretrained)


def resnet34(pretrained=False):
    return _resnet(BasicBlock, [3, 4, 6, 3], 'resnet34', pretrained)


def resnet50(pretrained=False):
    return _resnet(Bottleneck, [3, 4, 6, 3], 'resnet50', pretrained)


def resnet101(pretrained=False):
    return _resnet(Bottleneck, [3, 4, 23, 3], 'resnet101', pretrained)


def resnet152(pretrained=False):
    return _resnet(Bottleneck, [3, 8, 36, 3], 'resnet152', pretrained)


if __name__ == "__main__":

    import torch
    from torchsummary import summary

    model = resnet50()
    summary(model, (3, 224, 224), 2, device="cpu")

    x = torch.rand(2, 3, 224, 224)
    out = model(x)
    [print(stage_out.shape) for stage_out in out]
