# -*- coding: utf-8 -*-
"""
# @file name  : vgg.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-04
# @brief      : VGG模型类
# @reference  : https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py
"""

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ['VGG', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19']

model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


def conv3x3(in_channels, out_channels, stride=1, norm=False):
    layers = [
        nn.Conv2d(in_channels,
                  out_channels,
                  kernel_size=3,
                  stride=stride,
                  padding=1)
    ]
    if norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU(inplace=True))

    return layers


class VGG(nn.Module):
    def __init__(self, features, init_weights=True):
        super(VGG, self).__init__()
        self.layer0 = features[0]
        self.layer1 = features[1]
        self.layer2 = features[2]
        self.layer3 = features[3]
        self.layer4 = features[4]

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        c1 = self.layer0(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return c2, c3, c4, c5

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


def make_layers(cfg, norm=False):
    stages = []
    in_channels = 3
    for u in cfg:
        layers = []
        for v in u:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.extend(conv3x3(in_channels, v, norm=norm))
                in_channels = v
        stages.append(nn.Sequential(*layers))

    return nn.Sequential(*stages)


cfgs = {
    'D': [[64, 64], ['M', 128, 128], ['M', 256, 256, 256],
          ['M', 512, 512, 512], ['M', 512, 512, 512]],
    'E': [[64, 64], ['M', 128, 128], ['M', 256, 256, 256, 256],
          ['M', 512, 512, 512, 512], ['M', 512, 512, 512, 512]],
}


def _vgg(cfg, norm, name, pretrained=False):
    if pretrained:
        model = VGG(make_layers(cfgs[cfg], norm), init_weights=False)
        model_weights = model_zoo.load_url(model_urls[name])
        state_dict = {
            k: v
            for k, v in model.state_dict().items()
            if 'num_batches_tracked' not in k
        }
        state_dict = {k: v for k, v in zip(state_dict, model_weights.values())}
        model.load_state_dict(state_dict)
    else:
        model = VGG(make_layers(cfgs[cfg], norm))

    return model


def vgg16(pretrained=False):
    return _vgg('D', False, 'vgg16', pretrained)


def vgg16_bn(pretrained=False):
    return _vgg('D', True, 'vgg16_bn', pretrained)


def vgg19(pretrained=False):
    return _vgg('E', False, 'vgg19', pretrained)


def vgg19_bn(pretrained=False):
    return _vgg('E', True, 'vgg19_bn', pretrained)


if __name__ == "__main__":

    import torch
    from torchsummary import summary

    model = vgg16_bn()
    summary(model, (3, 224, 224), 2, device="cpu")

    x = torch.rand(2, 3, 224, 224)
    outs = model(x)
    [print(stage_outs.shape) for stage_outs in outs]
