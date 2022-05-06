# -*- coding: utf-8 -*-
"""
# @file name  : shufflenet.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-04-21
# @brief      : ShuffleNet模型类
# @reference  : https://github.com/pytorch/vision/blob/main/torchvision/models/shufflenetv2.py
"""

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = [
    'ShuffleNetV2', 'shufflenetv2_x0_5', 'shufflenetv2_x1_0',
    'shufflenetv2_x1_5', 'shufflenetv2_x2_0'
]

model_urls = {
    'shufflenetv2_x0.5':
    'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
    'shufflenetv2_x1.0':
    'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
}


def conv3x3(in_channels, out_channels, stride=1, groups=1, act=True):
    return nn.Sequential(
        nn.Conv2d(in_channels,
                  out_channels,
                  kernel_size=3,
                  stride=stride,
                  padding=1,
                  groups=groups,
                  bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True) if act else nn.Identity(),
    )


def conv1x1(in_channels, out_channels, stride=1, act=True):
    return nn.Sequential(
        nn.Conv2d(in_channels,
                  out_channels,
                  kernel_size=1,
                  stride=stride,
                  bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True) if act else nn.Identity(),
    )


def channel_split(x, split):
    """split a tensor into two pieces along channel dimension
    Args:
        x: input tensor
        split:(int) channel size for each pieces
    """
    assert x.size(1) == split * 2
    return torch.split(x, split, dim=1)


def channel_shuffle(x, groups):
    """channel shuffle operation
    Args:
        x: input tensor
        groups: input branch number
    """
    b, c, h, w = x.size()
    group_channels = int(c / groups)

    x = x.view(b, groups, group_channels, h, w)
    x = x.transpose(1, 2).contiguous()
    x = x.view(b, -1, h, w)

    return x


class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ShuffleUnit, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        shortcut, residual = [], []
        split_channels = int(out_channels / 2)
        if stride != 1 or in_channels != out_channels:
            shortcut.extend([
                # dw-linear
                conv3x3(in_channels,
                        in_channels,
                        stride=stride,
                        groups=in_channels,
                        act=False),
                # pw
                conv1x1(in_channels, split_channels),
            ])
            # pw
            residual.append(conv1x1(in_channels, split_channels))
        else:
            # pw
            residual.append(conv1x1(split_channels, split_channels))
        residual.extend([
            # dw-linear
            conv3x3(split_channels,
                    split_channels,
                    stride=stride,
                    groups=split_channels,
                    act=False),
            # pw
            conv1x1(split_channels, split_channels),
        ])
        self.shortcut = nn.Sequential(*shortcut)
        self.residual = nn.Sequential(*residual)

    def forward(self, x):
        if self.stride == 1 and self.in_channels == self.out_channels:
            shortcut, residual = channel_split(x, int(self.in_channels / 2))
        else:
            shortcut = x
            residual = x

        shortcut = self.shortcut(shortcut)
        residual = self.residual(residual)
        out = torch.cat([shortcut, residual], dim=1)

        return channel_shuffle(out, 2)


class ShuffleNetV2(nn.Module):
    def __init__(self, out_channels, init_weights=True):
        super(ShuffleNetV2, self).__init__()
        self.conv1 = conv3x3(3, 24, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage2 = self._make_stage(24, out_channels[0], 3)
        self.stage3 = self._make_stage(out_channels[0], out_channels[1], 7)
        self.stage4 = self._make_stage(out_channels[1], out_channels[2], 3)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.maxpool(c1)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)

        return c2, c3, c4, c5

    def _make_stage(self, in_channels, out_channels, repeat):
        layers = [ShuffleUnit(in_channels, out_channels, 2)]

        for _ in range(repeat):
            layers.append(ShuffleUnit(out_channels, out_channels, 1))

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


def _shufflenetv2(out_channels, name, pretrained=False):
    if pretrained:
        model = ShuffleNetV2(out_channels, init_weights=False)
        model_weights = model_zoo.load_url(model_urls[name])
        state_dict = {
            k:
            model_weights[k] if k in model_weights else model.state_dict()[k]
            for k in model.state_dict()
        }
        model.load_state_dict(state_dict)
    else:
        model = ShuffleNetV2(out_channels)

    return model


def shufflenetv2_x0_5(pretrained=False):
    return _shufflenetv2([48, 96, 192, 1024], 'shufflenetv2_x0_5', pretrained)


def shufflenetv2_x1_0(pretrained=False):
    return _shufflenetv2([116, 232, 464, 1024], 'shufflenetv2_x1_0',
                         pretrained)


def shufflenetv2_x1_5():
    return ShuffleNetV2([176, 352, 704, 1024])


def shufflenetv2_x2_0():
    return ShuffleNetV2([244, 488, 976, 2048])


if __name__ == "__main__":

    import torch
    from torchsummary import summary

    model = shufflenetv2_x1_0()
    summary(model, (3, 224, 224), 2, device="cpu")

    x = torch.rand(2, 3, 224, 224)
    out = model(x)
    [print(stage_out.shape) for stage_out in out]
