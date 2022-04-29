# -*- coding: utf-8 -*-
"""
# @file name  : efficientnet.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-04-22
# @brief      : EfficientNet模型类
# @reference  : https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py
"""

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ['efficientnetv2_s', 'efficientnetv2_m', 'efficientnetv2_l']

model_urls = {
    'efficientnetv2_s':
    'https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth',
    'efficientnetv2_m':
    'https://download.pytorch.org/models/efficientnet_v2_m-dc08266a.pth',
    'efficientnetv2_l':
    'https://download.pytorch.org/models/efficientnet_v2_l-59c71312.pth',
}

# SiLU (Swish) activation function
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, in_channels, squeeze_channels, reduction=4):
        super(SELayer, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, squeeze_channels // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(squeeze_channels // reduction, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


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
        nn.SiLU(inplace=True) if act else nn.Identity(),
    )


def conv1x1(in_channels, out_channels, stride=1, act=True):
    return nn.Sequential(
        nn.Conv2d(in_channels,
                  out_channels,
                  kernel_size=1,
                  stride=stride,
                  bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True) if act else nn.Identity(),
    )


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t, fused=False):
        super(MBConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        layers = []
        exp_channels = int(in_channels * t)
        if fused:
            if t != 1:
                layers.extend([
                    # fused
                    conv3x3(in_channels, exp_channels, stride=stride),
                    # pw-linear
                    conv1x1(exp_channels, out_channels, act=False),
                ])
            else:
                # fused
                layers.append(conv3x3(in_channels, out_channels,
                                      stride=stride))
        else:
            if t != 1:
                # pw
                layers.append(conv1x1(in_channels, exp_channels))
            layers.extend([
                # dw
                conv3x3(exp_channels,
                        exp_channels,
                        stride=stride,
                        groups=exp_channels),
                # attention
                SELayer(exp_channels, in_channels),
                # pw-linear
                conv1x1(exp_channels, out_channels, act=False),
            ])
        self.residual = nn.Sequential(*layers)

    def forward(self, x):
        residual = self.residual(x)
        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x

        return residual


class EfficientNetv2(nn.Module):
    def __init__(self, out_channels, repeat, init_weights=True):
        super(EfficientNetv2, self).__init__()
        self.stage0 = conv3x3(3, out_channels[0], 2)
        self.stage1 = self._make_stage(out_channels[0], out_channels[0],
                                       repeat[0], 1, 1, True)
        self.stage2 = self._make_stage(out_channels[0], out_channels[1],
                                       repeat[1], 2, 4, True)
        self.stage3 = self._make_stage(out_channels[1], out_channels[2],
                                       repeat[2], 2, 4, True)
        self.stage4 = self._make_stage(out_channels[2], out_channels[3],
                                       repeat[3], 2, 4)
        self.stage5 = self._make_stage(out_channels[3], out_channels[4],
                                       repeat[4], 1, 6)
        self.stage6 = self._make_stage(out_channels[4], out_channels[5],
                                       repeat[5], 2, 6)
        self.stage7 = self._make_stage(
            out_channels[5], out_channels[6], repeat[6], 1,
            6) if len(out_channels) == 7 else nn.Identity()

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        c1 = self.stage1(self.stage0(x))
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage5(self.stage4(c3))
        c5 = self.stage7(self.stage6(c4))

        return c2, c3, c4, c5

    def _make_stage(self,
                    in_channels,
                    out_channels,
                    repeat,
                    stride,
                    t,
                    fused=False):
        layers = [MBConv(in_channels, out_channels, stride, t, fused)]

        for _ in range(1, repeat):
            layers.append(MBConv(out_channels, out_channels, 1, t, fused))

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


def _efficientnetv2(out_channels, repeat, name, pretrained=False):
    if pretrained:
        model = EfficientNetv2(out_channels, repeat, init_weights=False)
        model_weights = model_zoo.load_url(model_urls[name])
        state_dict = {
            k:
            model_weights[k] if k in model_weights else model.state_dict()[k]
            for k in model.state_dict()
        }
        model.load_state_dict(state_dict)
    else:
        model = EfficientNetv2(out_channels, repeat)

    return model


def efficientnetv2_s(pretrained=False):
    return _efficientnetv2([24, 48, 64, 128, 160, 256], [2, 4, 4, 6, 9, 15],
                           'efficientnetv2_s', pretrained)


def efficientnetv2_m(pretrained=False):
    return _efficientnetv2([24, 48, 80, 160, 176, 304, 512],
                           [3, 5, 5, 7, 14, 18, 5], 'efficientnetv2_m',
                           pretrained)


def efficientnetv2_l(pretrained=False):
    return _efficientnetv2([32, 64, 96, 192, 224, 384, 640],
                           [4, 7, 7, 10, 19, 25, 7], 'efficientnetv2_l',
                           pretrained)


if __name__ == '__main__':

    import torch
    from torchsummary import summary

    model = efficientnetv2_s()
    summary(model, (3, 224, 224), 2, device="cpu")

    x = torch.rand(2, 3, 224, 224)
    out = model(x)
    [print(stage_out.shape) for stage_out in out]
