# -*- coding: utf-8 -*-
"""
# @file name  : mobilenet.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-04-20
# @brief      : MobileNet模型类
# @reference  : https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py
"""

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ['MobileNetV2', 'mobilenetv2']

model_urls = {
    'mobilenetv2':
    'https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth',
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
        nn.ReLU6(inplace=True) if act else nn.Identity(),
    )


def conv1x1(in_channels, out_channels, stride=1, act=True):
    return nn.Sequential(
        nn.Conv2d(in_channels,
                  out_channels,
                  kernel_size=1,
                  stride=stride,
                  bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True) if act else nn.Identity(),
    )


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t):
        super(MBConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        layers = []
        exp_channels = int(in_channels * t)
        if t != 1:
            # pw
            layers.append(conv1x1(in_channels, exp_channels))
        layers.extend([
            # dw
            conv3x3(exp_channels,
                    exp_channels,
                    stride=stride,
                    groups=exp_channels),
            # pw-linear
            conv1x1(exp_channels, out_channels, act=False),
        ])
        self.residual = nn.Sequential(*layers)

    def forward(self, x):
        residual = self.residual(x)
        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x

        return residual


class MobileNetV2(nn.Module):
    def __init__(self, init_weights=True):
        super(MobileNetV2, self).__init__()
        self.stage0 = conv3x3(3, 32, 2)
        self.stage1 = MBConv(32, 16, 1, 1)
        self.stage2 = self._make_stage(16, 24, 2, 2, 6)
        self.stage3 = self._make_stage(24, 32, 3, 2, 6)
        self.stage4 = self._make_stage(32, 64, 4, 2, 6)
        self.stage5 = self._make_stage(64, 96, 3, 1, 6)
        self.stage6 = self._make_stage(96, 160, 3, 2, 6)
        self.stage7 = MBConv(160, 320, 1, 6)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        c1 = self.stage1(self.stage0(x))
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage5(self.stage4(c3))
        c5 = self.stage7(self.stage6(c4))

        return c2, c3, c4, c5

    def _make_stage(self, in_channels, out_channels, repeat, stride, t):
        layers = [MBConv(in_channels, out_channels, stride, t)]

        for _ in range(1, repeat):
            layers.append(MBConv(out_channels, out_channels, 1, t))

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


def mobilenetv2(pretrained=False):
    if pretrained:
        model = MobileNetV2(init_weights=False)
        model_weights = model_zoo.load_url(model_urls['mobilenetv2'])
        state_dict = {
            k: v
            for k, v in zip(model.state_dict(), model_weights.values())
        }
        model.load_state_dict(state_dict)
    else:
        model = MobileNetV2()

    return model


if __name__ == "__main__":

    import torch
    from torchsummary import summary

    model = mobilenetv2()
    summary(model, (3, 224, 224), 2, device="cpu")

    x = torch.rand(2, 3, 224, 224)
    out = model(x)
    [print(stage_out.shape) for stage_out in out]
