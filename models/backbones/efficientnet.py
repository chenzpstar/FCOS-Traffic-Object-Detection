# -*- coding: utf-8 -*-
"""
# @file name  : efficientnet.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-04-22
# @brief      : EfficientNet模型类
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
    def __init__(self, in_channels, out_channels, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, in_channels // reduction, bias=False),
            SiLU(),
            nn.Linear(in_channels // reduction, out_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def conv3x3(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels,
                  out_channels,
                  kernel_size=3,
                  stride=stride,
                  padding=1,
                  bias=False),
        nn.BatchNorm2d(out_channels),
        SiLU(),
    )


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t, fused=False):
        super(MBConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        exp_channels = int(in_channels * t)
        if fused:
            self.residual = nn.Sequential(
                # fused
                nn.Conv2d(in_channels,
                          exp_channels,
                          kernel_size=3,
                          stride=stride,
                          padding=1,
                          bias=False),
                nn.BatchNorm2d(exp_channels),
                SiLU(),
                SELayer(in_channels, exp_channels),
                # pw-linear
                nn.Conv2d(exp_channels,
                          out_channels,
                          kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Sequential(
                # pw
                nn.Conv2d(in_channels, exp_channels, kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(exp_channels),
                SiLU(),
                # dw
                nn.Conv2d(exp_channels,
                          exp_channels,
                          kernel_size=3,
                          stride=stride,
                          padding=1,
                          groups=exp_channels,
                          bias=False),
                nn.BatchNorm2d(exp_channels),
                SiLU(),
                SELayer(in_channels, exp_channels),
                # pw-linear
                nn.Conv2d(exp_channels,
                          out_channels,
                          kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = self.residual(x)
        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x

        return residual


class EfficientNetv2(nn.Module):
    def __init__(self, scale='s', init_weights=True):
        super(EfficientNetv2, self).__init__()
        if scale == 's':
            self.stage0 = conv3x3(3, 24, 2)
            self.stage1 = self._make_stage(24, 24, 2, 1, 1, fused=True)
            self.stage2 = self._make_stage(24, 48, 4, 2, 4, fused=True)
            self.stage3 = self._make_stage(48, 64, 4, 2, 4, fused=True)
            self.stage4 = self._make_stage(64, 128, 6, 2, 4)
            self.stage5 = self._make_stage(128, 160, 9, 1, 6)
            self.stage6 = self._make_stage(160, 272, 15, 2, 6)
            self.stage7 = nn.Sequential()

        elif scale == 'm':
            self.stage0 = conv3x3(3, 24, 2)
            self.stage1 = self._make_stage(24, 24, 3, 1, 1, fused=True)
            self.stage2 = self._make_stage(24, 48, 5, 2, 4, fused=True)
            self.stage3 = self._make_stage(48, 80, 5, 2, 4, fused=True)
            self.stage4 = self._make_stage(80, 160, 7, 2, 4)
            self.stage5 = self._make_stage(160, 176, 14, 1, 6)
            self.stage6 = self._make_stage(176, 304, 18, 2, 6)
            self.stage7 = self._make_stage(304, 512, 5, 1, 6)

        elif scale == 'l':
            self.stage0 = conv3x3(3, 32, 2)
            self.stage1 = self._make_stage(24, 32, 4, 1, 1, fused=True)
            self.stage2 = self._make_stage(32, 64, 7, 2, 4, fused=True)
            self.stage3 = self._make_stage(64, 96, 7, 2, 4, fused=True)
            self.stage4 = self._make_stage(96, 192, 10, 2, 4)
            self.stage5 = self._make_stage(192, 224, 19, 1, 6)
            self.stage6 = self._make_stage(224, 384, 25, 2, 6)
            self.stage7 = self._make_stage(384, 640, 7, 1, 6)

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
        layers = []
        layers.append(MBConv(in_channels, out_channels, stride, t, fused))

        while repeat - 1:
            layers.append(MBConv(out_channels, out_channels, 1, t, fused))
            repeat -= 1

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


def efficientnetv2_s(pretrained=False):
    if pretrained:
        model = EfficientNetv2(scale='s', init_weights=False)
        model_weights = model_zoo.load_url(model_urls['efficientnetv2_s'])
        state_dict = {
            k:
            model_weights[k] if k in model_weights else model.state_dict()[k]
            for k in model.state_dict()
        }
        model.load_state_dict(state_dict)
    else:
        model = EfficientNetv2(scale='s')

    return model


def efficientnetv2_m(pretrained=False):
    if pretrained:
        model = EfficientNetv2(scale='m', init_weights=False)
        model_weights = model_zoo.load_url(model_urls['efficientnetv2_m'])
        state_dict = {
            k:
            model_weights[k] if k in model_weights else model.state_dict()[k]
            for k in model.state_dict()
        }
        model.load_state_dict(state_dict)
    else:
        model = EfficientNetv2(scale='m')

    return model


def efficientnetv2_l(pretrained=False):
    if pretrained:
        model = EfficientNetv2(scale='l', init_weights=False)
        model_weights = model_zoo.load_url(model_urls['efficientnetv2_l'])
        state_dict = {
            k:
            model_weights[k] if k in model_weights else model.state_dict()[k]
            for k in model.state_dict()
        }
        model.load_state_dict(state_dict)
    else:
        model = EfficientNetv2(scale='l')

    return model


if __name__ == '__main__':

    import torch
    from torchsummary import summary

    model = efficientnetv2_s()
    summary(model, (3, 224, 224), 2, device="cpu")

    x = torch.rand(2, 3, 224, 224)
    out = model(x)
    [print(stage_out.shape) for stage_out in out]
