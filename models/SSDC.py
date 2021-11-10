# -*- coding: utf-8 -*-
"""
@Date:   2020/6/15 10:00
@Author: Pangpd
@FileName: SSDC.py
@IDE: PyCharm
@Description:
SSDC-DenseNet: A Cost-Effective End-to-End Spectral-Spatial DualChannel
Dense Network for Hyperspectral Image Classiﬁcation
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class InputLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, bias=False):
        super(InputLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu((self.conv(x)))


# BN-Relu-Conv-Dropout
class BnReLuConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, bias=False):
        super(BnReLuConv2d, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=bias)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(self.relu(self.bn(x)))
        x = self.dropout(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size, stride, padding=0, bias=False):
        super(BasicBlock, self).__init__()
        self.conv_1 = BnReLuConv2d(in_channels, growth_rate, kernel_size=kernel_size, stride=stride, padding=padding,
                                   bias=bias)

        self.conv_2 = BnReLuConv2d(in_channels + growth_rate, growth_rate, kernel_size=kernel_size,
                                   stride=stride, padding=padding, bias=bias)

        self.conv_3 = BnReLuConv2d(in_channels + growth_rate * 2, growth_rate, kernel_size=kernel_size, stride=stride,
                                   padding=padding,
                                   bias=bias)

    def forward(self, x):
        x_1 = self.conv_1(x)
        x = torch.cat((x, x_1), 1)  # 卷积2的输入
        x_2 = self.conv_2(x)
        x = torch.cat((x, x_2), 1)  # 卷积3的输入
        x_3 = self.conv_3(x)
        x = torch.cat((x, x_3), 1)
        return x


class MultiSclaeBlock(nn.Module):
    def __init__(self, in_channels=128, growth_rate=32):
        super(MultiSclaeBlock, self).__init__()
        self.conv_1 = BasicBlock(in_channels, growth_rate, kernel_size=1, stride=1, bias=False)
        self.conv_2 = BasicBlock(in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.avgpool = nn.AvgPool2d(3, stride=2)

    def forward(self, x):
        x_1 = self.conv_1(x)
        x_1 = self.avgpool(x_1)
        x_2 = self.conv_2(x)
        x_2 = self.avgpool(x_2)
        x = torch.cat((x_1, x_2), 1)
        return x


class FeatureFusionBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_classes):
        super(FeatureFusionBlock, self).__init__()

        self.bn_1 = nn.BatchNorm2d(in_channels)
        self.relu_1 = nn.ReLU(inplace=True)
        self.conv_1 = nn.Conv2d(in_channels, 64, kernel_size=1, stride=1)
        self.conv_2 = BasicBlock(64, growth_rate, kernel_size=1, stride=1)
        self.global_avgpool = nn.AvgPool2d(4, 4)
        self.bn_2 = nn.BatchNorm2d(160)
        self.relu_2 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(160, n_classes)

    def forward(self, x):
        x = self.relu_1(self.bn_1(x))
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.global_avgpool(x)
        x = self.relu_2(self.bn_2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SSDC(nn.Module):

    def __init__(self, n_bands, n_classes, growth_rate=32):
        super(SSDC, self).__init__()
        self.conv_1 = InputLayer(n_bands, 128)
        self.multi_conv = MultiSclaeBlock(in_channels=128, growth_rate=growth_rate)  # 3x3x448
        self.feature_fusion_conv = FeatureFusionBlock(448, growth_rate=growth_rate, n_classes=n_classes)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.multi_conv(x)
        x = self.feature_fusion_conv(x)
        return x
