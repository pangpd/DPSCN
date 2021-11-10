# -*- coding: utf-8 -*-
"""
@Date:   2020/6/14 16:50
@Author: Pangpd
@FileName: SSRN_BAK.py
@IDE: PyCharm
@Description: 
"""
import math

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os


class InputLayer(nn.Module):
    def __init__(self, input_channels, init_channels, kernel_size, stride, bias=False):
        super(InputLayer, self).__init__()
        self.conv = nn.Conv3d(input_channels, init_channels, kernel_size=kernel_size, stride=stride, bias=bias)
        self.bn = nn.BatchNorm3d(init_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.conv_1 = Conv3dBnReLu(in_channels, out_channels, kernel_size=(85, 1, 1), stride=(1, 1, 1),
                                   padding=0)
        self.conv_2 = Conv3dBnReLu(1, in_channels, kernel_size=(128, 3, 3), stride=(1, 1, 1),
                                   padding=0)  # 3x3

    def forward(self, x):
        x = self.conv_1(x)  # 1x1  -> 7x7x1,128
        x = torch.reshape(x, (x.shape[0], x.shape[2], x.shape[1], x.shape[3], x.shape[4]))
        x = self.conv_2(x)  # 3x3卷积
        return x


class BnReLu(nn.Module):
    def __init__(self, in_channels):
        super(BnReLu, self).__init__()
        self.bn = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(x))


class Conv3dBnReLu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv3dBnReLu, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class BnReLuConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(BnReLuConv3d, self).__init__()
        self.bn = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)

    def forward(self, x):
        return self.conv(self.relu(self.bn(x)))


class BuildBlock(nn.Module):
    def __init__(self, in_channels, out_channels, feature):
        super(BuildBlock, self).__init__()
        self.feature = feature
        if feature == 'spectral':  # 1x1
            self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=(7, 1, 1), stride=(1, 1, 1),
                                    padding=(3, 0, 0), bias=False)
            self.conv_2 = BnReLuConv3d(in_channels, out_channels, kernel_size=(7, 1, 1), stride=(1, 1, 1),
                                       padding=(3, 0, 0))
            self.conv_3 = BnReLuConv3d(in_channels, out_channels, kernel_size=(7, 1, 1), stride=(1, 1, 1),
                                       padding=(3, 0, 0))
            self.conv_4 = BnReLuConv3d(in_channels, out_channels, kernel_size=(7, 1, 1), stride=(1, 1, 1),
                                       padding=(3, 0, 0))

        if feature == 'spatial':  # 3x3
            self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), stride=(1, 1, 1),
                                    padding=(0, 1, 1))

            self.conv_2 = BnReLuConv3d(in_channels, out_channels, kernel_size=(1, 3, 3), stride=(1, 1, 1),
                                       padding=(0, 1, 1))

            self.conv_3 = BnReLuConv3d(in_channels, out_channels, kernel_size=(1, 3, 3), stride=(1, 1, 1),
                                       padding=(0, 1, 1))

            self.conv_4 = BnReLuConv3d(in_channels, in_channels, kernel_size=(1, 3, 3), stride=(1, 1, 1),
                                       padding=(0, 1, 1))

    def forward(self, x):
        if self.feature == 'spectral':  # 3x3
            # 输入：(7x7x97,24)
            res_1 = self.conv_1(x)
            res_1 = self.conv_2(res_1)
            x = x + res_1
            res_2 = self.conv_3(x)
            res_2 = self.conv_4(res_2)
            x = x + res_2
        if self.feature == 'spatial':
            # 输入：(5x5x1,24)
            res_1 = self.conv_1(x)
            res_1 = self.conv_2(res_1)
            x = x + res_1

            res_2 = self.conv_3(x)
            res_2 = self.conv_4(res_2)
            x = x + res_2

        return x


class SSRN(nn.Module):

    def __init__(self, n_bands, n_classes):
        super(SSRN, self).__init__()
        self.n_bands = n_bands
        self.conv1 = InputLayer(1, 24, kernel_size=(7, 1, 1), stride=(2, 1, 1), bias=False)

        self.spectral_conv = BuildBlock(24, 24, feature='spectral')  # 光谱特征提取
        self.bn_relu_1 = BnReLu(24)
        self.bn_relu_2 = BnReLu(24)
        self.trans_conv = TransitionLayer(24, 128)  # reshape 转换层
        self.spatial_conv = BuildBlock(24, 24, feature='spatial')  # 空间特征提取
        self.bn_relu_3 = BnReLu(24)
        self.bn_relu_4 = BnReLu(24)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        #self.avgpool = nn.AvgPool3d((1, 7, 7), stride=(1, 1, 1))
        self.dropout = nn.Dropout3d(0.5)
        self.fc = nn.Linear(24, n_classes)

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
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.spectral_conv(x)
        x = self.bn_relu_1(x)
        x = self.bn_relu_2(x)
        x = self.trans_conv(x)
        x = self.spatial_conv(x)
        x = self.bn_relu_3(x)
        x = self.bn_relu_4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
