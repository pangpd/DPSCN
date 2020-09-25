# -*- coding: utf-8 -*-
"""
@Date:   2020/6/11 12:40
@Author: Pangpd
@FileName: My_2D_DpnNet.py
@IDE: PyCharm
@Description: 尝试使用DPN的思想构建一个1x1卷积组合的模块  (全部使用2D卷积)
"""
import math

import torch
import torch.nn as nn


class InputLayer(nn.Module):
    def __init__(self, input, output, kernel_size):
        super(InputLayer, self).__init__()
        self.conv = nn.Conv2d(input, output, kernel_size=kernel_size, bias=False)
        # self.bn = nn.BatchNorm2d(output)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


# BN-Relu-Conv_1x1
class BnReLuConv2d(nn.Module):
    def __init__(self, input, output, kernel_size=1, stride=1, padding=0):
        super(BnReLuConv2d, self).__init__()
        self.bn = nn.BatchNorm2d(input)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(input, output, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

    def forward(self, x):
        return self.conv(self.relu(self.bn(x)))


class DualPathBlock(nn.Module):
    def __init__(self, init_channels, middle_channels, res_channels, dense_channels):
        super(DualPathBlock, self).__init__()
        self.init_channels = init_channels
        self.res_channels = res_channels  # g1
        self.dense_channels = dense_channels  # g2
        self.conv_1 = BnReLuConv2d(init_channels, middle_channels, kernel_size=1)  # 1x1
        self.conv_2 = BnReLuConv2d(middle_channels, middle_channels, kernel_size=1)  # 1x1
        self.conv_3 = BnReLuConv2d(init_channels + self.dense_channels, middle_channels, kernel_size=1)  # 1x1
        self.conv_4 = BnReLuConv2d(middle_channels, middle_channels, kernel_size=1)  # 1x1

    def forward(self, x):
        x_1 = x[:, :self.res_channels, :, :]  # 前半部分用于残差连接 C1
        x_2 = x[:, self.res_channels:, :, :]  # 后半部分用于密度连接 C2

        unit_1 = self.conv_1(x)
        unit_1 = self.conv_2(unit_1)
        out_1_1 = unit_1[:, :self.res_channels, :, :]  # 前半部分用于残差连接 C1
        out_1_2 = unit_1[:, self.res_channels:, :, :]  # 后半部分用于密度连接 C1

        res_1 = x_1 + out_1_1
        dense_1 = torch.cat([x_2, out_1_2], dim=1)
        x = torch.cat([res_1, dense_1], dim=1)

        unit_2 = self.conv_3(x)
        unit_2 = self.conv_4(unit_2)
        out_2_1 = unit_2[:, :self.res_channels, :, :]  # 前半部分用于残差连接
        out_2_2 = unit_2[:, self.res_channels:, :, :]  # 后半部分用于密度连接

        res_2 = res_1 + out_2_1
        dense_2 = torch.cat([dense_1, out_2_2], dim=1)

        return torch.cat([res_2, dense_2], dim=1)


class MyDpnNet(nn.Module):
    def __init__(self, input_bands, init_channels, middle_channels, res_rate, num_classes):
        super(MyDpnNet, self).__init__()
        self.input_bands = input_bands  # 输入图像的波段数
        self.init_channels = init_channels  # 初始卷积核个数
        self.middle_channels = middle_channels  #
        self.res_channels = math.ceil(middle_channels * res_rate)  # C1
        self.dense_channels = middle_channels - self.res_channels  # C2

        self.conv1 = InputLayer(input_bands, init_channels, kernel_size=1)  # 第一个1x1卷积,HSI降维

        self.spectral_conv = DualPathBlock(init_channels, middle_channels, self.res_channels,
                                           self.dense_channels)  # 光谱特提取

        self.spatial_conv = BnReLuConv2d(init_channels + self.dense_channels * 2,
                                         init_channels + self.dense_channels * 2,
                                         kernel_size=3)  # 3x3空间特征提取

        self.spe_spa_conv = DualPathBlock(init_channels + self.dense_channels * 2, middle_channels, self.res_channels,
                                          self.dense_channels)  # 空-谱融合特征

        self.final_conv = BnReLuConv2d(init_channels + self.dense_channels * 4, num_classes, kernel_size=1)  # 最终的1x1卷积
        self.final_bn = nn.BatchNorm2d(num_classes)
        self.final_relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2))  # 2x2平均池化
        # self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.global_avgpool = nn.AvgPool2d(3, 3)

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

    def forward(self, x):

        x = self.conv1(x)
        x = self.spectral_conv(x)
        x = self.spatial_conv(x)
        x = self.spe_spa_conv(x)
        x = self.final_conv(x)
        x = self.final_relu(self.final_bn(x))
        x = self.avgpool(x)
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        return x
