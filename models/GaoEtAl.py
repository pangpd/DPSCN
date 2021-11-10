# -*- coding: utf-8 -*-
"""
@Date:   2020/4/22 23:23
@Author: Pangpd
@FileName: GaoEtAl.py
@IDE: PyCharm
@Description:
    Convolutional neural network for spectral–spatial classification
    of hyperspectral images
"""

import torch
import torch.nn as nn
from torch.nn import init
import math
import torch.nn.functional as F


class Conv2dBnReLu(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(Conv2dBnReLu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x


class GaoEtAl(nn.Module):

    def __init__(self, input_channels, n_classes, growth_rate):
        super(GaoEtAl, self).__init__()
        self.input_channels = input_channels

        self.conv0 = Conv2dBnReLu(input_channels, growth_rate, kernel_size=3, bias=False)  # 3x3卷积

        self.conv1 = Conv2dBnReLu(growth_rate, growth_rate*2, kernel_size=1, bias=False)  # 第一个1x1卷积
        self.conv1_1 = Conv2dBnReLu(growth_rate*2, growth_rate*2, kernel_size=1, bias=False)  # 1x1卷积
        self.conv1_2 = Conv2dBnReLu(growth_rate*2, growth_rate*2, kernel_size=1, bias=False)  # 1x1卷积
        self.conv1_3 = Conv2dBnReLu(growth_rate*4, growth_rate*2, kernel_size=1, bias=False)  # 1x1卷积
        self.conv1_4 = Conv2dBnReLu(growth_rate*2, growth_rate*2, kernel_size=1, bias=False)  # 1x1卷积

        self.final_conv = Conv2dBnReLu(growth_rate*8, growth_rate*2, kernel_size=1, bias=False)  # 最后的1x1卷积
        self.avgpool = nn.AvgPool2d((2, 2), stride=(2, 2))
        self.global_avgpool = nn.AvgPool2d(3, 3)
        self.fc = nn.Linear(growth_rate*2, n_classes)

        # # 定义权值初始化
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         torch.nn.init.kaiming_normal(m.weight.data)
        #         if m.bias is not None:
        #             m.bias.data.zero_()
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #     elif isinstance(m, nn.Linear):
        #         torch.nn.init.normal_(m.weight.data, 0, 0.01)
        #         m.bias.data.zero_()
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
        x = self.conv0(x)  # 3x3
        x = self.conv1(x)  # 第一个1x1卷积

        x_dens1 = self.conv1_1(x)  # SC-FR模型开始 1x1卷积
        x_dens1 = self.conv1_2(x_dens1)  # 1x1
        x_dens1 = torch.cat((x, x_dens1), 1)  # 第一个dens块的输出

        x_dens2 = self.conv1_3(x_dens1)  # 1x1
        x_dens2 = self.conv1_4(x_dens2)  # 1x1

        x = torch.cat((x, x_dens1, x_dens2), 1)  # 第二个dens块的输出

        x = self.final_conv(x)
        x = self.avgpool(x)
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
