# -*- coding: utf-8 -*-
"""
@Date:   2020/6/3 15:58
@Author: Pangpd
@FileName: FDSSC.py
@IDE: PyCharm
@Description:
    A Fast Dense Spectral–Spatial Convolution Network
    Framework for Hyperspectral Images Classification
    (FDSSC)
"""
import math

import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import torch.nn.functional as F


class spectral_conv(nn.Module):
    def __init__(self, input, output):
        super(spectral_conv, self).__init__()
        self.bn = nn.BatchNorm3d(input)
        self.prelu = nn.PReLU()
        self.spectral_conv = nn.Conv3d(input, output, (7, 1, 1), stride=(1, 1, 1), padding=(3, 0, 0), bias=False)

    def forward(self, x):
        x = self.bn(x)
        x = self.prelu(x)
        x = self.spectral_conv(x)
        return x


class spatial_conv(nn.Module):
    def __init__(self, input, output):
        super(spatial_conv, self).__init__()
        self.bn = nn.BatchNorm3d(input)
        self.prelu = nn.PReLU()
        self.spectral_conv = nn.Conv3d(input, output, (1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)

    def forward(self, x):
        x = self.bn(x)
        x = self.prelu(x)
        x = self.spectral_conv(x)
        return x


class FDSSC(nn.Module):

    def __init__(self, input_channels, n_classes):
        super(FDSSC, self).__init__()
        self.input_channels = input_channels

        self.conv0 = nn.Conv3d(1, 24, (7, 1, 1), stride=(2, 1, 1), bias=False)  # (x,y,z)三维深度上的步长是x，行方向步长是y，列方向步长z

        self.conv1_1 = spectral_conv(24, 12)

        self.conv1_2 = spectral_conv(36, 12)

        self.conv1_3 = spectral_conv(48, 12)

        self.features_size = self._get_trans_size().shape[2]

        self.bn1 = nn.BatchNorm3d(60)
        self.prelu1 = nn.PReLU()
        self.tran1 = nn.Conv3d(60, 200, (self.features_size, 1, 1), stride=(1, 1, 1),
                               bias=False)  # reshape之前 1x1xb,200卷积

        self.bn2 = nn.BatchNorm3d(1)
        self.prelu2 = nn.PReLU()
        self.tran2 = nn.Conv3d(1, 24, (200, 3, 3), stride=(1, 1, 1), bias=False)  # reshape之前 3x3x200,24卷积

        self.conv2_1 = spatial_conv(24, 12)

        self.conv2_2 = spatial_conv(36, 12)

        self.conv2_3 = spatial_conv(48, 12)

        self.final_bn = nn.BatchNorm3d(60)
        self.final_prelu = nn.PReLU()

        self.avgpool = nn.AvgPool3d((1, 7, 7), stride=(1, 1, 1))
        self.fc = nn.Linear(60, n_classes)
        self.dropout = nn.Dropout3d(0.5)

        # 定义权值初始化
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
        x_0 = x.unsqueeze(1)
        x_0 = self.conv0(x_0)  # [32, 24, 97, 9, 9]
        x_1_1 = self.conv1_1(x_0)  # 第一个dens块 [32, 12, 97, 9, 9]

        x_1_2_iuput = torch.cat((x_0, x_1_1), 1)  # 第二个dens块的输入
        x_1_2 = self.conv1_2(x_1_2_iuput)  # [32, 12, 97, 9, 9]

        x_1_3_iuput = torch.cat((x_0, x_1_1, x_1_2), 1)  # 第三个dens块的输入
        x_1_3 = self.conv1_3(x_1_3_iuput)

        x_tran1_input = torch.cat((x_0, x_1_1, x_1_2, x_1_3), 1)  # reshape之前 1x1xb的输入 [32, 60, 97, 9, 9]

        x_tran1_input = self.prelu1(self.bn1(x_tran1_input))
        x_tran1 = self.tran1(x_tran1_input)  # torch.Size([32, 200, 1, 9, 9])
        x_tran1 = torch.reshape(x_tran1, (x_tran1.shape[0], x_tran1.shape[2], x_tran1.shape[1], x_tran1.shape[3],
                                          x_tran1.shape[4]))  # torch.Size([32, 1, 200, 9, 9])

        x_tran1 = self.prelu2(self.bn2(x_tran1))
        x_1 = self.tran2(x_tran1)  # [32, 24, 1, 7, 7]

        x_2_1 = self.conv2_1(x_1)  # 空间特征提取：第一个dens块

        x_2_2_iuput = torch.cat((x_1, x_2_1), 1)  # 第二个dens块的输入
        x_2_2 = self.conv2_2(x_2_2_iuput)

        x_3_iuput = torch.cat((x_1, x_2_1, x_2_2), 1)  # 第三个dens块的输入
        x_2_3 = self.conv2_3(x_3_iuput)

        x_2 = torch.cat((x_1, x_2_1, x_2_2, x_2_3), 1)  # torch.Size([32, 60, 1, 7, 7])
        x = self.final_prelu(self.final_bn(x_2))
        x = self.avgpool(x)  # torch.Size([32, 60, 1, 1, 1])
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def _get_trans_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels, 9, 9))  # self.patch_size
            x_0 = self.conv0(x)
            x_1_1 = self.conv1_1(x_0)
            x_1_2_iuput = torch.cat((x_0, x_1_1), 1)
            x_1_2 = self.conv1_2(x_1_2_iuput)
            x_1_3_iuput = torch.cat((x_0, x_1_1, x_1_2), 1)
            x_1_3 = self.conv1_3(x_1_3_iuput)
            x_tran1_input = torch.cat((x_0, x_1_1, x_1_2, x_1_3), 1)

        return x_tran1_input
