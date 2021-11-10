# -*- coding: utf-8 -*-
"""
@Date:   2020/9/15 16:39
@Author: Pangpd
@FileName: MLP.py
@IDE: PyCharm
@Description:  多层感知机(MLP)
"""
import math

import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, n_bands, n_classes):
        super(MLP, self).__init__()
        self.fc_1 = nn.Linear(n_bands, int(n_bands * 2 / 3.) + 10)
        self.relu_1 = nn.ReLU(inplace=True)
        self.fc_2 = nn.Linear(int(n_bands * 2 / 3.) + 10, n_classes)

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
        x = x.squeeze(dim=-1).squeeze(dim=-1)
        x = x.unsqueeze(1)
        x = self.fc_1(x)

        x = self.relu_1(x)
        x = x.view(x.size(0), -1)
        x = self.fc_2(x)
        return x
