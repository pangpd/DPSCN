# -*- coding: utf-8 -*-
"""
@Date:   2020/9/19 14:57
@Author: Pangpd
@FileName: test_show_train_test_map.py
@IDE: PyCharm
@Description: 
"""
import sys

import numpy as np
import os
from math import ceil

from utils import data_preprocess
from utils.show_maps import show_label

np.set_printoptions(linewidth=sys.maxsize)
np.set_printoptions(threshold=sys.maxsize)


def random_unison(a, b, rstate=None):
    assert len(a) == len(b)
    p = np.random.RandomState(seed=rstate).permutation(len(a))
    return a[p], b[p]


def split_data_percent(pixels, labels, train_samples, val_samples, rand_state=None):
    train_set_size = []  # 存储每类地物训练样本数
    for cl in np.unique(labels):
        pixels_cl = len(pixels[labels == cl])  # 第i类地物样本总数
        train_pixels_cl = min(ceil(pixels_cl * 0.3), train_samples)  # 计算第i类 min(地物样本数*0.3,T)的数量
        train_set_size.append(train_pixels_cl)  # 存储每类地物的训练样本数

    val_set_size = [ceil(i * val_samples) for i in train_set_size]  # 存储每类地物的验证样本数

    pixels_number = np.unique(labels, return_counts=1)[1]  # 全部样本数

    tr_size = int(sum(train_set_size))
    val_size = int(sum(val_set_size))
    te_size = int(sum(pixels_number)) - tr_size - val_size
    sizetr = np.array([tr_size] + list(pixels.shape)[1:])
    sizeval = np.array([val_size] + list(pixels.shape)[1:])
    sizete = np.array([te_size] + list(pixels.shape)[1:])

    X_train = np.empty((sizetr))
    y_train = np.empty((tr_size), dtype=int)
    X_val = np.empty((sizeval))
    y_val = np.empty((val_size), dtype=int)
    X_test = np.empty((sizete))
    y_test = np.empty((te_size), dtype=int)
    trcont = 0;
    valcont = 0;
    tecont = 0;

    for cl in np.unique(labels):
        pixels_cl = pixels[labels == cl]
        labels_cl = labels[labels == cl]
        pixels_cl, labels_cl = random_unison(pixels_cl, labels_cl, rstate=rand_state)
        for cont, (a, b) in enumerate(zip(pixels_cl, labels_cl)):
            if cont < train_set_size[cl]:
                X_train[trcont, :, :, :] = a
                y_train[trcont] = b
                trcont += 1
            elif cont < train_set_size[cl] + val_set_size[cl]:
                X_val[valcont, :, :, :] = a
                y_val[valcont] = b
                valcont += 1
            else:
                X_test[tecont, :, :, :] = a
                y_test[tecont] = b
                tecont += 1

    X_train, y_train = random_unison(X_train, y_train, rstate=rand_state)
    # X_test, y_test = random_unison(X_test, y_test, rstate=rand_state)
    X_val, y_val = random_unison(X_val, y_val, rstate=rand_state)
    return X_train, y_train, X_val, y_val, X_test, y_test


def sampling_joint(labels):
    train = {}
    test = {}
    row_labels = labels.reshape(np.prod(labels.shape[:2]), )  # 转为行标签
    train_set_size = [29, 762, 435, 146, 232, 394, 16, 235, 10, 470, 1424, 328, 132, 728, 291, 57]
    m = max(row_labels)
    for i in range(m):
        indices = [j for j, x in enumerate(row_labels.ravel().tolist()) if x == i + 1]  # 第i类地物的全部索引
        np.random.shuffle(indices)
        per_train_num = train_set_size[i]
        test[i] = indices[:-per_train_num]
        train[i] = indices[-per_train_num:]
    train_indices = []
    test_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    return train_indices, test_indices


if __name__ == '__main__':
    dataset = 'IP'
    root_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
    data_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "..")), 'data')
    path = r'D:\\UseTools\OneDrive\codes\New-Research\\figures'
    save_path = os.path.join(path, "test_map_" + dataset)  # 保存路径

    data, labels, num_class, label_names = data_preprocess.loadData(data_path, dataset)
    train_indices, test_indices = sampling_joint(labels)
    # row_labels = labels.reshape(np.prod(labels.shape[:2]), )  # 转为行标签
    x = np.ravel(labels).reshape(-1, 1)  # 预测的结果也拉成一列
    for i in range(len(train_indices)):
        x[train_indices[i]] = 0
    test_map = x.reshape(labels.shape[0], labels.shape[1])  # 再次将结果拉成一列
    show_label(test_map, test_map, 16, save_path)
