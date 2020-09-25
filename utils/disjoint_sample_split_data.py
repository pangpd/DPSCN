# -*- coding: utf-8 -*-
"""
@Date:   2020/9/12 8:57
@Author: Pangpd
@FileName: disjoint_sample_split_data.py
@IDE: PyCharm
@Description:  在HSI的标记样本上随机取样
"""
import copy
import os
import numpy as np
from math import ceil
import scipy.io as sio
import torch
from sklearn.model_selection import train_test_split

from utils.data_preprocess import loadData, createImageCubes
from utils.hyper_pytorch import HyperData


def load_disjoint_data(data_path, dataset, components):
    data, all_labels, num_class, label_names = loadData(data_path, dataset, num_components=components)
    if dataset == 'IP':
        disjoint_labels = sio.loadmat(os.path.join(data_path, 'indianpines_disjoint_dset.mat'))[
            'indianpines_disjoint_dset']
        train_lables = copy.deepcopy(disjoint_labels)
        for i, val in enumerate([0, 2, 3, 5, 6, 8, 10, 11, 12, 14, 1, 4, 7, 9, 13, 15, 16]): train_lables[
            disjoint_labels == i] = val
        del disjoint_labels
        all_labels[train_lables != 0] = 0
        test_labels = all_labels
    elif dataset == 'PU':
        test_labels = sio.loadmat(os.path.join(data_path, 'TSpavia_fixed.mat'))['TSpavia_fixed']
        train_lables = sio.loadmat(os.path.join(data_path, 'TRpavia_fixed.mat'))['TRpavia_fixed']
    else:
        print("NO DISJOING DATA")
        exit()
    return data, test_labels, train_lables, num_class


def load_disjoint_hyper(data_path, dataset, spatial_size=7, val_percent=0.1, batch_size=24, components=None,
                        rand_state=None):
    data, test_labels, train_labels, num_classes = load_disjoint_data(data_path, dataset, components=components)
    bands = data.shape[-1]
    test_pixels, test_labels = createImageCubes(data, test_labels, windowSize=spatial_size, removeZeroLabels=True)
    train_pixels, train_labels = createImageCubes(data, train_labels, windowSize=spatial_size, removeZeroLabels=True)
    X_train, y_train = split_disjoint_data(train_pixels, train_labels, rand_state=rand_state)
    X_test, y_test = split_disjoint_data(test_pixels, test_labels, rand_state=rand_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_percent, stratify=y_train,
                                                      random_state=rand_state)
    # total = 0
    # for i in range(num_classes):
    #     print(i, ",训：", sum(y_train == i), ",验：", sum(y_val == i), ",测：", sum(y_test == i))
    #     a = sum(y_train == i) + sum(y_val == i) + sum(y_test == i)
    #     total = a + total
    #     print("当前类共计：", a)
    # print("共计：", total)
    del test_pixels, test_labels, train_pixels, train_labels
    train_hyper = HyperData((np.transpose(X_train, (0, 3, 1, 2)).astype("float32"), y_train), None)
    test_hyper = HyperData((np.transpose(X_test, (0, 3, 1, 2)).astype("float32"), y_test), None)
    val_hyper = HyperData((np.transpose(X_val, (0, 3, 1, 2)).astype("float32"), y_val), None)
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_hyper, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_hyper, batch_size=batch_size, shuffle=False, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_hyper, batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader, val_loader, num_classes, bands


# 划分不相交样本,验证由训练划分
def split_disjoint_data(pixels, labels, rand_state=None):
    # 存储每类地物训练样本数
    train_set_size = []
    for cl in np.unique(labels):
        pixels_cl = len(pixels[labels == cl])  # 第i类地物样本总数
        train_set_size.append(pixels_cl)  # 存储每类地物的训练样本数
    tr_size = int(sum(train_set_size))
    sizetr = np.array([tr_size] + list(pixels.shape)[1:])
    cont = 0
    X_pixels = np.empty((sizetr))
    y_labels = np.empty((tr_size), dtype=int)
    for cl in np.unique(labels):
        pixels_cl = pixels[labels == cl]
        labels_cl = labels[labels == cl]
        pixels_cl, labels_cl = random_unison(pixels_cl, labels_cl, rstate=rand_state)
        for i, (a, b) in enumerate(zip(pixels_cl, labels_cl)):
            X_pixels[cont, :, :, :] = a
            y_labels[cont] = b
            cont += 1
    X_pixels, y_labels = random_unison(X_pixels, y_labels, rstate=rand_state)
    return X_pixels, y_labels


def load_joint_hyper(data_path, dataset, spatial_size=7, val_percent=0.5, batch_size=24, components=None,
                     rand_state=None):
    data, labels, num_classes, label_names = loadData(data_path, dataset, num_components=components)
    bands = data.shape[-1]
    pixels, labels = createImageCubes(data, labels, windowSize=spatial_size, removeZeroLabels=True)
    if dataset == 'IP':
        train_set_size = [29, 762, 435, 146, 232, 394, 16, 235, 10, 470, 1424, 328, 132, 728, 291, 57]
    elif dataset == 'PU':
        train_set_size = [548, 540, 392, 524, 265, 532, 375, 514, 231]  # 3921
    else:
        print("Error")
        exit()
    X_train, y_train, X_test, y_test = split_joint_data(pixels, labels, train_set_size, rand_state=rand_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_percent, stratify=y_train,
                                                      random_state=rand_state)
    del pixels, labels
    train_hyper = HyperData((np.transpose(X_train, (0, 3, 1, 2)).astype("float32"), y_train), None)
    test_hyper = HyperData((np.transpose(X_test, (0, 3, 1, 2)).astype("float32"), y_test), None)
    val_hyper = HyperData((np.transpose(X_val, (0, 3, 1, 2)).astype("float32"), y_val), None)
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_hyper, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_hyper, batch_size=batch_size, shuffle=False, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_hyper, batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader, val_loader, num_classes, bands


# 划分相交样本,验证由训练划分
def split_joint_data(pixels, labels, train_samples, rand_state=None):
    train_set_size = train_samples  # 存储每类地物训练样本数
    pixels_number = np.unique(labels, return_counts=1)[1]  # 全部样本数
    tr_size = int(sum(train_set_size))
    te_size = int(sum(pixels_number)) - tr_size
    sizetr = np.array([tr_size] + list(pixels.shape)[1:])
    sizete = np.array([te_size] + list(pixels.shape)[1:])

    X_train = np.empty((sizetr))
    y_train = np.empty((tr_size), dtype=int)
    X_test = np.empty((sizete))
    y_test = np.empty((te_size), dtype=int)
    trcont = 0;
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
            else:
                X_test[tecont, :, :, :] = a
                y_test[tecont] = b
                tecont += 1

    X_train, y_train = random_unison(X_train, y_train, rstate=rand_state)
    # X_test, y_test = random_unison(X_test, y_test, rstate=rand_state)
    return X_train, y_train, X_test, y_test


def random_unison(a, b, rstate=None):
    assert len(a) == len(b)
    p = np.random.RandomState(seed=rstate).permutation(len(a))
    return a[p], b[p]


def random_single(a, rstate=None):
    return a[np.random.RandomState(seed=rstate).permutation(len(a))]
