# -*- coding: utf-8 -*-
"""
@Date:   2020/4/27 12:08
@Author: Pangpd
@FileName: main.py
@IDE: PyCharm
@Description:
"""
import os
import sys
import time
from utils.auxiliary import save_acc_loss
from utils.data_preprocess import load_hyper
from utils.auxiliary import get_logger
from utils.disjoint_sample_split_data import load_disjoint_hyper, load_joint_hyper
from utils.hyper_pytorch import *
from datetime import datetime
import torch
import torch.nn.parallel
from utils.evaluate import reports, stats
from utils.start import test, train, predict


np.set_printoptions(linewidth=400)
np.set_printoptions(threshold=sys.maxsize)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# -------------------------定义超参数--------------------------
data_path = os.path.join(os.getcwd(), 'data')  # 数据集路径
dataset = 'KSC'  # 数据集
splitmethod = 'disjoint'  # 划分数据集的方式
seed = 1014
nums_repeat = 10  # 每组实验次数
spatial_size = 9  # 邻域像素块
res_rate = 0.75
epochs = 200
inplanes = 64  # 初始通道数
middle_channels = 32
train_percent = 200  # 训练样本/比例
val_percent = 0.5
batch_size = 1
components = None  # 主成分数，当前代码没有使用
learn_rate = 0.01
momentum = 0.9
weight_decay = 1e-4


def main():
    # ----------------------定义日志格式---------------------------
    time_str = datetime.strftime(datetime.now(), '%m-%d_%H-%M-%S')
    log_path = os.path.join(os.getcwd(), "logs")  # logs目录
    log_dir = os.path.join(log_path, time_str)  # log组根目录

    oa_list = []
    aa_list = []
    kappa_list = []
    each_acc_list = []
    train_time_list = []
    test_time_list = []
    for iter in range(nums_repeat):
        torch.cuda.empty_cache()
        group_log_dir = os.path.join(log_dir, "Experiment_" + str(iter + 1))  # logs组目录
        if not os.path.exists(group_log_dir):
            os.makedirs(group_log_dir)
        group_logger = get_logger(str(iter + 1), group_log_dir)
        random_state = seed + iter
        print('-------------------------------------------Iter %s----------------------------------' % (iter + 1))
        oa, aa, kappa, each_acc, train_time, test_time = start(group_log_dir, random_state, logger=group_logger)
        oa_list.append(oa)
        aa_list.append(aa)
        kappa_list.append(kappa)
        each_acc_list.append(each_acc)
        train_time_list.append(train_time)
        test_time_list.append(test_time)

    stats_oa, stats_aa, stats_kappa, stats_each_acc, stats_train_time, \
    stats_test_time = stats(oa_list, aa_list, kappa_list, each_acc_list, train_time_list, test_time_list)

    stats_logger = get_logger('final', log_dir)
    stats_logger.debug(f'''Initial parameters:
             dataset:         {dataset}
             splitmethod:     {splitmethod}
             Epochs:          {epochs}
             init planes:     {inplanes}
             res_rate:        {res_rate}
             middle channels: {middle_channels}
             spatial size:    {spatial_size}
             components:      {components}
             batch size:      {batch_size}
             Learning rate:   {learn_rate}
             momentum:        {momentum}
             weight decay:    {weight_decay}       
             train percent:   {train_percent}
             val percent:     {val_percent}
     ''')
    stats_logger.info('------------------------------------本组实验结果---------------------------------------------------')
    stats_logger.info("OA均值:%f   总体标准差:%f   样本标准差:%f" %
                      (stats_oa['av_oa'], stats_oa['ov_std_oa'], stats_oa['samp_std_oa']))
    stats_logger.info("AA均值:%f   总体标准差:%f   样本标准差:%f " %
                      (stats_aa['av_aa'], stats_aa['ov_std_aa'], stats_aa['samp_std_aa']))
    stats_logger.info("kappa均值:%f  总体标准差:%f   样本标准差:%f" %
                      (stats_kappa['av_kappa'], stats_kappa['ov_std_kappa'], stats_kappa['samp_std_kappa']))

    stats_logger.info("每类地物分类均值:      %s" % (stats_each_acc['av_each_acc']))
    stats_logger.info("每类地物分类总体标准差:%s" % (stats_each_acc['ov_std_each_acc']))
    stats_logger.info("每类地物分类样本标准差:%s" % (stats_each_acc['samp_std_each_acc']))
    stats_logger.info("训练时间均值:%f  总体标准差:%f  样本标准差:%f;  测试时间均值:%f  总体标准差:%f     样本标准差:%f" % (
        stats_train_time['av_train_time'], stats_train_time['ov_std_train_time'],
        stats_train_time['samp_std_train_time']
        , stats_test_time['av_test_time'], stats_test_time['ov_std_test_time'], stats_test_time['samp_std_test_time']))


def start(group_log_dir, random_state, logger):
    if splitmethod == 'disjoint':
        train_loader, test_loader, val_loader, num_classes, n_bands \
            = load_disjoint_hyper(data_path, dataset, spatial_size, val_percent, batch_size,
                                  components=components, rand_state=random_state)
    elif splitmethod == "joint":
        train_loader, test_loader, val_loader, num_classes, n_bands \
            = load_joint_hyper(data_path, dataset, spatial_size, val_percent, batch_size,
                                  components=components, rand_state=random_state)
    else:
        train_loader, test_loader, val_loader, num_classes, n_bands = load_hyper(data_path, dataset, spatial_size,
                                                                                 train_percent, val_percent,
                                                                                 batch_size, components=components,
                                                                                 rand_state=random_state)
    use_cuda = torch.cuda.is_available()
    if use_cuda: torch.backends.cudnn.benchmark = True

    if res_rate == 0:
        model = My_2D_res0_DpnNet(n_bands, inplanes, middle_channels, res_rate, num_classes)
    elif res_rate == 1:
        model = My_2D_res1_DpnNet(n_bands, inplanes, middle_channels, res_rate, num_classes)
    else:
        model = My_3D_DpnNet(n_bands, inplanes, middle_channels, res_rate, num_classes)

    if use_cuda:
        model = model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), learn_rate, momentum=momentum, weight_decay=weight_decay,
                                nesterov=True)
    best_acc = -1
    best_loss = 1
    # 定义两个数组,记录训练损失和验证损失
    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []
    train_start_time = time.time()  # 返回当前的时间戳
    for epoch in range(epochs):
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        valid_loss, valid_acc = test(val_loader, model, criterion, epoch, use_cuda)
        valid_loss_list.append(valid_loss)
        valid_acc_list.append(valid_acc)
        logger.info('Epoch: %03d   Train Loss: %f Train Accuracy: %f   Valid Loss: %f Valid Accuracy: %f' % (
            epoch, train_loss, train_acc, valid_loss, valid_acc))

        # save model
        # if valid_acc > best_acc  :
        if valid_loss < best_loss and valid_acc >= best_acc:
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': valid_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, group_log_dir + "/best_model.pth.tar")
            best_acc = valid_acc
            best_loss = valid_loss
    train_end_time = time.time()

    checkpoint = torch.load(group_log_dir + "/best_model.pth.tar")
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    # 测试
    test_start_time = time.time()
    test_loss, test_acc = test(test_loader, model, criterion, epoch, use_cuda)
    test_end_time = time.time()
    logger.info("Final:   Loss: %s  Accuracy: %s", test_loss, test_acc)

    # 预测
    predict_values = np.argmax(predict(test_loader, model, use_cuda), axis=1)  # 预测结果
    labels_values = np.array(test_loader.dataset.__labels__())  # 实际标签值
    classification, confusion, oa, aa, kappa, each_acc = reports(predict_values, labels_values)
    train_time = train_end_time - train_start_time
    test_time = test_end_time - test_start_time
    logger.debug('classification:\n %s\n confusion:\n%s\n ' % (classification, confusion))
    logger.info('AA: %f, OA: %f, kappa: %f\n each_acc: %s' % (aa, oa, kappa, each_acc))
    logger.info("Train time:%s , Test time:%s", train_time, test_time)
    save_acc_loss(train_acc_list, train_loss_list, valid_acc_list, valid_loss_list, group_log_dir)

    return oa, aa, kappa, each_acc, train_time, test_time


def adjust_learning_rate(optimizer, epoch, learn_rate):
    lr = learn_rate * (0.1 ** (epoch // 50)) * (0.1 ** (epoch // 225))  # 每隔25个epoch更新学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
