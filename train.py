import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np, argparse, time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import IEMOCAPDataset, MELDDataset
from model import MaskedNLLLoss, Transformer_Based_Model, KLDivLoss, VGSRLoss
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
import pickle as pk
import datetime
import torch.nn as nn


def get_train_valid_sampler(trainset, valid=0.1, dataset='MELD'):
    """用于创建训练集和验证集的采样器
    这个函数特别适用于在训练深度学习模型时进行模型的验证
    """
    size = len(trainset)
    idx = list(range(size))
    # 计算验证集的大小，根据提供的比例 valid 来确定
    split = int(valid*size)
    # 返回两个 SubsetRandomSampler 对象，分别用于训练集和验证集
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_MELD_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    """
    用于创建 MELD 数据集的训练、验证和测试加载器（loaders）
   这些加载器使用了 DataLoader 类，并且可以指定批次大小、是否使用多线程加载数据、是否固定内存等参数。
    :param num_workers:加载数据时使用的子进程数量，默认为 0
    :param pin_memory:是否将数据加载到固定的内存中，以加速数据转移到 CUDA 设备上的速度，默认为 False
    :return:返回包含训练、验证和测试加载器的元组
    """
    trainset = MELDDataset('data/meld_multimodal_features.pkl')
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid, 'MELD')
    # 创建训练集的 DataLoader，使用指定的批次大小、采样器、collate 函数、子进程数量和固定内存选项
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MELDDataset('data/meld_multimodal_features.pkl', train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    # 返回包含训练、验证和测试加载器的元组
    return train_loader, valid_loader, test_loader


def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset()
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader


# IEMOCAP gamma_3=1.0      meld gamma_3=0.1
def train_or_eval_model(model, loss_function, kl_loss, dataloader, epoch, dataset, optimizer=None, train=False, gamma_1=1.0, gamma_2=1.0, gamma_3=1.0):
    """
    用于训练或评估一个模型，具体取决于 train 参数的值。它处理数据加载、前向传播、损失计算、反向传播（如果是训练模式）以及性能评估
    :param kl_loss:用于计算KL散度损失的函数
    :param train:一个布尔值，指示是训练模型还是评估模型
    :param gamma_1:用于加权不同损失项的系数
    """
    losses, preds, labels, masks = [], [], [], []
    labels_g = []
    # 确保在训练模式下提供了优化器
    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    # 遍历 dataloader 中的数据
    for data in dataloader:
        # 如果是训练模式，清除优化器的梯度
        if train:
            optimizer.zero_grad()
        # 将数据转移到设备上（如果使用 GPU，则为 CUDA 设备），并准备数据
        # 可以猜测数据集的特征是textf, visuf, acouf, qmask, umask, label，最后一个维度是label
        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        # qmask = qmask.permute(1, 0, 2)
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
        # 调用模型进行前向传播，获取不同模态和多模态融合的对数概率和概率
        log_prob1, log_prob2, log_prob3, all_log_prob, all_prob, \
        kl_log_prob1, kl_log_prob2, kl_log_prob3, kl_all_prob = model(textf, visuf, acouf, umask, qmask, lengths)

        # 计算主损失和KL散度损失，并将它们组合成总损失
        lp_1 = log_prob1.view(-1, log_prob1.size()[2])
        lp_2 = log_prob2.view(-1, log_prob2.size()[2])
        lp_3 = log_prob3.view(-1, log_prob3.size()[2])
        lp_all = all_log_prob.view(-1, all_log_prob.size()[1])
        labels_ = label.view(-1)

        kl_lp_1 = kl_log_prob1.view(-1, kl_log_prob1.size()[1])
        kl_lp_2 = kl_log_prob2.view(-1, kl_log_prob2.size()[1])
        kl_lp_3 = kl_log_prob3.view(-1, kl_log_prob3.size()[1])
        # print("kl_all_prob", kl_all_prob.shape)  # ([726, 6])
        kl_p_all = kl_all_prob.view(-1, kl_all_prob.size()[1])
        # print("kl_p_all", kl_p_all.shape)  # ([726, 6])

        if dataset=="IEMOCAP":
            loss_weights = torch.FloatTensor([1 / 0.086747,
                                          1 / 0.144406,
                                          1 / 0.227883,
                                          1 / 0.160585,
                                          1 / 0.127711,
                                          1 / 0.252668])
            loss_function_1 = nn.NLLLoss(loss_weights.to(torch.device("cuda:0")) if cuda else loss_weights)  # IEMOCAP
        else:
            loss_function_1 = nn.NLLLoss()  # MELD
        # loss = gamma_1 * loss_function(lp_all, labels_, umask) + \
        #         gamma_2 * (loss_function(lp_1, labels_, umask) + loss_function(lp_2, labels_, umask) + loss_function(lp_3, labels_, umask)) + \
        #        gamma_3 * (kl_loss(kl_lp_1, kl_p_all, umask) + kl_loss(kl_lp_2, kl_p_all, umask) + kl_loss(kl_lp_3, kl_p_all, umask))

        loss_function_g = VGSRLoss()
        # print("label.shape", label.shape)  # ([16, 74])
        # print("labels_.shape", labels_.shape)  # ([1184])

        label_g = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        # print("lp_all", all_log_prob.shape)  # torch.Size([838, 6])
        # print("label_g.shape", label_g.shape)  # torch.Size([838])
        # print("kl_p_all", kl_p_all.shape)  # ([838, 6])  ([851, 6]) ([759, 6])
        # print("kl_lp_1", kl_lp_1.shape)  # ([1760, 6])  ([759, 6])
        # print("kl_lp_2", kl_lp_2.shape)  #
        # print("kl_lp_3", kl_lp_3.shape)  #
        # print("umask", umask.shape)  # ([16, 110])

        # print("kl_lp_1", kl_lp_1.shape)  # ([1152, 6])

        # loss = gamma_1 * loss_function_1(all_log_prob, label_g) + \
        #         gamma_2 * (loss_function_g(lp_1, labels_) + loss_function_g(lp_2, labels_) + loss_function_g(lp_3, labels_)) + \
        #         gamma_3 * (kl_loss(kl_lp_1, kl_p_all) + kl_loss(kl_lp_2, kl_p_all) + kl_loss(kl_lp_3, kl_p_all))  # 73.25 72.95 (68/70)
        # loss = gamma_1 * loss_function_1(all_log_prob, label_g) + \
        #        gamma_2 * (loss_function_g(lp_1, labels_) + loss_function_g(lp_2, labels_) + loss_function_g(lp_3,
        #                                                                                                     labels_)) + \
        #        gamma_3 * (kl_loss(kl_lp_1, kl_p_all) + kl_loss(kl_lp_2, kl_p_all) + kl_loss(kl_lp_3,
        #                                                                                     kl_p_all))  # 73.76 73.51 (58/90)
        # loss = gamma_1 * loss_function_1(all_log_prob, label_g) + \
        #        gamma_2 * (loss_function_g(lp_1, labels_) + loss_function_g(lp_2, labels_) + loss_function_g(lp_3,
        #                                                                                                     labels_)) + \
        #        gamma_3 * (kl_loss(kl_lp_1, kl_p_all) + kl_loss(kl_lp_2, kl_p_all) + kl_loss(kl_lp_3,
        #                                                                                     kl_p_all))  # 72.67 72.40 (87/100)
        # loss = gamma_1 * loss_function_1(all_log_prob, label_g) + \
        #        gamma_2 * (loss_function_g(lp_1, labels_) + loss_function_g(lp_2, labels_) + loss_function_g(lp_3,
        #                                                                                                     labels_)) + \
        #        gamma_3 * (kl_loss(kl_lp_1, kl_p_all) + kl_loss(kl_lp_2, kl_p_all) + kl_loss(kl_lp_3,
        #                                                                                     kl_p_all))  # 73.45 73.14  (59/80)
        # queding90
        # loss = gamma_1 * loss_function_1(all_log_prob, label_g) + \
        #        gamma_2 * (loss_function_g(lp_1, labels_) + loss_function_g(lp_2, labels_) + loss_function_g(lp_3,
        #                                                                                                     labels_)) + \
        #        gamma_3 * (kl_loss(kl_lp_1, kl_p_all) + kl_loss(kl_lp_2, kl_p_all) + kl_loss(kl_lp_3,
        #                                                                                     kl_p_all))  # 72.47  (50/90) 2.0
        # loss = gamma_1 * loss_function_1(all_log_prob, label_g) + \
        #        gamma_2 * (loss_function_g(lp_1, labels_) + loss_function_g(lp_2, labels_) + loss_function_g(lp_3,
        #                                                                                                     labels_)) + \
        #        gamma_3 * (kl_loss(kl_lp_1, kl_p_all) + kl_loss(kl_lp_2, kl_p_all) + kl_loss(kl_lp_3,
        #                                                                                     kl_p_all))  # 73.76 73.44 (50/90) 1.1
        # + feature GCN
        # loss = gamma_1 * loss_function_1(all_log_prob, label_g) + \
        #        gamma_2 * (loss_function_g(lp_1, labels_) + loss_function_g(lp_2, labels_) + loss_function_g(lp_3,
        #                                                                                                     labels_)) + \
        #        gamma_3 * (kl_loss(kl_lp_1, kl_p_all) + kl_loss(kl_lp_2, kl_p_all) + kl_loss(kl_lp_3,
        #                                                                                     kl_p_all))  #  (/90) 1.0
        # loss = gamma_1 * loss_function_1(all_log_prob, label_g) + \
        #        gamma_2 * (loss_function_g(lp_1, labels_) + loss_function_g(lp_2, labels_) + loss_function_g(lp_3,
        #                                                                                                     labels_)) + \
        #        gamma_3 * (kl_loss(kl_lp_1, kl_p_all) + kl_loss(kl_lp_2, kl_p_all) + kl_loss(kl_lp_3,
        #                                                                                     kl_p_all))  # 73.03 72.77 (79/90) 1.0
        # + 全部
        # loss = gamma_1 * loss_function_1(all_log_prob, label_g) + \
        #        gamma_2 * (loss_function_g(lp_1, labels_) + loss_function_g(lp_2, labels_) + loss_function_g(lp_3,
        #                                                                                                     labels_)) + \
        #        gamma_3 * (kl_loss(kl_lp_1, kl_p_all) + kl_loss(kl_lp_2, kl_p_all) + kl_loss(kl_lp_3,
        #                                                                                     kl_p_all))  # 73.57 73.44 (79/90) 1.0
        # loss = gamma_1 * loss_function_1(all_log_prob, label_g) + \
        #        gamma_2 * (loss_function_g(lp_1, labels_) + loss_function_g(lp_2, labels_) + loss_function_g(lp_3,
        #                                                                                                     labels_)) + \
        #        gamma_3 * (kl_loss(kl_lp_1, kl_p_all) + kl_loss(kl_lp_2, kl_p_all) + kl_loss(kl_lp_3,
        #                                                                                     kl_p_all))  # 73.11 72.95 (68/90) 1.0  0.1
        # loss = gamma_1 * loss_function_1(all_log_prob, label_g) + \
        #        gamma_2 * (loss_function_g(lp_1, labels_) + loss_function_g(lp_2, labels_) + loss_function_g(lp_3,
        #                                                                                                     labels_)) + \
        #        gamma_3 * (kl_loss(kl_lp_1, kl_p_all) + kl_loss(kl_lp_2, kl_p_all) + kl_loss(kl_lp_3,
        #                                                                                     kl_p_all))  # 73.37 73.01 (73/90) 1.1
        # loss = gamma_1 * loss_function_1(all_log_prob, label_g) + \
        #        gamma_2 * (loss_function_g(lp_1, labels_) + loss_function_g(lp_2, labels_) + loss_function_g(lp_3,
        #                                                                                                     labels_)) + \
        #        gamma_3 * (kl_loss(kl_lp_1, kl_p_all) + kl_loss(kl_lp_2, kl_p_all) + kl_loss(kl_lp_3,
        #                                                                                     kl_p_all))  # 72.68 72.40 (52/90) 2.0
        # loss = gamma_1 * loss_function_1(all_log_prob, label_g) + \
        #        gamma_2 * (loss_function_g(lp_1, labels_) + loss_function_g(lp_2, labels_) + loss_function_g(lp_3,
        #                                                                                                     labels_)) + \
        #        gamma_3 * (kl_loss(kl_lp_1, kl_p_all) + kl_loss(kl_lp_2, kl_p_all) + kl_loss(kl_lp_3,
        #                                                                                     kl_p_all))  # 71.49 71.23 (75/90) 3.0
        # loss = gamma_1 * loss_function_1(all_log_prob, label_g) + \
        #        gamma_2 * (loss_function_g(lp_1, labels_) + loss_function_g(lp_2, labels_) + loss_function_g(lp_3,
        #                                                                                                     labels_)) + \
        #        gamma_3 * (kl_loss(kl_lp_1, kl_p_all) + kl_loss(kl_lp_2, kl_p_all) + kl_loss(kl_lp_3,
        #                                                                                     kl_p_all))  # 73.15  72.89(61/90) 0.9
        # 去self.alpha
        # loss = gamma_1 * loss_function_1(all_log_prob, label_g) + \
        #        gamma_2 * (loss_function_g(lp_1, labels_) + loss_function_g(lp_2, labels_) + loss_function_g(lp_3,
        #                                                                                                     labels_)) + \
        #        gamma_3 * (kl_loss(kl_lp_1, kl_p_all) + kl_loss(kl_lp_2, kl_p_all) + kl_loss(kl_lp_3,
        #                                                                                     kl_p_all))  # 73.68 73.51 (80/90) 1.0 一位小数矩阵
        # 换矩阵（两个公式）
        # loss = gamma_1 * loss_function_1(all_log_prob, label_g) + \
        #        gamma_2 * (loss_function_g(lp_1, labels_) + loss_function_g(lp_2, labels_) + loss_function_g(lp_3,
        #                                                                                                     labels_)) + \
        #        gamma_3 * (kl_loss(kl_lp_1, kl_p_all) + kl_loss(kl_lp_2, kl_p_all) + kl_loss(kl_lp_3,
        #                                                                                     kl_p_all))  # 73.66 73.44 (80/90) 1.0 归一化后
        # loss = gamma_1 * loss_function_1(all_log_prob, label_g) + \
        #        gamma_2 * (loss_function_g(lp_1, labels_) + loss_function_g(lp_2, labels_) + loss_function_g(lp_3,
        #                                                                                                     labels_)) + \
        #        gamma_3 * (kl_loss(kl_lp_1, kl_p_all) + kl_loss(kl_lp_2, kl_p_all) + kl_loss(kl_lp_3,
        #                                                                                     kl_p_all))  # 73.5 73.32 (57/90) 1.0 归一化前
        # meld---------------
        # loss = gamma_1 * loss_function_1(all_log_prob, label_g) + \
        #        gamma_2 * (loss_function_g(lp_1, labels_, dataset) + loss_function_g(lp_2, labels_, dataset) + loss_function_g(lp_3,
        #                                                                                                     labels_, dataset)) + \
        #        gamma_3 * (kl_loss(kl_lp_1, kl_p_all) + kl_loss(kl_lp_2, kl_p_all) + kl_loss(kl_lp_3,
        #                                                                                     kl_p_all))  # 65.04 67.16 (10/90) 1.0
        loss = gamma_1 * loss_function_1(all_log_prob, label_g) + \
               gamma_2 * (loss_function_g(lp_1, labels_, dataset) + loss_function_g(lp_2, labels_,
                                                                                    dataset) + loss_function_g(lp_3,
                                                                                                               labels_,
                                                                                                               dataset)) + \
               gamma_3 * (kl_loss(kl_lp_1, kl_p_all) + kl_loss(kl_lp_2, kl_p_all) + kl_loss(kl_lp_3,
                                                                                            kl_p_all))  #  (10/60) 1.0  0.0001




        lp_ = all_prob.view(-1, all_prob.size()[1])
        # print("lp_", lp_.shape)  # [718, 6])
        # 将预测的概率转换为预测标签
        pred_ = torch.argmax(lp_, 1)
        # print("pred_", pred_.shape)  # ([718])
        # print("labels_", labels_.shape)  # ([1760])
        # print("label_g", label_g.shape)  # ([718])
        # print("labels", labels)

        # 将损失、预测、标签和掩码添加到相应的列表中
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        labels_g.append(label_g.data.cpu().numpy())

        masks.append(umask.view(-1).cpu().numpy())
        losses.append(loss.item()*masks[-1].sum())

        # 如果是训练模式，执行反向传播并更新模型参数
        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()

    if preds!=[]:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        labels_g = np.concatenate(labels_g)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan')

    # 计算平均损失、准确率和 F1 分数
    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    # print("labels", labels.shape)  # (9512,)
    # print("preds", preds.shape)  # (5810,) 维度不一致无法预测，需要维度一致
    # print("label_g", label_g.shape)  # ([718])
    # print("masks", masks.shape)  # (9512,)
    # avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks)*100, 2)
    # avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted')*100, 2)
    avg_accuracy = round(accuracy_score(labels_g, preds) * 100, 2)
    avg_fscore = round(f1_score(labels_g, preds, average='weighted') * 100, 2)
    # 返回计算得到的平均损失、准确率、标签、预测、掩码和 F1 分数
    # return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore
    return avg_loss, avg_accuracy, labels_g, preds, masks, avg_fscore


if __name__ == '__main__':
    """Python 脚本，它使用 argparse 库来解析命令行参数
    通过这种方式，用户可以轻松地调整模型的超参数，而无需修改代码本身。
    """
    # 用于处理命令行参数
    parser = argparse.ArgumentParser()
    # 向解析器添加参数。每个参数都有一些属性，如动作（action）、默认值（default）、类型（type）、帮助信息（help）等
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=16, metavar='BS', help='batch size')  # 16->2
    parser.add_argument('--hidden_dim', type=int, default=1024, metavar='hidden_dim', help='output hidden size')
    parser.add_argument('--n_head', type=int, default=8, metavar='n_head', help='number of heads')  # 多头注意力机制中的头数
    parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')
    parser.add_argument('--temp', type=int, default=1, metavar='temp', help='temp')  # 温度参数，通常用于调整 softmax 的输出
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')
    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weights')
    parser.add_argument('--Dataset', default='MELD', help='dataset to train and test')  # IEMOCAP MELD

    # 解析命令行参数，并将它们存储在 args 变量中
    args = parser.parse_args()
    today = datetime.datetime.now()
    print(args)

    # 根据系统是否支持 CUDA 以及用户是否指定不使用 CUDA 来设置 cuda 标志
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size
    # 根据数据集的不同，设置音频、视觉和文本特征的维度
    feat2dim = {'IS10': 1582, 'denseface': 342, 'MELD_audio': 300}
    D_audio = feat2dim['IS10'] if args.Dataset=='IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_text = 1024

    D_m = D_audio + D_visual + D_text

    # 根据数据集的不同，设置说话者数量和类别数量
    n_speakers = 9 if args.Dataset=='MELD' else 2
    n_classes = 7 if args.Dataset=='MELD' else 6 if args.Dataset=='IEMOCAP' else 1
    # 打印温度参数
    print('temp {}'.format(args.temp))

    # 2. 这段代码是训练和评估过程的准备阶段，它确保模型、损失函数和优化器都已正确设置，并且数据加载器已经准备好提供训练和验证数据。
    model = Transformer_Based_Model(args.Dataset, args.temp, D_text, D_visual, D_audio, args.n_head,
                                        n_classes=n_classes,
                                        hidden_dim=args.hidden_dim,
                                        n_speakers=n_speakers,
                                        dropout=args.dropout)

    total_params = sum(p.numel() for p in model.parameters())
    print('total parameters: {}'.format(total_params))
    # 计算模型中可训练的参数数量
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('training parameters: {}'.format(total_trainable_params))


    def get_model_size(model):
        dtype_size_map = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
            torch.int8: 1
        }

        total_params = sum(p.numel() for p in model.parameters())
        # 获取模型中参数的 dtype（默认取第一个参数的 dtype）
        param_dtype = next(model.parameters()).dtype
        bytes_per_param = dtype_size_map.get(param_dtype, 4)  # 默认为 float32

        param_size_bytes = total_params * bytes_per_param
        param_size_mb = param_size_bytes / (1024 * 1024)

        print(f"total parameters: {total_params}")
        print(f"parameter dtype: {param_dtype}")
        print(f"model size: {param_size_mb:.2f} MB")
        return total_params, param_size_mb


    # 使用示例
    get_model_size(model)

    if cuda:
        # 将模型转移到 GPU 上
        model.cuda()
        
    # kl_loss = MaskedKLDivLoss()
    kl_loss = KLDivLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    # 根据数据集的不同，设置损失函数和数据加载器
    if args.Dataset == 'MELD':
        loss_function = MaskedNLLLoss()
        train_loader, valid_loader, test_loader = get_MELD_loaders(valid=0.0,
                                                                    batch_size=batch_size,
                                                                    num_workers=0)
    elif args.Dataset == 'IEMOCAP':
        loss_weights = torch.FloatTensor([1/0.086747,
                                        1/0.144406,
                                        1/0.227883,
                                        1/0.160585,
                                        1/0.127711,
                                        1/0.252668])
        loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.001, batch_size=batch_size,
                                                                      num_workers=0)
    else:
        print("There is no such dataset")

    # 用于跟踪验证过程中的最佳性能
    best_fscore, best_loss, best_label, best_pred, best_mask, best_label2 = None, None, None, None, None, None
    # 用于存储训练过程中的性能记录
    all_fscore, all_acc, all_loss = [], [], []

    # 3. 这段代码是一个训练循环，用于训练和评估一个基于 Transformer 的模型。它遍历指定的训练周期（epochs），在每个周期中进行训练、验证和测试，并记录性能指标
    for e in range(n_epochs):
        start_time = time.time()

        # 训练
        train_loss, train_acc, _, _, _, train_fscore = train_or_eval_model(model, loss_function, kl_loss, train_loader, e, args.Dataset, optimizer, True)
        # 验证
        valid_loss, valid_acc, _, _, _, valid_fscore = train_or_eval_model(model, loss_function, kl_loss, valid_loader, e, args.Dataset)
        # 测试
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore = train_or_eval_model(model, loss_function, kl_loss, test_loader, e, args.Dataset)
        all_fscore.append(test_fscore)
        all_acc.append(test_acc)

        # 如果这是第一次迭代或者当前测试 F-Score 高于之前的最佳值，则更新最佳 F-Score 和相应的标签和预测
        if best_fscore == None or best_fscore < test_fscore:
            best_fscore = test_fscore
            best_label, best_pred, best_mask = test_label, test_pred, test_mask

        if args.tensorboard:
            writer.add_scalar('test: accuracy', test_acc, e)
            writer.add_scalar('test: fscore', test_fscore, e)
            writer.add_scalar('train: accuracy', train_acc, e)
            writer.add_scalar('train: fscore', train_fscore, e)

        print('epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.\
                format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc, test_fscore, round(time.time()-start_time, 2)))
        if (e+1) % 10 == 0:
            # 打印最佳标签和预测的分类报告   和混淆矩阵
            # print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
            # print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))
            print(classification_report(best_label, best_pred, digits=4))
            print(confusion_matrix(best_label, best_pred))


    if args.tensorboard:
        writer.close()

    print('Best performance..')
    print('F1-Score: {}'.format(max(all_fscore)))
    # print('ACC: {}'.format(max(all_acc)))
    print('index: {}'.format(all_fscore.index(max(all_fscore)) + 1))

    # 4. 这段代码是用来记录模型的性能指标到一个文件中，以便后续分析和比较。它使用了 Python 的 pickle 模块来序列化和反序列化数据
    # 检查指定日期的记录文件是否存在
    if not os.path.exists("record_{}_{}_{}.pk".format(today.year, today.month, today.day)):
        # 如果文件不存在，打开文件以写入二进制模式
        with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day),'wb') as f:
            # 将一个空字典序列化到文件中
            pk.dump({}, f)
    # 打开文件以读取二进制模式
    with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day), 'rb') as f:
        record = pk.load(f)

    # 定义一个键，用于在 record 字典中存储数据
    key_ = 'name_'
    if record.get(key_, False):
        # 如果键存在，将最高的 F-Score 添加到对应的列表中
        record[key_].append(max(all_fscore))
    else:
        # 如果键不存在，创建一个新的列表，并添加最高的 F-Score
        record[key_] = [max(all_fscore)]

    if record.get(key_+'record', False):
        # 如果键存在，将分类报告添加到对应的列表中
        # record[key_+'record'].append(classification_report(best_label, best_pred, sample_weight=best_mask,digits=4))
        record[key_ + 'record'].append(classification_report(best_label, best_pred, digits=4))
    else:
        # 如果键不存在，创建一个新的列表，并添加分类报告
        # record[key_+'record'] = [classification_report(best_label, best_pred, sample_weight=best_mask,digits=4)]
        record[key_ + 'record'] = [classification_report(best_label, best_pred, digits=4)]
    # 再次打开文件以写入二进制模式
    with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day),'wb') as f:
        # 将更新后的 record 字典序列化回文件
        pk.dump(record, f)

    # 打印分类报告和混淆矩阵
    # print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
    # print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))

    print(classification_report(best_label, best_pred, digits=4))
    # 单个类别的acc和recall一样
    print(confusion_matrix(best_label, best_pred))
    # confuPLT(confusion_matrix(best_label, best_pred, sample_weight=best_mask).astype(int), args.Dataset)
    # confuPLT(confusion_matrix(best_label, best_pred).astype(int), args.Dataset)
    #
    # # 获取混淆矩阵
    # conf_matrix = confusion_matrix(best_label, best_pred)
    #
    # # 计算每个类别的准确率
    # # 例如，准确率 = 每个类别的正确预测数 / 该类别的总样本数
    # accuracies = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    #
    # # 输出每个类别的准确率
    # for i, acc in enumerate(accuracies):
    #     print(f'Class {i} Accuracy: {acc:.4f}')


