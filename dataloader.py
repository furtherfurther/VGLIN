import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
import numpy as np


class IEMOCAPDataset(Dataset):
    """这个类用于加载和处理 IEMOCAP 数据集，这是一个用于情感分析的多模态数据集，包含了视频、音频、文本和标签"""
    def __init__(self, train=True):
        """
        :param train: 用于指定加载训练集或测试集
        """
        # 使用 pickle 加载预处理后的 IEMOCAP 数据集特征。这些特征包括视频 ID、说话者、标签、文本、视觉特征、音频特征、句子等
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.roberta2, self.roberta3, self.roberta4, \
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open('data/iemocap_multimodal_features.pkl', 'rb'), encoding='latin1')
        # 根据 train 参数的值，选择训练集或测试集的视频 ID
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]

        text = torch.from_numpy(np.array(self.videoText[vid])).float()
        visual = torch.from_numpy(np.array(self.videoVisual[vid])).float()
        audio = torch.from_numpy(np.array(self.videoAudio[vid])).float()

        speaker = torch.FloatTensor(
            [[1, 0] if x == 'M' else [0, 1] for x in self.videoSpeakers[vid]]
        )

        mask = torch.FloatTensor([1] * len(self.videoLabels[vid]))
        labels = torch.LongTensor(self.videoLabels[vid])

        return text, visual, audio, speaker, mask, labels, vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        """用于在数据加载时对数据进行批处理, collate_fn 方法确保了在创建批次时数据具有一致的形状"""
        # 将数据转换为 pandas 数据框
        dat = pd.DataFrame(data)
        # 对不同类型的数据进行不同的处理。文本、视觉、音频特征使用 pad_sequence 进行填充，以确保它们具有相同的长度。说话者标签和视频标签则直接转换为列表
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]


class MELDDataset(Dataset):
    """这个类用于加载和处理 MELD 数据集，这是一个多模态对话数据集，包含了文本、视觉、音频和标签"""
    def __init__(self, path, train=True):
        """
        :param path: 包含预处理数据的文件路径
        :param train: 用于指定加载训练集或测试集
        """
        # 使用 pickle 加载指定路径下的预处理数据。这些数据包括视频 ID、说话者、标签、文本、视觉特征、音频特征、句子等
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
        self.roberta2, self.roberta3, self.roberta4, \
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid, _ = pickle.load(open(path, 'rb'))

        # 根据 train 参数的值，选择训练集或测试集的视频 ID
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]

        # 使用 from_numpy 替代 FloatTensor，先转换为 numpy 数组
        text = torch.from_numpy(np.array(self.videoText[vid])).float()
        visual = torch.from_numpy(np.array(self.videoVisual[vid])).float()
        audio = torch.from_numpy(np.array(self.videoAudio[vid])).float()

        # 如果 self.videoSpeakers[vid] 包含的是字符串 'M'/'F' 或其他格式，需要相应处理
        # 假设 speaker 特征已经是数值格式，直接转换
        speaker = torch.from_numpy(np.array(self.videoSpeakers[vid])).float()

        # 创建全1的 mask 张量
        mask = torch.FloatTensor([1] * len(self.videoLabels[vid]))

        # 标签转换为 LongTensor
        labels = torch.LongTensor(self.videoLabels[vid])

        return text, visual, audio, speaker, mask, labels, vid

    def __len__(self):
        return self.len

    def return_labels(self):
        """用于返回所有样本的标签"""
        return_label = []
        for key in self.keys:
            # 将每个视频 ID 对应的标签添加到列表中
            return_label+=self.videoLabels[key]
        return return_label

    def collate_fn(self, data):
        """用于在数据加载时对数据进行批处理"""
        # 将数据转换为 pandas 数据框
        dat = pd.DataFrame(data)
        # 对不同类型的数据进行不同的处理。文本、视觉、音频特征使用 pad_sequence 进行填充，以确保它们具有相同的长度。说话者特征和视频标签则直接转换为列表
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]
