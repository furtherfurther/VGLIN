import torch 
import torch.nn as nn

from torch.nn import Parameter

import math

from itertools import permutations

from graphgcn import GraphGCN
""""
融合部分，使用了graphgcn
"""


class GraphConvolution(nn.Module):
    """
    包含了常见的图卷积操作，但带有一些特别的功能，如变种（variant）和残差连接（residual）
    """
    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__()
        # 如果使用变种（variant），则输入特征维度翻倍
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features 
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        # 权重参数，输入特征数 x 输出特征数
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        # 初始化权重参数
        self.reset_parameters()

    def reset_parameters(self):
        # 使用均匀分布初始化权重参数
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, h0, lamda, alpha, l):
        """
            参数：
                - input: 输入特征矩阵（N x in_features）
                - adj: 图的邻接矩阵（N x N），可以是稀疏矩阵
                - h0: 初始节点特征（N x in_features）
                - lamda: 正则化参数
                - alpha: 平衡系数，用于混合邻居信息和原始特征
                - l: 另一个用于计算 `theta` 的参数

            返回：
                - output: 经过图卷积操作后的输出特征矩阵（N x out_features）
        """
        # 计算 theta，用于平衡邻居信息和原始信息的比重
        theta = math.log(lamda/l+1)
        # 计算邻居节点的加权特征（相当于卷积的核心部分）
        hi = torch.spmm(adj, input)  # 使用稀疏矩阵乘法，adj 为邻接矩阵
        # 处理变种（variant）情况：拼接当前节点特征和上一层特征
        if self.variant:
            support = torch.cat([hi, h0], 1)  # 拼接邻居特征和初始特征
            r = (1-alpha)*hi+alpha*h0  # 混合邻居信息和原始特征
        else:
            support = (1-alpha)*hi+alpha*h0  # 混合邻居信息和原始特征
            r = support  # 如果没有变种，直接使用支持的特征作为 r
        # 计算最终输出：权重的线性组合和残差信息的平衡
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        # 如果使用残差连接，将输入加到输出中
        if self.residual:
            output = output+input
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, dia_len):
        """
        x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        tmpx = torch.zeros(0).cuda()
        tmp = 0
        for i in dia_len:
            a = x[tmp:tmp+i].unsqueeze(1)
            a = a + self.pe[:a.size(0)]
            tmpx = torch.cat([tmpx,a], dim=0)
            tmp = tmp+i
        #x = x + self.pe[:x.size(0)]
        tmpx = tmpx.squeeze(1)
        return self.dropout(tmpx)


class GCN(nn.Module):
    def __init__(self, n_dim, nhidden, dropout, lamda, alpha, variant, return_feature, use_residue, 
                new_graph='full',n_speakers=2, modals=['a', 'v', 'l'], use_speaker=True, use_modal=False, num_L=3, num_K=4):
        super(GCN, self).__init__()
        self.return_feature = return_feature  # True是否返回特征
        self.use_residue = use_residue
        self.new_graph = new_graph  # 是否使用新的图结构

        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

        # 多模态输入配置：音频、视频、语言
        self.modals = modals
        self.modal_embeddings = nn.Embedding(3, n_dim)  # 对模态的嵌入
        self.speaker_embeddings = nn.Embedding(n_speakers, n_dim)  # 对说话人的嵌入
        self.use_speaker = use_speaker  # 是否使用说话人嵌入
        self.use_modal = use_modal  # 是否使用模态嵌入
        self.use_position = False  # 暂时不使用位置编码

        # 网络中的全连接层
        self.fc1 = nn.Linear(n_dim, nhidden)    # 输入维度到隐藏层的映射

        # 图卷积参数
        self.num_L = num_L
        self.num_K = num_K
        self.act_fn = nn.ReLU()
        self.hyperedge_weight = nn.Parameter(torch.ones(1000))  # 超边权重
        self.EW_weight = nn.Parameter(torch.ones(5200))
        self.hyperedge_attr1 = nn.Parameter(torch.rand(nhidden))  # 超边特征1
        self.hyperedge_attr2 = nn.Parameter(torch.rand(nhidden))  # 超边特征2
        # 图卷积层
        for kk in range(num_K):
            setattr(self, 'conv%d' %(kk+1), GraphGCN(nhidden, nhidden))

    def forward(self, a, v, l, dia_len, qmask, epoch):
        qmask = torch.cat([qmask[:x, i, :] for i, x in enumerate(dia_len)], dim=0)
        spk_idx = torch.argmax(qmask, dim=-1)  # 获取说话人索引
        spk_emb_vector = self.speaker_embeddings(spk_idx)  # 获取说话人嵌入向量
        # 如果使用说话人嵌入，将其加入到语言模态中
        if self.use_speaker:
            if 'l' in self.modals:
                l += spk_emb_vector
        # 如果使用位置编码，则将其加入各个模态的特征中
        if self.use_position:
            if 'l' in self.modals:
                l = self.l_pos(l, dia_len)
            if 'a' in self.modals:
                a = self.a_pos(a, dia_len)
            if 'v' in self.modals:
                v = self.v_pos(v, dia_len)
        # 如果使用模态嵌入，则将其添加到各个模态的特征中
        if self.use_modal:  
            emb_idx = torch.LongTensor([0, 1, 2]).to("cuda:0")
            emb_vector = self.modal_embeddings(emb_idx)  # 获取模态嵌入

            if 'a' in self.modals:
                a += emb_vector[0].reshape(1, -1).expand(a.shape[0], a.shape[1])
            if 'v' in self.modals:
                v += emb_vector[1].reshape(1, -1).expand(v.shape[0], v.shape[1])
            if 'l' in self.modals:
                l += emb_vector[2].reshape(1, -1).expand(l.shape[0], l.shape[1])

        # 创建图卷积需要的邻接矩阵和特征矩阵
        gnn_edge_index, gnn_features = self.create_gnn_index(a, v, l, dia_len, self.modals)
        x1 = self.fc1(gnn_features)  
        out = x1
        gnn_out = x1
        # 进行多次图卷积操作
        for kk in range(self.num_K):
            gnn_out = gnn_out + getattr(self, 'conv%d' %(kk+1))(gnn_out, gnn_edge_index)
        # 将卷积结果和初始结果拼接
        out2 = torch.cat([out, gnn_out], dim=1)
        if self.use_residue:
            out2 = torch.cat([gnn_features, out2], dim=-1)
        # 逆转特征并返回
        out1 = self.reverse_features(dia_len, out2)

        return out1

    def reverse_features(self, dia_len, features):
        # 将特征按对话长度分割并重新排列
        l=[]
        a=[]
        v=[]
        for i in dia_len:
            ll = features[0:1*i]
            aa = features[1*i:2*i]
            vv = features[2*i:3*i]
            features = features[3*i:]
            l.append(ll)
            a.append(aa)
            v.append(vv)
        tmpl = torch.cat(l,dim=0)
        tmpa = torch.cat(a,dim=0)
        tmpv = torch.cat(v,dim=0)
        features = torch.cat([tmpl, tmpa, tmpv], dim=-1)
        return features

    def create_gnn_index(self, a, v, l, dia_len, modals):
        # 创建图卷积所需的邻接矩阵（边的索引）
        num_modality = len(modals)
        node_count = 0
        index =[]
        tmp = []
        
        for i in dia_len:
            nodes = list(range(i*num_modality))  # 为每个模态创建节点
            nodes = [j + node_count for j in nodes] 
            nodes_l = nodes[0:i*num_modality//3]
            nodes_a = nodes[i*num_modality//3:i*num_modality*2//3]
            nodes_v = nodes[i*num_modality*2//3:]
            index = index + list(permutations(nodes_l,2)) + list(permutations(nodes_a,2)) + list(permutations(nodes_v,2))
            # 创建图中的超边（连接不同模态的节点）
            Gnodes=[]
            for _ in range(i):
                Gnodes.append([nodes_l[_]] + [nodes_a[_]] + [nodes_v[_]])
            for ii, _ in enumerate(Gnodes):
                tmp = tmp +  list(permutations(_, 2))
            # 拼接特征
            if node_count == 0:
                ll = l[0:0+i]
                aa = a[0:0+i]
                vv = v[0:0+i]
                features = torch.cat([ll, aa, vv], dim=0)
                temp = 0+i
            else:
                ll = l[temp:temp+i]
                aa = a[temp:temp+i]
                vv = v[temp:temp+i]
                features_temp = torch.cat([ll, aa, vv], dim=0)
                features =  torch.cat([features, features_temp], dim=0)
                temp = temp+i
            node_count = node_count + i*num_modality
        edge_index = torch.cat([torch.LongTensor(index).T, torch.LongTensor(tmp).T],1).to("cuda:0")
        return edge_index, features
