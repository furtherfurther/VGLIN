import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model_gcn import GCN


iemocap_similarity_matrix = torch.tensor([
    [1.0000, 0.5913, 0.9622, 0.6478, 0.9873, 0.3891],  # happy
    [0.5913, 1.0000, 0.2541, 1.0000, 0.7266, 0.9869],  # sad
    [0.9622, 0.2541, 1.0000, 0.3417, 0.8625, 0.0000],  # neutral
    [0.6478, 1.0000, 0.3417, 1.0000, 0.7373, 0.9250],  # angry
    [0.9873, 0.7266, 0.8625, 0.7373, 1.0000, 0.5806],  # excited
    [0.3891, 0.9869, 0.0000, 0.9250, 0.5806, 1.0000]   # frustrated
], dtype=torch.float32)  

meld_similarity_matrix = torch.tensor([
    [1.0000, 0.8756, 0.0000, 0.2662, 0.9425, 0.0086, 0.3716],  # Neutral
    [0.8756, 1.0000, 0.5731, 0.7446, 0.9792, 0.5770, 0.7632],  # Surprise
    [0.0000, 0.5731, 1.0000, 0.9438, 0.4474, 1.0000, 0.7768],  # Fear
    [0.2662, 0.7446, 0.9438, 1.0000, 0.6347, 0.9745, 0.9852],  # Sadness
    [0.9425, 0.9792, 0.4474, 0.6347, 1.0000, 0.4558, 0.6995],  # Joy
    [0.0086, 0.5770, 1.0000, 0.9745, 0.4558, 1.0000, 0.8395],  # Disgust
    [0.3716, 0.7632, 0.7768, 0.9852, 0.6995, 0.8395, 1.0000]   # Angry
], dtype=torch.float32)


class VGSRLoss(nn.Module):
    def __init__(self, gamma=2.0, size_average=True):
        super(VGSRLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, logits, labels, dataset, matrix=iemocap_similarity_matrix_new):
        """
        :param logits: 模型的预测结果
        :param labels: 真实标签
        :param similarity_matrix: 各类别之间的相似性矩阵
        """
        if dataset == "IEMOCAP":
            matrix = iemocap_similarity_matrix
        elif dataset == "MELD":
            matrix = meld_similarity_matrix
        similarity_matrix = matrix.to(labels.device)
        labels = labels.view(-1)

        log_p = F.log_softmax(logits, dim=-1)
        pt = torch.exp(log_p)
        sub_pt = 1 - pt

        # 类别相似性加权
        similarity_weights = similarity_matrix[labels]  # 从相似性矩阵中获取每个类别的加权

        # 计算 Focal Loss，并通过相似性加权
        focal_loss = - (sub_pt ** self.gamma) * log_p * similarity_weights.unsqueeze(1)

        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class KLDivLoss(nn.Module):
    """计算 KL 散度损失时考虑一个掩码（mask），以确保只对掩码中为真（即非零）的部分计算损失"""
    def __init__(self):
        super(KLDivLoss, self).__init__()
        # reduction='sum' 表示在计算损失时将所有样本的损失相加
        self.loss = nn.KLDivLoss(reduction='sum')

    def forward(self, log_pred, target):
        """
        :param log_pred: 模型输出的对数概率
        :param target: 目标概率分布
        :param mask: 一个布尔掩码，用于指示哪些数据点应该被考虑在损失计算中
        """
        # 将掩码 mask 重塑为列向量，以便与 log_pred 和 target 进行元素级别的乘法操作。

        # 计算损失，但只对掩码中为真的位置计算。这是通过将 log_pred 和 target 分别与 mask_ 相乘来实现的，这样只有在掩码为真的位置的损失才会被计算
        # 通过将总损失除以掩码中为真的元素的数量来归一化
        # print("log_pred.size(0)", log_pred.size(0))
        loss = self.loss(log_pred, target) / log_pred.size(0)
        return loss


class MaskedNLLLoss(nn.Module):
    """在计算负对数似然损失（Negative Log Likelihood Loss, NLLLoss）时考虑一个掩码（mask），以确保只对掩码中为真（即非零）的部分计算损失"""
    def __init__(self, weight=None):
        # 它接收一个可选参数 weight，这个参数可以用于对不同的类别赋予不同的权重
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        # 如果提供了 weight 参数，它将被用于加权损失。
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')

    def forward(self, pred, target, mask):
        """
        :param pred: 模型输出的对数概率
        :param target: 目标类别的索引
        :param mask: 一个布尔掩码，用于指示哪些数据点应该被考虑在损失计算中
        """
        mask_ = mask.view(-1, 1)
        if type(self.weight) == type(None):
            loss = self.loss(pred * mask_, target) / torch.sum(mask)
        else:
            # 将总损失除以权重和掩码的乘积的总和来归一化。这里使用 .squeeze() 方法去除单一维度，因为 self.weight 可能是一维的
            # 在 Python 中，反斜杠 \ 用于表示行连接符，它可以将长行代码分成多行,下面也可以写成 ..., target) / torch.sum(self.weight[target]...
            loss = self.loss(pred * mask_, target) \
                   / torch.sum(self.weight[target] * mask_.squeeze())
        return loss


def gelu(x):
    """
    它实现了Gaussian Error Linear Unit（高斯误差线性单元）激活函数。GELU 是一种近年来在深度学习领域中流行起来的激活函数，
    特别是在自然语言处理（NLP）任务中。GELU可以看作是介于ReLU和Sigmoid/Tanh激活函数之间的折衷方案，它结合了ReLU的非饱和特性和Sigmoid/Tanh的平滑特性。
    可以之间调用 nn.GELU()
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
    """
    这个类实现了 Transformer 模型中的前馈网络（Feed-Forward Network，FFN），也称为位置感知前馈网络（Positionwise Feed-Forward Network），
    它通过两层线性变换和激活函数来增加模型的非线性处理能力。残差连接和层归一化有助于避免训练过程中的梯度消失问题。
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        :param d_model: 模型的特征维度
        :param d_ff: 前馈网络中间层的维度
        :param dropout:
        """
        super(PositionwiseFeedForward, self).__init__()
        # 创建一个线性层，用于将输入从 d_model 维度转换到 d_ff 维度
        self.w_1 = nn.Linear(d_model, d_ff)
        # 创建另一个线性层，用于将中间层的输出从 d_ff 维度转换回 d_model 维度
        self.w_2 = nn.Linear(d_ff, d_model)
        # 创建一个层归一化（Layer Normalization）层，用于对输入的特征进行归一化处理
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        # 将 GELU 激活函数赋值给 self.actv 成员变量
        self.actv = gelu
        # 分别创建两个 Dropout 层，用于在训练过程中防止过拟合
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        # 首先对输入 x 进行层归一化，然后通过第一个线性层 w_1，接着应用 GELU 激活函数，最后通过 Dropout 层
        inter = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
        # 将中间结果 inter 通过第二个线性层 w_2，然后应用第二个 Dropout 层
        output = self.dropout_2(self.w_2(inter))
        # 将 Dropout 后的输出与原始输入 x 相加，实现残差连接，然后返回结果
        return output + x


class MultiHeadedAttention(nn.Module):
    """
    多头注意力机制（Multi-Head Attention），这是 Transformer 模型中的一个关键组件。多头注意力允许模型在不同的表示子空间中并行地学习信息
    """
    def __init__(self, head_count, model_dim, dropout=0.1):
        """
        :param head_count: 注意力头的数量
        :param model_dim: 模型的特征维度
        """
        # 确保模型维度可以被头的数量整除，这是多头注意力机制的要求
        assert model_dim % head_count == 0
        # 计算每个头的维度
        self.dim_per_head = model_dim // head_count
        # 保存头的数量
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        # 创建三个线性层，用于对键（key）、值（value）和查询（query）进行变换
        # model_dim =  head_count * self.dim_per_head
        self.linear_k = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_v = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_q = nn.Linear(model_dim, head_count * self.dim_per_head)
        # Softmax 层，用于计算注意力权重
        self.softmax = nn.Softmax(dim=-1)
        # Dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)
        # 用于在最后将多头的输出合并回模型的维度
        self.linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None):
        # 获取批次大小，即batch_size
        batch_size = key.size(0)
        # 获取每个头的维度和头的数量
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        # 两个辅助函数，用于调整张量的形状以适应多头注意力的计算
        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(batch_size, -1, head_count * dim_per_head)

        # 对键、值和查询进行线性变换，并调整形状以分离不同的头
        key = self.linear_k(key).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        query = self.linear_q(query).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        #  对查询进行缩放，这是多头注意力中的一个常见操作，有助于稳定梯度
        query = query / math.sqrt(dim_per_head)
        # 计算查询和键的点积，得到注意力分数
        scores = torch.matmul(query, key.transpose(2, 3))

        # 如果提供了掩码，则将其应用于注意力分数，以屏蔽无关的信息
        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e10)

        #  通过 Softmax 层计算注意力权重
        attn = self.softmax(scores)
        # 将 Dropout 应用于注意力权重
        drop_attn = self.dropout(attn)
        # 计算加权的值，得到上下文向量
        context = torch.matmul(drop_attn, value).transpose(1, 2).\
                    contiguous().view(batch_size, -1, head_count * dim_per_head)
        # 将上下文向量通过一个线性层，输出最终的多头注意力结果
        output = self.linear(context)
        # 返回多头注意力的输出
        return output


class PositionalEncoding(nn.Module):
    """位置编码（Positional Encoding），Transformer 的一个组件，
    它通过将每个位置的正弦和余弦函数值加到输入序列上，给模型提供序列中单词的位置信息"""
    def __init__(self, dim, max_len=512):
        """
        :param dim: 位置编码的维度
        :param max_len: 位置编码的最大长度
        """
        super(PositionalEncoding, self).__init__()
        # 用于存储位置编码
        pe = torch.zeros(max_len, dim)
        # 创建一个从 0 到 max_len 的序列，并将它们转换为列向量
        position = torch.arange(0, max_len).unsqueeze(1)
        # 计算除数项，用于后续的正弦和余弦函数中。这个除数项是为了在不同维度上对位置信息进行缩放
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        # 分别计算位置编码的奇数和偶数维度上的正弦和余弦值
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        # 在位置编码张量前添加一个维度，使其形状变为 (1, max_len, dim)，以便于与批次中的序列进行广播相加
        pe = pe.unsqueeze(0)
        # 将位置编码注册为模块的缓冲区，这样它就不会被视为模型参数，不会在训练过程中更新
        self.register_buffer('pe', pe)

    def forward(self, x, speaker_emb):
        # 获取输入 x 的序列长度
        L = x.size(1)
        # 从位置编码中取出与输入序列长度相匹配的部分
        pos_emb = self.pe[:, :L]
        # 将位置编码和说话人嵌入（如果有的话）加到输入 x 上，为模型提供位置信息
        x = x + pos_emb + speaker_emb
        # 返回添加了位置编码的输出
        return x


class TransformerEncoderLayer(nn.Module):
    """这个类实现了 Transformer 模型中的一个编码器层，它由自注意力机制（Self-Attention）和前馈网络（Feed-Forward Network）
    组成，并且包含了层归一化（Layer Normalization）和残差连接"""
    def __init__(self, d_model, heads, d_ff, dropout):
        """
        :param d_model: 模型的特征维度
        :param heads: 多头注意力机制中的头数
        :param d_ff: 前馈网络中间层的维度
        :param dropout:
        """
        super(TransformerEncoderLayer, self).__init__()
        # 用于实现自注意力机制
        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        # 用于实现前馈网络
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        # 用于归一化输入特征
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, inputs_a, inputs_b, mask):
        # 检查 inputs_a 和 inputs_b 是否相等。如果相等，表示它们是同一输入的不同表示，这通常用于 Transformer 的解码器部分，其中 inputs_b 是编码器的输出
        if inputs_a.equal(inputs_b):
            # 如果不是第一次迭代（通常用于训练过程中的增量学习），则对 inputs_b 应用层归一
            if (iter != 0):
                inputs_b = self.layer_norm(inputs_b)
            else:
                inputs_b = inputs_b

            # 调整掩码的形状，以匹配自注意力机制的输入要求
            mask = mask.unsqueeze(1)
            # 应用自注意力机制，计算上下文信息
            context = self.self_attn(inputs_b, inputs_b, inputs_b, mask=mask)
        else:
            #  如果 inputs_a 和 inputs_b 不相等，表示它们是不同的输入，这通常用于 Transformer 的编码器部分
            if (iter != 0):
                inputs_b = self.layer_norm(inputs_b)
            else:
                inputs_b = inputs_b

            mask = mask.unsqueeze(1)
            context = self.self_attn(inputs_a, inputs_a, inputs_b, mask=mask)

        # 应用 Dropout 并添加残差连
        out = self.dropout(context) + inputs_b
        # 将残差连接的输出传递给前馈网络，返回最终的编码器层输出
        return self.feed_forward(out)


class TransformerEncoder(nn.Module):
    """这个类实现了 Transformer 模型中的编码器部分，由多个编码器层（TransformerEncoderLayer）组成，
    并且包含了位置编码（PositionalEncoding）和 Dropout 层"""
    def __init__(self, d_model, d_ff, heads, layers, dropout=0.1):
        """
        :param d_model: 模型的特征维度
        :param d_ff: 前馈网络中间层的维度
        :param heads: 多头注意力机制中的头数
        :param layers: 编码器中的层数
        :param dropout:
        """
        super(TransformerEncoder, self).__init__()
        # 保存模型的特征维度、编码器中的层数
        self.d_model = d_model
        self.layers = layers
        # 用于给输入序列添加位置编码
        self.pos_emb = PositionalEncoding(d_model)
        # 创建一个模块列表，包含 layers 个 TransformerEncoderLayer 实例
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_a, x_b, mask, speaker_emb):
        # 检查 x_a 和 x_b 是否相等。如果相等，表示它们是同一输入的不同表示，这通常用于 Transformer 的解码器部分，其中 x_b 是编码器的输出
        if x_a.equal(x_b):
            # 将位置编码和说话人嵌入（如果有的话）加到 x_b 上
            x_b = self.pos_emb(x_b, speaker_emb)
            x_b = self.dropout(x_b)
            # 遍历所有的编码器层
            for i in range(self.layers):
                # 对每个编码器层，将 x_b 作为输入传递进去，并使用掩码（mask.eq(0)）来屏蔽无关的信息
                x_b = self.transformer_inter[i](i, x_b, x_b, mask.eq(0))
        # 如果 x_a 和 x_b 不相等，表示它们是不同的输入，这通常用于 Transformer 的编码器部分
        else:
            # 分别将位置编码和说话人嵌入加到 x_a 和 x_b 上
            x_a = self.pos_emb(x_a, speaker_emb)
            x_a = self.dropout(x_a)
            x_b = self.pos_emb(x_b, speaker_emb)
            x_b = self.dropout(x_b)
            for i in range(self.layers):
                #  对每个编码器层，将 x_a 和 x_b 作为输入传递进去，并使用掩码（mask.eq(0)）来屏蔽无关的信
                x_b = self.transformer_inter[i](i, x_a, x_b, mask.eq(0))
        return x_b


class EnhancedFilterModule(nn.Module):
    def __init__(self, hidden_size, dataset):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        if dataset == 'MELD':
            # 将权重初始化为单位矩阵
            self.gate[0].weight.data.copy_(torch.eye(hidden_size, hidden_size))
            # 冻结权重，使其在训练过程中不更新
            self.gate[0].weight.requires_grad = False

    def forward(self, x):
        gate = self.gate(x)
        out = gate * x
        return out


class Multimodal_GatedFusion(nn.Module):
    """这个类实现了多模态数据的门控融合，它允许模型结合来自不同模态的特征，并根据每个模态的重要性自动调整其贡献
    它结合了门控机制和加权融合的方法。
    """
    def __init__(self, hidden_size):
        """
        :param hidden_size: 特征的维度
        """
        super(Multimodal_GatedFusion, self).__init__()
        # 1. 用于变换特征
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        # 2. 门控机制：创建一个 Softmax 层，用于在倒数第二个维度上计算门控系数的归一化
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, a, b, c):
        # 3. 特征的处理和拼接
        # 接收三个不同模态的输入特征 a、b 和 c. 将输入特征扩展一个维度，以便进行拼接
        a_new = a.unsqueeze(-2)
        b_new = b.unsqueeze(-2)
        c_new = c.unsqueeze(-2)
        # 4. 计算门控系数
        # 将三个模态的特征拼接在一起，形成一个新张量
        utters = torch.cat([a_new, b_new, c_new], dim=-2)
        # 过线性层处理每个模态的特征，并将它们拼接在一起
        utters_fc = torch.cat([self.fc(a).unsqueeze(-2), self.fc(b).unsqueeze(-2), self.fc(c).unsqueeze(-2)], dim=-2)
        # 计算门控系数的 Softmax，这将为每个模态的特征分配一个权重，表示其在融合过程中的重要
        utters_softmax = self.softmax(utters_fc)

        # 5. 加权融合
        #  将门控系数与拼接的特征相乘，得到加权后的特征表示
        utters_three_model = utters_softmax * utters
        # 将加权后的特征在指定维度上求和，得到最终的融合表示
        final_rep = torch.sum(utters_three_model, dim=-2, keepdim=False)
        # 返回融合后的特征表示
        return final_rep


def simple_batch_graphify(features, lengths, no_cuda):
    node_features = []
    batch_size = features.size(1)
    for j in range(batch_size):
        node_features.append(features[:lengths[j], j, :])

    node_features = torch.cat(node_features, dim=0)

    if not no_cuda:
        node_features = node_features.to("cuda:0")
    return node_features


class GlobalSemanticPreservingTransformerInteraction(nn.Module):
    def __init__(self, hidden_dim, dataset=None, n_head=None, dropout=None):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.q_q = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.k_q = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.v_q = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)

        self.q_q_gate = EnhancedFilterModule(hidden_dim, dataset)
        self.k_q_gate = EnhancedFilterModule(hidden_dim, dataset)
        self.v_q_gate = EnhancedFilterModule(hidden_dim, dataset)

        self.features_reduce = nn.Linear(3 * hidden_dim, hidden_dim)

    def forward(self, q, k, v, u_mask, spk_embeddings):
        q_q_out = self.q_q(q, q, u_mask, spk_embeddings)
        k_q_out = self.k_q(k, q, u_mask, spk_embeddings)
        v_q_out = self.v_q(v, q, u_mask, spk_embeddings)

        q_q_out = self.q_q_gate(q_q_out)
        k_q_out = self.k_q_gate(k_q_out)
        v_q_out = self.v_q_gate(v_q_out)

        out = self.features_reduce(torch.cat([q_q_out, k_q_out, v_q_out], dim=-1))

        return out


class Transformer_Based_Model(nn.Module):
    """这个类实现了一个基于 Transformer 的多模态情感分类模型，它处理文本、视觉和音频数据，并预测情感类别"""
    """这里面糅合了蒸馏模型
    def __init__(self, dataset, temp, D_text, D_visual, D_audio, n_head,
                 n_classes, hidden_dim, n_speakers, dropout):
    """
    def __init__(self, dataset, temp, D_text, D_visual, D_audio, n_head,
                 n_classes, hidden_dim, n_speakers, dropout, D_g=1024, graph_hidden_size=1024, num_L = 3, num_K = 4, modals='avl'):
        """
        :param dataset:表示使用的数据集的名称或类型。模型可能会根据不同的数据集调整其处理方式或融合机制。
        :param temp:温度参数，通常用于调整softmax的平滑程度
        :param D_text:输入的文本特征的维度
        :param D_visual:输入的视觉特征的维度
        :param D_audio:输入的音频特征的维度
        :param n_classes:输出的情感类别数
        :param hidden_dim:模型的隐藏层维度。这个参数指定了Transformer编码器、卷积层、嵌入层等组件的内部表示维度。
        :param n_speakers: 对话中的说话者数量。它指定了对话中可能存在的不同说话者的数量，并用于说话者嵌入层的大小以及说话者信息的处理。
        """
        super(Transformer_Based_Model, self).__init__()
        # 保存温度参数，通常用于调整 softmax 的平滑程度
        self.temp = temp
        self.n_classes = n_classes
        self.n_speakers = n_speakers
        self.dropout = dropout
        """
         2. n_speakers的使用：说话者掩码和索引调整
         如果 n_speakers 是 2 或 9，代码中会设置不同的 padding_idx 和填充策略
        """
        if self.n_speakers == 2:
            padding_idx = 2
        if self.n_speakers == 9:
            padding_idx = 9
        #  创建一个嵌入层，用于将说话者索引转换为嵌入向量
        """
        1. n_speakers的使用：嵌入层大小:
        n_speakers 参数用于定义说话者嵌入层 (self.speaker_embeddings) 的大小。嵌入层的输入是说话者的索引（从 0 到 n_speakers），
        它的输出是每个说话者的嵌入向量。模型使用这些嵌入向量来表示对话中的不同说话者。
        
        这里的 n_speakers + 1 是因为通常在 nn.Embedding 中会保留一个额外的索引用于填充（padding）。" padding_idx 参数用于指定哪个索引值
        是用于填充的" ，因此我们通常需要提供比实际说话者数量多一个的大小。
        """
        self.speaker_embeddings = nn.Embedding(n_speakers+1, hidden_dim, padding_idx)
        
        # Temporal convolutional layers
        # 创建三个一维卷积层，用于将不同模态的输入特征转换为模型所需的隐藏维度
        # textf_input的f指的是features
        self.textf_input = nn.Conv1d(D_text, hidden_dim, kernel_size=1, padding=0, bias=False)
        self.acouf_input = nn.Conv1d(D_audio, hidden_dim, kernel_size=1, padding=0, bias=False)
        self.visuf_input = nn.Conv1d(D_visual, hidden_dim, kernel_size=1, padding=0, bias=False)

        # Multimodal-level Gated Fusion, 用于多模态级别的门控融合
        self.last_gate = Multimodal_GatedFusion(hidden_dim)

        # Emotion Classifier
        # 创建三个顺序模型，每个模态一个，用于情感分类
        self.t_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
            )
        self.a_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
            )
        self.v_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
            )

        self.t_out = GlobalSemanticPreservingTransformerInteraction(hidden_dim, dataset, n_head, dropout)
        self.a_out = GlobalSemanticPreservingTransformerInteraction(hidden_dim, dataset, n_head, dropout)
        self.v_out = GlobalSemanticPreservingTransformerInteraction(hidden_dim, dataset, n_head, dropout)

        # return_feature没有使用, 其他采用默认参数
        self.graph_model = GCN(n_dim=D_g, nhidden=graph_hidden_size,
                               dropout=self.dropout, lamda=0.5, alpha=0.1, variant=True,
                               return_feature=True, use_residue=False, n_speakers=n_speakers,
                               modals='avl', use_speaker=True, use_modal=False, num_L=num_L,
                               num_K=num_K)
        self.multi_modal = True
        self.att_type = 'concat_DHT'
        self.modals = self.modals = [x for x in modals]  # a, v, l
        self.use_residue = False

        if self.multi_modal:
            self.dropout_ = nn.Dropout(self.dropout)
            self.hidfc = nn.Linear(graph_hidden_size, n_classes)
            if self.att_type == 'concat_subsequently':
                if self.use_residue:
                    self.smax_fc = nn.Linear((D_g+graph_hidden_size)*len(self.modals), n_classes)
                else:
                    self.smax_fc = nn.Linear((graph_hidden_size)*len(self.modals), n_classes)
            elif self.att_type == 'concat_DHT':
                if self.use_residue:
                    self.smax_fc = nn.Linear((D_g+graph_hidden_size*2)*len(self.modals), n_classes)
                else:
                    # print("len(self.modals)", len(self.modals))  # 3
                    # print("graph_hidden_size", graph_hidden_size)  # 1024
                    # print("n_classes", n_classes)  # 6
                    self.smax_fc = nn.Linear((graph_hidden_size*2)*len(self.modals), n_classes)

            elif self.att_type == 'gated':
                if len(self.modals) == 3:
                    print("len(self.modals)", len(self.modals))
                    print("graph_hidden_size", len(graph_hidden_size))
                    self.smax_fc = nn.Linear(100*len(self.modals), graph_hidden_size)
                else:
                    self.smax_fc = nn.Linear(100, graph_hidden_size)
            else:
                self.smax_fc = nn.Linear(D_g+graph_hidden_size*len(self.modals), graph_hidden_size)

        self.all_output_layer = nn.Linear(hidden_dim, n_classes)

    def forward(self, textf, visuf, acouf, u_mask, qmask, dia_len, epoch=None):
        # print(qmask.shape)
        # print("textf", textf.shape)  # ([110, 16, 1024])
        # print("visuf", visuf.shape)  # ([110, 16, 342])
        # print("acouf", acouf.shape)  # ([110, 16, 1582])
        """
        def forward(self, textf, visuf, acouf, u_mask, qmask, dia_len):
        :param textf: 文本模态的输入特征，f是features
        :param visuf: 视觉模态的输入特征
        :param acouf:音频模态的输入特征
        :param u_mask:通常用于标识有效或无效的模态数据区域，例如处理填充数据或选择性关注特征。
        :param qmask:用于指示对话中说话者的变化或每个时间步的说话者索引，帮助模型理解不同时间步中的说话者信息。
        :param dia_len: 对话长度
        :return:
        """
        # 通过 qmask（一个用于指示说话者变化的掩码）找到每个时间步的说话者索引
        # 因为下面的加入函数需要的格式不同，故将train_or_eval_model的中qmask维度变换注释掉，在这里进行修改
        spk_idx = torch.argmax(qmask, -1)
        spk_idx = torch.argmax(qmask.permute(1, 0, 2), -1)
        origin_spk_idx = spk_idx
        """
         2. n_speakers的使用：对话处理中的应用
         n_speakers 还用于调整对话长度（dia_len）后的说话者索引。这样做的目的是在对话结束之后，用特定的索引值来填充
         那些不再参与对话的时间步。这通常用于在批处理操作中保持张量的形状一致。
        """
        if self.n_speakers == 2:
            # dia_len 是一个包含每个对话长度的列表，i 是对话的索引，x 是该对话的长度
            for i, x in enumerate(dia_len):
                """
                ??????
                """
                spk_idx[i, x:] = (2*torch.ones(origin_spk_idx[i].size(0)-x)).int().cuda()
        if self.n_speakers == 9:
            for i, x in enumerate(dia_len):
                spk_idx[i, x:] = (9*torch.ones(origin_spk_idx[i].size(0)-x)).int().cuda()
        # 将说话者索引转换为嵌入向量
        spk_embeddings = self.speaker_embeddings(spk_idx)

        # Temporal convolutional layers
        # 通过一维卷积层 textf_input、acouf_input、visuf_input 对输入特征进行处理，并将特征维度转换为模型所需的维度
        textf = self.textf_input(textf.permute(1, 2, 0)).transpose(1, 2)
        acouf = self.acouf_input(acouf.permute(1, 2, 0)).transpose(1, 2)
        visuf = self.visuf_input(visuf.permute(1, 2, 0)).transpose(1, 2)
        # print("text", textf.shape)
        # print("acouf", acouf.shape)
        # print("vis", visuf.shape)

        t_transformer_out = self.t_out(textf, acouf, visuf, u_mask, spk_embeddings)
        a_transformer_out = self.a_out(acouf, textf, visuf, u_mask, spk_embeddings)
        v_transformer_out = self.v_out(visuf, textf, acouf, u_mask, spk_embeddings)

        # GCN
        # print("a_transformer_out", a_transformer_out.shape)  # ([16, 83, 1024])
        # print("v_transformer_out", v_transformer_out.shape)  #
        # print("t_transformer_out", t_transformer_out.shape)  #
        features_a = simple_batch_graphify(a_transformer_out.permute(1, 0, 2), dia_len, False)
        features_v = simple_batch_graphify(v_transformer_out.permute(1, 0, 2), dia_len, False)
        features_l = simple_batch_graphify(t_transformer_out.permute(1, 0, 2), dia_len, False)
        # print("features_a", features_a.shape)  # ([758, 1024])
        # print("features_v", features_v.shape)  #
        # print("features_l", features_l.shape)  #
        # print("dia_len", len(dia_len))  # 16
        # print("qmask", qmask.shape)  # ([62, 16, 2])
        # print("epoch", epoch)  #

        emotions_feat = self.graph_model(features_a, features_v, features_l, dia_len, qmask, epoch)
        emotions_feat = self.dropout_(emotions_feat)
        emotions_feat = nn.ReLU()(emotions_feat)
        #  如果你的张量有多个维度（例如批次维度和类别维度），你需要指定哪一个维度表示类别，以便正确计算 Softmax, 下面是2
        # print("emotions_feat", emotions_feat.shape)  # ([782, 6144])
        # print("self.smax_fc(emotions_feat)", self.smax_fc(emotions_feat).shape)  # ([736, 6])
        log_prob = F.log_softmax(self.smax_fc(emotions_feat), 1)
        prob = F.softmax(self.smax_fc(emotions_feat), 1)

        # Multimodal-level Gated Fusion, 对所有模态融合后的特征进行最终的门控融合
        all_transformer_out = self.last_gate(t_transformer_out, a_transformer_out, v_transformer_out)

        # Emotion Classifier
        # 通过情感分类器处理特征，得到每个模态和多模态融合后的情感预测（既有单模态又有多模态）
        t_final_out = self.t_output_layer(t_transformer_out)
        a_final_out = self.a_output_layer(a_transformer_out)
        v_final_out = self.v_output_layer(v_transformer_out)
        all_final_out = self.all_output_layer(all_transformer_out)
        t_final_out_1 = self.t_output_layer(features_l)
        a_final_out_1 = self.a_output_layer(features_a)
        v_final_out_1 = self.v_output_layer(features_v)
        # print("t_final_out", t_final_out.shape)  # ([16, 74, 6])
        # print("t_final_out_1", t_final_out_1.shape)  # ([758, 6])

        # 计算softmax和log_softmax，用于后续的损失计算和概率解释
        """"
        2 是指定的维度。在这个上下文中，2 表示对最后一个维度（通常是类别维度）进行对数 Softmax 操作。
        
        在分类任务中，我们通常在类别维度上应用 Softmax，因为我们希望得到每个类别的概率分布。
        如果你的张量有多个维度（例如批次维度和类别维度），你需要指定哪一个维度表示类别，以便正确计算 Softmax。
        """
        t_log_prob = F.log_softmax(t_final_out, 2)  # 2->1
        a_log_prob = F.log_softmax(a_final_out, 2)
        v_log_prob = F.log_softmax(v_final_out, 2)
        t_log_prob_1 = F.log_softmax(t_final_out_1, 1)  # 2->1
        a_log_prob_1 = F.log_softmax(a_final_out_1, 1)
        v_log_prob_1 = F.log_softmax(v_final_out_1, 1)

        all_log_prob = F.log_softmax(all_final_out, 2)
        all_prob = F.softmax(all_final_out, 2)

        # 使用温度参数 self.temp 调整的对数概率和概率，这通常用于知识蒸馏或其他正则化技术
        kl_t_log_prob = F.log_softmax(t_final_out /self.temp, 2)
        kl_a_log_prob = F.log_softmax(a_final_out /self.temp, 2)
        kl_v_log_prob = F.log_softmax(v_final_out /self.temp, 2)
        kl_t_log_prob_1 = F.log_softmax(t_final_out_1 / self.temp, 1)
        kl_a_log_prob_1 = F.log_softmax(a_final_out_1 / self.temp, 1)
        kl_v_log_prob_1 = F.log_softmax(v_final_out_1 / self.temp, 1)

        kl_all_prob = F.softmax(all_final_out /self.temp, 2)
        # print("all_final_out", all_final_out.shape)  # ([16, 74, 6])
        # print("kl_all_prob", kl_all_prob.shape)  # ([16, 74, 6])
        # print("F.softmax(self.smax_fc(emotions_feat) /self.temp, 1)", F.softmax(self.smax_fc(emotions_feat) /self.temp, 1).shape) 无法拆分成klLoss所需要的三维，只能二维，只能再弄一个klLoss函数
        kl_all_prob_1 = F.softmax(self.smax_fc(emotions_feat) /self.temp, 1)
        # print("emotions_feat", emotions_feat.shape)  # ([758, 6144])
        # print("self.smax_fc(emotions_feat)", self.smax_fc(emotions_feat).shape)  # ([758, 6])
        # print("kl_t_log_prob", kl_t_log_prob.shape)  # ([16, 74, 6])
        # print("kl_t_log_prob_1", kl_t_log_prob_1.shape)  # ([758, 6])
        # print("kl_log_prob_1", kl_all_prob_1.shape)  # ([758, 6])
        # print("emotions_feat", emotions_feat.shape)
        # print("self.smax_fc(emotions_feat)", self.smax_fc(emotions_feat).shape)  # ([764, 6])
        # print("kl_all_prob", kl_all_prob.shape)  # ([16, 94, 6])
        # print("kl_all_prob_1", kl_all_prob_1.shape)  # ([761, 6])
        # 返回不同模态和多模态融合后的情感分类的对数概率、概率以及温度调整后的概率
        # return t_log_prob, a_log_prob, v_log_prob, all_log_prob, all_prob, \
        #        kl_t_log_prob, kl_a_log_prob, kl_v_log_prob, kl_all_prob
        return t_log_prob, a_log_prob, v_log_prob, log_prob, prob, \
               kl_t_log_prob_1, kl_a_log_prob_1, kl_v_log_prob_1, kl_all_prob_1
