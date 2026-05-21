import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


class FeatureAttentionMask(nn.Module):
    """根据特征内容动态生成掩码得分"""

    def __init__(self, in_channels):
        super(FeatureAttentionMask, self).__init__()
        self.mask_layer = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(),
            nn.Linear(in_channels // 4, in_channels),
            nn.Sigmoid()  # 输出 0-1 之间的权重
        )

    def forward(self, x):
        # x shape: [N, in_channels]
        return self.mask_layer(x)


class GraphGCN(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='add'):
        super(GraphGCN, self).__init__(aggr=aggr)
        self.gate = torch.nn.Linear(2 * in_channels, 1)
        # 新增动态掩码层
        self.dynamic_mask = FeatureAttentionMask(in_channels)

    def forward(self, x, edge_index):
        # 1. 计算动态特征得分并应用掩码
        mask_weights = self.dynamic_mask(x)
        x_masked = x * mask_weights

        # 2. 正常的消息传递过程
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x_masked)

    def message(self, x_i, x_j, edge_index, size):
        # 计算边的门控值 (保留原有逻辑)
        z = torch.cat([x_i, x_j], dim=-1)
        g = torch.sigmoid(self.gate(z))

        # 节点度归一化
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return norm.view(-1, 1) * g * x_j