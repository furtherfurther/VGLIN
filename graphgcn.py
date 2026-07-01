import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


class FeatureAttentionMask(nn.Module):
  

    def __init__(self, in_channels):
        super(FeatureAttentionMask, self).__init__()
        self.mask_layer = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(),
            nn.Linear(in_channels // 4, in_channels),
            nn.Sigmoid() 
        )

    def forward(self, x):
        # x shape: [N, in_channels]
        return self.mask_layer(x)


class GraphGCN(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='add'):
        super(GraphGCN, self).__init__(aggr=aggr)
        self.gate = torch.nn.Linear(2 * in_channels, 1)
       
        self.dynamic_mask = FeatureAttentionMask(in_channels)

    def forward(self, x, edge_index):
       
        mask_weights = self.dynamic_mask(x)
        x_masked = x * mask_weights

      
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x_masked)

    def message(self, x_i, x_j, edge_index, size):
       
        z = torch.cat([x_i, x_j], dim=-1)
        g = torch.sigmoid(self.gate(z))

        
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return norm.view(-1, 1) * g * x_j
