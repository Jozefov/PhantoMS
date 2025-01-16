import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCNLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(GCNLayer, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.relu(x)
        return x