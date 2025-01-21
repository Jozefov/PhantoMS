import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class GCNLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """
        GCN Layer with optional embedding collection.

        Args:
            in_channels (int): Number of input features per node.
            out_channels (int): Number of output features per node.
        """
        super(GCNLayer, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, batch, collect_embeddings=False):
        """
        Forward pass through the GCN layer.

        Args:
            x (torch.Tensor): Node feature matrix of shape [num_nodes, in_channels].
            edge_index (torch.LongTensor): Graph connectivity in COO format with shape [2, num_edges].
            batch (torch.Tensor): Batch vector which assigns each node to a specific graph in the batch.
            collect_embeddings (bool): Flag to determine whether to collect embeddings.

        Returns:
            tuple:
                - x (torch.Tensor): Updated node features of shape [num_nodes, out_channels].
                - embedding (torch.Tensor or None): Pooled graph embedding of shape [batch_size, out_channels] if
                  collect_embeddings is True, else None.
        """
        x = self.conv(x, edge_index)
        x = self.relu(x)

        if collect_embeddings:
            x_pooled = global_mean_pool(x, batch)  # Shape: [batch_size, out_channels]
            x_pooled = x_pooled.detach().cpu()
            return x, x_pooled
        else:
            return x, None