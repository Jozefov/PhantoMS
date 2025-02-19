import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv
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


class GATLayer(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, heads: int = 4, concat: bool = True,
                 dropout: float = 0.6):
        """
        GAT Layer with optional embedding collection.

        Args:
            in_channels (int): Number of input features per node.
            hidden_channels (int): Desired output dimension per head.
            heads (int): Number of attention heads.
            concat (bool): Whether to concatenate the heads' outputs.
            dropout (float): Dropout rate on attention coefficients.
        """
        super(GATLayer, self).__init__()
        self.heads = heads
        # With concat=True, output dimension will be heads * hidden_channels.
        self.conv = GATConv(in_channels, hidden_channels, heads=heads, concat=concat, dropout=dropout)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, batch, collect_embeddings=False):
        # Compute convolution output (pre-activation)
        x_conv = self.conv(x, edge_index)  # shape: [num_nodes, heads * hidden_channels]

        out_dict = None
        if collect_embeddings:
            # Reshape into [num_nodes, heads, hidden_channels]
            x_heads = x_conv.view(x_conv.size(0), self.heads, -1)
            # Compute overall pooled embeddings using mean and max pooling.
            overall_mean = global_mean_pool(x_heads.mean(dim=1), batch)  # [batch, hidden_channels]
            overall_max = global_max_pool(x_heads.mean(dim=1), batch)  # [batch, hidden_channels]

            # Compute head-specific embeddings with both pooling methods.
            head_mean_embeddings = {}
            head_max_embeddings = {}
            for h in range(self.heads):
                head_mean_embeddings[f"head_{h + 1}_mean"] = global_mean_pool(x_heads[:, h, :], batch)
                head_max_embeddings[f"head_{h + 1}_max"] = global_max_pool(x_heads[:, h, :], batch)

            # Combine the overall and per-head pooled embeddings into one dictionary.
            out_dict = {"overall_mean": overall_mean, "overall_max": overall_max}
            out_dict.update(head_mean_embeddings)
            out_dict.update(head_max_embeddings)

        # Now apply the activation function
        x = self.relu(x_conv)
        return x, out_dict


class GINLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """
        GIN Layer with optional embedding collection.

        Args:
            in_channels (int): Number of input features per node.
            out_channels (int): Number of output features per node.
        """
        super(GINLayer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.conv = GINConv(self.mlp)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, batch, collect_embeddings=False):
        """
        Forward pass through the GIN layer.

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


class SAGELayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalize: bool = True):
        """
        SAGE Layer with optional embedding collection.

        Args:
            in_channels (int): Number of input features per node.
            out_channels (int): Number of output features per node.
            normalize (bool): Whether to normalize the output features.
        """
        super(SAGELayer, self).__init__()
        self.conv = SAGEConv(in_channels, out_channels, normalize=normalize)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, batch, collect_embeddings=False):
        """
        Forward pass through the SAGE layer.

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