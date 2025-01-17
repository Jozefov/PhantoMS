import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from massspecgym.models.base import Stage
from phantoms.layers.gcn_layers import GCNLayer
from phantoms.heads.retrieval_heads import RetrievalHead
from phantoms.optimizations.loss_functions import MSELoss
from massspecgym.models.retrieval.base import RetrievalMassSpecGymModel


class GNNRetrievalModel(RetrievalMassSpecGymModel):
    def __init__(
            self,
            hidden_channels: int = 128,
            out_channels: int = 4096,  # Fingerprint size
            node_feature_dim: int = 1039,  # Adjust based on your 'spec.x' feature size
            *args,
            **kwargs
    ):
        """GNN-based retrieval model for MSn spectral trees."""
        super().__init__(*args, **kwargs)

        # Define GCN Layers
        self.gcn1 = GCNLayer(node_feature_dim, hidden_channels)
        self.gcn2 = GCNLayer(hidden_channels, hidden_channels)

        # Define Retrieval Head
        self.head = RetrievalHead(hidden_channels, out_channels)

        # Define Loss Function
        self.loss_fn = MSELoss()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Pass through GCN layers
        x = self.gcn1(x, edge_index)
        x = self.gcn2(x, edge_index)

        # Global Mean Pooling
        x = global_mean_pool(x, batch)  # Shape: [batch_size, hidden_channels]

        # Pass through Retrieval Head
        x = self.head(x)  # Shape: [batch_size, fp_size]

        return x

    def step(self, batch: dict, stage: Stage) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a single step of training/validation/testing.

        Args:
            batch (dict): Batch of data.
            stage (str): Stage identifier ('train', 'val', 'test').

        Returns:
            dict: Dictionary containing loss and scores.
        """
        data = batch['spec']  # PyG DataBatch
        fp_true = batch['mol']  # True fingerprints, shape: [batch_size, fp_size]
        cands = batch['candidates']  # Candidate fingerprints, shape: [total_candidates, fp_size]
        batch_ptr = batch['batch_ptr']  # Number of candidates per sample, shape: [batch_size]

        # Forward pass
        fp_pred = self.forward(data)  # Shape: [batch_size, fp_size]

        # Compute loss
        loss = self.loss_fn(fp_pred, fp_true)

        # Compute similarity scores
        fp_pred_repeated = fp_pred.repeat_interleave(batch_ptr, dim=0)  # Shape: [total_candidates, fp_size]
        scores = F.cosine_similarity(fp_pred_repeated, cands)  # Shape: [total_candidates]

        return {'loss': loss, 'scores': scores}