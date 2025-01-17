import torch
import torch.nn.functional as F
import torch.nn as nn
from massspecgym.models.base import Stage
from torch_geometric.nn import global_mean_pool
from phantoms.layers.gcn_layers import GCNLayer
from phantoms.heads.retrieval_heads import SkipConnectionRetrievalHead
from phantoms.optimizations.loss_functions import BCEWithLogitsLoss
from massspecgym.models.retrieval.base import RetrievalMassSpecGymModel

class GNNRetrievalSkipConnections(RetrievalMassSpecGymModel):
    def __init__(
        self,
        hidden_channels: int = 2048,
        out_channels: int = 4096,  # Fingerprint size
        node_feature_dim: int = 1039,
        dropout_rate: float = 0.2,
        bottleneck_factor: float = 1.0,
        num_skipblocks: int = 3,
        num_gcn_layers: int = 3,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.num_gcn_layers = num_gcn_layers

        # Define GCN Layers dynamically
        self.gcn_layers = nn.ModuleList()
        in_channels = node_feature_dim
        for i in range(num_gcn_layers):
            self.gcn_layers.append(GCNLayer(in_channels, hidden_channels))
            in_channels = hidden_channels

        # Define Skip Connection Retrieval Head
        self.head = SkipConnectionRetrievalHead(
            input_size=hidden_channels,
            hidden_size=hidden_channels,
            output_size=out_channels,
            bottleneck_factor=bottleneck_factor,
            num_skipblocks=num_skipblocks,
            dropout_rate=dropout_rate
        )

        # Define Loss Function
        self.loss_fn = BCEWithLogitsLoss()  # Binary vector prediction

    def forward(self, data, collect_embeddings=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Pass through GCN layers
        embeddings = {}
        for idx, gcn in enumerate(self.gcn_layers, 1):
            x = gcn(x, edge_index)
            if collect_embeddings:
                x_pooled = global_mean_pool(x, batch)
                embeddings[f'gcn{idx}'] = x_pooled.detach().cpu()

        # Pass through Retrieval Head
        x, head_embeddings = self.head(x, collect_embeddings=collect_embeddings)  # Shape: [batch_size, fp_size]
        if collect_embeddings and head_embeddings is not None:
            for layer_name, embed in head_embeddings.items():
                embeddings[f'head_{layer_name}'] = embed

        if collect_embeddings:
            return x, embeddings  # Return output and embeddings
        else:
            return x  # Return output only


    def get_embeddings(self, data):

        _, embeddings = self.forward(data, collect_embeddings=True)
        return embeddings

    def step(self, batch: dict, stage: Stage) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a single step of training/validation/testing.

        Args:
            batch (dict): Batch of data.
            stage (str): Stage identifier ('train', 'val', 'test').

        Returns:
            dict: Dictionary containing loss and scores.
        """
        data = batch['spec']          # PyG DataBatch
        fp_true = batch['mol']        # True fingerprints, shape: [batch_size, fp_size]
        cands = batch['candidates']   # Candidate fingerprints, shape: [total_candidates, fp_size]
        batch_ptr = batch['batch_ptr']  # Number of candidates per sample, shape: [batch_size]

        # Forward pass
        fp_pred = self.forward(data)

        # Compute loss
        loss = self.loss_fn(fp_pred, fp_true)

        # Compute similarity scores
        fp_pred_repeated = fp_pred.repeat_interleave(batch_ptr, dim=0)
        scores = F.cosine_similarity(fp_pred_repeated, cands)

        return {'loss': loss, 'scores': scores}