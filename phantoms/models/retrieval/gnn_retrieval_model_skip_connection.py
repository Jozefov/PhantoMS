import torch
import torch.nn.functional as F
import torch.nn as nn
from massspecgym.models.base import Stage
from torch_geometric.nn import global_mean_pool
from phantoms.layers.gcn_layers import GCNLayer, GATLayer, GINLayer, SAGELayer
from phantoms.heads.retrieval_heads import SkipConnectionRetrievalHead
from phantoms.optimizations.loss_functions import MSELoss, CosineSimilarityLoss
from massspecgym.models.retrieval.base import RetrievalMassSpecGymModel
from phantoms.utils.constants import ELEMENTS
from phantoms.utils.data import encode_formula, smiles_to_formula
from typing import List, Optional

class GNNRetrievalSkipConnections(RetrievalMassSpecGymModel):
    def __init__(self, hidden_channels: int = 2048,
                 out_channels: int = 2048,
                 node_feature_dim: int = 1024,
                 dropout_rate: float = 0.2,
                 bottleneck_factor: float = 1.0,
                 num_skipblocks: int = 3,
                 num_gcn_layers: int = 3,
                 use_formula: bool = False,
                 formula_embedding_dim: int = 64,
                 gnn_layer_type: str = 'GCNConv',
                 log_only_loss_at_stages: Optional[list] = [Stage.TRAIN],
                 *args, **kwargs):
        super().__init__(log_only_loss_at_stages=log_only_loss_at_stages, *args, **kwargs)
        self.use_formula = use_formula
        self.formula_embedding_dim = formula_embedding_dim
        self.num_gcn_layers = num_gcn_layers
        self.gnn_layer_type = gnn_layer_type

        layer_mapping = {
            'GCNConv': GCNLayer,
            'GATConv': GATLayer,
            'GINConv': GINLayer,
            'SAGEConv': SAGELayer
        }
        if self.gnn_layer_type not in layer_mapping:
            raise ValueError(f"Unsupported gnn_layer_type '{self.gnn_layer_type}'. Supported types: {list(layer_mapping.keys())}")

        LayerClass = layer_mapping[self.gnn_layer_type]
        self.gcn_layers = nn.ModuleList()
        in_channels = node_feature_dim
        # For GAT, read and store the number of heads (default = 4)
        if self.gnn_layer_type == "GATConv":
            self.nheads = kwargs.get("nheads", 4)
        for i in range(num_gcn_layers):
            if self.gnn_layer_type == "GATConv":
                # Create a GAT layer that outputs (nheads * hidden_channels) per node.
                self.gcn_layers.append(GATLayer(in_channels, hidden_channels, heads=self.nheads))
                # Update in_channels to match the concatenated output dimension.
                in_channels = self.nheads * hidden_channels
            else:
                self.gcn_layers.append(LayerClass(in_channels, hidden_channels))
                in_channels = hidden_channels

        # Set head_input_size to be the final GNN output dimension.
        if self.use_formula:
            self.formula_encoder = nn.Sequential(
                nn.Linear(len(ELEMENTS), self.formula_embedding_dim),
                nn.ReLU(),
                nn.Linear(self.formula_embedding_dim, self.formula_embedding_dim),
                nn.ReLU()
            )
            head_input_size = in_channels + self.formula_embedding_dim
        else:
            head_input_size = in_channels

        self.head = SkipConnectionRetrievalHead(
            input_size=head_input_size,
            hidden_size=hidden_channels,
            output_size=out_channels,
            bottleneck_factor=bottleneck_factor,
            num_skipblocks=num_skipblocks,
            dropout_rate=dropout_rate
        )
        # self.loss_fn = MSELoss()
        self.loss_fn = CosineSimilarityLoss()

    def forward(self, data, collect_embeddings=False, smiles_batch: Optional[List[str]] = None):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        embeddings = {}

        for idx, layer in enumerate(self.gcn_layers, 1):
            x, layer_embed = layer(x, edge_index, batch, collect_embeddings=collect_embeddings)
            if collect_embeddings and layer_embed is not None:
                if self.gnn_layer_type == "GATConv":
                    # Save both mean and max pooled overall embeddings.
                    embeddings[f"gnn_{idx}_overall_mean"] = layer_embed["overall_mean"]
                    embeddings[f"gnn_{idx}_overall_max"] = layer_embed["overall_max"]
                    # Save both mean and max pooled embeddings for each head.
                    for j in range(1, self.nheads + 1):
                        embeddings[f"gnn_{idx}_head_{j}_mean"] = layer_embed[f"head_{j}_mean"]
                        embeddings[f"gnn_{idx}_head_{j}_max"] = layer_embed[f"head_{j}_max"]
                else:
                    embeddings[f"gcn_{idx}"] = layer_embed

        x_pooled = global_mean_pool(x, batch)

        if self.use_formula:
            if smiles_batch is None:
                raise ValueError("smiles_batch must be provided when use_formula is enabled.")
            formulas = [smiles_to_formula(smiles) for smiles in smiles_batch]
            formula_encodings = torch.stack([encode_formula(formula) for formula in formulas]).to(x.device)
            formula_encodings = self.formula_encoder(formula_encodings)
            combined = torch.cat([x_pooled, formula_encodings], dim=1)
        else:
            combined = x_pooled

        x_out, head_embeddings = self.head(combined, collect_embeddings=collect_embeddings)
        if collect_embeddings and head_embeddings is not None:
            for layer_name, embed in head_embeddings.items():
                embeddings[f'head_{layer_name}'] = embed
        if collect_embeddings:
            return x_out, embeddings
        else:
            return x_out

    def get_embeddings(self, batch):
        data = batch['spec']
        smiles = batch.get('smiles', None) if self.use_formula else None
        _, embeddings = self.forward(data, collect_embeddings=True, smiles_batch=smiles)
        return embeddings

    def step(self, batch: dict, stage: Stage) -> tuple[torch.Tensor, torch.Tensor]:
        data = batch['spec']
        fp_true = batch['mol']
        cands = batch['candidates']
        batch_ptr = batch['batch_ptr']
        if self.use_formula:
            smiles = batch['smiles']
            if smiles is None:
                raise ValueError("Batch missing 'smiles' key required for formula encoding.")
            fp_pred = self.forward(data, smiles_batch=smiles)
        else:
            fp_pred = self.forward(data)
        loss = self.loss_fn(fp_pred, fp_true)
        fp_pred_repeated = fp_pred.repeat_interleave(batch_ptr, dim=0)
        scores = F.cosine_similarity(fp_pred_repeated, cands)
        return {'loss': loss, 'scores': scores}

