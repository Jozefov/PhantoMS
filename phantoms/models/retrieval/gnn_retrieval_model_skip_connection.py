import torch
import torch.nn.functional as F
import torch.nn as nn
from massspecgym.models.base import Stage
from torch_geometric.nn import global_mean_pool
from phantoms.layers.gcn_layers import GCNLayer
from phantoms.heads.retrieval_heads import SkipConnectionRetrievalHead
from phantoms.optimizations.loss_functions import BCEWithLogitsLoss
from massspecgym.models.retrieval.base import RetrievalMassSpecGymModel
from phantoms.utils.constants import ELEMENTS
from phantoms.utils.data import encode_formula, smiles_to_formula
from typing import List, Optional

class GNNRetrievalSkipConnections(RetrievalMassSpecGymModel):
    def __init__(
        self,
        hidden_channels: int = 2048,
        out_channels: int = 2048,  # Fingerprint size
        node_feature_dim: int = 1024,
        dropout_rate: float = 0.2,
        bottleneck_factor: float = 1.0,
        num_skipblocks: int = 3,
        num_gcn_layers: int = 3,
        use_formula: bool = False,
        formula_embedding_dim: int = 64,
        *args,
        **kwargs
    ):
        """
        GNN-based retrieval model with optional skip connections and molecular formula integration.

        Args:
            hidden_channels (int): Number of hidden channels in all layers except input and output layers.
            out_channels (int): Dimension of the output fingerprint vector.
            node_feature_dim (int): Dimension of input node features.
            dropout_rate (float): Dropout rate.
            bottleneck_factor (float): Factor to control bottleneck in SkipBlocks.
            num_skipblocks (int): Number of SkipBlocks in the retrieval head.
            num_gcn_layers (int): Number of GCN layers.
            use_formula (bool): Whether to integrate molecular formula information.
            formula_embedding_dim (int): Dimension for molecular formula encoding.
        """

        super().__init__(*args, **kwargs)

        self.use_formula = use_formula
        self.formula_embedding_dim = formula_embedding_dim

        self.num_gcn_layers = num_gcn_layers

        # Define GCN Layers dynamically
        self.gcn_layers = nn.ModuleList()
        in_channels = node_feature_dim
        for i in range(num_gcn_layers):
            self.gcn_layers.append(GCNLayer(in_channels, hidden_channels))
            in_channels = hidden_channels

        # If using formula, define formula encoder
        if self.use_formula:
            self.formula_encoder = nn.Sequential(
                nn.Linear(len(ELEMENTS), self.formula_embedding_dim),
                nn.ReLU(),
                nn.Linear(self.formula_embedding_dim, self.formula_embedding_dim),
                nn.ReLU()
            )
            head_input_size = hidden_channels + self.formula_embedding_dim
        else:
            head_input_size = hidden_channels

        # Define Skip Connection Retrieval Head
        self.head = SkipConnectionRetrievalHead(
            input_size=head_input_size,
            hidden_size=hidden_channels,
            output_size=out_channels,
            bottleneck_factor=bottleneck_factor,
            num_skipblocks=num_skipblocks,
            dropout_rate=dropout_rate
        )

        # Define Loss Function
        self.loss_fn = BCEWithLogitsLoss()  # Binary vector prediction

    def forward(self, data, collect_embeddings=False, smiles_batch: Optional[List[str]] = None):
        """
        Forward pass to predict molecular fingerprint from spectral tree and (optionally) molecular formula.

        Args:
            data (torch_geometric.data.Data): Batched spectral trees.
            collect_embeddings (bool): If True, collect embeddings from each layer.
            smiles_batch (Optional[List[str]]): List of SMILES strings for the batch. Required if use_formula is True.

        Returns:
            torch.Tensor: Predicted molecular fingerprints. Shape: [batch_size, fp_size]
            Optional[dict]: If collect_embeddings is True, returns a dictionary of embeddings.
        """

        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Pass through GCN layers
        embeddings = {}
        for idx, gcn in enumerate(self.gcn_layers, 1):
            x, gnn_embeddings = gcn(x, edge_index, batch, collect_embeddings=collect_embeddings)
            if collect_embeddings and gnn_embeddings is not None:
                embeddings[f'gcn{idx}'] = gnn_embeddings  # e.g., 'gcn1': [batch_size, hidden_channels]

        x_pooled = global_mean_pool(x, batch)

        if self.use_formula:
            if smiles_batch is None:
                raise ValueError("smiles_batch must be provided when use_formula is enabled.")
            # Convert SMILES to formulas
            formulas = [smiles_to_formula(smiles) for smiles in smiles_batch]

            # Encode formulas and move to the same device as x
            formula_encodings = torch.stack([encode_formula(formula) for formula in formulas]).to(x.device)
            formula_encodings = self.formula_encoder(formula_encodings)  # [batch_size, formula_embedding_dim]

            # Concatenate GCN output with formula encodings
            combined = torch.cat([x_pooled, formula_encodings],
                                 dim=1)  # [batch_size, hidden_channels + formula_embedding_dim]
        else:
            combined = x_pooled  # [batch_size, hidden_channels]

        # Pass through Retrieval Head
        x, head_embeddings = self.head(combined, collect_embeddings=collect_embeddings)  # [batch_size, fp_size]
        if collect_embeddings and head_embeddings is not None:
            for layer_name, embed in head_embeddings.items():
                embeddings[f'head_{layer_name}'] = embed

        if collect_embeddings:
            return x, embeddings  # [batch_size, fp_size], dict
        else:
            return x


    def get_embeddings(self, data, smiles_batch: Optional[List[str]] = None):
        """
        Extract embeddings from each layer.

        Args:
            data (dict): Batch of data.
            smiles_batch (Optional[List[str]]): List of SMILES strings, required if use_formula is True, for bonus task.

        Returns:
            dict: Dictionary containing embeddings from each layer.
        """
        _, embeddings = self.forward(data, collect_embeddings=True, smiles_batch=smiles_batch)
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

        if self.use_formula:
            smiles = batch['smiles']
            if smiles is None:
                raise ValueError("Batch does not contain 'smiles' key required for formula encoding.")
            fp_pred = self.forward(data, smiles_batch=smiles)  # Shape: [batch_size, fp_size]
        else:
            fp_pred = self.forward(data)  # Shape: [batch_size, fp_size]

        # Compute loss
        loss = self.loss_fn(fp_pred, fp_true)

        # Compute similarity scores
        fp_pred_repeated = fp_pred.repeat_interleave(batch_ptr, dim=0)  # Shape: [total_candidates, fp_size]
        scores = F.cosine_similarity(fp_pred_repeated, cands)  # Shape: [total_candidates]

        return {'loss': loss, 'scores': scores}
