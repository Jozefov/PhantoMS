import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from phantoms.models.denovo.positional_encoding import PositionalEncoding

class MoleculeDecoder(pl.LightningModule):
    def __init__(self, vocab_size: int, d_model: int = 1024, nhead: int = 4,
                 num_decoder_layers: int = 4, dropout: float = 0.1,
                 pad_token_id: int = 0, max_len: int = 200):
        super().__init__()
        self.d_model = d_model
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
            dropout=dropout, activation="relu", batch_first=False
        )
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    def forward(self, tgt_input):
        emb = self.embedding(tgt_input) * math.sqrt(self.d_model)
        emb = self.pos_encoder(emb)
        batch = tgt_input.size(1)
        memory = torch.zeros(1, batch, self.d_model, device=tgt_input.device)
        seq_len = tgt_input.size(0)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=tgt_input.device), diagonal=1)
        output = self.decoder(tgt=emb, memory=memory, tgt_mask=causal_mask)
        logits = self.fc_out(output)
        return logits
    def training_step(self, batch, batch_idx):
        logits = self.forward(batch["input"])
        loss = self.criterion(logits.view(-1, logits.size(-1)), batch["target"].view(-1))
        self.log("loss", loss)
        return loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)