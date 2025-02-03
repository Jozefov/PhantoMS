import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as T
from typing import List, Optional, Dict, Tuple
from torch_geometric.nn import GATConv, global_mean_pool

from massspecgym.models.base import Stage
from massspecgym.models.de_novo.base import DeNovoMassSpecGymModel

from phantoms.utils.custom_tokenizers import ByteBPETokenizerWithSpecialTokens
from phantoms.utils.constants import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN
from phantoms.models.denovo.positional_encoding import PositionalEncoding

from torch.nn import TransformerDecoder, TransformerDecoderLayer

class GATDeNovoTransformer(DeNovoMassSpecGymModel):
    """
    Transformer model for de novo SMILES generation with a GAT encoder and a multi-layer Transformer decoder.

    When collect_embeddings=True in the forward pass, the returned dictionary (under key "embeddings") contains:
      - "gnn_i": global mean pooled output after GAT layer i.
      - "gnn_i_head_j": per-head pooled outputs from GAT layer i (for each head j).
      - "encoder_projection": output of encoder_fc after (optionally) concatenating the formula branch.
      - For each decoder layer i:
            • "decoder_layer_i": mean-pooled output of that decoder layer.
            • "decoder_layer_i_head_j": projection of the mean per head j of that layer (each projected to d_model).

    Decoding supports greedy (beam_width=1) and beam search (beam_width>1).
    """

    def __init__(
            self,
            input_dim: int,  # e.g., 4000
            d_model: int = 1024,  # desired model dimension (e.g., 1024)
            nhead: int = 8,
            num_gat_layers: int = 3,
            num_decoder_layers: int = 6,
            num_gat_heads: int = 8,
            gat_dropout: float = 0.6,
            smiles_tokenizer: ByteBPETokenizerWithSpecialTokens = None,
            start_token: str = SOS_TOKEN,
            end_token: str = EOS_TOKEN,
            pad_token: str = PAD_TOKEN,
            unk_token: str = UNK_TOKEN,
            dropout: float = 0.1,
            max_smiles_len: int = 200,
            k_predictions: int = 1,
            temperature: T.Optional[float] = 1.0,
            pre_norm: bool = False,
            chemical_formula: bool = False,
            formula_embedding_dim: int = 64,  # used if chemical_formula is True
            log_only_loss_at_stages: Optional[list] = [Stage.TRAIN],
            test_beam_width: int = 1,  # default greedy during training; can be set higher at test
            *args, **kwargs
    ):
        super().__init__(log_only_loss_at_stages=log_only_loss_at_stages, *args, **kwargs)
        if smiles_tokenizer is None:
            raise ValueError("Must provide a ByteBPETokenizerWithSpecialTokens instance.")
        self.smiles_tokenizer = smiles_tokenizer
        self.vocab_size = self.smiles_tokenizer.get_vocab_size()
        for tok in [start_token, end_token, pad_token, unk_token]:
            if tok not in self.smiles_tokenizer.get_vocab():
                raise ValueError(f"Special token '{tok}' not in tokenizer vocab")
        self.start_token_id = self.smiles_tokenizer.token_to_id(start_token)
        self.end_token_id = self.smiles_tokenizer.token_to_id(end_token)
        self.pad_token_id = self.smiles_tokenizer.token_to_id(pad_token)
        self.unk_token_id = self.smiles_tokenizer.token_to_id(unk_token)

        self.d_model = d_model
        self.max_smiles_len = max_smiles_len
        self.k_predictions = k_predictions
        self.temperature = temperature if k_predictions > 1 else None
        self.chemical_formula = chemical_formula
        self.test_beam_width = test_beam_width
        self.nhead = nhead  # number of heads

        # --- GAT Encoder ---
        self.gat_layers = nn.ModuleList()
        # Use a GATConv with concatenation; its output shape will be [num_nodes, nhead * (d_model/nhead)] = [num_nodes, d_model]
        self.gat_layers.append(
            GATConv(
                in_channels=input_dim,
                out_channels=d_model // num_gat_heads,
                heads=num_gat_heads,
                dropout=gat_dropout,
                add_self_loops=True
            )
        )
        for _ in range(num_gat_layers - 1):
            self.gat_layers.append(
                GATConv(
                    in_channels=d_model,
                    out_channels=d_model // num_gat_heads,
                    heads=num_gat_heads,
                    dropout=gat_dropout,
                    add_self_loops=True
                )
            )

        # --- Encoder Projection & Formula Integration ---
        if self.chemical_formula:
            # Formula encoder: maps from 128 to formula_embedding_dim
            self.formula_encoder = nn.Sequential(
                nn.Linear(128, formula_embedding_dim),
                nn.ReLU(),
                nn.Linear(formula_embedding_dim, formula_embedding_dim),
                nn.ReLU()
            )
            # After concatenation, input dimension is (d_model + formula_embedding_dim)
            self.encoder_fc = nn.Linear(d_model + formula_embedding_dim, d_model)
        else:
            self.encoder_fc = nn.Linear(d_model, d_model)

        # --- Transformer Decoder ---
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="relu",
            batch_first=False,
            norm_first=pre_norm
        )
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_smiles_len)
        self.decoder_embed = nn.Embedding(self.vocab_size, d_model, padding_idx=self.pad_token_id)
        self.decoder_fc = nn.Linear(d_model, self.vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)

        # --- Per-head projections for decoder layers ---
        # One projection per decoder layer to map each head's vector (d_model//nhead) to d_model.
        self.decoder_head_projs = nn.ModuleList([
            nn.Linear(d_model // nhead, d_model) for _ in range(num_decoder_layers)
        ])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.encoder_fc.weight)
        nn.init.zeros_(self.encoder_fc.bias)
        nn.init.normal_(self.decoder_embed.weight, mean=0, std=self.d_model ** -0.5)
        nn.init.xavier_uniform_(self.decoder_fc.weight)
        nn.init.zeros_(self.decoder_fc.bias)

    def step(self, batch: dict, stage: Stage = Stage.NONE) -> dict:
        ret = self.forward(batch, collect_embeddings=False)
        loss = ret["loss"]
        self.log(f"{stage.to_pref()}loss", loss, prog_bar=True, batch_size=batch["spec"].num_graphs)
        if stage not in self.log_only_loss_at_stages:
            mols_pred = self.decode_smiles(batch, beam_width=self.test_beam_width)
            ret["mols_pred"] = mols_pred
        else:
            ret["mols_pred"] = None
        return ret

    def forward(self, batch: dict, collect_embeddings: bool = False) -> dict:
        """
        Forward pass with teacher forcing.
        If collect_embeddings is True, a dictionary with intermediate embeddings is returned.
        """
        embeddings: Dict[str, torch.Tensor] = {}
        spec = batch["spec"]
        smiles_list = batch["mol"]
        x, edge_index, batch_idx = spec.x, spec.edge_index, spec.batch

        # --- GAT Encoder with per-head extraction ---
        for i, gat in enumerate(self.gat_layers, 1):
            x = gat(x, edge_index)  # shape: [num_nodes, d_model] (concatenated output)
            x = F.elu(x)
            # Overall pooled output (always needed)
            # pooled = global_mean_pool(x, batch_idx)  # [batch, d_model]
            if collect_embeddings:
                pooled = global_mean_pool(x, batch_idx)  # [batch, d_model]
                embeddings[f"gnn_{i}"] = pooled.detach()
                # Also extract per-head embeddings
                head_dim = self.d_model // self.nhead  # compute only if needed
                x_heads = x.view(x.size(0), self.nhead, head_dim)  # reshape to [num_nodes, nhead, head_dim]
                for h in range(self.nhead):
                    head_h = x_heads[:, h, :]  # [num_nodes, head_dim]
                    pooled_head = global_mean_pool(head_h, batch_idx)  # [batch, head_dim]
                    # Store each head’s pooled embedding
                    embeddings[f"gnn_{i}_head_{h + 1}"] = pooled_head.detach()
        gnn_out = global_mean_pool(x, batch_idx)  # final overall pooled output
        # if collect_embeddings:
        #     embeddings["gnn_pool"] = gnn_out.detach()

        # --- Formula Integration ---
        if self.chemical_formula and ("formula" in batch):
            formula = batch["formula"].float().to(x.device)
            formula_enc = self.formula_encoder(formula)  # [batch, formula_embedding_dim]
            combined = torch.cat([gnn_out, formula_enc], dim=1)  # [batch, d_model + formula_embedding_dim]
        else:
            combined = gnn_out
        # if collect_embeddings:
        #     embeddings["encoder_input"] = combined.detach()
        encoder_proj = self.encoder_fc(combined)  # [batch, d_model]
        if collect_embeddings:
            embeddings["encoder_projection"] = encoder_proj.detach()
        memory = encoder_proj.unsqueeze(0)  # [1, batch, d_model]

        # --- Transformer Decoder Teacher Forcing ---
        encoded_smiles = self.smiles_tokenizer.encode_batch(smiles_list)  # list of lists of token IDs
        smiles_ids = torch.tensor(encoded_smiles, dtype=torch.long, device=x.device)
        tgt_input = smiles_ids[:, :-1]
        tgt_output = smiles_ids[:, 1:]
        tgt_input = tgt_input.transpose(0, 1).contiguous()  # [seq_len-1, batch]
        tgt_output = tgt_output.transpose(0, 1).contiguous()  # [seq_len-1, batch]
        tgt_key_padding_mask = (tgt_input == self.pad_token_id).transpose(0, 1)
        tgt_embed = self.decoder_embed(tgt_input) * math.sqrt(self.d_model)
        tgt_embed = self.pos_encoder(tgt_embed)
        tgt_len = tgt_input.size(0)
        causal_mask = torch.triu(torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=x.device), diagonal=1)

        # --- Transformer Decoder: iterate layer-by-layer ---
        decoder_input = tgt_embed
        if collect_embeddings:
            for i, layer in enumerate(self.transformer_decoder.layers, 1):
                decoder_input = layer(decoder_input, memory, tgt_mask=causal_mask,
                                      tgt_key_padding_mask=tgt_key_padding_mask)
                pooled_dec = decoder_input.mean(dim=0)  # [batch, d_model]
                embeddings[f"decoder_layer_{i}"] = pooled_dec.detach()
                d_head = self.d_model // self.nhead
                reshaped = decoder_input.view(decoder_input.size(0), decoder_input.size(1), self.nhead, d_head)
                head_means = reshaped.mean(dim=0)  # [batch, nhead, d_head]
                for h in range(self.nhead):
                    head_emb = self.decoder_head_projs[i - 1](head_means[:, h, :])  # [batch, d_model]
                    embeddings[f"decoder_layer_{i}_head_{h + 1}"] = head_emb.detach()
        else:
            # If not collecting embeddings, simply run the whole decoder:
            decoder_input = self.transformer_decoder(
                tgt=tgt_embed, memory=memory, tgt_mask=causal_mask, tgt_key_padding_mask=tgt_key_padding_mask
            )

        logits = self.decoder_fc(decoder_input)
        logits = logits.transpose(0, 1).contiguous()  # [batch, seq_len-1, vocab_size]
        tgt_output = tgt_output.transpose(0, 1).contiguous()  # [batch, seq_len-1]
        loss = self.criterion(logits.view(-1, self.vocab_size), tgt_output.view(-1))
        ret = {"loss": loss}
        if collect_embeddings:
            ret["embeddings"] = embeddings
        return ret

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1)

    def beam_search_decode(self, memory: torch.Tensor, beam_width: int, max_length: int) -> List[int]:
        """
        Performs beam search decoding for a single sample.
        memory: [d_model] tensor.
        Returns the best sequence (list of token IDs).
        """
        device = memory.device
        memory = memory.unsqueeze(0).unsqueeze(0)  # [1, 1, d_model]
        beam = [([self.start_token_id], 0.0, False)]
        for _ in range(max_length):
            new_beam = []
            for seq, log_prob, finished in beam:
                if finished:
                    new_beam.append((seq, log_prob, finished))
                    continue
                seq_tensor = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(1)  # [seq_len, 1]
                tgt_embed = self.decoder_embed(seq_tensor) * math.sqrt(self.d_model)
                tgt_embed = self.pos_encoder(tgt_embed)
                seq_len = seq_tensor.size(0)
                causal_mask = self._generate_square_subsequent_mask(seq_len).to(device)
                output = self.transformer_decoder(tgt=tgt_embed, memory=memory, tgt_mask=causal_mask,
                                                  memory_key_padding_mask=None)
                logits = self.decoder_fc(output[-1])  # [1, vocab_size]
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)  # [vocab_size]
                topk = torch.topk(log_probs, beam_width)
                for token, token_log_prob in zip(topk.indices.tolist(), topk.values.tolist()):
                    new_seq = seq + [token]
                    new_log_prob = log_prob + token_log_prob
                    new_finished = (token == self.end_token_id)
                    new_beam.append((new_seq, new_log_prob, new_finished))
            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = new_beam[:beam_width]
            if all(candidate[2] for candidate in beam):
                break
        best_seq, _, _ = max(beam, key=lambda x: x[1])
        return best_seq

    def decode_smiles(self, batch: dict, beam_width: Optional[int] = None) -> List[List[str]]:
        """
        Generate SMILES for each sample in the batch.
        If beam_width is not provided, uses self.test_beam_width.
        Returns a list (length = batch_size) of lists (each containing one generated SMILES).
        """
        if beam_width is None:
            beam_width = self.test_beam_width
        spec = batch["spec"]
        x, edge_index, batch_idx = spec.x, spec.edge_index, spec.batch
        for gat in self.gat_layers:
            x = gat(x, edge_index)
            x = F.elu(x)
        x = global_mean_pool(x, batch_idx)
        if self.chemical_formula and ("formula" in batch):
            formula = batch["formula"].float().to(x.device)
            formula_enc = self.formula_encoder(formula)
            x = torch.cat([x, formula_enc], dim=1)
        memory_all = self.encoder_fc(x)
        batch_size = memory_all.size(0)
        device = memory_all.device
        decoded_list = []
        if beam_width <= 1:
            generated_tokens = torch.full((1, batch_size), self.start_token_id, dtype=torch.long, device=device)
            finished = [False] * batch_size
            decoded_sequences = [[] for _ in range(batch_size)]
            for _ in range(self.max_smiles_len):
                tgt_embed = self.decoder_embed(generated_tokens) * math.sqrt(self.d_model)
                tgt_embed = self.pos_encoder(tgt_embed)
                tgt_len = tgt_embed.size(0)
                causal_mask = self._generate_square_subsequent_mask(tgt_len).to(device)
                output = self.transformer_decoder(tgt=tgt_embed, memory=memory_all.unsqueeze(0), tgt_mask=causal_mask)
                last_logits = self.decoder_fc(output[-1])
                next_token = torch.argmax(last_logits, dim=-1).unsqueeze(0)
                generated_tokens = torch.cat([generated_tokens, next_token], dim=0)
                for i in range(batch_size):
                    if not finished[i]:
                        token_id = next_token[0, i].item()
                        if token_id == self.end_token_id:
                            finished[i] = True
                        else:
                            decoded_sequences[i].append(token_id)
                if all(finished):
                    break
            for seq in decoded_sequences:
                text = self.smiles_tokenizer.decode(seq, skip_special_tokens=True)
                decoded_list.append([text])
        else:
            for i in range(batch_size):
                mem = memory_all[i]
                best_seq = self.beam_search_decode(mem, beam_width, self.max_smiles_len)
                text = self.smiles_tokenizer.decode(best_seq, skip_special_tokens=True)
                decoded_list.append([text])
        return decoded_list

    def get_embeddings(self, batch: dict, smiles_batch: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Run a forward pass with collect_embeddings=True and return a dictionary of intermediate embeddings.
        The returned dictionary contains keys:
          - "gnn_i" for each GAT layer overall pooled output,
          - "gnn_i_head_j" for per-head pooled outputs of each GAT layer,
          - "encoder_projection" for the output of encoder_fc,
          - "decoder_layer_i" for overall pooled output of each decoder layer,
          - "decoder_layer_i_head_j" for per-head pooled outputs from each decoder layer.
        """
        ret = self.forward(batch, collect_embeddings=True)
        return ret.get("embeddings", {})

    def load_pretrained_decoder(self, pretrained_state_dict: dict):
        """
        Loads pretrained decoder weights (e.g., from MoleculeDecoder) into the decoder parts of this model.
        Only keys starting with "decoder", "pos_encoder", "decoder_embed", or "decoder_fc" are updated.
        """
        model_dict = self.state_dict()
        pretrained_keys = {
            k: v for k, v in pretrained_state_dict.items()
            if k in model_dict and (
                    k.startswith("decoder") or
                    k.startswith("pos_encoder") or
                    k.startswith("decoder_embed") or
                    k.startswith("decoder_fc")
            )
        }
        model_dict.update(pretrained_keys)
        self.load_state_dict(model_dict)
        print("Pretrained decoder weights loaded into GATDeNovoTransformer.")