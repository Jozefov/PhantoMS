import torch.nn as nn


class RetrievalHead(nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int):
        super(RetrievalHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
            nn.Sigmoid()  # Fingerprint bits between 0 and 1
        )

    def forward(self, x):
        return self.fc(x)

class SkipBlock(nn.Module):
    def __init__(self, in_features, hidden_features, bottleneck_factor=1.0, use_dropout=True, dropout_rate=0.2):
        super(SkipBlock, self).__init__()

        self.batchNorm1 = nn.BatchNorm1d(in_features)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate) if use_dropout else nn.Identity()
        self.hidden1 = nn.utils.weight_norm(
            nn.Linear(in_features, int(hidden_features * bottleneck_factor)),
            name='weight', dim=0
        )

        self.batchNorm2 = nn.BatchNorm1d(int(hidden_features * bottleneck_factor))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate) if use_dropout else nn.Identity()
        self.hidden2 = nn.utils.weight_norm(
            nn.Linear(int(hidden_features * bottleneck_factor), in_features),
            name='weight', dim=0
        )


    def forward(self, x, collect_embeddings=False):
        identity = x
        hidden = self.batchNorm1(x)
        hidden = self.relu1(hidden)
        hidden = self.dropout1(hidden)
        hidden1 = self.hidden1(hidden)  # First hidden layer

        hidden = self.batchNorm2(hidden1)
        hidden = self.relu2(hidden)
        hidden = self.dropout2(hidden)
        hidden2 = self.hidden2(hidden)  # Second hidden layer

        out = hidden2 + identity  # Skip connection

        if collect_embeddings:
            # Return a dictionary of embeddings
            return out, {'hidden1': hidden1.detach().cpu(), 'hidden2': hidden2.detach().cpu()}
        else:
            return out, None



class SkipConnectionRetrievalHead(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        bottleneck_factor: float = 1.0,
        num_skipblocks: int = 3,
        dropout_rate: float = 0.2
    ):
        super(SkipConnectionRetrievalHead, self).__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        # Initial layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)

        # Skip blocks
        self.skipblocks = nn.ModuleList([
            SkipBlock(hidden_size, hidden_size, bottleneck_factor, use_dropout=True, dropout_rate=dropout_rate)
            for _ in range(num_skipblocks)
        ])

        # Final layers
        self.fc_final = nn.Linear(hidden_size, hidden_size)
        self.bn_final = nn.BatchNorm1d(hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x, collect_embeddings=False):
        embeddings = {}

        # Initial layer
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        if collect_embeddings:
            embeddings['head_fc1'] = x.detach().cpu()

        # Skip blocks
        for idx, skipblock in enumerate(self.skipblocks, 1):
            x, skip_embeds = skipblock(x, collect_embeddings=collect_embeddings)
            if collect_embeddings and skip_embeds is not None:
                for embed_name, embed in skip_embeds.items():
                    embeddings[f'skipblock_{idx}_{embed_name}'] = embed

        # Final layers
        x = self.fc_final(x)
        x = self.bn_final(x)
        x = self.relu(x)
        x = self.dropout(x)
        if collect_embeddings:
            embeddings['head_fc_final'] = x.detach().cpu()

        out = self.output_layer(x)
        # if collect_embeddings:
        #     embeddings['head_output'] = out.detach().cpu()

        if collect_embeddings:
            return out, embeddings
        else:
            return out, None