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