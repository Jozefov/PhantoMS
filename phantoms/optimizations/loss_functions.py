import torch.nn as nn
import torch

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, inputs, targets):
        return self.loss_fn(inputs, targets)


class BCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(BCEWithLogitsLoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        return self.loss_fn(inputs, targets)


class CosineSimilarityLoss(nn.Module):
    def __init__(self, margin=0.0, reduction='mean'):
        super(CosineSimilarityLoss, self).__init__()
        self.loss_fn = nn.CosineEmbeddingLoss(margin=margin, reduction=reduction)

    def forward(self, input1, input2):
        # Ensure inputs have a batch dimension. If a single example is provided, unsqueeze to add batch.
        if input1.dim() == 1:
            input1 = input1.unsqueeze(0)
        if input2.dim() == 1:
            input2 = input2.unsqueeze(0)
        # Create a target tensor of ones (we want the embeddings to be similar).
        target = torch.ones(input1.size(0), device=input1.device)
        return self.loss_fn(input1, input2, target)