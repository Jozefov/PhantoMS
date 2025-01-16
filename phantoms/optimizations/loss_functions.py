import torch.nn as nn


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