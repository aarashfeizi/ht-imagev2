import torch.nn as nn
import torch.nn.functional as F
import torch
import utils

EPS = 1e-7

class COV_Loss(nn.Module):
    def __init__(self, dim, margin=1.0):
        super().__init__()
        self.dim = dim
        self.margin = margin

    def forward(self, batch):
        if len(batch.shape) != 2:
            batch = batch.reshape(-1, self.dim)

        n, d = batch.shape
        batch_cov = utils.torch_get_cov(batch)
        not_same_feat_loss = (batch_cov.sum() - batch_cov.diag().sum()) / (d * (d - 1))
        stds = torch.sqrt(batch_cov.diag() + EPS)
        same_feat_loss = (F.relu(-stds + self.margin).sum() / d)

        return not_same_feat_loss + same_feat_loss
        # todo add eps to main diagonal and subtract margin