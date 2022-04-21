import torch.nn as nn
import torch.nn.functional as F
import torch
import utils

EPS = 1e-7

class COV_Loss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, batch):
        # n, d = batch.shape
        batch_cov = utils.torch_get_cov(batch)
        not_same_feat_loss = batch_cov.sum() - batch_cov.diag.sum()
        stds = torch.sqrt(batch_cov.diag() + EPS)
        same_feat_loss = F.relu(-stds + self.margin).sum()

        return not_same_feat_loss + same_feat_loss
        # todo add eps to main diagonal and subtract margin