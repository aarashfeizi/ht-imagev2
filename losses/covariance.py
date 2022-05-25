import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

EPS = 1e-7


class COV_Loss(nn.Module):
    def __init__(self, dim, margin=1.0, static_mean=False):
        super().__init__()
        self.dim = dim
        self.means = torch.zeros(size=(self.dim,))
        self.datapoint_num = 0
        self.margin = margin
        self.static_mean = static_mean

    def forward(self, batch):
        if len(batch.shape) != 2:
            batch = batch.reshape(-1, self.dim)

        n, d = batch.shape

        if self.static_mean:
            if batch.device.type == 'cuda':
                self.means = self.means.cuda()

            batch_cov, new_means = utils.torch_get_cov_with_previous(batch, self.means, self.datapoint_num)
            self.update_means(new_means=new_means, new_size=(self.datapoint_num + n))
        else:
            batch_cov = utils.torch_get_cov(batch)


        batch_cov_2 = (batch_cov * batch_cov)
        cov_loss = (batch_cov_2.sum() - batch_cov_2.diag().sum()) / (d * (d - 1))
        stds = torch.sqrt(batch_cov.diag() + EPS)
        var_loss = (F.relu(-stds + self.margin).sum() / d)

        return cov_loss, var_loss

    def reset_means(self):
        self.means = torch.zeros(size=(self.dim,))
        self.datapoint_num = 0

    def update_means(self, new_means, new_size):
        self.means = new_means.detach().cpu()
        self.datapoint_num = new_size

    def get_means(self):
        return self.means
