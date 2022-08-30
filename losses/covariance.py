import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

from torch import nn
import torch.distributed as dist



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


class VICReg_Loss(nn.Module):
    def __init__(self, sim_coeff, std_coeff, cov_coeff, batch_size, num_features):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.batch_size = batch_size
        self.num_features = num_features

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


    def forward(self, x, y):
        repr_loss = F.mse_loss(x, y)

        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.batch_size - 1)
        cov_y = (y.T @ y) / (self.batch_size - 1)
        cov_loss = self.off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + self.off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )

        return loss

class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]