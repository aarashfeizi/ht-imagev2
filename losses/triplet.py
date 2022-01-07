import torch.nn as nn
import torch.nn.functional as F
import torch
import utils

class TripletMargin_Loss(nn.Module):
    def __init__(self, margin):
        self.margin = margin
        pass

    def forward(self, batch, lbls):
        pass