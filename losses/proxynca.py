import torch
import torch.nn.functional as F
import utils

class ProxyNCA_prob(torch.nn.Module):
    def __init__(self, args):
        torch.nn.Module.__init__(self)
        sz_embed = args.get('emb_size')
        self.proxies = torch.nn.Parameter(torch.randn(args.get('nb_classes'), sz_embed) / 8)
        self.scale = args.get('scale')

    def forward(self, X, T):
        P = self.proxies

        # note: self.scale is equal to sqrt(1/T)
        # in the paper T = 1/9, therefore, scale = sart(1/(1/9)) = sqrt(9) = 3
        #  we need to apply sqrt because the pairwise distance is calculated as norm^2
        P = self.scale * F.normalize(P, p=2, dim=-1)
        X = self.scale * F.normalize(X, p=2, dim=-1)

        D = utils.pairwise_distance(
            torch.cat(
                [X, P]
            ),
            squared=True
        )[:X.size()[0], X.size()[0]:]

        T = utils.binarize_and_smooth_labels(
            T=T, nb_classes=len(P), smoothing_const=0
        )
        loss = torch.sum(- T * F.log_softmax(-D, -1), -1)
        loss = loss.mean()
        return loss

