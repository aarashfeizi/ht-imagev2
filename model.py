import torch
import torch.nn as nn
import torch.nn.functional as F
import backbones

class TopModule(nn.Module):
    def __init__(self, args, encoder):
        super(TopModule, self).__init__()
        self.metric = 'cosine'
        self.encoder = encoder

    def get_preds(self, embeddings):
        if self.metric == 'cosine':
            norm_embeddings = F.normalize(embeddings, p=2)
            cosine_sim = torch.matmul(norm_embeddings, norm_embeddings.T)
            preds = (cosine_sim + 1) / 2
            preds = torch.clamp(preds, min=0.0, max=1.0)
        else:
            raise Exception(f'{self.metric} not supported in Top Module')

        return preds, cosine_sim


    def forward(self, imgs):
        embeddings = self.encoder(imgs)
        preds, sims = self.get_preds(embeddings)

        return preds, sims, embeddings

def get_top_module(args):
    encoder = backbones.get_bb_network(args)
    return TopModule(args, encoder)