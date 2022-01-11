import torch
import torch.nn as nn
import torch.nn.functional as F
import backbones
import utils


class TopModule(nn.Module):
    def __init__(self, args, encoder):
        super(TopModule, self).__init__()
        self.metric = args.get('metric')
        self.encoder = encoder
        self.logits_net = None

        if args.get('metric') == 'mlp':
            self.logits_net = nn.Sequential(nn.Linear(in_features=2 * args.get('emb_size'),
                                                      out_features=args.get('emb_size')),
                                            nn.ReLU(),
                                            nn.Linear(in_features=args.get('emb_size'),
                                                      out_features=1))

    def get_preds(self, embeddings):
        if self.metric == 'cosine':
            norm_embeddings = F.normalize(embeddings, p=2)
            cosine_sim = torch.matmul(norm_embeddings, norm_embeddings.T)
            preds = (cosine_sim + 1) / 2
            preds = torch.clamp(preds, min=0.0, max=1.0)
        elif self.metric == 'euclidean':
            euclidean_dist = utils.pairwise_distance(embeddings)
            euclidean_sim = -1 * euclidean_dist
            euclidean_sim = euclidean_sim / self.temperature

            preds = torch.clamp(preds, min=0.0, max=1.0)
        elif self.metric == 'mlp':
            logits =
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