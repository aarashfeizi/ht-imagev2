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
        self.temeperature = args.get('temperature')

        if args.get('metric') == 'mlp':
            self.logits_net = nn.Sequential(nn.Linear(in_features=2 * args.get('emb_size'),
                                                      out_features=args.get('emb_size')),
                                            nn.ReLU(),
                                            nn.Linear(in_features=args.get('emb_size'),
                                                      out_features=1))

    def get_preds(self, embeddings):
        if self.metric == 'cosine':
            norm_embeddings = F.normalize(embeddings, p=2)
            sims = torch.matmul(norm_embeddings, norm_embeddings.T)
            preds = (sims + 1) / 2 # maps (-1, 1) to (0, 1)

            preds = torch.clamp(preds, min=0.0, max=1.0)
        elif self.metric == 'euclidean':
            euclidean_dist = utils.pairwise_distance(embeddings)

            euclidean_dist = euclidean_dist / self.temperature

            preds = 2 * nn.functional.sigmoid(-euclidean_dist) # maps (0, +inf) to (1, 0)
            sims = -euclidean_dist
            # preds = torch.clamp(preds, min=0.0, max=1.0)
        elif self.metric == 'mlp':
            bs = embeddings.shape[0]
            indices = torch.tensor([[i, j] for i in range(bs) for j in range(bs)]).flatten()
            logits = self.logits_net(embeddings[indices].reshape(bs * bs, -1))

            sims = logits / self.temperature
            preds = nn.functional.sigmoid(sims)
        else:
            raise Exception(f'{self.metric} not supported in Top Module')

        return preds, sims


    def forward(self, imgs):
        embeddings = self.encoder(imgs)
        preds, sims = self.get_preds(embeddings)

        return preds, sims, embeddings

def get_top_module(args):
    encoder = backbones.get_bb_network(args)
    return TopModule(args, encoder)