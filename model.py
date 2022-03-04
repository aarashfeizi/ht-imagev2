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

    def forward(self, imgs):
        embeddings = self.encoder(imgs)
        # preds, sims = self.get_preds(embeddings)

        return embeddings

def get_top_module(args):
    encoder = backbones.get_bb_network(args)
    return TopModule(args, encoder)