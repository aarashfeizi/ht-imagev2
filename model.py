import torch
import torch.nn as nn
import torch.nn.functional as F
import backbones
import utils

from functools import partial

from timm.models.vision_transformer import VisionTransformer


class SimTrans(nn.Module):
    def __init__(self, in_size=7, in_channels=2048, emb_dim=768, depth=12, num_heads=6): # depth 3?
        super(SimTrans, self).__init__()
        self.transformer = VisionTransformer(img_size=in_size, patch_size=1, in_chans=in_channels, num_classes=1000, embed_dim=emb_dim, depth=depth,
                 num_heads=num_heads, mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6))

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