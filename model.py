import torch
import torch.nn as nn
import torch.nn.functional as F
import backbones
import utils

from functools import partial

from timm.models.vision_transformer import VisionTransformer

FEATURE_MAP_SIZES = {1: (256, 56, 56),
                     2: (512, 28, 28),
                     3: (1024, 14, 14),
                     4: (2048, 7, 7)}

class SimTrans(nn.Module):
    def __init__(self, in_size=7, in_channels=2048, emb_dim=768, depth=12, num_heads=6): # depth 3?
        super(SimTrans, self).__init__()
        self.transformer = VisionTransformer(img_size=in_size, patch_size=1, in_chans=in_channels, num_classes=1000, embed_dim=emb_dim, depth=depth,
                 num_heads=num_heads, mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6))


class Projector(nn.Module):

    def __init__(self, input_channels, output_channels, pool=None, kernel_size=-1):
        super(Projector, self).__init__()
        self.op = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1)
        if pool == 'avg':
            assert kernel_size != -1
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)
        elif pool == 'max':
            assert kernel_size != -1
            self.pool = nn.MaxPool2d(kernel_size=kernel_size)
        else:
            self.pool = None

    def forward(self, x):
        x = self.op(x)
        if self.pool:
            x = self.pool(x)
        return x


class TopModule(nn.Module):
    def __init__(self, args, encoder):
        super(TopModule, self).__init__()
        self.metric = args.get('metric')
        self.encoder = encoder
        self.logits_net = None
        self.temeperature = args.get('temperature')
        self.projs = []
        if args.get('ml_emb'):
            assert args.get('emb_size') % 4 == 0
            partial_emb_size = args.get('emb_size') / 4
            for i in range(1, 5):
                self.projs.append(Projector(input_channels=FEATURE_MAP_SIZES[i][0],
                                            output_channels=partial_emb_size,
                                            pool='avg',
                                            kernel_size=FEATURE_MAP_SIZES[i][1]))


        if args.get('metric') == 'mlp':
            self.logits_net = nn.Sequential(nn.Linear(in_features=2 * args.get('emb_size'),
                                                      out_features=args.get('emb_size')),
                                            nn.ReLU(),
                                            nn.Linear(in_features=args.get('emb_size'),
                                                      out_features=1))

    def get_normal_embeddings(self, imgs):
        """

        :param imgs:
        :return: return a (B, dim) tensor, where dim is the emb_dim
        """
        return self.encoder(imgs)

    def get_multilayer_embeddings(self, imgs):
        """

        :param imgs:
        :return: return a (B, dim) tensor, where dim is the emb_dim
        """
        embeddings, activations = self.encoder(imgs, is_feat=True)
        smaller_embs = []
        for a, p in zip(activations, self.projs):
            smaller_embs.append(p(a))

        embeddings = torch.cat(smaller_embs, dim=1).squeeze(dim=-1).squeeze(dim=-1)
        return embeddings


    def forward(self, imgs):
        embeddings = None
        if len(self.projs) == 0:
            embeddings = self.get_normal_embeddings(imgs)
        else: #partial embeddings
            embeddings = self.get_multilayer_embeddings(imgs)
        # preds, sims = self.get_preds(embeddings)

        return embeddings



def get_top_module(args):
    encoder = backbones.get_bb_network(args)
    return TopModule(args, encoder)