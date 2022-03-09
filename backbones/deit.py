from collections import OrderedDict
from functools import partial

import torch
from timm.models.vision_transformer import VisionTransformer as VisionTransformerT
from torch import nn


class VisionTransformer(VisionTransformerT):

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        # if self.dist_token is None:
        #     return self.pre_logits(x[:, 0])
        # else:
        #     return x[:, 0], x[:, 1]

        return x

    def forward(self, x):
        x = self.forward_features(x)
        # if self.head_dist is not None:
        #     x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
        #     if self.training and not torch.jit.is_scripting():
        #         # during inference, return the average of both classifier predictions
        #         return x, x_dist
        #     else:
        #         return (x + x_dist) / 2
        # else:
        #     x = self.head(x)
        return self.pre_logits(x[:, 0])

    def forward_get_rep_and_all(self, x):
        x = self.forward_features(x)

        return self.pre_logits(x[:, 0]), x[:, 1:]

    def set_pre_logits(self, representation_size=None):
        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(self.embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()


def deit_small_patch16(args, pretrained=True):
    # model = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224')
    model = VisionTransformer(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
                              qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

    if pretrained:
        checkpoint = torch.load('./backbones/deit_small_patch16_224-cd65a155.pth')
        model.load_state_dict(checkpoint['model'])

    model.set_pre_logits(args.get('emb_size'))

    return model
