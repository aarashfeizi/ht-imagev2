from functools import partial

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer

import backbones

FEATURE_MAP_SIZES = {1: (256, 56, 56),
                     2: (512, 28, 28),
                     3: (1024, 14, 14),
                     4: (2048, 7, 7)}


class SimTrans(nn.Module):
    def __init__(self, in_size=7, in_channels=2048, emb_dim=768, depth=12, num_heads=6):  # depth 3?
        super(SimTrans, self).__init__()
        self.transformer = VisionTransformer(img_size=in_size, patch_size=1, in_chans=in_channels, num_classes=1000,
                                             embed_dim=emb_dim, depth=depth,
                                             num_heads=num_heads, mlp_ratio=4.,
                                             norm_layer=partial(nn.LayerNorm, eps=1e-6))


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


class GeneralTopLevelModule(nn.Module):
    def __init__(self, args, encoder):
        super(GeneralTopLevelModule, self).__init__()
        self.metric = args.get('metric')
        self.encoder = encoder
        self.logits_net = None
        self.temeperature = args.get('temperature')
        self.multi_layer_emb = args.get('multlayer_emb')
        self.equal_layer_contrib = args.get('eq_layer_contrib')
        self.proj_layer1 = None
        self.proj_layer2 = None
        self.proj_layer3 = None
        self.proj_layer4 = None

        self.projs = []

        self.final_projector = None

        self.attQ_layer1 = None
        self.attQ_layer2 = None
        self.attQ_layer3 = None
        self.attQ_layer4 = None

        self.attQs = []
        self.atts = []

    def forward(self, imgs):
        embeddings = self.encoder(imgs)
        return embeddings

    def forward_with_activations(self, imgs):
        embeddings, activations = self.encoder(imgs, is_feat=True)  # returns embeddings, [f1, f2, f3, f4]
        return embeddings, activations


class SingleEmbTopModule(GeneralTopLevelModule):
    def __init__(self, args, encoder):
        super(SingleEmbTopModule, self).__init__(args, encoder)

        if self.multi_layer_emb:
            if self.equal_layer_contrib:
                assert args.get('emb_size') % 4 == 0
                partial_emb_size = args.get('emb_size') // 4

                self.proj_layer1 = Projector(input_channels=FEATURE_MAP_SIZES[1][0],
                                             output_channels=partial_emb_size,
                                             pool='avg',
                                             kernel_size=FEATURE_MAP_SIZES[1][1])

                self.proj_layer2 = Projector(input_channels=FEATURE_MAP_SIZES[2][0],
                                             output_channels=partial_emb_size,
                                             pool='avg',
                                             kernel_size=FEATURE_MAP_SIZES[2][1])

                self.proj_layer3 = Projector(input_channels=FEATURE_MAP_SIZES[3][0],
                                             output_channels=partial_emb_size,
                                             pool='avg',
                                             kernel_size=FEATURE_MAP_SIZES[3][1])

                self.proj_layer4 = Projector(input_channels=FEATURE_MAP_SIZES[4][0],
                                             output_channels=partial_emb_size,
                                             pool='avg',
                                             kernel_size=FEATURE_MAP_SIZES[4][1])

                self.projs = [self.proj_layer1,
                              self.proj_layer2,
                              self.proj_layer3,
                              self.proj_layer4]
            else:
                big_emb_size = 0
                for k, v in FEATURE_MAP_SIZES.items():
                    big_emb_size += v[0]

                self.final_projector = nn.Linear(big_emb_size, args.get('emb_size'))

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

    def get_multilayer_embeddings_equal(self, imgs):
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

    def get_multilayer_embeddings_unequal(self, imgs):
        """

        :param imgs:
        :return: return a (B, dim) tensor, where dim is the emb_dim
        """
        embeddings, activations = self.encoder(imgs, is_feat=True)
        smaller_embs = []
        for a in activations:
            B, C, H, W = a.shape
            smaller_embs.append(a.reshape(B, C, -1).mean(dim=-1))

        embeddings = torch.cat(smaller_embs, dim=-1)

        return embeddings

    def forward(self, imgs):
        embeddings = None
        if self.multi_layer_emb:  # partial embeddings
            if self.equal_layer_contrib:
                embeddings = self.get_multilayer_embeddings_equal(imgs)
            else:
                embeddings = self.get_multilayer_embeddings_unequal(imgs)
                embeddings = self.final_projector(embeddings)
        else:
            embeddings = self.get_normal_embeddings(imgs)
        # preds, sims = self.get_preds(embeddings)

        return embeddings


class FanInOutAtt(nn.Module):
    def __init__(self, in_channels, out_channels=0):
        super(FanInOutAtt, self).__init__()
        if out_channels == 0:
            out_channels = in_channels // 2
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=in_channels,
                               kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        return x


class MultiEmbTopModule(GeneralTopLevelModule):
    def __init__(self, args, encoder):
        super(MultiEmbTopModule, self).__init__(args, encoder)

        self.maxpool_8 = nn.MaxPool2d((8, 8))
        self.maxpool_4 = nn.MaxPool2d((4, 4))
        self.maxpool_2 = nn.MaxPool2d((2, 2))
        self.maxpool_1 = nn.MaxPool2d((1, 1))

        self.identity = nn.Identity()

        self.maxpool_layers = {56: self.maxpool_8,
                               28: self.maxpool_4,
                               14: self.maxpool_2,
                               7: self.identity}

        big_emb_size = 0

        self.att_layer1 = BatchMultiHeadAttention(emb_size=FEATURE_MAP_SIZES[1][0], heads=args.get('ml_self_att_head_number'))
        self.att_layer2 = BatchMultiHeadAttention(emb_size=FEATURE_MAP_SIZES[2][0], heads=args.get('ml_self_att_head_number'))
        self.att_layer3 = BatchMultiHeadAttention(emb_size=FEATURE_MAP_SIZES[3][0], heads=args.get('ml_self_att_head_number'))
        self.att_layer4 = BatchMultiHeadAttention(emb_size=FEATURE_MAP_SIZES[4][0], heads=args.get('ml_self_att_head_number'))

        self.atts = [self.att_layer1,
                     self.att_layer2,
                     self.att_layer3,
                     self.att_layer4]

        self.layer_to_use = args.get('ml_self_att_layers_to_use')  # 4

        if self.layer_to_use < 4:
            self.maxpool_8 = None
            self.maxpool_layers[56] = None
            self.att_layer1 = None
            self.atts[0] = None

        if self.layer_to_use < 3:
            self.maxpool_4 = None
            self.att_layer2 = None
            self.atts[1] = None

            self.maxpool_2 = self.identity
            self.maxpool_layers[28] = None
            self.maxpool_layers[14] = self.identity

        if self.layer_to_use < 2:
            self.maxpool_2 = self.identity
            self.maxpool_layers[14] = self.identity
            self.att_layer3 = None
            self.atts[2] = None

        # self.attQ_layer1 = FanInOutAtt(in_channels=FEATURE_MAP_SIZES[1][0])
        # self.attQ_layer2 = FanInOutAtt(in_channels=FEATURE_MAP_SIZES[2][0])
        # self.attQ_layer3 = FanInOutAtt(in_channels=FEATURE_MAP_SIZES[3][0])
        # self.attQ_layer4 = FanInOutAtt(in_channels=FEATURE_MAP_SIZES[4][0])

        # self.attQs = [self.attQ_layer1,
        #               self.attQ_layer2,
        #               self.attQ_layer3,
        #               self.attQ_layer4]
        #
        thresh = 4 - self.layer_to_use
        for k, v in FEATURE_MAP_SIZES.items():
            if k > thresh:
                big_emb_size += v[0]

        self.final_projector = nn.Linear(big_emb_size, args.get('emb_size'))

        self.l2normalize = not args.get('cov')

        if self.multi_layer_emb:
            assert args.get('emb_size') % 4 == 0
            partial_emb_size = args.get('emb_size') // 4

            self.proj_layer1 = Projector(input_channels=FEATURE_MAP_SIZES[1][0],
                                         output_channels=partial_emb_size,
                                         pool='avg',
                                         kernel_size=FEATURE_MAP_SIZES[1][1])

            self.proj_layer2 = Projector(input_channels=FEATURE_MAP_SIZES[2][0],
                                         output_channels=partial_emb_size,
                                         pool='avg',
                                         kernel_size=FEATURE_MAP_SIZES[2][1])

            self.proj_layer3 = Projector(input_channels=FEATURE_MAP_SIZES[3][0],
                                         output_channels=partial_emb_size,
                                         pool='avg',
                                         kernel_size=FEATURE_MAP_SIZES[3][1])

            self.proj_layer4 = Projector(input_channels=FEATURE_MAP_SIZES[4][0],
                                         output_channels=partial_emb_size,
                                         pool='avg',
                                         kernel_size=FEATURE_MAP_SIZES[4][1])

            self.projs = [self.proj_layer1,
                          self.proj_layer2,
                          self.proj_layer3,
                          self.proj_layer4]

    def forward(self, imgs, is_feat=False, get_pairwise_acts=False):
        embeddings, activations = self.encoder(imgs, is_feat=True)

        # heatmaps = []

        layer_embeddings = []
        batch_size = 0
        new_activations = []
        all_new_acts = []
        for idx, act in enumerate(activations):
            B, C, H0, W0 = act.shape
            if batch_size == 0:
                batch_size = B
            if self.maxpool_layers[H0] is None:
                continue

            act = self.maxpool_layers[H0](act)
            _, _, H, W = act.shape
            act = act.reshape(B, C, H * W)
            act = act.transpose(-1, -2)
            new_act = self.atts[idx](act) # att_act's shape is (B, B, H*W, C)


            # act1Q = self.attQs[idx](act)
            # act1Q = act1Q.reshape(B, 1, C, H * W)
            # act1Q = act1Q.transpose(3, 2)
            #
            # act2K = act.reshape(1, B, C, H * W)
            # heatmap = (act1Q @ act2K) / np.sqrt(C)
            # # heatmap is (B, B, H*W, H*W) the attention coefficients of every image according to another image
            #
            # heatmap = heatmap.softmax(dim=-1)
            #
            # act = act.reshape(1, B, C, H * W)
            # act = act.transpose(3, 2)
            # att_act =
            # activations is being updated to a list of tensors with size (B, B, C, H*W) -> activations of every image according to another image's activations

            # new_act = new_act + act  # add original with attention activation

            if get_pairwise_acts:
                all_new_acts.append(new_act.transpose(-1, -2).reshape(B, B, C, H, W))

            new_activations.append(
                torch.diagonal(new_act.transpose(-1, -2).reshape(B, B, C * H * W)).transpose(0, 1).reshape(B, C, H, W))

            layer_embeddings.append(new_act.transpose(-1, -2).mean(dim=-1))

            # heatmaps.append(heatmap)

        # create all_layer embeddings
        all_embeddings = torch.cat(layer_embeddings, dim=2)

        all_embeddings = self.final_projector(all_embeddings.reshape(batch_size * batch_size, -1)).reshape(batch_size,
                                                                                                           batch_size,
                                                                                                           -1)

        # normalize embeddings
        if self.l2normalize:
            all_embeddings = all_embeddings / all_embeddings.norm(dim=-1, keepdim=True).reshape(batch_size, batch_size,
                                                                                                1)

        # Find cosine similarities between embeddings as predictions
        # cosine_similarities is a (B, B) matrix, ranging from -1 to 1
        # cosine_similarities = (all_embeddings * all_embeddings.transpose(0, 1)).sum(dim=-1)
        #
        # predictions = (cosine_similarities + 1) / 2

        # todo currently, outputed final embeddings from the model are NOT being used. Maybe use concatenating embeddings and passing it to an mlp for difference?
        if is_feat:
            all_activations = {'org': activations[4 - self.layer_to_use:], 'att': new_activations}
            return all_embeddings, all_activations
        elif get_pairwise_acts:
            org_activations = [a.repeat(batch_size, 1, 1, 1) for a in activations[4 - self.layer_to_use:]]
            org_to_return = []
            for a in org_activations:
                B2, C, H, W = a.shape
                assert B2 == batch_size * batch_size
                org_to_return.append(a.reshape(batch_size, batch_size, C, H, W))

            all_activations = {'org': org_to_return, 'att': all_new_acts}
            return all_embeddings, all_activations
        else:
            return all_embeddings

    def forward_with_activations(self, imgs):
        embeddings, activations = self.forward(imgs, is_feat=True, get_pairwise_acts=False)
        return embeddings, activations

    def forward_with_pairwise_activations(self, imgs):
        embeddings, activations = self.forward(imgs, is_feat=False, get_pairwise_acts=True)
        return embeddings, activations

def get_top_module(args):
    encoder = backbones.get_bb_network(args)
    if args.get('ml_self_att'):
        return MultiEmbTopModule(args, encoder)
    else:
        return SingleEmbTopModule(args, encoder)


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, heads=1, k=True, q=True, v=True, o=True):
        """

        :param emb_size: total embedding size
        :param heads: number of heads
        :param k: include w_k
        :param q: include w_q
        :param v: include w_v
        :param o: include w_o
        """
        super().__init__()
        self.emb_size = emb_size
        self.heads = heads
        self.head_emb_size = self.emb_size / self.heads
        assert self.emb_size % self.heads == 0
        self.softmax = nn.Softmax(dim=-1)
        self.w_k = nn.Identity()
        self.w_q = nn.Identity()
        self.w_v = nn.Identity()
        self.w_o = nn.Identity()

        if k:
            self.w_k = nn.Linear(self.emb_size, self.emb_size)
        if q:
            self.w_q = nn.Linear(self.emb_size, self.emb_size)
        if v:
            self.w_v = nn.Linear(self.emb_size, self.emb_size)

        if o and heads > 1:
            self.w_o = nn.Linear(self.emb_size, self.emb_size)

    def multi_head_reshape(self, x):
        B, N, D = x.shape
        assert D == self.emb_size
        x = x.reshape(B, N, self.heads, -1)
        x = x.transpose(1, 2)
        return x

    def reverse_multi_head_reshape(self, x):
        B, H, N, d = x.shape
        assert d == self.head_emb_size
        assert H == self.heads
        x = x.transpose(1, 2)
        x = x.reshape(B, N, -1)

        return x

    def forward(self, x):
        """

        :param x: tensor with shape (batch_size, sequences, emb_size)
        :return: attended tensor with shape (batch_size, sequences, emb_size)
        """
        B, N, D = x.shape
        assert D == self.emb_size
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        if self.heads != 1:
            Q = self.multi_head_reshape(Q)
            K = self.multi_head_reshape(K)
            V = self.multi_head_reshape(V)

        attention = Q @ K.transpose(-1, -2)
        attention /= np.sqrt(self.head_emb_size)

        att_V = attention @ V

        if self.heads != 1:
            att_V = self.reverse_multi_head_reshape(att_V)

        return self.w_o(att_V)

class BatchMultiHeadAttention(nn.Module):
    def __init__(self, emb_size, heads=1, k=True, q=True, v=True, o=True):
        """

        :param emb_size: total embedding size
        :param heads: number of heads
        :param k: include w_k
        :param q: include w_q
        :param v: include w_v
        :param o: include w_o
        """
        super().__init__()
        self.emb_size = emb_size
        self.heads = heads
        self.head_emb_size = self.emb_size / self.heads
        assert self.emb_size % self.heads == 0
        self.softmax = nn.Softmax(dim=-1)
        self.w_k = nn.Identity()
        self.w_q = nn.Identity()
        self.w_v = nn.Identity()
        self.w_o = nn.Identity()

        if k:
            self.w_k = nn.Linear(self.emb_size, self.emb_size)
        if q:
            self.w_q = nn.Linear(self.emb_size, self.emb_size)
        if v:
            self.w_v = nn.Linear(self.emb_size, self.emb_size)

        if o and heads > 1:
            self.w_o = nn.Linear(self.emb_size, self.emb_size)

    def multi_head_reshape(self, x):
        B, _, N, D = x.shape
        assert D == self.emb_size
        x = x.reshape(B, 1, N, self.heads, -1)
        x = x.transpose(-2, -3) # N and heads
        return x

    def reverse_multi_head_reshape(self, x):
        B1, B2, H, N, d = x.shape
        assert d == self.head_emb_size
        assert H == self.heads
        assert B1 == B2
        x = x.transpose(-2, -3) # N and heads
        x = x.reshape(B1, B2, N, -1)

        return x

    def forward(self, x):
        """

        :param x: tensor with shape (batch_size, sequences, emb_size)
        :return: attended tensor with shape (batch_size, sequences, emb_size)
        """
        B, N, D = x.shape

        assert D == self.emb_size
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        Q = Q.reshape(B, 1, N, D)
        K = K.reshape(B, 1, N, D)
        V = V.reshape(B, 1, N, D)

        if self.heads != 1:
            Q = self.multi_head_reshape(Q)
            K = self.multi_head_reshape(K)
            V = self.multi_head_reshape(V)

        attention = Q @ K.transpose(-1, -2).transpose(0, 1) # attention is (B, B, H, N, N)
        attention /= np.sqrt(self.head_emb_size)

        att_V = attention @ V

        if self.heads != 1:
            att_V = self.reverse_multi_head_reshape(att_V)

        return self.w_o(att_V)
