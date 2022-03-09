from backbones.resnet import *
from backbones.deit import *
import timm
import torch

NETWORKS = {
    'resnet50': resnet50,
    'deit_small': deit_small_patch16
}

def get_bb_network(args):
    net = NETWORKS[args.get('backbone')](args, pretrained=True)
    return net

