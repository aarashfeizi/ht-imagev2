from backbones.resnet import *
import timm

NETWORKS = {
    'resnet50': resnet50
}

def get_bb_network(args):
    net = NETWORKS[args.get('backbone')](args, pretrained=True)
    return net


