import os

import torch
import torch.nn as nn
from torch.nn import Parameter
from torchvision.models import ResNet as tResNet

import backbones.pooling as pooling

__all__ = ['ResNet', 'resnet18', 'resnet50']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(tResNet):

    def __init__(self, block, layers, num_classes, four_dim=False, pooling_method='spoc', output_dim=0,
                 layer_norm=False):
        super(ResNet, self).__init__(block, layers)
        self.gradients = None
        self.activations = None

        print(f'Pooling method: {pooling_method}')
        if pooling_method == 'spoc':
            self.pool = self.avgpool
        elif pooling_method == 'gem':
            self.pool = pooling.GeM()
        elif pooling_method == 'mac':
            self.pool = pooling.MAC()
        elif pooling_method == 'rmac':
            self.pool = pooling.RMAC()
        else:
            raise Exception(f'Pooling method {pooling_method} not implemented... :(')

        previous_output = self.layer4[-1].conv3.out_channels if type(self.layer4[-1]) == Bottleneck else self.layer4[
            -1].conv2.out_channels

        if layer_norm and previous_output == 2048:
            self.layer_norm = nn.LayerNorm([previous_output, 7, 7], elementwise_affine=False)
        elif layer_norm:
            raise Exception('Layer Norm not defined for outputs of size unequal to 2048 in ./backbones/resnet.py')
        else:
            self.layer_norm = None

        if output_dim != 0:
            self.last_conv = nn.Conv2d(in_channels=previous_output, out_channels=output_dim,
                                       kernel_size=(1, 1), stride=(1, 1))
        else:
            self.last_conv = None

        self.rest = nn.Sequential(self.conv1,
                                  self.bn1,
                                  self.relu,
                                  self.maxpool,
                                  self.layer1,
                                  self.layer2,
                                  self.layer3,
                                  self.layer4)

    def activations_hook(self, grad):
        self.gradients = grad.clone()

    def set_to_eval(self):
        for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.modules()):
            module.eval()
            module.train = lambda _: None
        return True

    def forward(self, x, is_feat=False, hook=False):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        f0 = x

        x = self.layer1(x)
        f1 = x
        x = self.layer2(x)
        f2 = x
        x = self.layer3(x)
        f3 = x
        x = self.layer4(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if self.last_conv is not None:
            x = self.last_conv(x)  # downsampling channels for dim reduction

        if hook:
            x.register_hook(self.activations_hook)
            self.activations = x.clone()

        f4 = x
        x = self.pool(x)

        feat = x
        x = torch.flatten(x, 1)

        if is_feat:
            return feat, [f1, f2, f3, f4]
        else:
            return x

    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self):
        return self.activations

    def load_my_state_dict(self, state_dict, four_dim):

        own_state = self.state_dict()
        for name, param in state_dict.items():

            if name not in own_state:
                continue
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data

            if four_dim and name == 'conv1.weight':
                print('Augmented zero initialized!!!')
                zeros = torch.zeros(size=[64, 1, 7, 7])
                param = torch.cat((param, zeros), axis=1)

            own_state[name].copy_(param)


def _resnet(arch, block, layers, pretrained, progress, num_classes, pooling_method='spoc',
            mask=False, fourth_dim=False, project_path='.', output_dim=0, pretrained_model='',
            layer_norm=False, **kwargs):
    model = ResNet(block, layers, num_classes, four_dim=(mask and fourth_dim),
                   pooling_method=pooling_method, output_dim=output_dim,
                   layer_norm=layer_norm, **kwargs)

    # if pretrained and pretrained_model != '':
    #     arch += '-' + pretrained_model
    #     print(f'loading {arch} from pretrained')
    #     if pretrained_model == 'byol':
    #         pretrained_path = os.path.join(project_path, f'models/pretrained_{arch}.pth')
    #         state_dict = torch.load(pretrained_path, map_location='cuda:0')['online_network_state_dict']
    #
    #     elif pretrained_model == 'simclr':
    #         pretrained_path = os.path.join(project_path, f'models/pretrained_{arch}.pth.tar')
    #         state_dict = torch.load(pretrained_path, map_location='cuda:0')['state_dict']
    #
    #     elif pretrained_model == 'swav' or pretrained_model == 'dino':
    #         pretrained_path = os.path.join(project_path, f'models/pretrained_{arch}.pt')
    #         state_dict = torch.load(pretrained_path)['model_state_dict']
    #     else:
    #         pretrained_path = os.path.join(project_path, pretrained_model)
    #         state_dict = torch.load(pretrained_path)['model_state_dict']
    #
    #     model.load_my_state_dict(state_dict, four_dim=(mask and fourth_dim))
    #     print('pretrained loaded!')
    #     return model

    if pretrained:
        pretrained_path = os.path.join(project_path, 'backbones/', f'pretrained_{arch}.pt')

        if os.path.exists(pretrained_path):
            print(f'loading {arch} from pretrained')
            state_dict = torch.load(pretrained_path)['model_state_dict']
        else:
            raise Exception(f'Model {arch} not found in {pretrained_path}')
            # state_dict = load_state_dict_from_url(model_urls[arch],
            #                                             progress=progress)
            # state_dict = torch.load('/Users/aarash/Downloads/resnet50-19c8e357.pth', map_location=None)

        model.load_my_state_dict(state_dict, four_dim=(mask and fourth_dim))
        print('pretrained loaded!')

    return model


def resnet18(args, pretrained=False, progress=True, num_classes=1, mask=False, fourth_dim=False, output_dim=0,
             **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, num_classes,
                   project_path=args.project_path,
                   mask=mask, fourth_dim=fourth_dim, pooling_method=args.pooling, output_dim=output_dim,
                   pretrained_model=args.pretrained_model, **kwargs)


def simple_resnet50(args, pretrained=False, progress=True, num_classes=1, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, num_classes,
                   project_path=args.project_path,
                   pretrained_model=args.pretrained_model, output_dim=args.dim_reduction, **kwargs)


def simple_resnet18(args, pretrained=False, progress=True, num_classes=1, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, num_classes,
                   project_path=args.project_path,
                   pretrained_model=args.pretrained_model, output_dim=args.dim_reduction, **kwargs)


def resnet50(args, pretrained=False, progress=True, num_classes=1, mask=False, fourth_dim=False, output_dim=0,
             **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, num_classes,
                   project_path=args.get('project_path'),
                   mask=mask, fourth_dim=fourth_dim, pooling_method='spoc',
                   output_dim=args.get('emb_size'), layer_norm=args.get('lnorm'), **kwargs)
    # return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, num_classes, project_path=args.get('project_path'),
    #                mask=mask, fourth_dim=fourth_dim, pooling_method='spoc', output_dim=output_dim, pretrained_model=args.pretrained_model, **kwargs)
