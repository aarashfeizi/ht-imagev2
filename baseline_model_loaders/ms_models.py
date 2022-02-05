import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class LinearNorm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearNorm, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        # self.fc.apply(weights_init_kaiming) # it's going to be loaded from a pretrained model

    def forward(self, x):
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x

class ResNet50(nn.Module):

    def __init__(self):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(pretrained=True)

        for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
            module.eval()
            module.train = lambda _: None

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.model.fc(x)  --remove
        return x

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'last_linear' in i:
                continue
            self.model.state_dict()[i].copy_(param_dict[i])



def build_model():
    backbone = ResNet50()
    head = LinearNorm(in_channels=2048, out_channels=512)

    model = nn.Sequential(OrderedDict([
        ('backbone', backbone),
        ('head', head)
    ]))

    return model

