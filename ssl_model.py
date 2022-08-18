from functools import partial

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer

import backbones


class SSL_MODEL(nn.Module):
    def __init__(self, backbone, emb_size, num_classes, freeze_backbone=False):
        self.encoder = backbone
        self.emb_size = emb_size

        if freeze_backbone:            
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        self.classifier = nn.Sequential([nn.Linear(emb_size, num_classes), 
                                        nn.Softmax()])
    
    def forward_backbone(self, x):
        return self.encoder(x)
        
    def forward(self, x):
        x = self.forward_backbone(x)
        x_classes = self.classifier(x)
        return x_classes


