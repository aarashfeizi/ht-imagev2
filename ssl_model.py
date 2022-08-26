from functools import partial

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer

import backbones


class SSL_MODEL(nn.Module):
    def __init__(self, backbone, emb_size, num_classes=0, freeze_backbone=False, projector_sclaing=2):
        super(SSL_MODEL, self).__init__()
        self.encoder = backbone
        self.encoder.fc = nn.Identity()
        self.emb_size = emb_size

        if freeze_backbone and num_classes == 0:
            raise Exception('Backbone should not be freezed for a non-classification task!!')

        if freeze_backbone:            
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        if num_classes > 0:
            self.classifier = nn.Sequential(nn.Linear(emb_size, num_classes), 
                                            nn.Softmax(dim=-1))
            self.projector = None
        else:
            self.classifier = None
            assert (emb_size % projector_sclaing == 0)
            self.projector = nn.Sequential(nn.Linear(emb_size, emb_size // projector_sclaing))
        
    def forward_backbone(self, x):
        return self.encoder(x)
        
    def forward(self, x):
        x = self.forward_backbone(x)
        if self.classifier is not None:
            x = self.classifier(x)
        else: # self.projector is not None
            x = self.projector(x)
        return x


