import utils
import os
import torch
from torchvision.models import resnet50


def get_backbone(backbone_name, pretrained=False):
    if pretrained:
        model = utils.MODEL_BACKBONES[backbone_name](pretrained=True)
    else:
        model = utils.MODEL_BACKBONES[backbone_name]()
    return model

    
MODEL_URLS = {'byol': 'https://storage.googleapis.com/deepmind-byol/checkpoints/pretrain_res50x1.pkl',
                'simsiam': 'https://dl.fbaipublicfiles.com/simsiam/models/100ep/pretrain/checkpoint_0099.pth.tar',
                'dino': 'https://dl.fbaipublicfiles.com/dinogg/dino_resnet50_pretrain/dino_resnet50_pretrain_full_checkpoint.pth',
                'vicreg': None,
                'simclr': 'https://storage.cloud.google.com/simclr-gcs/checkpoints/ResNet50_1x.zip',
                'swav': None}

def save_pretreind_model(net, path):
    torch.save({'model_state_dict': net.state_dict()}, path)

def download_swav(net, checkpoint, save_path):
     model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
     save_pretreind_model(model, save_path)

def download_vicreg(net, checkpoint, save_path):
    model = torch.hub.load('facebookresearch/vicreg:main', 'resnet50')
    save_pretreind_model(model, save_path)

def download_simsiam(net, checkpoint, save_path):
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder up to before the embedding layer
        if k.startswith('module.encoder') and not k.startswith('module.encoder.fc'):
            # remove prefix
            state_dict[k[len("module.encoder."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    msg = net.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
    save_pretreind_model(net, save_path)

    
def download_dino(net, checkpoint, save_path):
    state_dict = checkpoint['student']
    for k in list(state_dict.keys()):
        # retain only encoder up to before the embedding layer
        if k.startswith('module.backbone'):
            # remove prefix
            state_dict[k[len("module.backbone."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    msg = net.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
    save_pretreind_model(net, save_path)

    
# def download_simclr():
save_ssl_download = {
    'swav': download_swav,
    'dino': download_dino,
    'vicreg': download_vicreg,
    'simsiam': download_simsiam
}


def load_ssl_weight_to_model(model, method_name, arch_name):
    import urllib
    import pathlib
    utils.make_dirs('ssl_backbones')

    checkpoint_path = os.path.join('ssl_backbones', f'{arch_name}_{method_name}.pth')

    if method_name == 'default':
        return model
    else:
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(f'{arch_name}_{method_name}.pth', map_location='cpu')
            msg = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else: 
            suffixes = pathlib.Path(MODEL_URLS[method_name])
            suffix = ''.join(suffixes)
            downloaded_chkp = None
            if MODEL_URLS[method_name] is not None:
                downloaded_chkp = urllib.urlretrieve(MODEL_URLS[method_name], "{arch_name}_{method_name}{suffix}")

            msg = save_ssl_download[method_name](model, downloaded_chkp, checkpoint_path)
        return msg