import utils
import os
import torch
import torch.nn as nn
from torchvision.models import resnet50


def get_backbone(backbone_name, pretrained=False):
    if pretrained:
        model = utils.MODEL_BACKBONES[backbone_name](pretrained=True)
    else:
        model = utils.MODEL_BACKBONES[backbone_name]()
    return model

    
MODEL_URLS = {'byol': 'https://storage.googleapis.com/deepmind-byol/checkpoints/pretrain_res50x1.pkl',
                'simsiam': 'https://dl.fbaipublicfiles.com/simsiam/models/100ep/pretrain/checkpoint_0099.pth.tar',
                'dino': 'https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain_full_checkpoint.pth',
                'vicreg': None,
                'simclr': 'https://storage.cloud.google.com/simclr-gcs/checkpoints/ResNet50_1x.zip',
                'swav': None,
                'barlow': None,
                'densecl': 'https://cloudstor.aarnet.edu.au/plus/s/hdAg5RYm8NNM2QP/download',
                'densecl_CC': 'https://cloudstor.aarnet.edu.au/plus/s/3GapXiWuVAzdKwJ/download'}

def load_state_dict_wo_fc(net, state_dict):
    for k in list(state_dict.keys()):
        # retain only encoder up to before the embedding layer
        if k.startswith('module.encoder') and not k.startswith('module.encoder.fc'):
            # remove prefix
            state_dict[k[len("module.encoder."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    msg = net.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
    return

def save_pretreind_model(net, path):
    torch.save({'model_state_dict': net.state_dict()}, path)

def download_swav(net, checkpoint, save_path):
     model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
     save_pretreind_model(model, save_path)

def download_vicreg(net, checkpoint, save_path):
    model = torch.hub.load('facebookresearch/vicreg:main', 'resnet50')
    save_pretreind_model(model, save_path)

def download_barlow(net, checkpoint, save_path):
    model = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
    save_pretreind_model(model, save_path)    

def download_simsiam(net, checkpoint, save_path):
    state_dict = checkpoint['state_dict']
    load_ssl_weight_to_model(net, state_dict)
    save_pretreind_model(net, save_path)
    
def download_dino(net, checkpoint, save_path):
    state_dict = checkpoint['student']
    load_ssl_weight_to_model(net, state_dict)
    save_pretreind_model(net, save_path)
    
def download_densecl(net, checkpoint, save_path):
    state_dict = checkpoint['state_dict']
    load_ssl_weight_to_model(net, state_dict)
    save_pretreind_model(net, save_path)


# def download_simclr():
save_ssl_download = {
    'swav': download_swav,
    'dino': download_dino,
    'vicreg': download_vicreg,
    'simsiam': download_simsiam,
    'barlow': download_barlow,
    'densecl': download_densecl,
    'densecl_CC': download_densecl,
}


def load_ssl_weight_to_model(model, method_name, arch_name):
    import urllib.request as req
    import pathlib
    utils.make_dirs('ssl_backbones')

    checkpoint_path = os.path.join('ssl_backbones', f'{arch_name}_{method_name}.pth')

    if method_name == 'default':
        return model
    else:
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else: 
            downloaded_chkp = None
            if MODEL_URLS[method_name] is not None:
                suffixes = pathlib.Path(MODEL_URLS[method_name]).suffixes
                suffix = ''.join(suffixes)
                print(f'Downloading... {arch_name}_{method_name}{suffix}')
                downloaded_chkp_path = req.urlretrieve(MODEL_URLS[method_name], f"{arch_name}_{method_name}{suffix}")[0]
                downloaded_chkp = torch.load(downloaded_chkp_path, map_location='cpu')
                
            save_ssl_download[method_name](model, downloaded_chkp, checkpoint_path)
            
        return model
    
def set_net_to_eval(net):
    for module in filter(lambda m: type(m) == nn.BatchNorm2d, net.modules()):
        module.eval()
        module.train = lambda _: None
    return True