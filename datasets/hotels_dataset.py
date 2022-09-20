from hashlib import new
import os

import torch

import utils
from datasets.base_dataset import BaseDataset

def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


class HotelsDataset(BaseDataset):
    def __init__(self, args, mode, filename='', transform=None, get_paths=False, pairwise_labels=False, classification=False, ssl=False, **kwargs):
        super(HotelsDataset, self).__init__(args, mode, filename, transform, get_paths, pairwise_labels, classification, ssl, **kwargs)

        self.lbl2idx = {l: i for i, l in enumerate(self.data_dict.keys())}

        self.rename_labels()

class HotelsDataset_SSL(HotelsDataset):
    def __init__(self, args, mode, filename='', transform=None, get_paths=False, pairwise_labels=False, classification=False, ssl=False, **kwargs):
        super(HotelsDataset_SSL, self).__init__(args, mode, filename, transform, get_paths, pairwise_labels, classification, ssl, **kwargs)

        self.ssl_aug = kwargs.get('ssl_aug', False)
        self.multi_crop = kwargs.get('multi_crop', False)

        if self.ssl_aug:
            assert not self.multi_crop
        elif self.multi_crop:
            assert not self.ssl_aug

        if self.ssl_aug:
            self.random_crop_resize_transform = kwargs['random_crop_resize_transform']
            self.mask_in_transform = kwargs['mask_in_transform']
            self.rest_transform = kwargs['rest_transform']
        else:
            self.random_crop_resize_transform = None
            self.mask_in_transform = None
            self.rest_transform = None
    
    def update_transforms(self, transform_dict):
        if transform_dict is not None:
            self.ssl_aug = True
        self.random_crop_resize_transform = transform_dict['random_crop_resize_transform']
        self.mask_in_transform = transform_dict['mask_in_transform']
        self.rest_transform = transform_dict['rest_transform']
    
    
    def __ssl_transform_img(self, img):
        if self.ssl_aug:
            img_corpped_resized = self.random_crop_resize_transform(img)
            img1 = self.rest_transform(img_corpped_resized)
            img2 = self.rest_transform(self.mask_in_transform(img_corpped_resized))
        else:
            img1 = self.transform(img)
            img2 = self.transform(img)

        return img1, img2

    def __getitem__(self, idx):
        swap_label = 0.0
        img_path = os.path.join(self.root, self.path_list[idx])
        lbl = self.label_list[idx]
        if type(lbl) is not torch.Tensor:
            lbl = torch.tensor(lbl, dtype=torch.int64)

        img = utils.open_img(img_path)

        img1_transformed, img2_transformed = self.__ssl_transform_img(img)
        imgs = torch.stack([img1_transformed, img2_transformed], dim=0)
        lbls = torch.stack([lbl, lbl], dim=0)
        img_paths = [img_path, img_path]
        

        if self.swap_prob > 0:
            raise Exception('Not implemented')
        else:
            if self.get_paths:
                return imgs, lbls, img_paths
            else:
                return imgs, lbls

