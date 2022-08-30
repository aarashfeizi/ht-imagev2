from hashlib import new
import os

import torch

import utils
from datasets.base_dataset import BaseDataset


class HotelsDataset(BaseDataset):
    def __init__(self, args, mode, filename='', transform=None, get_paths=False, pairwise_labels=False, classification=False, ssl=False, **kwargs):
        super(HotelsDataset, self).__init__(args, mode, filename, transform, get_paths, pairwise_labels, classification, ssl, **kwargs)

        self.lbl2idx = {l: i for i, l in enumerate(self.data_dict.keys())}

        self.rename_labels()

class HotelsDataset_SSL(HotelsDataset):
    def __init__(self, args, mode, filename='', transform=None, get_paths=False, pairwise_labels=False, classification=False, ssl=False, **kwargs):
        super(HotelsDataset_SSL, self).__init__(args, mode, filename, transform, get_paths, pairwise_labels, classification, ssl, **kwargs)

        self.random_crop_resize_transform = kwargs['random_crop_resize_transform']
        self.mask_in_transform = kwargs['mask_in_transform']
        self.rest_transform = kwargs['rest_transform']
    
    def __create_local_global_crop__(self, img):
        img_corpped_resized = self.random_crop_resize_transform(img)
        img1 = self.rest_transform(img_corpped_resized)
        img2 = self.rest_transform(self.mask_in_transform(img_corpped_resized))

        return img1, img2

    def __getitem__(self, idx):
        swap_label = 0.0
        img_path = os.path.join(self.root, self.path_list[idx])
        lbl = self.label_list[idx]
        if type(lbl) is not torch.Tensor:
            lbl = torch.tensor(lbl, dtype=torch.int64)

        img = utils.open_img(img_path)

        img1_transformed, img2_transformed = self.__create_local_global_crop__(img)
        imgs = [img1_transformed, img2_transformed]
        lbls = [lbl, lbl]
        img_paths = [img_path, img_path]
        

        if self.swap_prob > 0:
            raise Exception('Not impelemnted')
        else:
            if self.get_paths:
                return imgs, lbls, img_paths
            else:
                return imgs, lbls

