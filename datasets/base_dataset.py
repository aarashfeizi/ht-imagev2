import imp
import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from sklearn.preprocessing import OneHotEncoder as OHE

import utils

class BaseDataset(Dataset):
    def __init__(self, args, mode, filename='', transform=None, get_paths=False, pairwise_labels=False, classification=False):
        if filename == '':
            self.data_file_path = args.get(f'{mode}_file')
        else:
            self.data_file_path = filename

        self.root = args.get('dataset_path')
        if args.get('aug_swap') > 1 and mode == 'train':
            self.swap_prob = args.get('aug_swap_prob')
            assert self.swap_prob > 0
        else:
            self.swap_prob = 0.0

        self.classification = classification
        self.get_paths = get_paths
        self.path_list = []
        self.label_list = []
        self.transform = transform
        self.data_dict = None
        self.make_data_dict()
        self.lbl2idx = None
        self.onehotencoder = None
        self.sample_pairwise = pairwise_labels
        self.labels = list(self.data_dict.keys())
        self.pairwise_labels_path = args.get(f'{mode}_pairwise_label_path')
        if pairwise_labels and self.pairwise_labels_path is not None:
            self.pairwise_labels = np.load(self.pairwise_labels_path)
            assert self.pairwise_labels.shape[0] == len(self.label_list)
        else:
            self.pairwise_labels = None
    
    def set_onehotencoder(self, ohe):
        self.onehotencoder = ohe
    
    def get_onehotencoder(self):
        return self.onehotencoder
    
    def set_lbl2idx(self, new_lbl2idx, ohe=None):
        self.make_data_dict()
        self.set_onehotencoder(ohe)
        self.lbl2idx = new_lbl2idx
        self.rename_labels()
        return
    
    def get_lbl2idx(self):
        return self.lbl2idx

    def rename_labels(self):
        if self.lbl2idx is None:
            raise Exception(f'lbl2idx is None!!')
        print('Relabelling labels to sequential numbers!')
        new_data_dict = {}
        for key, value in self.data_dict.items():
            new_data_dict[self.lbl2idx[key]] = value

        self.data_dict = new_data_dict
        self.label_list = [self.lbl2idx[l] for l in self.label_list]
        self.labels = list(self.data_dict.keys())

        if self.classification and self.onehotencoder is None:
            self.onehotencoder = OHE()
            self.label_list = self.onehotencoder.fit_transform(torch.tensor(self.label_list).reshape(-1, 1)).toarray()

        else:
            print('OHE was set!!')
            self.label_list = self.onehotencoder.transform(np.array(self.label_list).reshape(-1, 1)).toarray()

        return

    def make_data_dict(self):
        data_dict = {}

        if self.data_file_path.endswith('.txt'):
            self.__read_txt()
        elif self.data_file_path.endswith('.csv'):
            self.__read_csv()
        else:
            raise Exception(f'{self.data_file_path}: File type not supported')

        # make sure label_list are all float
        float_label_list = [float(l) for l in self.label_list]
        self.label_list = float_label_list

        for l in self.label_list:
            data_dict[l] = []

        for i, label in enumerate(self.label_list):
            data_dict[label].append(i)

        self.data_dict = data_dict
        return data_dict

    def __read_txt(self):
        path = os.path.join(self.root, self.data_file_path)
        with open(path, 'r') as f:
            for line in f:
                _path, _label = re.split(r",| ", line.strip())
                self.path_list.append(_path)
                self.label_list.append(_label)

    def __read_csv(self):
        path = os.path.join(self.root, self.data_file_path)
        file = pd.read_csv(path)
        self.path_list = list(file['image'])
        self.label_list = list(file['label'])

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        swap_label = 0.0
        img_path = os.path.join(self.root, self.path_list[idx])
        if type(self.label_list[idx]) is not torch.Tensor:
            lbl = torch.tensor(self.label_list[idx], dtype=torch.float32)
        else:
            lbl = self.label_list[idx]

        img = utils.open_img(img_path)

        if self.swap_prob > 0:
            if self.transform is not None:
                img, swap_label = self.do_swap_transform(img)
            swap_label = torch.tensor(swap_label, dtype=torch.float32)
        else:
            if self.transform is not None:
                img = self.transform(img)

        if self.swap_prob > 0:
            if self.get_paths:
                return img, lbl, swap_label, img_path
            else:
                return img, lbl, swap_label
        else:
            if self.get_paths:
                return img, lbl, img_path
            else:
                return img, lbl


    def do_swap_transform(self, img):
        swapped = 0.0
        img = self.transform[0](img)  # all transforms before swapping
        if torch.rand(1) < self.swap_prob:
            img = self.transform[1](img)  # swapping
            swapped = 1.0
        img = self.transform[2](img)  # all transforms after swapping

        return img, swapped