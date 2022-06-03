import os
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

import utils

class BaseDataset(Dataset):
    def __init__(self, args, mode, filename='', transform=None, get_paths=False, pairwise_labels=False):
        if filename == '':
            self.data_file_path = args.get(f'{mode}_file')
        else:
            self.data_file_path = filename

        self.root = args.get('dataset_path')
        self.get_paths = get_paths
        self.path_list = []
        self.label_list = []
        self.transform = transform
        self.data_dict = self.make_data_dict()
        self.lbl2idx = None
        self.labels = list(self.data_dict.keys())
        self.pairwise_labels_path = args.get(f'pairwise_label_path')
        if pairwise_labels and self.pairwise_labels_path is not None:
            self.pairwise_labels = np.load(self.pairwise_labels_path)
            assert self.pairwise_labels.shape[0] == len(self.label_list)
        else:
            self.pairwise_labels = None

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
        img_path = os.path.join(self.root, self.path_list[idx])
        lbl = torch.tensor(self.label_list[idx], dtype=torch.float32)

        img = utils.open_img(img_path)

        if self.transform is not None:
            img = self.transform(img)

        if self.get_paths:
            return img, lbl, img_path
        else:
            return img, lbl