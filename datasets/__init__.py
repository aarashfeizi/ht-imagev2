import os
import re

import pandas as pd
import torch
from torch.utils.data.dataset import Dataset

import utils

class BaseDataset(Dataset):
    def __init__(self, args, mode, filename='', transform=None):
        if filename == '':
            self.data_file_path = args.get(f'{mode}_file')
        else:
            self.data_file_path = filename

        self.root = args.get('dataset_path')
        self.path_list = []
        self.label_list = []
        self.transform = transform
        self.data_dict = self.__make_data_dict()

        self.labels = len(self.data_dict.keys())

    def __make_data_dict(self):
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
        with open(self.data_file_path, 'r') as f:
            for line in f:
                _path, _label = re.split(r",| ", line.strip())
                self.path_list.append(_path)
                self.label_list.append(_label)

    def __read_csv(self):
        file = pd.read_csv(self.data_file_path)
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

        return img, lbl