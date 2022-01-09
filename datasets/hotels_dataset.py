import os
import re

import pandas as pd
import torch
from datasets import BaseDataset

import utils

class HotelsDataset(BaseDataset):
    def __init__(self, args, mode, filename='', transform=None):
        super().__init__(args, mode, filename, transform)

        self.data_dict = self.__make_data_dict()

        self.lbl2idx = {l: i for i, l in enumerate(self.data_dict.keys())}

        self.rename_labels()

        self.labels = list(self.data_dict.keys())

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