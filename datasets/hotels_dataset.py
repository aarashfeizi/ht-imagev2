from hashlib import new
import os

import torch

import utils
from datasets.base_dataset import BaseDataset


class HotelsDataset(BaseDataset):
    def __init__(self, args, mode, filename='', transform=None, get_paths=False, pairwise_labels=False, classification=False):
        super(HotelsDataset, self).__init__(args, mode, filename, transform, get_paths, pairwise_labels, classification)

        self.lbl2idx = {l: i for i, l in enumerate(self.data_dict.keys())}

        self.rename_labels()
