# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com


import copy
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data.sampler import Sampler


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - dataset (BaseDataSet).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, dataset, batch_size, num_instances, k_dec_freq=0, **kwargs):
        self.data_dict = dataset.data_dict
        self.batch_size = batch_size
        self.K = num_instances
        self.k_dec_freq = k_dec_freq
        self.k_counter = 0
        self.num_labels_per_batch = self.batch_size // self.K
        self.max_iters = (dataset.__len__() // batch_size)
        self.labels = dataset.labels
        self.pairwise_labels = dataset.pairwise_labels

    def __len__(self):
        return self.max_iters

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"|Sampler| iters {self.max_iters}| K {self.K}| K_dec_freq {self.k_dec_freq}| M {self.batch_size}|"

    def prepare_batch(self):
        raise NotImplemented()

    def update_K(self):
        if self.k_dec_freq > 0 and self.K > 2:
            self.k_counter += 1
            if self.k_counter % self.k_dec_freq == 0:
                    self.K -= 1  # get it more difficult
                    self.num_labels_per_batch = self.batch_size // self.K
                    self.k_counter = 0

    def __iter__(self):
        batch_idxs_dict, avai_labels = self.prepare_batch()
        self.update_K()
        for _ in range(self.max_iters):
            batch = []
            if len(avai_labels) < self.num_labels_per_batch:
                batch_idxs_dict, avai_labels = self.prepare_batch()
                self.update_K()

            selected_labels = random.sample(avai_labels, self.num_labels_per_batch)
            for label in selected_labels:
                batch_idxs = batch_idxs_dict[label].pop(0)
                batch.extend(batch_idxs)
                if len(batch_idxs_dict[label]) == 0:
                    avai_labels.remove(label)
            yield batch