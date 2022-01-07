import argparse
import json
import logging
import os
import random
import sys

import numpy as np
import torch
from PIL import Image

import datasets, samplers
from torch.utils.data import DataLoader


class Global_Config_File:
    def __init__(self, config_file, args):
        config_file = config_file
        args = vars(args)

        self.global_config_file = {}

        self.__merge_config_files(args=args, config=config_file)

        self.__initialize_env()

    def __initialize_env(self):
        self.global_config_file['tensorboard_path'] = 'tensorboard_' + self.global_config_file.get('dataset')
        self.global_config_file['save_path'] = 'save_' + self.global_config_file.get('dataset')

        make_dirs(self.global_config_file['tensorboard_path'])
        make_dirs(self.global_config_file['save_path'])

    def __merge_config_files(self, args, config):
        for key, value in args.items():
            self.global_config_file[key] = value

        for key, value in config.items(): # overrites args if specified in config file
            self.global_config_file[key] = value

    def __str__(self):
        return str(self.global_config_file)

    def get(self, key):
        res = self.global_config_file.get(key, '#')

        if res == '#':
            raise Exception(f'No key {key} found in config or args')

        return res

    def set(self, key, value):
        if key in self.global_config_file:
            print(f'Overwriting {key} from {self.global_config_file[key]} to {value}')
        self.global_config_file[key] = value

        return

def get_logger(): # params before: logname, env
    # if env == 'hlr' or env == 'local':
    #     logging.basicConfig(filename=os.path.join('logs', logname + '.log'),
    #                         filemode='a', format='%(asctime)s - %(message)s', level=logging.INFO)
    # else:
    logging.basicConfig(stream=sys.stdout,
                        filemode='a', format='%(asctime)s - %(message)s', level=logging.INFO)
    return logging.getLogger()

def make_dirs(path):
    if os.path.exists(path):
        return
    else:
        os.makedirs(path)
        return

def load_config(config_name):
    config = json.load(open(config_name))

    def eval_json(config):
        for k in config:
            if type(config[k]) != dict:
                config[k] = eval(config[k])
            else:
                eval_json(config[k])

    eval_json(config)
    return config


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seed_all(seed, cuda=False):
    if not seed:
        seed = 10

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return True


def open_img(path):
    img = Image.open(path).convert('RGB')
    return img


def get_data(args, mode, file_name=''):
    dataset = datasets.BaseDataset(args, mode, file_name)
    sampler = samplers.RandomIdentitySampler(dataset=dataset,
                                             batch_size=args.get('batch_size'),
                                             num_instances=args.get('num_inst_per_class'))
    dataloader = DataLoader(dataset=dataset, shuffle=False, num_workers=args.get('workers'), batch_sampler=sampler,
                                  pin_memory=args.get('pin_memory'), worker_init_fn=seed_worker)

    return dataloader

def get_all_data(args, dataset_config, mode):
    all_sets = dataset_config[f'all_{mode}_files']
    loaders = []
    for ds_name in all_sets:
        loaders.append(get_data(args, dataset_config, mode, ds_name))

    return loaders

def pairwise_distance(a, squared=False, diag_to_max=False):
    """Computes the pairwise distance matrix with numerical stability."""
    pairwise_distances_squared = torch.add(
        a.pow(2).sum(dim=1, keepdim=True).expand(a.size(0), -1),
        torch.t(a).pow(2).sum(dim=0, keepdim=True).expand(a.size(0), -1)
    ) - 2 * (
                                     torch.mm(a, torch.t(a))
                                 )

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = torch.clamp(
        pairwise_distances_squared, min=0.0
    )

    # Get the mask where the zero distances are at.
    error_mask = torch.le(pairwise_distances_squared, 0.0)
    # print(error_mask.sum())
    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = torch.sqrt(
            pairwise_distances_squared + error_mask.float() * 1e-16
        )

    # Undo conditionally adding 1e-16.
    pairwise_distances = torch.mul(
        pairwise_distances,
        (error_mask == False).float()
    )

    # Explicitly set diagonals to zero or diag_to_max.
    mask_offdiagonals = 1 - torch.eye(
        *pairwise_distances.size(),
        device=pairwise_distances.device
    )
    pairwise_distances = torch.mul(pairwise_distances, mask_offdiagonals)

    if diag_to_max:
        max_value = pairwise_distances.max().item() + 10
        pairwise_distances.fill_diagonal_(max_value)

    return pairwise_distances

def binarize_and_smooth_labels(T, nb_classes, smoothing_const=0):
    import sklearn.preprocessing
    T = T.cpu().numpy()
    T = sklearn.preprocessing.label_binarize(
        T, classes=range(0, nb_classes)
    )
    T = T * (1 - smoothing_const)
    T[T == 0] = smoothing_const / (nb_classes - 1)
    T = torch.FloatTensor(T).cuda()

    return T
