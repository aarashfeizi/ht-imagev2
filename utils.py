import json
import logging
import os
import random
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

import datasets
import samplers


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

        for key, value in config.items():  # overrites args if specified in config file
            self.global_config_file[key] = value

    def __str__(self):
        return str(self.global_config_file)

    def get(self, key):
        res = self.global_config_file.get(key, '#')

        if res == '#':
            print(f'No key {key} found in config or args!!!')
            res = None

        return res

    def set(self, key, value):
        if key in self.global_config_file:
            print(f'Overwriting {key} from {self.global_config_file[key]} to {value}')
        self.global_config_file[key] = value

        return


class TransformLoader:

    def __init__(self, args, image_size=224, rotate=0,
                 normalize_param=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                 jitter_param=dict(Brightness=0.4, Contrast=0.4, Color=0.4),
                 color_jitter_param=dict(brightness=0.5, hue=0.5, contrast=0.5, saturation=0.5),
                 scale=[0.5, 1.0],
                 resize_size=256):
        # hotels v5 train small mean: tensor([0.5791, 0.5231, 0.4664])
        # hotels v5 train small std: tensor([0.2512, 0.2581, 0.2698])

        # hotels v5 train mean: tensor([0.5805, 0.5247, 0.4683])
        # hotels v5 train std: tensor([0.2508, 0.2580, 0.2701])

        self.image_size = image_size
        self.first_resize = resize_size
        if args.get('normalize_param') is None:
            self.normalize_param = normalize_param
        else:
            self.normalize_param = args.get('normalize_param')

        if args.get('color_jitter_param') is None:
            self.color_jitter_param = color_jitter_param
        else:
            self.color_jitter_param = args.get('color_jitter_param')

        self.jitter_param = jitter_param
        self.rotate = rotate
        self.scale = scale
        self.random_erase_prob = 0.0

    def parse_transform(self, transform_type):

        method = getattr(transforms, transform_type)
        if transform_type == 'RandomResizedCrop':
            return method(self.image_size, scale=self.scale, ratio=[1.0, 1.0])
        elif transform_type == 'CenterCrop':
            return method(self.image_size)
        elif transform_type == 'Resize':
            return method([self.first_resize, self.first_resize])  # 256 by 256
        elif transform_type == 'Normalize':
            return method(**self.normalize_param)
        elif transform_type == 'RandomRotation':
            return method(self.rotate)
        elif transform_type == 'ColorJitter':
            return method(**self.color_jitter_param)
        elif transform_type == 'RandomErasing':
            return method(p=self.random_erase_prob, scale=(0.1, 0.75), ratio=(0.3, 3.3))  # TODO RANDOM ERASE!!!
        elif transform_type == 'RandomHorizontalFlip':
            return method(p=0.5)
        else:
            return method()

    def get_composed_transform(self, mode='train',
                               color_jitter=False,
                               random_erase=0.0):
        transform_list = []

        if mode == 'train':
            transform_list = ['Resize', 'RandomResizedCrop', 'RandomHorizontalFlip']
        else:
            transform_list = ['Resize', 'CenterCrop']

        if color_jitter and mode == 'train':
            transform_list.extend(['ColorJitter'])

        transform_list.extend(['ToTensor'])

        if mode == 'train' and random_erase > 0.0:
            self.random_erase_prob = random_erase
            transform_list.extend(['RandomErasing'])

        transform_list.extend(['Normalize'])

        transform_funcs = [self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)

        return transform, transform_list


def get_logger():  # params before: logname, env
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


def get_data(args, mode, file_name='', transform=None):
    dataset = datasets.load_dataset(args, mode, file_name, transform=transform)
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


def get_model_name(args):
    name = ''

    if args.get('cuda'):
        gpu_ids = args.get("gpu_ids").replace(',','')
        name += f'gpu{gpu_ids}_'

    name += 'model_%s_lss%s_bs%d_k%d_lr%f_bblr%f' % (args.get('dataset'),
                                                 args.get('loss'),
                                                 args.get('batch_size'),
                                                 args.get('num_inst_per_class'),
                                                 args.get('learning_rate'),
                                                 args.get('bb_learning_rate'))
    if args.get('loss') == 'pnpp':
        name += '_prxlr%f' % (args.get('proxypncapp_lr'),
                              )
    return name
