from tkinter import Image
from datasets.base_dataset import BaseDataset, BaseDataset_SSL
from datasets.hotels_dataset import HotelsDataset, HotelsDataset_SSL
from datasets.imagenet import ImageNet

import torchvision.datasets as tv_datasets

DATASETS = {
    'hotels': HotelsDataset,
    'hotels_small': HotelsDataset,
    'hotelid-val': HotelsDataset,
    'hotelid-test': HotelsDataset,
    'hotelid-val-ssl': HotelsDataset_SSL,
    'hotelid-test-ssl': HotelsDataset_SSL,
    'cub-val': BaseDataset,
    'cub-test': BaseDataset,
    'cub-val-ssl': BaseDataset_SSL,
    'cub-test': BaseDataset_SSL,
    'mini-imagenet-val': BaseDataset,
    'mini-imagenet-test': BaseDataset
}

PREDEFINED_DATASETS = {
    'imagenet': ImageNet,
    'imagenet100': ImageNet,
    'cifar100': tv_datasets.CIFAR100,
    'places': tv_datasets.Places365,
}


def load_dataset(args, mode, filename, transform, for_heatmap=False, pairwise_labels=False, classification=False, ssl=False, **kwargs):
    if ssl:
        if 'val' in mode or \
            'test' in mode:
            raise Exception('evaluation datasets shouldn\'t have ssl setting')
        dataset_name = args.get('dataset') + '-ssl'
    else:
        dataset_name = args.get('dataset')

    if dataset_name in PREDEFINED_DATASETS:
        if 'imagenet' in dataset_name:
            return PREDEFINED_DATASETS[dataset_name](args.get('dataset_path'), transform=transform, split=mode, **kwargs)
        else:
            is_train = 'train' == mode
            return PREDEFINED_DATASETS[dataset_name](args.get('dataset_path'), train=is_train, transform=transform)
    else:
        return DATASETS[dataset_name](args, mode, filename, transform, get_paths=for_heatmap,
                                             pairwise_labels=pairwise_labels, classification=classification, ssl=ssl, **kwargs)
