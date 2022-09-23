from datasets.base_dataset import BaseDataset
from datasets.hotels_dataset import HotelsDataset, HotelsDataset_SSL

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
    'mini-imagenet-val': BaseDataset,
    'mini-imagenet-test': BaseDataset
}

PREDEFINED_DATASETS = {
    'imagenet': tv_datasets.ImageNet,
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
        return PREDEFINED_DATASETS[dataset_name](args.get('dataset_path'), transform=transform, split=mode)
    else:
        return DATASETS[dataset_name](args, mode, filename, transform, get_paths=for_heatmap,
                                             pairwise_labels=pairwise_labels, classification=classification, ssl=ssl, **kwargs)
