from datasets.base_dataset import BaseDataset
from datasets.hotels_dataset import HotelsDataset

DATASETS = {
    'hotels': HotelsDataset,
    'hotels_small': HotelsDataset,
    'cub': BaseDataset
}


def load_dataset(args, mode, filename, transform):
    return DATASETS[args.get('dataset')](args, mode, filename, transform)
