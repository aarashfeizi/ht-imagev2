from datasets.base_dataset import BaseDataset
from datasets.hotels_dataset import HotelsDataset

DATASETS = {
    'hotels': HotelsDataset,
    'hotels_small': HotelsDataset,
    'hotelid-val': HotelsDataset,
    'hotelid-test': HotelsDataset,
    'cub': BaseDataset
}


def load_dataset(args, mode, filename, transform, for_heatmap=False, pairwise_label=False):
    return DATASETS[args.get('dataset')](args, mode, filename, transform, get_paths=for_heatmap,
                                         pairwise_labels=pairwise_label)
