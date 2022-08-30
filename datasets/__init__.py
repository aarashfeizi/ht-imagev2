from msilib import datasizemask
from datasets.base_dataset import BaseDataset
from datasets.hotels_dataset import HotelsDataset, HotelsDataset_SSL

DATASETS = {
    'hotels': HotelsDataset,
    'hotels-ssl': HotelsDataset_SSL,
    'hotels_small': HotelsDataset,
    'hotelid-val': HotelsDataset,
    'hotelid-test': HotelsDataset,
    'cub': BaseDataset
}


def load_dataset(args, mode, filename, transform, for_heatmap=False, pairwise_labels=False, classification=False, ssl=False, **kwargs):
    if ssl:
        if 'val' in args.get('dataset') or \
            'test' in args.get('dataset'):
            raise Exception('evaluation datasets shouldn\'t have ssl setting')
        dataset_name = args.get('dataset') + '-ssl'
    else:
        dataset_name = args.get('dataset')

    return DATASETS[dataset_name](args, mode, filename, transform, get_paths=for_heatmap,
                                         pairwise_labels=pairwise_labels, classification=classification, ssl=ssl, **kwargs)
