import copy
import json
import logging
import os
import random
import sys

import cv2
import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader

import datasets
import metrics
# from torchvision import transforms
import transforms
from samplers.my_sampler import BalancedTripletSampler, KBatchSampler, DataBaseSampler, DrawHeatmapSampler, \
    HardTripletSampler, Draw2XHeatmapSampler

SHARING_STRATEGY = "file_system"
torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)


class Global_Config_File:
    def __init__(self, config_file, args, init_tb=True):
        config_file = config_file
        args = vars(args)

        self.global_config_file = {}

        self.__merge_config_files(args=args, config=config_file)

        if init_tb:
            self.__initialize_env()

    def __initialize_env(self):
        self.global_config_file['tensorboard_path'] = os.path.join(self.global_config_file.get('log_path'), 'tensorboard_' + self.global_config_file.get('dataset'))
        self.global_config_file['save_path'] = os.path.join(self.global_config_file.get('log_path'), 'save_' + self.global_config_file.get('dataset'))

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
        self.random_swap = args.get('aug_swap')
        self.random_swap_prob = args.get('aug_swap_prob')
        self.random_mask_prob = args.get('aug_mask_prob')

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
        elif transform_type == 'RandomSwap':
            return method(size=self.random_swap, p=1, mask_prob=self.random_mask_prob) # self.random_swap_prob is applied in __getitem__()
        elif transform_type == 'RandomHorizontalFlip':
            return method(p=0.5)
        else:
            return method()

    def get_composed_transform(self, mode='train',
                               color_jitter=False,
                               random_erase=0.0):
        transform_list = []
        before_swap_transform_list = []
        swap_transform_list = []

        if mode == 'train':
            transform_list = ['Resize', 'RandomResizedCrop', 'RandomHorizontalFlip']
        else:
            transform_list = ['Resize', 'CenterCrop']

        if color_jitter and mode == 'train':
            transform_list.extend(['ColorJitter'])

        if self.random_swap != 1 and mode == 'train': # random_swap is number of crops on each edge
            before_swap_transform_list = [t for t in transform_list]
            swap_transform_list = ['RandomSwap']
            transform_list = []

        transform_list.extend(['ToTensor'])

        if mode == 'train' and random_erase > 0.0:
            self.random_erase_prob = random_erase
            transform_list.extend(['RandomErasing'])

        transform_list.extend(['Normalize'])

        if len(swap_transform_list) > 0:
            transform_funcs = [self.parse_transform(x) for x in before_swap_transform_list]
            transform_before_swap = transforms.Compose(transform_funcs)

            transform_funcs = [self.parse_transform(x) for x in swap_transform_list]
            transform_swap = transforms.Compose(transform_funcs)

            transform_funcs = [self.parse_transform(x) for x in transform_list]
            transform_after_swap = transforms.Compose(transform_funcs)

            return [transform_before_swap, transform_swap, transform_after_swap], \
                   [before_swap_transform_list, swap_transform_list, transform_list]

        else:
            transform_funcs = [self.parse_transform(x) for x in transform_list]
            transform = transforms.Compose(transform_funcs)

            # if 'RandomSwap' in transform_list:
            #     transform_list_wo_swap = transform_list.remove('RandomSwap')
            #     transform_funcs_wo_swap = [self.parse_transform(x) for x in transform_list_wo_swap]
            #     transform_wo_swap = transforms.Compose(transform_funcs_wo_swap)
            #     return [transform, transform_wo_swap], \
            #            [transform_list, transform_list_wo_swap]

            return transform, transform_list


def get_logger():  # params before: logname, env
    # if env == 'hlr' or env == 'local':
    #     logging.basicConfig(filename=os.path.join('logs', logname + '.log'),
    #                         filemode='a', format='%(asctime)s - %(message)s', level=logging.INFO)
    # else:
    logging.basicConfig(stream=sys.stdout,
                        filemode='a', format='%(asctime)s - %(message)s', level=logging.INFO)
    return logging.getLogger()


def make_dirs(path, force=False):
    i = 1
    if os.path.exists(path):
        if force:
            new_path = path
            while os.path.exists(new_path):
                new_path = path + f'_v{i}'
                i += 1
            os.makedirs(new_path)
            return new_path
        else:
            return path
    else:
        os.makedirs(path)
        return path


def load_json(config_name):
    config = json.load(open(config_name))

    return config

def save_json(json_obj, fp):
    with open(fp, 'w') as f:
        json.dump(json_obj, f)

    return

def seed_worker(worker_id):
    torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)
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


def get_data(args, mode, file_name='', transform=None, sampler_mode='kbatch',
             pairwise_labels=False, **kwargs):  # 'kbatch', 'balanced_triplet', 'db'
    SAMPLERS = {'kbatch': KBatchSampler,
                'balanced_triplet': BalancedTripletSampler,
                'hard_triplet': HardTripletSampler,
                'db': DataBaseSampler,
                'heatmap': DrawHeatmapSampler,
                'heatmap2x': Draw2XHeatmapSampler}

    mode_splits = mode.split('_')
    eval_mode = mode_splits[0]
    if eval_mode != 'train':
        if len(mode_splits) > 1 and mode_splits[1] == 'pairwise':
            assert pairwise_labels
        else:
            assert not pairwise_labels

    dataset = datasets.load_dataset(args, eval_mode, file_name,
                                    transform=transform,
                                    for_heatmap=sampler_mode.startswith('heatmap'),
                                    pairwise_labels=pairwise_labels)

    sampler = SAMPLERS[sampler_mode](dataset=dataset,
                                     batch_size=args.get('batch_size'),
                                     num_instances=args.get('num_inst_per_class'),
                                     k_dec_freq=args.get('k_dec_freq'),
                                     use_pairwise_label=pairwise_labels,
                                     **kwargs)

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
    loss_specific_args = ['LOSS_margin',
                          'NCA_scale',
                          'LOSS_alpha',
                          'ARCFACE_scale',
                          'CIR_m',
                          'CIR_gamma',
                          'LOSS_lr',
                          'LOSS_temp',
                          'SOFTTRPL_cpc']

    name = ''

    if args.get('cuda'):
        gpu_ids = args.get("gpu_ids").replace(',', '')
        name += f'gpu{gpu_ids}_'

    name += f"wbn_{args.get('epochs')}ep_" \
            f"{args.get('dataset')}_" \
            f"{args.get('backbone')}_" \
            f"{args.get('metric')}_" \
            f"bs{args.get('batch_size')}_" \
            f"k{args.get('num_inst_per_class')}_" \
            f"lr{args.get('learning_rate'):.2}"


    if args.get('new_lr_coef') != 1.0:
        coef = args.get('new_lr_coef')
        name += f'-newLRcoef{coef}'

    if args.get('early_stopping_tol') > 0:
        tol = args.get('early_stopping_tol')
        name += f'-est{tol}'

    if args.get('aug_swap') != 1:
        swap_size = args.get('aug_swap')
        swap_prob = args.get('aug_swap_prob')
        mask_prob = args.get('aug_mask_prob')
        if mask_prob <= 0:
            masking = ''
        else:
            masking = f'mask{mask_prob}'

        swap_loss = ''
        if args.get('swap_coef') > 0.0:
            swap_coef = args.get('swap_coef')
            swap_loss = f'{swap_coef}SwLss'

        name += f'-{swap_prob}swap{swap_size}{masking}{swap_loss}'

    if args.get('optimizer') != 'adam':
        opt = args.get('optimizer').upper()
        name += f'-opt{opt}'

    if args.get('cov'):
        cov_coef = ''
        var_coef = ''

        if args.get("cov_coef") != 1.0:
            cov_coef = args.get("cov_coef")

        if args.get("var_coef") != 1.0:
            var_coef = args.get("var_coef")

        if args.get('cov_static_mean'):
            name += f'-{cov_coef}cov9-{var_coef}var'
        else:
            name += f'-{cov_coef}cov4-{var_coef}var'


    name += f"_{args.get('loss')}"

    if args.get('pairwise_labels'):
        if args.get('num_inst_per_class') != 2:
            raise Exception('Pairwise_labels only support k = 2')
        name += f'-PairLbl'
        if args.get('eval_with_pairwise'):
            name += '-EvPr'

    if args.get('with_bce'):
        name += f'-bce_bw{args.get("bce_weight")}'

    if args.get('ml_self_att'):
        layers_to_use = args.get('ml_self_att_layers_to_use')
        ltu = ''
        if layers_to_use < 4:
            ltu = f'{layers_to_use}'
        name += f"_M{args.get('ml_self_att_head_number')}LocSelfAtt{ltu}"

        if args.get('only_att'):
            name += 'OnlyATT'


    if args.get('multlayer_emb'):
        name += f'-MLTEMB'

        if args.get('eq_layer_contrib'):
            name += f'-EQ'

    for n in loss_specific_args:
        if args.get(n) is not None:
            name += f'-{n}{args.get(n)}'

    if args.get('lnorm'):
        name += f"_n"

    if args.get('metric') != 'cosine':
        name += f"_temp{args.get('temperature')}"

    if args.get('k_inc_freq') != 0:
        name += f"_Kinc{args.get('k_inc_freq')}"


    if args.get('k_dec_freq') != 0:
        name += f"_Kdec{args.get('k_dec_freq')}"

    if args.get('xname') != '':
        name += f"_{args.get('xname')}"

    return name


def balance_labels(pairwise_batch, k):
    balanced_batch = np.array([], dtype=np.float32)
    bs = pairwise_batch.shape[0]
    for i in range(bs // k):
        idx = i * k
        balanced_inst_list = pairwise_batch[[idx, idx], [idx + 1, idx + 2]]
        balanced_batch = np.concatenate([balanced_batch, balanced_inst_list])

    return torch.tensor(balanced_batch, dtype=torch.float32)


def load_model(net, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=next(net.parameters()).device)
    net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint.get("epoch", -1)
    print(f'Retrieving model {checkpoint_path} from epoch {checkpoint["epoch"]}...')
    return net, epoch


def save_model(net, epoch, val_kind, save_path):
    # best_model_name = 'model-epoch-' + str(epoch) + '-val-auroc-' + str(val_auc) + '.pt'

    if type(net) == torch.nn.DataParallel:
        netmod = net.module
    else:
        netmod = net

    # torch.save({'epoch': epoch, 'model_state_dict': netmod.state_dict()},
    #            save_path + '/' + best_model_name)

    best_model_name = f'best_model_{val_kind}.pt'
    torch.save({'epoch': epoch, 'model_state_dict': netmod.state_dict()},
               save_path + '/' + best_model_name)

    return best_model_name


def get_faiss_knn(reps, k=1500, gpu=False, metric='cosine'):  # method "cosine" or "euclidean"
    assert reps.dtype == np.float32
    valid = False

    D, I, self_D = None, None, None

    d = reps.shape[1]
    if metric == 'euclidean':
        index_function = faiss.IndexFlatL2
    elif metric == 'cosine':
        index_function = faiss.IndexFlatIP
    else:
        raise Exception(f'get_faiss_knn unsupported method {metric}')

    if gpu:
        try:
            index_flat = index_function(d)
            res = faiss.StandardGpuResources()
            index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
            index_flat.add(reps)  # add vectors to the index
            print('Using GPU for KNN!!'
                  ' Thanks FAISS! :)')
        except:
            print('Didn\'t fit it GPU, No gpus for faiss! :( ')
            index_flat = index_function(d)
            index_flat.add(reps)  # add vectors to the index
    else:
        print('No gpus for faiss! :( ')
        index_flat = index_function(d)
        index_flat.add(reps)  # add vectors to the index

    assert (index_flat.ntotal == reps.shape[0])

    while not valid:
        print(f'get_faiss_knn metric is: {metric} for top {k}')

        D, I = index_flat.search(reps, k)

        D_notself = []
        I_notself = []

        self_distance = []
        max_dist = np.array(D.max(), dtype=np.float32)
        for i, (i_row, d_row) in enumerate(zip(I, D)):
            if len(np.where(i_row == i)[0]) > 0:  # own index in returned indices
                self_distance.append(d_row[np.where(i_row == i)])
                I_notself.append(np.delete(i_row, np.where(i_row == i)))
                D_notself.append(np.delete(d_row, np.where(i_row == i)))
            else:
                self_distance.append(max_dist)
                I_notself.append(np.delete(i_row, len(i_row) - 1))
                D_notself.append(np.delete(d_row, len(i_row) - 1))

        self_D = np.array(self_distance, dtype=np.float32)
        D = np.array(D_notself, dtype=np.int32)
        I = np.array(I_notself, dtype=np.int32)
        if len(self_D) == D.shape[0]:
            valid = True
        else:  # self was not found for all examples
            print(f'self was not found for all examples, going from k={k} to k={k * 2}')
            k *= 2

    return D, I, self_D


def get_recall_at_k(img_feats, img_lbls, sim_matrix=None, metric='cosine', Kset=[1, 2, 4, 8], pairwise_labels=None):
    all_lbls = np.unique(img_lbls)

    num = img_lbls.shape[0]

    is_pairwise = pairwise_labels is not None

    if is_pairwise:
        assert pairwise_labels.shape[0] == num
        assert pairwise_labels.shape[1] == num

    k_max = min(1500, img_lbls.shape[0])

    if sim_matrix is None:
        if not img_feats.flags['C_CONTIGUOUS']:
            img_feats = np.ascontiguousarray(img_feats)
        _, I, self_D = get_faiss_knn(img_feats, k=k_max, gpu=True, metric=metric)
    else:
        minval = np.min(sim_matrix) - 1.
        self_D = -(np.diag(sim_matrix))
        sim_matrix -= np.diag(np.diag(sim_matrix))
        sim_matrix += np.diag(np.ones(num) * minval)
        I = (-sim_matrix).argsort()[:, :-1]

    recall_at_k = metrics.Accuracy_At_K(classes=np.array(all_lbls), ks=Kset, pairwise=is_pairwise)
    if not is_pairwise:

        for idx, lbl in enumerate(img_lbls):
            ret_lbls = img_lbls[I[idx]]
            recall_at_k.update(lbl, ret_lbls)

    else:

        for idx, lbl in enumerate(img_lbls):
            ret_lbls_1_or_0 = pairwise_labels[idx, :][I[idx]]
            class_lbls_1_0 = (img_lbls[I[idx]] == lbl).astype(int)
            recall_at_k.update(lbl, np.bitwise_and(ret_lbls_1_or_0, class_lbls_1_0).astype(np.float32))

    total = recall_at_k.get_all_metrics()
    return total


def make_batch_bce_labels(labels, diagonal_fill=None):
    """
    :param labels: e.g. tensor of size (N,1)
    :return: binary matrix of labels of size (N, N)
    """

    l_ = labels.repeat(len(labels)).reshape(-1, len(labels))
    l__ = labels.repeat_interleave(len(labels)).reshape(-1, len(labels))

    final_bce_labels = (l_ == l__).type(torch.float32)

    if diagonal_fill:
        final_bce_labels.fill_diagonal_(diagonal_fill)

    return final_bce_labels


def get_samples(l, k):
    if len(l) < k:
        to_ret = np.random.choice(l, k, replace=True)
    else:
        to_ret = np.random.choice(l, k, replace=False)

    return to_ret


def get_xs_ys(bce_labels, k=1):
    """

    :param bce_labels: tensor of (N, N) with 0s and 1s
    :param k: number of pos and neg samples per anch
    :return: an equal number of positive and negative pairs chosen randomly

    """
    xs = []
    ys = []
    bce_labels_copy = copy.deepcopy(bce_labels)
    bce_labels_copy.fill_diagonal_(-1)
    for i, row in enumerate(bce_labels_copy):
        neg_idx = torch.where(row == 0)[0]
        pos_idx = torch.where(row == 1)[0]

        ys.extend(get_samples(neg_idx, k))
        ys.extend(get_samples(pos_idx, k))
        xs.extend(get_samples([i], 2 * k))

    return xs, ys


def get_hard_xs_ys(bce_labels, a2n, k):
    """

    :param bce_labels: tensor of (N, N) with 0s and 1s
    :param a2n: dict, mapping every anchor idx to hard neg idxs
    :param k: number of pos and neg samples per anch
    :return:

    """
    xs = []
    ys = []
    bce_labels_copy = copy.deepcopy(bce_labels)
    bce_labels_copy.fill_diagonal_(-1)
    for i, row in enumerate(bce_labels_copy):
        neg_idx_chosen = a2n[i][:k]
        pos_idx = torch.where(row == 1)[0]

        ys.extend(neg_idx_chosen)
        ys.extend(get_samples(pos_idx, k))
        xs.extend(get_samples([i], 2 * k))

    return xs, ys


def calc_auroc(embeddings, labels, k=1, anch_2_hardneg_idx=None):
    """

    :param embeddings: all embeddings of a set to be tested
    :param labels: all labels of the set to be tested
    :return: the AUROC score, where random would score 1/(k + 1)
    """
    from sklearn.metrics import roc_auc_score
    bce_labels = make_batch_bce_labels(labels)
    similarities = cosine_similarity(embeddings)

    if anch_2_hardneg_idx is None:  # random
        xs, ys = get_xs_ys(bce_labels, k=k)
    else:
        xs, ys = get_hard_xs_ys(bce_labels, anch_2_hardneg_idx, k=k)

    true_labels = bce_labels[xs, ys]
    predicted_labels = similarities[xs, ys]

    return roc_auc_score(true_labels, predicted_labels), {'true_labels': true_labels,
                                                          'pred_labels': predicted_labels}


def transform_only_img(img_path):
    transform_only_img_func = transforms.Compose([transforms.Resize((256, 256)),
                                                  transforms.CenterCrop(224)])

    img = open_img(img_path)

    img = transform_only_img_func(img)

    return img


def get_avg_activations(acts, size=None):
    sample_act = list(acts)[0]
    if len(sample_act.shape) == 3:
        if size is None:
            max_size = 0
            for a in acts:
                if a.shape[1] > max_size:
                    max_size = a.shape[1]
        else:
            max_size = size

        reshaped_activations = []

        for batch_a in acts:
            temp_list = []
            if batch_a.shape[1] != max_size:
                for a in batch_a:
                    a = cv2.resize(np.float32(a), (max_size, max_size))
                    temp_list.append(a)
                reshaped_activations.append(np.stack(temp_list, axis=0))

    else: # len(sample_act.shape) == 2
        if size is None:
            max_size = 0
            for a in acts:
                if a.shape[0] > max_size:
                    max_size = a.shape[0]
        else:
            max_size = size

        reshaped_activations = []

        for a in acts:
            if a.shape[0] != max_size:
                a = cv2.resize(np.float32(a), (max_size, max_size))
            reshaped_activations.append(a)

    final_addition = copy.deepcopy(reshaped_activations[0])
    w_sum = 1
    # weighted average
    for i, fa in enumerate(reshaped_activations[1:], 1):
        w = pow(2, i)
        final_addition += fa * w
        w_sum += w

    # final_addition /= len(reshaped_activations)
    final_addition /= w_sum  # (1 + 2 + 3 + 4)

    return final_addition


def get_heatmaped_img(acts, img):
    """
    :param act: one activation layer (H, W)
    :param img: ndarray e.g. (224, 224)
    :return: merged_img
    """

    heatmap = __post_create_heatmap(acts, (img.shape[0], img.shape[1]))
    pic = merge_heatmap_img(img, heatmap)

    return pic

def concat_imgs(img_list):
    final_img = cv2.hconcat(img_list)
    return final_img


def reduce_normalize_activation(t, mode='avg'):
    """
    :param t: an ndarray of size (B, C, H, W)
    :param mode: ['avg', 'max']
    :return: a normalized and reduced ndarray of size (H, W) ndarray if B == 1, else (B, H, W)
    """

    B, C, H, W = t.shape
    t = np.maximum(t, 0)
    # t_max = t.max(axis=-1).max(axis=-1).max(axis=-1)  # activations between 0 and 1
    # t_max = t_max.repeat(H * W).reshape((B, H, W))

    t_max = t.max()  # normalize all tensors in B together (global max is used)


    if mode == 'avg':
        ret = t.mean(axis=1)
    elif mode == 'max':
        ret = t.max(axis=1)
    else:
        raise Exception('Not supported in reduce_activation in utils.py')

    ret /= t_max

    if B == 1:
        ret = ret.squeeze(axis=0)

    return ret

def get_double_heatmaps(list_of_activationsets, imgss):
    all_img_heatmaps = []
    for acts, imgs in zip(list_of_activationsets, imgss):
        dict_of_activations = {}
        for i, a in enumerate(acts, 1):
            a = a.detach().cpu().numpy()
            dict_of_activations[f'l{i}'] = reduce_normalize_activation(a, mode='max') # each element is a (B, H, W) matrix where B != 1

        # todo size being None causes 2 resizes instead of one
        dict_of_activations['all'] = get_avg_activations(dict_of_activations.values(), size=None)

        heatmaps_to_return = {}

        imgs = [np.array(img) for img in imgs]

        for layer_i, (label, act) in enumerate(dict_of_activations.items(), 1):
            assert len(act) == len(imgs)
            pics = []
            for one_act, one_img in zip(act, imgs):
                pics.append(get_heatmaped_img(one_act, one_img))
            heatmaps_to_return[label] = concat_imgs(pics)

        heatmaps_to_return['org'] = concat_imgs([img[:, :, :3] for img in imgs])
        all_img_heatmaps.append(heatmaps_to_return)

    return all_img_heatmaps


def get_all_heatmaps(list_of_activationsets, imgs):
    all_img_heatmaps = []
    for acts, img in zip(list_of_activationsets, imgs):
        img = np.array(img)
        dict_of_activations = {}
        for i, a in enumerate(acts, 1):
            a = a.detach().cpu().numpy()
            dict_of_activations[f'l{i}'] = reduce_normalize_activation(a, mode='max')

        # todo size being None causes 2 resizes instead of one
        dict_of_activations['all'] = get_avg_activations(dict_of_activations.values(), size=None)

        heatmaps_to_return = {}
        for layer_i, (label, act) in enumerate(dict_of_activations.items(), 1):
            pic = get_heatmaped_img(act, img)
            heatmaps_to_return[label] = pic

        heatmaps_to_return['org'] = img[:, :, :3]
        all_img_heatmaps.append(heatmaps_to_return)

    return all_img_heatmaps


def merge_heatmap_img(img, heatmap):
    pic = img.copy()
    cv2.addWeighted(heatmap, 0.4, img, 0.6, 0, pic)
    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)

    return pic


def __post_create_heatmap(heatmap, shape):
    # draw the heatmap
    plt.matshow(heatmap.squeeze())

    heatmap = cv2.resize(np.float32(heatmap), shape)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap


def get_a2n(ordered_lbls, ordered_idxs, all_labels):
    N, K = ordered_idxs.shape

    if ordered_lbls.shape != ordered_idxs.shape:
        ordered_lbls = ordered_lbls.reshape(ordered_idxs.shape)

    all_labels = all_labels.repeat(K).reshape(N, K)
    pos_mask = (all_labels == ordered_lbls).astype(np.int64)
    negative_idxs_of_idxs = pos_mask.argmin(axis=1)
    y_idxs = np.array([i for i in range(N)])
    negative_idxs = ordered_idxs[y_idxs, negative_idxs_of_idxs]

    a2n = {i: [row] for i, row in enumerate(negative_idxs)}

    return a2n


def get_preds(embeddings, metric='cosine', model=None, temperature=3):
    if metric == 'cosine':
        norm_embeddings = F.normalize(embeddings, p=2, dim=-1)
        if len(norm_embeddings.shape) == 2:
            sims = torch.matmul(norm_embeddings, norm_embeddings.T)
        else: # todo make sure it works!!!!
            sims = (norm_embeddings * norm_embeddings.transpose(0, 1)).sum(dim=-1)
        preds = (sims + 1) / 2  # maps (-1, 1) to (0, 1)

        preds = torch.clamp(preds, min=0.0, max=1.0)
    elif metric == 'euclidean':
        euclidean_dist = pairwise_distance(embeddings)

        euclidean_dist = euclidean_dist / temperature

        preds = 2 * nn.functional.sigmoid(-euclidean_dist)  # maps (0, +inf) to (1, 0)
        sims = -euclidean_dist
        # preds = torch.clamp(preds, min=0.0, max=1.0)
    elif metric == 'mlp':
        if model is None:
            raise Exception('No model provided for mlp distance')
        bs = embeddings.shape[0]
        indices = torch.tensor([[i, j] for i in range(bs) for j in range(bs)]).flatten()
        logits = model(embeddings[indices].reshape(bs * bs, -1))

        sims = logits / temperature
        preds = nn.functional.sigmoid(sims)
    else:
        raise Exception(f'{metric} not supported in Top Module')

    return preds, sims


def seperate_k_per_class(data, number_of_instances=3, number_of_classes=500, number_of_runs=1, img_label='image',
                         class_label='hotel_id'):
    to_return = []
    img_ids = np.array([i for i in range(len(data))])
    classes, class_sizes = np.unique(data[class_label], return_counts=True)
    classes = classes[class_sizes >= number_of_instances]
    # class_sizes = class_sizes[class_sizes >= number_of_instances]
    assert number_of_classes <= len(classes)
    for _ in range(number_of_runs):
        sampled_df = {img_label: [], class_label: []}
        if number_of_classes != 0:
            sampled_classes = np.random.choice(classes, number_of_classes, replace=False)
        else:
            sampled_classes = classes

        sampled_classes = sorted(sampled_classes)
        for c in sampled_classes:
            data_c = img_ids[data[class_label] == c]
            sampled_images = np.random.choice(data_c, number_of_instances, replace=False)
            sampled_df[img_label].extend(sorted(sampled_images))
            sampled_df[class_label].extend([c for _ in range(number_of_instances)])

        to_return.append(pd.DataFrame(data=sampled_df))

    return to_return

def get_class_plots(embeddings, labels, num_classes_2_draw=16, specific_labels=None, name='classembeddings'):
    """
    plots graphs with -> x axis: channels, y axis: avg channel value with a std of each value
    :param embeddings:
    :param labels:
    :param num_classes_2_draw:
    :param specific_labels:
    :param name:
    :return:
    """
    N, C = embeddings.shape
    X_axis = [i for i in range(C)]

    unique_classes = np.unique(labels)


    unique_classes = sorted(unique_classes)
    fig, axes = plt.subplots(4, 4, figsize=(128, 96), sharex=True, sharey=True)
    if specific_labels is not None:
        classes_to_iterate = specific_labels
        classes_to_iterate = sorted(classes_to_iterate)
    else:
        classes_to_iterate = unique_classes[:num_classes_2_draw]

    for idx, c in enumerate(classes_to_iterate):
        c_embs = embeddings[labels == c]
        c_embs_avg = c_embs.mean(axis=0)
        c_embs_std = c_embs.std(axis=0)

        axes[idx // 4][idx % 4].plot(X_axis, c_embs_avg)
        axes[idx // 4][idx % 4].fill_between(X_axis,
                                             c_embs_avg - c_embs_std,
                                             c_embs_avg + c_embs_std,
                                             alpha=.1)
        axes[idx // 4][idx % 4].set_title(f'Label {c}')

    if specific_labels is not None:
        num_classes_2_draw = len(specific_labels)
    plt.savefig(f'{num_classes_2_draw}_{name}.pdf')
    # plt.show()

def get_diag_3d_tensor(t):
    t_2d = torch.diagonal(t).T
    return t_2d

def torch_get_cov_with_previous(t, previous_mean, size):
    """
    get cov matrix
    :param t: a (N, D) tensor, with N samples and D feature types
    :param previous_mean: a (1, D) tensor with mean of all previous embeddings
    :param size: number of all embeddings up until now
    :return: cov tensor with size (D, D)
    """

    N, D = t.shape
    # t_mean = t.mean(dim=0, keepdim=True)
    t_sum = t.sum(dim=0, keepdim=True)
    t_mean = (previous_mean * size + t_sum) / (size + t.shape[0])
    t_prime = t - t_mean

    t_cov = (t_prime.T @ t_prime) / (N - 1)

    return t_cov, t_mean

def torch_get_cov(t):
    """
        get cov matrix
        :param t: a (N, D) tensor, with N samples and D feature types
        :return: cov tensor with size (D, D)
        """

    N, D = t.shape
    t_mean = t.mean(dim=0, keepdim=True)
    t_prime = t - t_mean

    t_cov = (t_prime.T @ t_prime) / (N - 1)

    return t_cov

def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, nn.BatchNorm2d):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, nn.BatchNorm2d) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)