import json
import logging
import os
import random
import sys

import cv2
import faiss
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

import datasets
import metrics
from samplers.my_sampler import BalancedTripletSampler, KBatchSampler, DataBaseSampler, DrawHeatmapSampler


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


def get_data(args, mode, file_name='', transform=None, sampler_mode='kbatch'):  # 'kbatch', 'balanced_triplet', 'db'
    SAMPLERS = {'kbatch': KBatchSampler,
                'balanced_triplet': BalancedTripletSampler,
                'db': DataBaseSampler,
                'heatmap': DrawHeatmapSampler}

    dataset = datasets.load_dataset(args, mode, file_name,
                                    transform=transform,
                                    for_heatmap= sampler_mode == 'heatmap')

    sampler = SAMPLERS[sampler_mode](dataset=dataset,
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
    loss_specific_args = ['LOSS_margin',
                          'NCA_scale',
                          'LOSS_alpha',
                          'ARCFACE_scale',
                          'CIR_m',
                          'CIR_gamma',
                          'PNPP_lr']

    name = ''

    if args.get('cuda'):
        gpu_ids = args.get("gpu_ids").replace(',', '')
        name += f'gpu{gpu_ids}_'

    name += 'wbn_%dep_%s_%s_bs%d_k%d_lr%f_bblr%f' % (args.get('epochs'),
                                                       args.get('dataset'),
                                                       args.get('metric'),
                                                       args.get('batch_size'),
                                                       args.get('num_inst_per_class'),
                                                       args.get('learning_rate'),
                                                       args.get('bb_learning_rate'))

    name += f"_{args.get('loss')}"
    for n in loss_specific_args:
        if args.get(n) is not None:
            name += f'-{n}{args.get(n)}'

    if args.get('lnorm'):
        name += f"_n"

    if args.get('metric') != 'cosine':
        name += f"_temp{args.get('temperature')}"
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


def save_model(net, epoch, val_acc, save_path):
    best_model_name = 'model-epoch-' + str(epoch) + '-val-acc-' + str(val_acc) + '.pt'
    torch.save({'epoch': epoch, 'model_state_dict': net.state_dict()},
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


def get_recall_at_k(img_feats, img_lbls, sim_matrix=None, metric='cosine'):
    all_lbls = np.unique(img_lbls)

    num = img_lbls.shape[0]

    k_max = min(1500, img_lbls.shape[0])

    if sim_matrix is None:
        _, I, self_D = get_faiss_knn(img_feats, k=k_max, gpu=True, metric=metric)
    else:
        minval = np.min(sim_matrix) - 1.
        self_D = -(np.diag(sim_matrix))
        sim_matrix -= np.diag(np.diag(sim_matrix))
        sim_matrix += np.diag(np.ones(num) * minval)
        I = (-sim_matrix).argsort()[:, :-1]

    recall_at_k = metrics.Accuracy_At_K(classes=np.array(all_lbls))

    for idx, lbl in enumerate(img_lbls):
        ret_lbls = img_lbls[I[idx]]
        recall_at_k.update(lbl, ret_lbls)

    total = recall_at_k.get_all_metrics()

    return total


def make_batch_bce_labels(labels):
    """
    :param labels: e.g. tensor of size (N,1)
    :return: binary matrix of labels of size (N, N)
    """
    l_ = labels.repeat(len(labels)).reshape(-1, len(labels))
    l__ = labels.repeat_interleave(len(labels)).reshape(-1, len(labels))

    final_bce_labels = (l_ == l__).type(torch.float32)

    return final_bce_labels


def transform_only_img(img_path):
    transform_only_img_func = transforms.Compose([transforms.Resize((256, 256)),
                                                       transforms.CenterCrop(224)])

    img = open_img(img_path)

    img = transform_only_img_func(img)

    return img

def get_avg_activations(acts, size=None):
    if size is None:
        max_size = 0
        for a in acts:
            if a.shape[0] > max_size:
                max_size = a.shape[0]
                print(max_size)
    else:
        max_size = size

    reshaped_activations = []

    for a in acts:
        if a.shape[0] != max_size:
            a = cv2.resize(np.float32(a), (max_size, max_size))

        reshaped_activations.append(a)



    final_addition = reshaped_activations[0]

    for fa in reshaped_activations[1:]:
        final_addition += fa

    final_addition /= len(reshaped_activations)

    return final_addition


def get_heatmaped_img(acts, img):
    """
    :param act: one activation layer (1, C, H, W)
    :param img: ndarray e.g. (224, 224)
    :return: merged_img
    """

    heatmap = __post_create_heatmap(acts, (img.shape[0], img.shape[1]))
    pic = merge_heatmap_img(img, heatmap)

    return pic



def get_all_heatmaps(list_of_activationsets, imgs):

    all_img_heatmaps = []
    for acts, img in zip(list_of_activationsets, imgs):
        img = np.array(img)
        dict_of_activations = {}
        for i, a in enumerate(acts, 1):
            dict_of_activations[f'l{i}'] = a.detach().cpu().numpy()

        # todo size being None causes 2 resizes instead of one
        dict_of_activations['all'] = get_avg_activations(acts, size=None)

        heatmaps_to_return = {}
        for layer_i, (label, act) in enumerate(dict_of_activations.items(), 1):

            acts_pos = np.maximum(act, 0)
            acts_pos /= np.max(acts_pos) # activations between 0 and 1

            max_act = acts_pos.max(axis=0).max(axis=0)
            pic = get_heatmaped_img(max_act, img)
            heatmaps_to_return[label] = pic

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
