import argparse
import json
import os
import pickle

import h5py
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from tqdm import tqdm

import baseline_model_loaders.ms_models as mm
import baseline_model_loaders.pnpp_models as pnpp
import baseline_model_loaders.proxy_anchor_models as pa
import baseline_model_loaders.softtriple_models as st
import baseline_model_loaders.sup_contrastive_models as sc
import model as htv2
# on hlr:
# python evaluation.py -chk ../SupContrast/save/SupCon/hotels_models/SupCon_hotels_resnet50_lr_0.01_decay_0.0001_bsz_32_temp_0.1_trial_0_cosine/last.pth -name SupCon_hotels_resnet50_lr_0.01_decay_0.0001_bsz_32_temp_0.1_trial_0_cosine/ --kset 1 2 4 8 10 100 1000 --model_type resnet50 -d hotels -dr ../../datasets/ --baseline supcontrastive --gpu_ids 6
import utils

ALPHA = 200.0

COLORS_VALUES_0255 = [
          (0.0, 0.0, 0.0, ALPHA),
          (1.0, 0.0, 103.0, ALPHA),
          (213.0, 255.0, 0.0, ALPHA),
          (255.0, 0.0, 86.0, ALPHA),
          (158.0, 0.0, 142.0, ALPHA),
          (14.0, 76.0, 161.0, ALPHA),
          (255.0, 229.0, 2.0, ALPHA),
          (0.0, 95.0, 57.0, ALPHA),
          (0.0, 255.0, 0.0, ALPHA),
          (149.0, 0.0, 58.0, ALPHA),
          (255.0, 147.0, 126.0, ALPHA),
          (164.0, 36.0, 0.0, ALPHA),
          (0.0, 21.0, 68.0, ALPHA),
          (145.0, 208.0, 203.0, ALPHA),
          (98.0, 14.0, 0.0, ALPHA),
          (107.0, 104.0, 130.0, ALPHA),
          (0.0, 0.0, 255.0, ALPHA),
          (0.0, 125.0, 181.0, ALPHA),
          (106.0, 130.0, 108.0, ALPHA),
          (0.0, 174.0, 126.0, ALPHA),
          (194.0, 140.0, 159.0, ALPHA),
          (190.0, 153.0, 112.0, ALPHA),
          (0.0, 143.0, 156.0, ALPHA),
          (95.0, 173.0, 78.0, ALPHA),
          (255.0, 0.0, 0.0, ALPHA),
          (255.0, 0.0, 246.0, ALPHA),
          (255.0, 2.0, 157.0, ALPHA),
          (104.0, 61.0, 59.0, ALPHA),
          (255.0, 116.0, 163.0, ALPHA),
          (150.0, 138.0, 232.0, ALPHA),
          (152.0, 255.0, 82.0, ALPHA),
          (167.0, 87.0, 64.0, ALPHA),
          (1.0, 255.0, 254.0, ALPHA),
          (255.0, 238.0, 232.0, ALPHA),
          (254.0, 137.0, 0.0, ALPHA),
          (189.0, 198.0, 255.0, ALPHA),
          (1.0, 208.0, 255.0, ALPHA),
          (187.0, 136.0, 0.0, ALPHA),
          (117.0, 68.0, 177.0, ALPHA),
          (165.0, 255.0, 210.0, ALPHA),
          (255.0, 166.0, 254.0, ALPHA),
          (119.0, 77.0, 0.0, ALPHA),
          (122.0, 71.0, 130.0, ALPHA),
          (38.0, 52.0, 0.0, ALPHA),
          (0.0, 71.0, 84.0, ALPHA),
          (67.0, 0.0, 44.0, ALPHA),
          (181.0, 0.0, 255.0, ALPHA),
          (255.0, 177.0, 103.0, ALPHA),
          (255.0, 219.0, 102.0, ALPHA),
          (144.0, 251.0, 146.0, ALPHA),
          (126.0, 45.0, 210.0, ALPHA),
          (189.0, 211.0, 147.0, ALPHA),
          (229.0, 111.0, 254.0, ALPHA),
          (222.0, 255.0, 116.0, ALPHA),
          (0.0, 255.0, 120.0, ALPHA),
          (0.0, 155.0, 255.0, ALPHA),
          (0.0, 100.0, 1.0, ALPHA),
          (0.0, 118.0, 255.0, ALPHA),
          (133.0, 169.0, 0.0, ALPHA),
          (0.0, 185.0, 23.0, ALPHA),
          (120.0, 130.0, 49.0, ALPHA),
          (0.0, 255.0, 198.0, ALPHA),
          (255.0, 110.0, 65.0, ALPHA),
          (232.0, 94.0, 190.0, ALPHA)]

# todo: visualize embeddings after training with each model to see how they look softtriple, proxy, htv2

dataset_choices = ['cars', 'cub', 'hotels', 'hotels_small', 'hotelid-val', 'hotelid-test']
BASELINE_MODELS = ['ours',
                   'softtriple',
                   'proxy-anchor',
                   'supcontrastive',
                   'proxyncapp',
                   'htv2',
                   'ms',
                   'resnet50']

DATASET_SIZES = {'cars': {'test': 8131},
                 'cub': {'test': 5924},
                 'hotels_small': {'val1_small': 3060,
                                  'val2_small': 2397,
                                  'val3_small': 2207,
                                  'val4_small': 2348,
                                  'test1_small': 1,  # 7390 is wrong
                                  'test2_small': 4944,
                                  'test3_small': 5146,
                                  'test4_small': 5919},
                 'hotels': {'val1': 22121,
                            'val2': 16095,
                            'val3': 15970,
                            'val4': 17981,
                            'test1': 51294,
                            'test2': 36537,
                            'test3': 35693,
                            'test4': 41437},
                 'hotelid-val': {'val1': 3704,
                                 'val2': 6612,
                                 'val3': 6595},

                 'hotelid-test': {'test1': 9013,
                                  'test2': 11110,
                                  'test3': 10973,
                                  'test4': 20220}
                 }

DATASET_MEANS = {'hotels': [0.5805, 0.5247, 0.4683],
                 'hotels_small': [0.5805, 0.5247, 0.4683],
                 'hotelid-val': [0.4620, 0.3980, 0.3292],
                 "hotelid-test": [0.4620, 0.3980, 0.3292],
                 'cub': None}

DATASET_STDS = {'hotels': [0.2508, 0.2580, 0.2701],
                'hotels_small': [0.2508, 0.2580, 0.2701],
                'hotelid-val': [0.2619, 0.2529, 0.2460],
                'hotelid-test': [0.2619, 0.2529, 0.2460],
                'cub': None}


def get_features_and_labels(args, model, loader):
    features = []
    labels = []
    lbl2idx = loader.dataset.lbl2idx
    idx2lbl = {v: k for k, v in lbl2idx.items()}
    with tqdm(total=len(loader), desc='Getting features...') as t:
        for idx, batch in enumerate(loader):
            img, lbl = batch
            if args.get('cuda'):
                f = model(img.cuda())
            else:
                f = model(img)

            if args.get('baseline') == 'softtriple':
                f = F.normalize(f, p=2, dim=1)

            features.append(f.cpu().detach().numpy())
            labels.append(lbl.apply_(idx2lbl.get))

            t.update()

    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def proxyanchor_load_model_resnet50(save_path, args):
    if args.get('cuda'):
        checkpoint = torch.load(save_path, map_location=torch.device(0))
    else:
        checkpoint = torch.load(save_path, map_location=torch.device('cpu'))

    net = pa.Resnet50(embedding_size=args.get('emb_size'),
                      pretrained=True,
                      is_norm=1,
                      bn_freeze=1)

    net.load_state_dict(checkpoint['model_state_dict'])

    if args.get('cuda'):
        net = net.cuda()

    return net


def supcontrastive_load_model_resnet50(save_path, args):
    if args.get('cuda'):
        checkpoint = torch.load(save_path, map_location=torch.device(0))
    else:
        checkpoint = torch.load(save_path, map_location=torch.device('cpu'))

    net = sc.resnet50()

    new_checkpoint = {}
    for key, value in checkpoint['model'].items():
        if key.startswith('encoder'):
            new_checkpoint[key[8:]] = value
        elif key.startswith('head'):
            pass

    net.load_state_dict(new_checkpoint)

    if args.get('cuda'):
        net = net.cuda()

    return net


def softtriple_load_model_resnet50(save_path, args):
    if args.get('cuda'):
        checkpoint = torch.load(save_path, map_location=torch.device(0))
    else:
        checkpoint = torch.load(save_path, map_location=torch.device('cpu'))

    net = timm.create_model('resnet50', num_classes=args.get('emb_size'))

    net.load_state_dict(checkpoint)

    if args.get('cuda'):
        net = net.cuda()

    return net


def proxyncapp_load_model_resnet50(save_path, args):
    if args.get('cuda'):
        checkpoint = torch.load(save_path, map_location=torch.device(0))
    else:
        checkpoint = torch.load(save_path, map_location=torch.device('cpu'))

    net = pnpp.get_model(args.get('emb_size'))

    if args.get('trained_with_mltp_gpu'):
        net = torch.nn.DataParallel(net)

    net.load_state_dict(checkpoint)

    if args.get('cuda'):
        net = net.cuda()

    return net


def htv2_load_model_resnet50(save_path, args):
    if args.get('cuda'):
        checkpoint = torch.load(save_path, map_location=torch.device(0))
    else:
        checkpoint = torch.load(save_path, map_location=torch.device('cpu'))

    net = htv2.get_top_module(args)

    # if args.get('trained_with_mltp_gpu'):
    #     net = torch.nn.DataParallel(net)

    net.load_state_dict(checkpoint['model_state_dict'])

    if args.get('cuda'):
        net = net.cuda()

    # if args.get('trained_with_mltp_gpu'):
    #     net = net.module.encoder
    # else:
    net = net.encoder

    return net


def softtriple_load_model_inception(save_path, args):
    if args.get('cuda'):
        checkpoint = torch.load(save_path, map_location=torch.device(0))
    else:
        checkpoint = torch.load(save_path, map_location=torch.device('cpu'))

    net = st.bninception(args.get('emb_size'))

    net.load_state_dict(checkpoint)

    if args.get('cuda'):
        net = net.cuda()

    return net


def ms_load_model_resnet50(save_path, args):
    if args.get('cuda'):
        checkpoint = torch.load(save_path, map_location=torch.device(0))
    else:
        checkpoint = torch.load(save_path, map_location=torch.device('cpu'))

    net = mm.build_model()

    net.load_state_dict(checkpoint['model'])

    if args.get('cuda'):
        net = net.cuda()

    return net


def resnet_load_model(save_path, args):
    # if args.get('cuda'):
    #     checkpoint = torch.load(save_path, map_location=torch.device(0))
    # else:
    #     checkpoint = torch.load(save_path, map_location=torch.device('cpu'))

    net = timm.create_model('resnet50', pretrained=True, num_classes=0)

    if args.get('cuda'):
        net = net.cuda()

    return net


def pca(features, emb_size):
    pca_model = PCA(n_components=emb_size)

    print(f'Performing PCA to reduce dim from {features.shape[1]} to {emb_size}')
    new_features = pca_model.fit_transform(features)

    return new_features


def check(args, all_data):
    val_keys = DATASET_SIZES[args.get('dataset')].keys()
    for provided_data, val_type in zip(all_data, val_keys):
        if provided_data[0].shape[0] != DATASET_SIZES[args.get('dataset')][val_type]:
            print(
                f'Val type {val_type} should be {DATASET_SIZES[args.get("dataset")][val_type]} images, but is {provided_data[0].shape[0]}')
            return False
    print(f'All sizes for {val_keys} were checked and are correct')
    return True


def fix_name(path: str):
    return path.replace('/', '_').split('.')[0]


def add_dicts(dict1, dict2):
    if dict1 is None or len(dict1.keys()) == 0:
        for k in dict2.keys():
            dict2[k] = list(dict2[k])

        return dict2

    assert len(dict1) == len(dict2)

    for k, v in dict2.items():
        dict1[k].extend(list(v))

    return dict1


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-cuda', '--cuda', default=False, action='store_true')
    parser.add_argument('-seed', '--seed', default=402, type=int)
    parser.add_argument('--run_times', default=1, type=int)
    parser.add_argument('-trained_with_mltp_gpu', '--trained_with_mltp_gpu', default=False, action='store_true')
    parser.add_argument('--eval_mode', default='val', help="val or test", choices=['val', 'test'])

    parser.add_argument('-gpu', '--gpu_ids', default='', help="gpu ids used to train")  # before: default="0,1,2,3"

    parser.add_argument('-X', '--X', nargs='+', default=[],
                        help="Different features for datasets (order important)")
    # parser.add_argument('-X_desc', '--X_desc', nargs='+', default=[],
    #                     help="Different features desc for datasets (order important)") # for h5 or npz files

    parser.add_argument('-Y', '--Y', nargs='+', default=[],
                        help="Different labels for datasets (order important)")
    # parser.add_argument('-Y_desc', '--Y_desc', nargs='+', default=[],
    #                     help="Different labels desc for datasets (order important)")  # for h5 or npz files

    parser.add_argument('-emb', '--emb_size', default=512, type=int)
    parser.add_argument('-b', '--batch_size', default=32, type=int)
    parser.add_argument('-w', '--workers', default=10, type=int)
    parser.add_argument('--pin_memory', default=False, action='store_true')

    parser.add_argument('-d', '--dataset', default=None, choices=dataset_choices)
    parser.add_argument('--num_inst_per_class', default=5, type=int)
    parser.add_argument('--lnorm', default=False, action='store_true')

    parser.add_argument('--config_path', default='config/', help="config_path for datasets")
    parser.add_argument('--project_path', default='./', help="current project path")

    parser.add_argument('-num_of_dataset', '--num_of_dataset', type=int, default=4,
                        help="number of hotels val_datasets to go through")
    parser.add_argument('--baseline', default='proxy-anchor', choices=BASELINE_MODELS)
    parser.add_argument('--backbone', default='resnet50', choices=['bninception', 'resnet50'])

    parser.add_argument('--pca_to_dim', default=False, action='store_true')
    parser.add_argument('--force_update', default=False, action='store_true')

    parser.add_argument('-chk', '--checkpoint', default=None, help='Path to checkpoint')
    parser.add_argument('--kset', nargs='+', default=[1, 2, 4, 8, 16, 32, 100])

    parser.add_argument('-elp', '--eval_log_path', default='./eval_logs')
    parser.add_argument('-name', '--name', default=None, type=str)

    parser.add_argument('--hard_neg', default=False, action='store_true')
    parser.add_argument('--project', default=False, action='store_true')
    parser.add_argument('--normalize_project', default=False, action='store_true')
    parser.add_argument('--project_no_labels', type=int, default=30)
    parser.add_argument('--project_labels_start', type=int, default=0)

    parser.add_argument('--eval_metric', default='auc', choices=['auc', 'ret'])


    parser.add_argument('--metric', default='cosine', choices=['cosine', 'euclidean'])

    args = parser.parse_args()

    all_data = []

    kset = [int(k) for k in args.kset]

    dataset_config = utils.load_config(os.path.join(args.config_path, args.dataset + '.json'))

    all_args = utils.Global_Config_File(args=args, config_file=dataset_config)


    print(str(all_args))
    utils.seed_all(all_args.get('seed'))

    if all_args.get('name') is None:
        raise Exception('Provide --name')

    eval_log_path = os.path.join(all_args.get('eval_log_path'),
                                 f'{all_args.get("eval_mode").upper()}_{all_args.get("dataset")}')

    utils.make_dirs(os.path.join(eval_log_path, 'cache/'))

    cache_path = os.path.join(eval_log_path, 'cache', all_args.get('name'))

    # provide model and extract embeddings here
    if len(all_args.get('X')) == 0:

        if os.path.exists(cache_path) and \
                not all_args.get('force_update'):

            for i in range(0, all_args.get('num_of_dataset')):
                val_set_name = fix_name(all_args.get(f'all_{all_args.get("eval_mode")}_files')[i])

                emb_data_path = os.path.join(cache_path, val_set_name + "_embs.npy")

                lbl_data_path = os.path.join(cache_path, val_set_name + "_lbls.npy")

                emb_data = np.load(emb_data_path)
                lbl_data = np.load(lbl_data_path)

                all_data.append((emb_data, lbl_data))
        else:
            if all_args.get('gpu_ids') != '':
                os.environ["CUDA_VISIBLE_DEVICES"] = all_args.get('gpu_ids')

            val_transforms, val_transforms_names = utils.TransformLoader(all_args).get_composed_transform(mode='val')
            # eval_datasets = []
            eval_ldrs = []

            if all_args.get('num_of_dataset') > len(all_args.get(f'all_{all_args.get("eval_mode")}_files')):
                raise Exception(
                    f"num_of_dataset ({all_args.get('num_of_dataset')}) is greater than all_val_files in specified in json file")

            for i in range(0, all_args.get('num_of_dataset')):
                val_set_name = all_args.get(f'all_{all_args.get("eval_mode")}_files')[i]
                eval_ldrs.append(utils.get_data(all_args, mode=all_args.get('eval_mode'),
                                                file_name=val_set_name,
                                                transform=val_transforms,
                                                sampler_mode='db'))

            # if 'hotels' in all_args.get('dataset'):
            #     for i in range(1, args.num_of_dataset + 1):
            #         eval_loaders.append(utils.get_data(all_args, mode='val', transform=val_transforms, sampler_mode='db'))
            #         eval_datasets.append(dataset.load(
            #             name=all_args.get('dataset'),
            #             root=all_args.get('data_root'),
            #             transform=dataset_loaders.utils.make_transform(
            #                 is_train=False, std=DATASET_STDS.get(all_args.get('dataset')),
            #                 mean=DATASET_MEANS.get(all_args.get('dataset'))),
            #             valset=i,
            #             small=('small' in all_args.get('dataset'))))
            # else:
            #     eval_datasets = [dataset_loaders.load(
            #         name=all_args.get('dataset'),
            #         root=all_args.get('data_root'),
            #         transform=dataset_loaders.utils.make_transform(
            #             is_train=False, std=DATASET_STDS.get(all_args.get('dataset')),
            #             mean=DATASET_MEANS.get(all_args.get('dataset'))
            #         ))]
            net = None
            if all_args.get('baseline') == 'proxy-anchor':
                net = proxyanchor_load_model_resnet50(all_args.get('checkpoint'), all_args)
            elif all_args.get('baseline') == 'supcontrastive':
                net = supcontrastive_load_model_resnet50(all_args.get('checkpoint'), all_args)
            elif all_args.get('baseline') == 'softtriple':
                if all_args.get('backbone') == 'resnet50':
                    net = softtriple_load_model_resnet50(all_args.get('checkpoint'), all_args)
                elif all_args.get('backbone') == 'bninception':
                    net = softtriple_load_model_inception(all_args.get('checkpoint'), all_args)
            elif all_args.get('baseline') == 'resnet50':
                net = resnet_load_model(all_args.get('checkpoint'), all_args)
            elif all_args.get('baseline') == 'proxyncapp':
                net = proxyncapp_load_model_resnet50(all_args.get('checkpoint'), all_args)
            elif all_args.get('baseline') == 'htv2':
                net = htv2_load_model_resnet50(all_args.get('checkpoint'), all_args)
            elif all_args.get('baseline') == 'ms':
                net = ms_load_model_resnet50(all_args.get('checkpoint'), all_args)

            assert net is not None
            net.eval()
            # eval_ldrs = []
            # for dtset in eval_datasets:
            #     eval_ldrs.append(torch.utils.data.DataLoader(
            #         dtset,
            #         batch_size=all_args.get('sz_batch'),
            #         shuffle=False,
            #         num_workers=all_args.get('nb_workers'),
            #         drop_last=False,
            #         pin_memory=True
            #     ))

            for ldr in eval_ldrs:
                with torch.no_grad():
                    features, labels = get_features_and_labels(all_args, net, ldr)
                    all_data.append((features, labels))

            utils.make_dirs(cache_path)
            for i in range(0, all_args.get('num_of_dataset')):
                val_set_name = fix_name(all_args.get(f'all_{all_args.get("eval_mode")}_files')[i])
                emb_data, lbl_data = all_data[i]

                emb_data_path = os.path.join(cache_path, val_set_name + "_embs.npy")
                lbl_data_path = os.path.join(cache_path, val_set_name + "_lbls.npy")

                np.save(emb_data_path, emb_data)
                np.save(lbl_data_path, lbl_data)

    else:  # X and Y should be provided
        for idx, (x, y) in enumerate(zip(all_args.get('X'), all_args.get('Y'))):
            if x.endswith('.pkl'):
                with open(x, 'rb') as f:
                    features = pickle.load(f)
            elif x.endswith('.npz'):  # tood
                features = np.load(x)
            elif x.endswith('.h5'):
                with h5py.File(x, 'r') as hf:
                    features = hf['data'][:]
            else:
                raise Exception(f'{x} data format not supported')

            if y.endswith('.pkl'):
                with open(y, 'rb') as f:
                    labels = pickle.load(f)
            elif y.endswith('.npz'):  # tood
                labels = np.load(y)
            elif y.endswith('.h5'):
                with h5py.File(y, 'r') as hf:
                    labels = hf['data'][:]
            else:
                raise Exception(f'{y} data format not supported')

            if torch.is_tensor(features):
                features = features.cpu().numpy()

            if torch.is_tensor(labels):
                labels = labels.cpu().numpy()

            all_data.append((features, labels))

    if check(all_args, all_data):
        print('Assertion completed!')

    results = f'{all_args.get("dataset")}\n'

    ordered_lbls_idxs = []
    if all_args.get('hard_neg'):
        for i in range(0, all_args.get('num_of_dataset')):
            val_set_name = all_args.get(f'all_{all_args.get("eval_mode")}_files')[i]

            o_l = np.load(all_args.get('all_ordered_lbls')[val_set_name])
            o_i = np.load(all_args.get('all_ordered_idxs')[val_set_name])
            ordered_lbls_idxs.append([o_l, o_i])
    auc_predictions = {}

    if all_args.get('hard_neg'):
        hard_neg_string = '_HN'
    else:
        hard_neg_string = ''

    different_seeds_auc = {}
    # if all_args.get('eval_metric').upper() == 'AUC':
    #     different_seeds_results = {}
    if all_args.get('eval_metric').upper() == 'AUC':
        seeds = [all_args.get('seed') * (i + 1) for i in range(all_args.get('run_times'))]
    else:
        seeds = [all_args.get('seed')]

    for seed in seeds:
        print(f'SEED = {seed}')
        for idx, (features, labels) in enumerate(all_data, 1):

            if features.shape[1] != all_args.get('emb_size'):
                if all_args.get('pca_to_dim'):
                    features = pca(features, all_args.get('emb_size'))
                else:
                    raise Exception(
                        f'--pca_to_dim is set to False and feature dim {features.shape[1]} not equal to expected dim {all_args.get("emb_size")}')

            if all_args.get('eval_metric').upper() == 'AUC':
                print('*' * 10)
                print(f'{idx}: Calc AUC_ROC')
                if all_args.get('hard_neg'):
                    a2n = utils.get_a2n(ordered_lbls_idxs[idx - 1][0], ordered_lbls_idxs[idx - 1][1], labels)
                else:
                    a2n = None
                auc, t_and_p_labels = utils.calc_auroc(features, torch.tensor(labels), anch_2_hardneg_idx=a2n)
                if idx not in auc_predictions.keys():
                    auc_predictions[idx] = {}

                auc_predictions[idx] = add_dicts(auc_predictions[idx], t_and_p_labels)

                print(f'{idx}: AUC_ROC:', auc)
                results += f'\n\n{idx}: AUC_ROC: {auc}\n\n'
                if idx not in different_seeds_auc.keys():
                    different_seeds_auc[idx] = []

                different_seeds_auc[idx].extend([auc])
                results += '*' * 20 + '\n'

            elif all_args.get('eval_metric').upper() == 'RET':
                print(f'{idx}: Calc Recall at {kset}')
                rec = utils.get_recall_at_k(features, labels,
                                            metric='cosine',
                                            sim_matrix=None,
                                            Kset=kset)
                # = evaluate_recall_at_k(features, labels, Kset=kset, metric=all_args.get('metric'))
                print(kset)
                print(rec)
                results += f'seed: {seed} - {idx}: Calc Recall at {kset}' + '\n' + str(kset) + '\n' + str(rec) + '\n'
                results += '*' * 20 + '\n\n'

    mean_stdvs = {}

    if all_args.get('eval_metric').upper() == 'AUC':

        for k, v in different_seeds_auc.items():
            mean_stdvs[k] = (np.mean(v), np.std(v))
            auc_predictions[k] = (auc_predictions[k], np.mean(v), np.std(v))
            results += f"AUCs for Eval {k}: {v}\n\n"

        results += f"\n***\nMean AUC and Std Dev over {len(seeds)} seeds: ({seeds}) \n{str(json.dumps(mean_stdvs))}\n\n"

        if len(auc_predictions) > 1:
            fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.2))
            fig.suptitle(f'{all_args.get("name")} {hard_neg_string}')
            for ax, (key, value) in zip([axes[0][0], axes[0][1], axes[1][0], axes[1][1]], auc_predictions.items()):
                ax.hist(value[0]['pred_labels'][value[0]['true_labels'] == 1], bins=100, color='g', alpha=0.5)
                ax.hist(value[0]['pred_labels'][value[0]['true_labels'] == 0], bins=100, color='r', alpha=0.5)
                ax.set_title(f'Test {key}: {value[1]:.3} +- {value[2]:.3}')
        else:
            title_name = list(auc_predictions.keys())[0]
            t_and_p_labels = auc_predictions[title_name]
            plt.hist(t_and_p_labels[0]['pred_labels'][t_and_p_labels[0]['true_labels'] == 1], bins=100, color='g',
                     alpha=0.5)
            plt.hist(t_and_p_labels[0]['pred_labels'][t_and_p_labels[0]['true_labels'] == 0], bins=100, color='r',
                     alpha=0.5)
            plt.title(f'{all_args.get("name")} {hard_neg_string}\nTest {title_name}: {t_and_p_labels[1]:.3} +- {t_and_p_labels[2]:.3}')

        plt.savefig(os.path.join(eval_log_path, all_args.get('name') + f"{hard_neg_string}_aucplot.pdf"))
        plt.clf()

    if all_args.get('project'):
        COLORS_VALUES_01 = []
        for c in COLORS_VALUES_0255:
            COLORS_VALUES_01.append((c[0] / 255,
                                     c[1] / 255,
                                     c[2] / 255,
                                     c[3] / 255))

        NUM_COLORS = all_args.get('project_no_labels')  # 30
        idx_start = all_args.get('project_labels_start')
        cm = plt.get_cmap('gist_rainbow')

        fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.2))
        fig.suptitle(f'{all_args.get("name")} {hard_neg_string}')
        drawn_labels = {}
        for idx, (ax, (features, labels), (key, value)) in enumerate(
                zip([axes[0][0], axes[0][1], axes[1][0], axes[1][1]],
                    all_data,
                    auc_predictions.items()), 1):
            # ax.set_prop_cycle(color=[cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])

            if all_args.get('normalize_project'):
                features = F.normalize(torch.tensor(features), p=2, dim=1).numpy()

            ax.set_prop_cycle(color=COLORS_VALUES_01[:NUM_COLORS])
            features_2d = pca(features, emb_size=2)

            chosen_unique_labels = sorted(np.unique(labels))[idx_start: idx_start + NUM_COLORS]
            print(f'Test {key} labels:', chosen_unique_labels)
            drawn_labels[key] = chosen_unique_labels

            features_2d_specific = features_2d[np.logical_and(labels <= chosen_unique_labels[-1],
                                                              labels >= chosen_unique_labels[0])]

            labels_specific = labels[np.logical_and(labels <= chosen_unique_labels[-1],
                                                    labels >= chosen_unique_labels[0])]

            u_lbls = np.unique(labels_specific)

            for l in u_lbls:
                ax.scatter(features_2d_specific[labels_specific == l][:, 0],
                           features_2d_specific[labels_specific == l][:, 1],
                           )
            ax.set_title(f'Test {key}: {value[1]:.3}')

        if all_args.get('normalize_project'):
            norm_string = 'norm_'
        else:
            norm_string = ''
        plt.savefig(os.path.join(eval_log_path, all_args.get('name') + f"{hard_neg_string}_scatter_{norm_string}{all_args.get('project_labels_start')}-{all_args.get('project_no_labels')}.pdf"))
        plt.clf()

        scatter_text_to_write = ''
        for k, v in drawn_labels.items():
            scatter_text_to_write += f'Test {k}: {v}' + '\n'

        with open(os.path.join(eval_log_path, all_args.get('name') + f"{hard_neg_string}_scatter_{norm_string}{all_args.get('project_labels_start')}-{all_args.get('project_no_labels')}.txt"), 'w') as f:
            f.write(scatter_text_to_write)

    with open(os.path.join(eval_log_path, all_args.get('name') + f"{hard_neg_string}.txt"), 'w') as f:
        f.write(results)


if __name__ == '__main__':
    main()
