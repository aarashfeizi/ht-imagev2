import argparse
import os
import pickle
import time

import faiss
import h5py
import numpy as np
import timm
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

import model as htv2
import baseline_model_loaders.pnpp_models as pnpp
import baseline_model_loaders.proxy_anchor_models as pa
import baseline_model_loaders.softtriple_models as st
import baseline_model_loaders.sup_contrastive_models as sc
import baseline_model_loaders.ms_models as mm
# on hlr:
# python evaluation.py -chk ../SupContrast/save/SupCon/hotels_models/SupCon_hotels_resnet50_lr_0.01_decay_0.0001_bsz_32_temp_0.1_trial_0_cosine/last.pth -name SupCon_hotels_resnet50_lr_0.01_decay_0.0001_bsz_32_temp_0.1_trial_0_cosine/ --kset 1 2 4 8 10 100 1000 --model_type resnet50 -d hotels -dr ../../datasets/ --baseline supcontrastive --gpu_ids 6
import utils

dataset_choices = ['cars', 'cub', 'hotels', 'hotels_small']
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
                                  'test1_small': 1, # 7390 is wrong
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
                            'test4': 41437}
                 }

DATASET_MEANS = {'hotels': [0.5805, 0.5247, 0.4683],
                 'hotels_small': [0.5805, 0.5247, 0.4683],
                 'cub': None}

DATASET_STDS = {'hotels': [0.2508, 0.2580, 0.2701],
                'hotels_small': [0.2508, 0.2580, 0.2701],
                'cub': None}


def get_features_and_labels(args, model, loader):
    features = []
    labels = []

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
            labels.append(lbl)

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

    if args.get('trained_with_mltp_gpu'):
        net = torch.nn.DataParallel(net)

    net.load_state_dict(checkpoint['model_state_dict'])

    if args.get('cuda'):
        net = net.cuda()
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

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-cuda', '--cuda', default=False, action='store_true')
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

    parser.add_argument('--metric', default='cosine', choices=['cosine', 'euclidean'])

    args = parser.parse_args()

    all_data = []

    kset = [int(k) for k in args.kset]

    dataset_config = utils.load_config(os.path.join(args.config_path, args.dataset + '.json'))

    all_args = utils.Global_Config_File(args=args, config_file=dataset_config)

    print(str(all_args))

    if all_args.get('name') is None:
        raise Exception('Provide --name')

    utils.make_dirs(os.path.join(all_args.get('eval_log_path'), 'cache/'))

    cache_path = os.path.join(all_args.get('eval_log_path'), 'cache', all_args.get('name'))

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

    for idx, (features, labels) in enumerate(all_data, 1):

        if features.shape[1] != all_args.get('emb_size'):
            if all_args.get('pca_to_dim'):
                features = pca(features, all_args.get('emb_size'))
            else:
                raise Exception(
                    f'--pca_to_dim is set to False and feature dim {features.shape[1]} not equal to expected dim {all_args.get("emb_size")}')


        print('*' * 10)
        print(f'{idx}: Calc AUC_ROC')
        if all_args.get('hard_neg'):
            a2n = utils.get_a2n(ordered_lbls_idxs[idx - 1][0], ordered_lbls_idxs[idx - 1][1], labels)
        else:
            a2n = None
        auc = utils.calc_auroc(features, torch.tensor(labels), anch_2_hardneg_idx=a2n)
        print(f'{idx}: AUC_ROC:', auc)
        results += f'\n\n{idx}: AUC_ROC: {auc}\n\n'
        results += '*' * 20


        print(f'{idx}: Calc Recall at {kset}')
        rec = utils.get_recall_at_k(features, labels,
                                    metric='cosine',
                                    sim_matrix=None,
                                    Kset=kset)
        # = evaluate_recall_at_k(features, labels, Kset=kset, metric=all_args.get('metric'))
        print(kset)
        print(rec)
        results += f'{idx}: Calc Recall at {kset}' + '\n' + str(kset) + '\n' + str(rec) + '\n'


    if all_args.get('hard_neg'):
        hard_neg_string = '_HN'
    else:
        hard_neg_string = ''
    with open(os.path.join(all_args.get('eval_log_path'), all_args.get('name') + f"{hard_neg_string}.txt"), 'w') as f:
        f.write(results)

    if all_args.get('force') or \
        not os.path.exists(cache_path):
        utils.make_dirs(cache_path)
        for i in range(0, all_args.get('num_of_dataset')):

            val_set_name = fix_name(all_args.get(f'all_{all_args.get("eval_mode")}_files')[i])
            emb_data, lbl_data = all_data[i]

            emb_data_path = os.path.join(cache_path, val_set_name + "_embs.npy")
            lbl_data_path = os.path.join(cache_path, val_set_name + "_lbls.npy")

            np.save(emb_data_path, emb_data)
            np.save(lbl_data_path, lbl_data)


if __name__ == '__main__':
    main()
