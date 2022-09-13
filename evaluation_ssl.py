import argparse
import json
import os
import pickle

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from tqdm import tqdm
import ssl_utils, arg_parser
import wandb
import torch.nn as nn

import ssl_model

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


DATASET_SIZES = {'cars': {'test': 8131},
                'cub-val': {'val': 2975},
                 'cub-test': {'test': 5924},
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
                                  'test4': 20220},
                'imagenet': {'test': 8131,
                            'val': 2975},
                 }

DATASET_MEANS = {'hotels': [0.5805, 0.5247, 0.4683],
                 'hotels_small': [0.5805, 0.5247, 0.4683],
                 'hotelid-val': [0.4620, 0.3980, 0.3292],
                 "hotelid-test": [0.4620, 0.3980, 0.3292],
                 'cub-val': None,
                 'cub-test': None,
                 'imagenet': None}

DATASET_STDS = {'hotels': [0.2508, 0.2580, 0.2701],
                'hotels_small': [0.2508, 0.2580, 0.2701],
                'hotelid-val': [0.2619, 0.2529, 0.2460],
                'hotelid-test': [0.2619, 0.2529, 0.2460],
                'cub-val': None,
                'cub-test': None,
                'imagenet': None}


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
            dict2[k] = list([dict2[k]])

        return dict2

    assert len(dict1) == len(dict2)

    for k, v in dict2.items():
        dict1[k].extend(list([v]))

    return dict1


def main():
    args = arg_parser.get_args_ssl_eval()
    
    all_data = []

    kset = [int(k) for k in args.kset]

    dataset_config = utils.load_json(os.path.join(
        args.config_path, args.dataset + '.json'))


    all_args_def = utils.Global_Config_File(args=args, config_file=dataset_config, init_tb=not args.wandb)
    all_args_def_ns = all_args_def.get_namespace()

    utils.seed_all(all_args_def_ns.seed)

    # Pass them to wandb.init
    # model_name = utils.get_model_name(all_args_def)
    if all_args_def.get('wandb'):
        wandb.init(config=all_args_def_ns, dir=os.path.join(all_args_def.get('log_path'), 'wandb/'))

        # Access all hyperparameter values through wandb.config
        all_args_ns_new = wandb.config
        all_args = utils.Global_Config_File(
            config_file={}, args=all_args_ns_new, init_tb=True)
    else:
        all_args = all_args_def

    logger = utils.get_logger()
    print(all_args)
    logger.info(all_args)


    print('successfull!')
    print(str(all_args))
    utils.seed_all(all_args.get('seed'))



    eval_log_path = os.path.join(all_args.get('eval_log_path'),
                                 f'{all_args.get("eval_mode").upper()}_{all_args.get("dataset")}')

    utils.make_dirs(os.path.join(eval_log_path, 'ssl_cache/'))
    net = None

    encoder = ssl_utils.get_backbone(all_args.get('backbone'),
                                 pretrained=(all_args.get('method_name') == 'default'))

    if all_args.get('ssl'):
        class_num = 0
    else:
        class_num = all_args.get('nb_classes')


    if all_args.get('checkpoint'):
        if all_args.get('name') is None:
            raise Exception('Provide --name')    
        checkpoint_name = os.path.split(all_args.get('checkpoint'))[1].split('.')[0]
        cache_path = os.path.join(eval_log_path, 'ssl_cache', f'{checkpoint_name}_' + all_args.get('name'))
    else:
        checkpoint_name = f"{all_args.get('backbone')}_{all_args.get('method_name')}"
        cache_path = os.path.join(eval_log_path, 'ssl_cache', f'{checkpoint_name}')


    pairwise_label_list = []
    for i in range(0, all_args.get('num_of_dataset')):
        if all_args.get('pairwise_lbls'):
            pairwise_lbl = all_args.get(f'all_pairwise_{all_args.get("eval_mode")}_labels')[i]
            pairwise_label_list.append(np.load(pairwise_lbl))
        else:
            pairwise_label_list.append(None)

    if os.path.exists(cache_path):
        #  and \
        #     not all_args.get('force_update'):

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

        if all_args.get('checkpoint'):
            net = ssl_model.SSL_MODEL(backbone=encoder,
                        emb_size=2048,
                        num_classes=class_num,
                        freeze_backbone=True,
                        projector_sclaing=all_args.get('ssl_projector_scale')) # freezes backbone when Linear Probing
    
            net, epoch = utils.load_model(net, checkpoint_path=all_args.get('checkpoint'))
            net = net.encoder # just keep the feature extracting module
        else:
            net = ssl_utils.get_backbone(all_args.get('backbone'),
                                        pretrained=(all_args.get('method_name') == 'default'))

            net = ssl_utils.load_ssl_weight_to_model(model=net,
                                                    method_name=all_args.get(
                                                        'method_name'),
                                                    arch_name=all_args.get('backbone'))
            net.fc = nn.Identity()


        if all_args.get('cuda'):
            net = net.cuda()
    
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

    if all_args.get('pairwise_lbls'):
        pairwise_lbls_string = '_PWL'
        if all_args.get('eval_pairwise_hard_neg'):
            pairwise_lbls_string += '_HN'
    else:
        pairwise_lbls_string = ''

    different_seeds_auc = {}
    different_recs = {}
    # if all_args.get('eval_metric').upper() == 'AUC':
    #     different_seeds_results = {}
    if all_args.get('eval_metric').upper() == 'AUC' or all_args.get('eval_metric').upper() == 'CONRET':
        seeds = [all_args.get('seed') * (i + 1) for i in range(all_args.get('run_times'))]
    else:
        seeds = [all_args.get('seed')]


    controlled_recall_files = all_args.get('controlled_recall_files')

    for j, seed in enumerate(seeds):
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
                pw_lbls = pairwise_label_list[idx - 1]
                if pw_lbls is not None:
                    pw_lbls = torch.tensor(pw_lbls)
                auc, t_and_p_labels = utils.calc_auroc(features,
                                                       torch.tensor(labels),
                                                       anch_2_hardneg_idx=a2n,
                                                       pairwise_labels=pw_lbls,
                                                       pairwise_hard_neg=all_args.get('eval_pairwise_hard_neg'))
                if idx not in auc_predictions.keys():
                    auc_predictions[idx] = {}

                auc_predictions[idx] = add_dicts(auc_predictions[idx], t_and_p_labels)

                print(f'{idx}: AUC_ROC:', auc)
                results += f'\n\n{idx}: AUC_ROC: {auc}\n\n'
                if idx not in different_seeds_auc.keys():
                    different_seeds_auc[idx] = []

                different_seeds_auc[idx].extend([auc])
                results += '*' * 20 + '\n'

            elif all_args.get('eval_metric').upper() == 'CONRET':
                mask_path = controlled_recall_files[idx - 1] + f'{j}.csv'
                idx_mask = np.array(pd.read_csv(os.path.join(all_args.get('dataset_path') + mask_path)).image)

                mask_name = fix_name(mask_path)

                features = features[idx_mask, :]
                labels = labels[idx_mask]

                print(f'{idx}: Calc Recall at {kset}')
                rec = utils.get_recall_at_k(features, labels,
                                            metric='cosine',
                                            sim_matrix=None,
                                            Kset=kset)
                # = evaluate_recall_at_k(features, labels, Kset=kset, metric=all_args.get('metric'))
                print(kset)
                print(rec)

                if idx not in different_recs:
                    different_recs[idx] = {}

                results += f'{mask_name} - {idx}: Calc Recall at {kset}' + '\n' + str(kset) + '\n' + str(rec) + '\n'
                results += '*' * 20 + '\n\n'

                different_recs[idx] = add_dicts(different_recs[idx], rec)
                #
                # if len(different_recs[idx]) > 0:
                #     for k, v in rec.items():
                #         different_recs[idx][k].append(v)
                # else:
                #     different_recs[idx].append(rec)

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
            fig.suptitle(f'{all_args.get("name")} {hard_neg_string}{pairwise_lbls_string}')
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
            plt.title(f'{all_args.get("name")} {hard_neg_string}{pairwise_lbls_string}\nTest {title_name}: {t_and_p_labels[1]:.3} +- {t_and_p_labels[2]:.3}')

        plt.savefig(os.path.join(eval_log_path, f'{checkpoint_name}_' + all_args.get('name') + f"{hard_neg_string}{pairwise_lbls_string}_aucplot.pdf"))
        plt.clf()

    if all_args.get('eval_metric').upper() == 'CONRET':
        print(different_recs)
        for k1, v1 in different_recs.items():
            mean_stdvs[k1] = {}
            for k, v in v1.items():
                mean_stdvs[k1][k] = (np.mean(v), np.std(v))
                v1[k] = (v1[k], np.mean(v), np.std(v))
                results += f"RECs for Eval split {k1}-{k}: {v}\n\n"

        results += f"\n***\nMean REC and Std Dev over {len(seeds)} files: \n{str(json.dumps(mean_stdvs))}\n\n"


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
        fig.suptitle(f'{all_args.get("name")} {hard_neg_string}{pairwise_lbls_string}')
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
        plt.savefig(os.path.join(eval_log_path, f'{checkpoint_name}_' + all_args.get('name') + f"{hard_neg_string}{pairwise_lbls_string}_scatter_{norm_string}{all_args.get('project_labels_start')}-{all_args.get('project_no_labels')}.pdf"))
        plt.clf()

        scatter_text_to_write = ''
        for k, v in drawn_labels.items():
            scatter_text_to_write += f'Test {k}: {v}' + '\n'

        with open(os.path.join(eval_log_path, f'{checkpoint_name}_' + all_args.get('name') + f"{hard_neg_string}{pairwise_lbls_string}_scatter_{norm_string}{all_args.get('project_labels_start')}-{all_args.get('project_no_labels')}.txt"), 'w') as f:
            f.write(scatter_text_to_write)

    with open(os.path.join(eval_log_path, f'{checkpoint_name}_' + all_args.get('name') + '_m' + all_args.get('eval_metric').upper() + f"{hard_neg_string}{pairwise_lbls_string}.txt"), 'w') as f:
        f.write(results)


if __name__ == '__main__':
    main()
