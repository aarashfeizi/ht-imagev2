from re import L
from tkinter import E
import arg_parser
import utils
import ssl_utils
import wandb
import os
import sys

import ssl_model

import torch
import losses
from trainer import Trainer

import numpy as np
import torch.nn as nn


def main():
    args = arg_parser.get_args_ssl()

    dataset_config = utils.load_json(os.path.join(
        args.config_path, args.dataset + '.json'))

    all_args_def = utils.Global_Config_File(
        args=args, config_file=dataset_config, init_tb=not args.wandb)
        
    all_args_def_ns = all_args_def.get_namespace()

    utils.seed_all(all_args_def_ns.seed)

    # Pass them to wandb.init
    # model_name = utils.get_model_name(all_args_def)
    if all_args_def.get('wandb'):
        wandb.init(config=all_args_def_ns, dir=os.path.join(
            all_args_def.get('log_path'), 'wandb/'))

        # Access all hyperparameter values through wandb.config
        all_args_ns_new = wandb.config
        all_args = utils.Global_Config_File(
            config_file={}, args=all_args_ns_new, init_tb=True)
    else:
        all_args = all_args_def

    logger = utils.get_logger()
    print(all_args)
    logger.info(all_args)

    train_transforms, train_transforms_names = utils.TransformLoader(
        all_args).get_composed_transform(mode='train', color_jitter=all_args.get('color_jitter'))

    val_transforms, val_transforms_names = utils.TransformLoader(
        all_args).get_composed_transform(mode='val')

    encoder = ssl_utils.get_backbone(all_args.get('backbone'),
                                 pretrained=(all_args.get('method_name') == 'default'))

    encoder = ssl_utils.load_ssl_weight_to_model(model=encoder,
                                             method_name=all_args.get(
                                                 'method_name'),
                                             arch_name=all_args.get('backbone'))
    if all_args.get('ssl'):
        class_num = 0
    else:
        class_num = all_args.get('nb_classes')
    
    net = ssl_model.SSL_MODEL(backbone=encoder,
                                emb_size=2048,
                                num_classes=class_num,
                                freeze_backbone=all_args.get('backbone_mode') == 'LP',
                                projector_sclaing=all_args.get('ssl_projector_scale')) # freezes backbone when Linear Probing

    print('successfull!')

    print('Train transforms: ', train_transforms_names)
    print('Val transforms: ', val_transforms_names)

    if all_args.get('loss') == 'CE':
        sampler_mode = 'classification'
    elif all_args.get('loss') == 'infonce':
        sampler_mode = 'ssl'                            
    else:
        raise Exception('Loss not supported!!')
    
    ssl_kwargs = {}
    if all_args.get('ssl'):
        if all_args.get('local_global_aug'):
            ssl_transforms, ssl_transforms_names = utils.TransformLoader(all_args, scale=[0.8, 1.0], rotate=90).get_composed_transform(mode='train-ssl')
            print('Train-SSL transforms: ', ssl_transforms_names)

            ssl_kwargs = {'random_crop_resize_transform': ssl_transforms[0],
                'mask_in_transform':ssl_transforms[1],
                'rest_transform': ssl_transforms[2]}

        ssl_kwargs['ssl_aug'] = all_args.get('local_global_aug')


    train_loader = utils.get_data(all_args, mode='train',
                                    transform=train_transforms,
                                    #   sampler_mode='kbatch',
                                    sampler_mode=sampler_mode,
                                    pairwise_labels=all_args.get('train_with_pairwise'),
                                    ssl=all_args.get('ssl'),
                                    **ssl_kwargs)

    train_lbl2idx = train_loader.dataset.get_lbl2idx()
    train_ohe = train_loader.dataset.get_onehotencoder()

    # if train_transforms_swap is not None:
    #     train_loader = utils.get_data(all_args, mode='train', transform=train_transforms, sampler_mode='kbatch')

    val2_loader = None
    val2_db_loader = None

    if sampler_mode == 'classification':
        val_classification_loader = utils.get_data(
            all_args, mode='val', transform=val_transforms, 
            sampler_mode='classification', lbl2idx=train_lbl2idx, 
            onehotencoder=train_ohe)
    else:
        val_classification_loader = None

    val_loader = utils.get_data(
        all_args, mode='val', transform=val_transforms, sampler_mode='balanced_triplet')
    val2_loader = utils.get_data(
        all_args, mode='val2', transform=val_transforms, sampler_mode='balanced_triplet')

    val_db_loader = utils.get_data(
        all_args, mode='val', transform=val_transforms, sampler_mode='db')
    val2_db_loader = utils.get_data(
        all_args, mode='val2', transform=val_transforms, sampler_mode='db')

    val_loader_pairwise = None
    val2_loader_pairwise = None
    val_db_loader_pairwise = None
    val2_db_loader_pairwise = None

    if all_args.get('eval_with_pairwise'):
        val_loader_pairwise = utils.get_data(all_args, mode='val_pairwise', transform=val_transforms, sampler_mode='balanced_triplet',
                                             pairwise_labels=all_args.get('eval_with_pairwise'))
        val2_loader_pairwise = utils.get_data(all_args, mode='val2_pairwise', transform=val_transforms, sampler_mode='balanced_triplet',
                                              pairwise_labels=all_args.get('eval_with_pairwise'))
        val_db_loader_pairwise = utils.get_data(all_args, mode='val_pairwise', transform=val_transforms, sampler_mode='db',
                                                pairwise_labels=all_args.get('eval_with_pairwise'))
        val2_db_loader_pairwise = utils.get_data(all_args, mode='val2_pairwise', transform=val_transforms, sampler_mode='db',
                                                 pairwise_labels=all_args.get('eval_with_pairwise'))

    test_loader = None
    if args.test:
        test_loader = utils.get_data(all_args, mode='test', transform=val_transforms, sampler_mode='balanced_triplet',
                                     pairwise_label=False)
        test_loader_pairwise = utils.get_data(all_args, mode='test-pairwise', transform=val_transforms, sampler_mode='balanced_triplet',
                                              pairwise_label=all_args.get('eval_with_pairwise'))

    loss = losses.get_loss(all_args)

    if all_args.get('cuda'):
        if all_args.get('gpu_ids') != '':
            os.environ["CUDA_VISIBLE_DEVICES"] = all_args.get('gpu_ids')
            logger.info(f"use gpu: {all_args.get('gpu_ids')} to train.")

        if torch.cuda.device_count() > 1:
            logger.info(
                f'torch.cuda.device_count() = {torch.cuda.device_count()}')
            net = nn.DataParallel(net)
        logger.info(f'Let\'s use {torch.cuda.device_count()} GPUs!')
        net.cuda()
        # if len(net.projs) != 0:
        #     for p in net.projs:
        #         p.cuda()
        loss.cuda()

    all_val_loader_names = ['val', 'val2']
    all_val_loaders = [val_loader, val2_loader]
    all_val_db_loaders = [val_db_loader, val2_db_loader]
    val_coef = 1
    if all_args.get('eval_with_pairwise'):
        val_coef = 2
        all_val_loader_names = ['val', 'val_pairwise', 'val2', 'val2_pairwise']
        all_val_loaders = [val_loader, val_loader_pairwise,
                           val2_loader, val2_loader_pairwise]
        all_val_db_loaders = [
            val_db_loader, val_db_loader_pairwise, val2_db_loader, val2_db_loader_pairwise]

    val_loaders_dict = {}
    val_db_loaders_dict = {}

    for i in range(all_args.get('number_of_val') * val_coef):
        val_loaders_dict[all_val_loader_names[i]] = all_val_loaders[i]
        val_db_loaders_dict[all_val_loader_names[i]] = all_val_db_loaders[i]

    if not all_args.get('test'):  # training
        trainer = Trainer(all_args, loss=loss, train_loader=train_loader,
                            val_loaders=val_loaders_dict,
                            val_db_loaders=val_db_loaders_dict,
                            force_new_dir=True,
                            optimizer=all_args.get('optimizer'))

        if all_args.get('loss') == 'CE':
            trainer.set_val_classification_loader(val_classification_loader)

        trainer.train(net, val=(not all_args.get('no_validation')))

    else:  # testing
        assert os.path.exists(all_args.get('ckpt_path'))
        trainer = Trainer(all_args, loss=loss, train_loader=None, val_loaders=val_loaders_dict,
                          val_db_loaders=val_db_loaders_dict, force_new_dir=False)
        net, epoch = utils.load_model(
            net, os.path.join(all_args.get('ckpt_path')))
        ssl_utils.set_net_to_eval(net)

        for val_name, val_loader in val_loaders_dict.items():
            if val_loader is None:
                continue
            with torch.no_grad():
                val_losses, val_acc, val_auroc_score = trainer.validate(
                    net, val_name, val_loader)
                embeddings, classes = trainer.get_embeddings(
                    net, data_loader=val_db_loaders_dict[val_name])

                r_at_k_score = utils.get_recall_at_k(embeddings, classes,
                                                     metric='cosine',
                                                     sim_matrix=None)

                all_val_losses = {lss_name: (lss / len(val_loader))
                                  for lss_name, lss in val_losses.items()}

                print(f'VALIDATION from saved in epoch {epoch}-> {val_name}_loss: ', all_val_losses,
                      f', {val_name}_acc: ', val_acc,
                      f', {val_name}_auroc: ', val_auroc_score,
                      f', {val_name}_R@K: ', r_at_k_score)


if __name__ == '__main__':
    main()
