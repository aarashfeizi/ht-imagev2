import os

import numpy as np
import torch
import torch.nn as nn

import arg_parser
import losses
import model
import utils
from trainer import Trainer

import wandb

def initiate_wandb(args):
    model_name = utils.get_model_name(args)
    wandb.init(project=f"{model_name}")
    wandb_config = utils.get_wandb_config(args)
    wandb.config = wandb_config


def main():
    args = arg_parser.get_args()
    dataset_config = utils.load_json(os.path.join(args.config_path, args.dataset + '.json'))

    all_args = utils.Global_Config_File(args=args, config_file=dataset_config)
    utils.seed_all(all_args.get('seed'))

    initiate_wandb(all_args)

    logger = utils.get_logger()
    print(args)
    logger.info(args)

    train_transforms, train_transforms_names = utils.TransformLoader(all_args).get_composed_transform(mode='train')
    # train_transforms_swap = None
    # if type(train_transforms) is list:
    #     train_transforms_swap = train_transforms[1]
    #     train_transforms = train_transforms[0]

    val_transforms, val_transforms_names = utils.TransformLoader(all_args).get_composed_transform(mode='val')

    print('Train transforms: ', train_transforms_names)
    print('Val transforms: ', val_transforms_names)

    train_loader = utils.get_data(all_args, mode='train',
                                  transform=train_transforms,
                                  sampler_mode='kbatch',
                                  pairwise_labels=all_args.get('train_with_pairwise'))

    # if train_transforms_swap is not None:
    #     train_loader = utils.get_data(all_args, mode='train', transform=train_transforms, sampler_mode='kbatch')

    val2_loader = None
    val2_db_loader = None

    if not all_args.get('hard_triplet'):
        val_loader = utils.get_data(all_args, mode='val', transform=val_transforms, sampler_mode='balanced_triplet')
        val2_loader = utils.get_data(all_args, mode='val2', transform=val_transforms, sampler_mode='balanced_triplet')

    else:
        if all_args.get('ordered_idxs') is not None:
            ordered_idxs = np.load(all_args.get('ordered_idxs'))
            ordered_lbls = np.load(all_args.get('ordered_lbls'))
        else:
            ordered_idxs = None
            ordered_lbls = None

        val_loader = utils.get_data(all_args, mode='val',
                                    transform=val_transforms,
                                    sampler_mode='hard_triplet',
                                    ordered_idxs=ordered_idxs,
                                    ordered_lbls=ordered_lbls)

    val_loader_4heatmap = utils.get_data(all_args, mode='val', transform=val_transforms, sampler_mode='heatmap')
    val_loader_4_2xheatmap = utils.get_data(all_args, mode='val', transform=val_transforms, sampler_mode='heatmap2x', triplet_path=all_args.get('triplet_path_heatmap2x'))

    val_db_loader = utils.get_data(all_args, mode='val', transform=val_transforms, sampler_mode='db')
    val2_db_loader = utils.get_data(all_args, mode='val2', transform=val_transforms, sampler_mode='db')

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

    net = model.get_top_module(args=all_args)

    loss = losses.get_loss(all_args)

    if all_args.get('cuda'):
        if all_args.get('gpu_ids') != '':
            os.environ["CUDA_VISIBLE_DEVICES"] = all_args.get('gpu_ids')
            logger.info(f"use gpu: {all_args.get('gpu_ids')} to train.")

        if torch.cuda.device_count() > 1:
            logger.info(f'torch.cuda.device_count() = {torch.cuda.device_count()}')
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
        all_val_loaders = [val_loader, val_loader_pairwise, val2_loader, val2_loader_pairwise]
        all_val_db_loaders = [val_db_loader, val_db_loader_pairwise, val2_db_loader, val2_db_loader_pairwise]

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

        if all_args.get('draw_heatmaps'):
            trainer.set_heatmap_loader(val_loader_4heatmap)
        if all_args.get('draw_heatmaps2x'):
            trainer.set_heatmap2x_loader(val_loader_4_2xheatmap)


        trainer.train(net, val=(not all_args.get('no_validation')))

    else:  # testing
        assert os.path.exists(all_args.get('ckpt_path'))
        trainer = Trainer(all_args, loss=loss, train_loader=None, val_loaders={'val': val_loader},
                          val_db_loaders={'val': val_db_loader}, force_new_dir=False)
        net, epoch = utils.load_model(net, os.path.join(all_args.get('ckpt_path')))
        net.encoder.set_to_eval()

        if all_args.get('draw_heatmaps'):
            trainer.set_heatmap_loader(val_loader_4heatmap)
            trainer.draw_heatmaps(net)

        if all_args.get('draw_heatmaps2x'):
            trainer.set_heatmap2x_loader(val_loader_4_2xheatmap)
            trainer.draw_heatmaps2x(net)

        for val_name, val_loader in val_loaders_dict.items():
            if val_loader is None:
                continue
            with torch.no_grad():
                val_losses, val_acc, val_auroc_score = trainer.validate(net, val_name, val_loader)
                embeddings, classes = trainer.get_embeddings(net, data_loader=val_db_loaders_dict[val_name])

                r_at_k_score = utils.get_recall_at_k(embeddings, classes,
                                                     metric='cosine',
                                                     sim_matrix=None)

                all_val_losses = {lss_name: (lss / len(val_loader)) for lss_name, lss in val_losses.items()}

                print(f'VALIDATION from saved in epoch {epoch}-> {val_name}_loss: ', all_val_losses,
                      f', {val_name}_acc: ', val_acc,
                      f', {val_name}_auroc: ', val_auroc_score,
                      f', {val_name}_R@K: ', r_at_k_score)


if __name__ == '__main__':
    main()
