import os

import numpy as np
import torch
import torch.nn as nn

import arg_parser
import losses
import model
import utils
from trainer import Trainer


def main():
    args = arg_parser.get_args()
    dataset_config = utils.load_config(os.path.join(args.config_path, args.dataset + '.json'))

    all_args = utils.Global_Config_File(args=args, config_file=dataset_config)
    utils.seed_all(all_args.get('seed'))

    logger = utils.get_logger()
    print(args)
    logger.info(args)

    train_transforms, train_transforms_names = utils.TransformLoader(all_args).get_composed_transform(mode='train')
    val_transforms, val_transforms_names = utils.TransformLoader(all_args).get_composed_transform(mode='val')

    print('Train transforms: ', train_transforms_names)
    print('Val transforms: ', val_transforms_names)

    train_loader = utils.get_data(all_args, mode='train', transform=train_transforms, sampler_mode='kbatch')
    if not all_args.get('hard_triplet'):
        val_loader = utils.get_data(all_args, mode='val', transform=val_transforms, sampler_mode='balanced_triplet')

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
    val_db_loader = utils.get_data(all_args, mode='val', transform=val_transforms, sampler_mode='db')
    test_loader = None
    if args.test:
        test_loader = utils.get_data(all_args, mode='test', transform=val_transforms, sampler_mode='balanced_triplet')

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
        loss.cuda()

    if not all_args.get('test'):  # training
        trainer = Trainer(all_args, loss=loss, train_loader=train_loader, val_loader=val_loader,
                          val_db_loader=val_db_loader, force_new_dir=True)
        trainer.train(net, val=(not all_args.get('no_validation')))

    else:  # testing
        assert os.path.exists(all_args.get('ckpt_path'))
        trainer = Trainer(all_args, loss=loss, train_loader=None, val_loader=val_loader,
                          val_db_loader=val_db_loader, force_new_dir=False)
        net, epoch = utils.load_model(net, os.path.join(all_args.get('ckpt_path')))
        net.encoder.set_to_eval()

        if all_args.get('draw_heatmaps'):
            trainer.set_heatmap_loader(val_loader_4heatmap)
            trainer.draw_heatmaps(net)

        with torch.no_grad():
            val_losses, val_acc, val_auroc_score = trainer.validate(net)
            embeddings, classes = trainer.get_embeddings(net)

            r_at_k_score = utils.get_recall_at_k(embeddings, classes,
                                                 metric='cosine',
                                                 sim_matrix=None)

            all_val_losses = {lss_name: (lss / len(trainer.val_loader)) for lss_name, lss in val_losses.items()}

            print(f'VALIDATION from saved in epoch {epoch}-> val_loss: ', all_val_losses,
                  f', val_acc: ', val_acc,
                  f', val_auroc: ', val_auroc_score,
                  f', val_R@K: ', r_at_k_score)


if __name__ == '__main__':
    main()
