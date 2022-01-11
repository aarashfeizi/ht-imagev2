import backbones
import losses
import model
import utils
import torch
import os, sys
import arg_parser

import timm

from trainer import Trainer
import torch.nn as nn


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

    train_loader = utils.get_data(all_args, mode='train', transform=train_transforms)
    val_loader = utils.get_data(all_args, mode='val', transform=val_transforms)
    test_loader = None
    if args.test:
        test_loader = utils.get_data(all_args, mode='test', transform=val_transforms)

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

    trainer = Trainer(all_args, loss=loss, train_loader=train_loader, val_loader=val_loader)

    trainer.train(net, val=True)



if __name__ == '__main__':
    main()