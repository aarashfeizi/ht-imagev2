import arg_parser, utils, ssl_utils
import wandb
import os, sys



def main():
    args = arg_parser.get_args_ssl()
    dataset_config = utils.load_json(os.path.join(args.config_path, args.dataset + '.json'))

    all_args_def = utils.Global_Config_File(args=args, config_file=dataset_config, init_tb=False)
    all_args_def_ns = all_args_def.get_namespace()

    utils.seed_all(all_args_def_ns.seed)

    # Pass them to wandb.init
    # model_name = utils.get_model_name(all_args_def)
    if all_args_def.get('wandb'):
        wandb.init(config=all_args_def_ns, dir=os.path.join(all_args_def.get('log_path'), 'wandb/'))

        # Access all hyperparameter values through wandb.config
        all_args_ns_new = wandb.config
        all_args = utils.Global_Config_File(config_file={}, args=all_args_ns_new, init_tb=True)
    else:
        all_args = all_args_def

    logger = utils.get_logger()
    print(all_args)
    logger.info(all_args)

    train_transforms, train_transforms_names = utils.TransformLoader(all_args).get_composed_transform(mode='train')
    
    val_transforms, val_transforms_names = utils.TransformLoader(all_args).get_composed_transform(mode='val')

    model = ssl_utils.get_backbone(all_args.get('backbone'),
                                            pretrained=(all_args.get('method_name') == 'default'))


    model = ssl_utils.load_ssl_weight_to_model(model=model,
                                                method_name=all_args.get('method_name'),
                                                arch_name=all_args.get('backbone'))

    print('successfull!')

    




if __name__ == '__main__':
    main()