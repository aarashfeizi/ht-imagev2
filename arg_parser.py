import argparse

DATASET_LIST = ['cub', 'hotels', 'hotels_small', 'cub', 'hotelid-val', 'hotelid-test']

LOSSES_LIST = ['pnpp',
               'bce',
               'hardbce',
               'trpl',
               'supcon',
               'proxy_nca',
               'proxy_anchor',
               'arcface',
               'angular',
               'circle',
               'multisim',
               'lifted',
               'softtriple',
               'CE',
               'infonce']

BACKBONE_LIST = ['resnet50', 'resnet18', 'deit_small']

METRIC_LIST = ['cosine',
               'euclidean',
               'mlp']

OPTIMIZER_LIST = ['adam', 'sam', 'sgd']

HYPER_PARAMS = ['optimizer',
                'learning_rate',
                'new_lr_coef',
                'bb_learning_rate',
                'weight_decay',
                'epochs',
                'batch_size',
                'emb_size',
                'multlayer_emb',
                'eq_layer_contrib',
                'ml_self_att',
                'train_with_pairwise',
                'eval_with_pairwise',
                'ml_self_att_head_number',
                'ml_self_att_layers_to_use',
                'only_att',
                'num_inst_per_class',
                'k_inc_freq',
                'k_dec_freq',
                'loss',
                'cov',
                'cov_static_mean',
                'cov_coef',
                'var_coef',
                'swap_coef',
                'with_bce',
                'bce_weight',
                'aug_swap',
                'aug_swap_prob',
                'aug_mask_prob',
                'temperature',
                'lnorm',
                'metric',
                'backbone']

SSL_MODELS = ['default',
                'swav',
                'simsiam',
                'byol',
                'unigrad',
                'simclr',
                'vicreg',
                'dino',
                'barlow',
                'densecl',
                'densecl_CC'] # _CC is pretrained on COCO (as opposed to ImageNet)

eval_dataset_choices = ['cars', 'cub', 'hotels', 'hotels_small', 'hotelid-val', 'hotelid-test']
eval_BASELINE_MODELS = ['ours',
                   'softtriple',
                   'proxy-anchor',
                   'supcontrastive',
                   'proxyncapp',
                   'htv2',
                   'ms',
                   'resnet50']

def get_args():
    parser = argparse.ArgumentParser()

    # logistics
    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--gpu_ids', default='', help="gpu ids used to train")  # before: default="0,1,2,3"
    parser.add_argument('--seed', type=int, default=402, help="set random seed")
    parser.add_argument('--config_path', default='config/', help="config_path for datasets")
    parser.add_argument('--project_path', default='./', help="project_path")
    parser.add_argument('--xname', default='', help="extra name to be added to name")
    parser.add_argument('--save_model', default=False, action='store_true', help="save model or not")
    parser.add_argument('--no_validation', default=False, action='store_true', help="save model or not")
    parser.add_argument('--early_stopping_tol', default=2, type=int, help="early stopping tolerance on validation")


    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--draw_heatmaps', default=False, action='store_true')
    parser.add_argument('--draw_heatmaps2x', default=False, action='store_true')
    parser.add_argument('--triplet_path_heatmap2x', default='jsons/triplets_for_heatmap2x.json')

    parser.add_argument('--workers', type=int, default=10, help="number of workers for data loading")
    parser.add_argument('--pin_memory', default=False, action='store_true', help="pinning memory for data loading")

    # data
    parser.add_argument('--dataset', default='hotels_small', choices=DATASET_LIST)
    parser.add_argument('--number_of_val', default=1, type=int)
    parser.add_argument('--hard_triplet', default=False, action='store_true')

    # learning
    parser.add_argument('--optimizer', default='adam', choices=OPTIMIZER_LIST, help='optimizer to use')
    parser.add_argument('--ckpt_path', default=None, help="path to the checkpoint file")
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--new_lr_coef', type=float, default=1.0)
    parser.add_argument('--bb_learning_rate', type=float, default=0.00001)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--emb_size', type=int, default=512)
    parser.add_argument('--multlayer_emb', default=False, action='store_true')
    parser.add_argument('--eq_layer_contrib', default=False, action='store_true')
    parser.add_argument('--ml_self_att', default=False, action='store_true')
    parser.add_argument('--train_with_pairwise', default=False, action='store_true')
    parser.add_argument('--eval_with_pairwise', default=False, action='store_true')
    parser.add_argument('--ml_self_att_head_number', type=int, default=4)
    parser.add_argument('--ml_self_att_layers_to_use', type=int, default=4)
    parser.add_argument('--only_att', default=False, action='store_true')
    parser.add_argument('--num_inst_per_class', type=int, default=2)
    parser.add_argument('--k_inc_freq', type=int, default=0)
    parser.add_argument('--k_dec_freq', type=int, default=0)
    parser.add_argument('--loss', default='pnpp', choices=LOSSES_LIST)
    parser.add_argument('--cov', default=False, action='store_true')
    parser.add_argument('--cov_static_mean', default=False, action='store_true')
    parser.add_argument('--cov_coef', type=float, default=1.0)
    parser.add_argument('--var_coef', type=float, default=1.0)
    parser.add_argument('--swap_coef', type=float, default=1.0)
    parser.add_argument('--with_bce', default=False, action='store_true')
    parser.add_argument('--bce_weight', type=float, default=1.0)
    parser.add_argument('--aug_swap', type=int, default=1)  # split image into (aug_swap * aug_swap squares) and shuffle them
    parser.add_argument('--aug_swap_prob', type=float, default=0.5)
    parser.add_argument('--aug_mask_prob', type=float, default=-1.0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--scale', type=float, default=3.0)
    parser.add_argument('--metric', default='cosine', choices=METRIC_LIST)
    parser.add_argument('--backbone', default='resnet50', choices=BACKBONE_LIST)
    parser.add_argument('--lnorm', default=False, action='store_true', help="Layer norm BEFORE creating embeddings")

    # loss specific defaults
    parser.add_argument('--LOSS_lr', type=float, default=None)
    parser.add_argument('--LOSS_margin', type=float, default=None)
    parser.add_argument('--LOSS_temp', type=float, default=None)
    parser.add_argument('--NCA_scale', type=float, default=None)
    parser.add_argument('--LOSS_alpha', type=float, default=None)
    parser.add_argument('--MS_beta', type=float, default=None)
    parser.add_argument('--MS_base', type=float, default=None)
    parser.add_argument('--LIFT_negmargin', type=float, default=None)
    parser.add_argument('--LIFT_posmargin', type=float, default=None)
    parser.add_argument('--ARCFACE_scale', type=float, default=None)
    parser.add_argument('--CIR_m', type=float, default=None)
    parser.add_argument('--CIR_gamma', type=float, default=None)
    parser.add_argument('--SOFTTRPL_cpc', type=int, default=None)
    parser.add_argument('--SOFTTRPL_lambda', type=float, default=None)
    parser.add_argument('--SOFTTRPL_gamma', type=float, default=None)

    # 'proxy_nca': ProxyNCALoss,  # softmax_scale=1,
    # 'proxy_anchor': ProxyAnchorLoss,  # num_classes, embedding_size, margin = 0.1, alpha = 32
    # 'arcface': ArcFaceLoss,  # num_classes, embedding_size, margin=28.6, scale=64,
    # 'angular': AngularLoss,  # alpha=40
    # 'circle': CircleLoss,  # m=0.4, gamma=80,
    # 'trpl': TripletMargin, # margin=0.05
    # 'supcon': SupConLoss, # temperature=0.1
    #     'multisim': pml_losses.MultiSimilarityLoss, # alpha=2, beta=50, base=0.5
    #     'lifted': pml_losses.LiftedStructureLoss # neg_margin=1, pos_margin=0,
    # 'softtriple': centers_per_class = 10, la = 20, gamma = 0.1, margin = 0.01


    args = parser.parse_args()

    return args

def get_args_eval():
    parser = argparse.ArgumentParser()

    parser.add_argument('-cuda', '--cuda', default=False, action='store_true')
    parser.add_argument('-seed', '--seed', default=402, type=int)
    parser.add_argument('--run_times', default=1, type=int)
    parser.add_argument('-trained_with_mltp_gpu', '--trained_with_mltp_gpu', default=False, action='store_true')
    parser.add_argument('--eval_mode', default='val', help="val or test", choices=['val', 'test', 'train'])

    parser.add_argument('--pairwise_lbls', default=False, action='store_true')
    parser.add_argument('--eval_pairwise_hard_neg', default=False, action='store_true')


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

    parser.add_argument('-d', '--dataset', default=None, choices=eval_dataset_choices)
    parser.add_argument('--num_inst_per_class', default=5, type=int)
    parser.add_argument('--lnorm', default=False, action='store_true')

    parser.add_argument('--config_path', default='config/', help="config_path for datasets")
    parser.add_argument('--project_path', default='./', help="current project path")

    parser.add_argument('-num_of_dataset', '--num_of_dataset', type=int, default=4,
                        help="number of hotels val_datasets to go through")
    parser.add_argument('--baseline', default='proxy-anchor', choices=eval_BASELINE_MODELS)
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
    parser.add_argument('--aug_swap', type=int, default=1) # always set to 1 for no partitioning and swapping

    parser.add_argument('--ml_self_att', default=False, action='store_true')
    parser.add_argument('--ml_self_att_head_number', type=int, default=4)
    parser.add_argument('--ml_self_att_layers_to_use', type=int, default=4)

    parser.add_argument('--eval_metric', default='auc', choices=['auc', 'ret', 'conret'])


    parser.add_argument('--metric', default='cosine', choices=['cosine', 'euclidean'])


    args = parser.parse_args()

    return args

def get_args_for_ordered_distance():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--gpu_ids', default='', help="gpu ids used to train")  # before: default="0,1,2,3"
    parser.add_argument('--config_path', default='config/', help="config_path for datasets")
    parser.add_argument('--project_path', default='./', help="project_path")



    parser.add_argument('--backbone', choices=BACKBONE_LIST)

    parser.add_argument('--dataset', choices=DATASET_LIST)
    parser.add_argument('--workers', type=int, default=10, help="number of workers for data loading")
    parser.add_argument('--pin_memory', default=False, action='store_true', help="pinning memory for data loading")
    parser.add_argument('--top_k', type=int, default=1000, help="number of top returns to save")


    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_inst_per_class', type=int, default=5)


    parser.add_argument('--eval_mode', choices=['val', 'test'])
    parser.add_argument('--num_of_dataset', default=4, type=int)

    args = parser.parse_args()

    return args

def get_args_ssl():
    parser = argparse.ArgumentParser()

    # logistics
    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--gpu_ids', default='', help="gpu ids used to train")  # before: default="0,1,2,3"
    parser.add_argument('--seed', type=int, default=402, help="set random seed")
    parser.add_argument('--config_path', default='config/', help="config_path for datasets")
    parser.add_argument('--project_path', default='./', help="project_path")
    parser.add_argument('--xname', default='', help="extra name to be added to name")
    parser.add_argument('--save_model', default=False, action='store_true', help="save model or not")
    parser.add_argument('--no_validation', default=False, action='store_true', help="save model or not")
    parser.add_argument('--early_stopping_tol', default=2, type=int, help="early stopping tolerance on validation")
    

    parser.add_argument('--test', default=False, action='store_true')

    parser.add_argument('--workers', type=int, default=10, help="number of workers for data loading")
    parser.add_argument('--pin_memory', default=False, action='store_true', help="pinning memory for data loading")

    # data
    parser.add_argument('--dataset', default='hotels_small', choices=DATASET_LIST)
    parser.add_argument('--number_of_val', default=1, type=int)
    parser.add_argument('--hard_triplet', default=False, action='store_true')

    # learning
    parser.add_argument('--method_name', default='default', choices=SSL_MODELS) # does not support byol and simclr
    parser.add_argument('--backbone_mode', default='LP', choices=['LP', 'FT']) # to 'finetune' or 'linear prob' a backbone
    parser.add_argument('--ssl', default=False, action='store_true')
    parser.add_argument('--color_jitter', default=False, action='store_true')
    parser.add_argument('--local_global_aug', default=False, action='store_true')
    parser.add_argument('--ssl_projector_scale', type=int, default=2)
    parser.add_argument('--optimizer', default='adam', choices=OPTIMIZER_LIST, help='optimizer to use')
    parser.add_argument('--ckpt_path', default=None, help="path to the checkpoint file")
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--new_lr_coef', type=float, default=1.0)
    parser.add_argument('--bb_learning_rate', type=float, default=0.00001)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--emb_size', type=int, default=512)
    parser.add_argument('--multlayer_emb', default=False, action='store_true')
    parser.add_argument('--eq_layer_contrib', default=False, action='store_true')
    parser.add_argument('--ml_self_att', default=False, action='store_true')
    parser.add_argument('--train_with_pairwise', default=False, action='store_true')
    parser.add_argument('--eval_with_pairwise', default=False, action='store_true')
    parser.add_argument('--ml_self_att_head_number', type=int, default=4)
    parser.add_argument('--ml_self_att_layers_to_use', type=int, default=4)
    parser.add_argument('--only_att', default=False, action='store_true')
    parser.add_argument('--num_inst_per_class', type=int, default=2)
    parser.add_argument('--k_inc_freq', type=int, default=0)
    parser.add_argument('--k_dec_freq', type=int, default=0)
    parser.add_argument('--loss', default='pnpp', choices=LOSSES_LIST)
    parser.add_argument('--cov', default=False, action='store_true')
    parser.add_argument('--cov_static_mean', default=False, action='store_true')
    parser.add_argument('--cov_coef', type=float, default=1.0)
    parser.add_argument('--var_coef', type=float, default=1.0)
    parser.add_argument('--swap_coef', type=float, default=1.0)
    parser.add_argument('--with_bce', default=False, action='store_true')
    parser.add_argument('--bce_weight', type=float, default=1.0)
    parser.add_argument('--aug_swap', type=int, default=1)  # split image into (aug_swap * aug_swap squares) and shuffle them
    parser.add_argument('--aug_swap_prob', type=float, default=0.5)
    parser.add_argument('--aug_mask_prob', type=float, default=-1.0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--scale', type=float, default=3.0)
    parser.add_argument('--metric', default='cosine', choices=METRIC_LIST)
    parser.add_argument('--backbone', default='resnet50', choices=BACKBONE_LIST)
    parser.add_argument('--lnorm', default=False, action='store_true', help="Layer norm BEFORE creating embeddings")

    # loss specific defaults
    parser.add_argument('--LOSS_lr', type=float, default=None)
    parser.add_argument('--LOSS_margin', type=float, default=None)
    parser.add_argument('--LOSS_temp', type=float, default=None)
    parser.add_argument('--NCA_scale', type=float, default=None)
    parser.add_argument('--LOSS_alpha', type=float, default=None)
    parser.add_argument('--MS_beta', type=float, default=None)
    parser.add_argument('--MS_base', type=float, default=None)
    parser.add_argument('--LIFT_negmargin', type=float, default=None)
    parser.add_argument('--LIFT_posmargin', type=float, default=None)
    parser.add_argument('--ARCFACE_scale', type=float, default=None)
    parser.add_argument('--CIR_m', type=float, default=None)
    parser.add_argument('--CIR_gamma', type=float, default=None)
    parser.add_argument('--SOFTTRPL_cpc', type=int, default=None)
    parser.add_argument('--SOFTTRPL_lambda', type=float, default=None)
    parser.add_argument('--SOFTTRPL_gamma', type=float, default=None)

    # 'proxy_nca': ProxyNCALoss,  # softmax_scale=1,
    # 'proxy_anchor': ProxyAnchorLoss,  # num_classes, embedding_size, margin = 0.1, alpha = 32
    # 'arcface': ArcFaceLoss,  # num_classes, embedding_size, margin=28.6, scale=64,
    # 'angular': AngularLoss,  # alpha=40
    # 'circle': CircleLoss,  # m=0.4, gamma=80,
    # 'trpl': TripletMargin, # margin=0.05
    # 'supcon': SupConLoss, # temperature=0.1
    #     'multisim': pml_losses.MultiSimilarityLoss, # alpha=2, beta=50, base=0.5
    #     'lifted': pml_losses.LiftedStructureLoss # neg_margin=1, pos_margin=0,
    # 'softtriple': centers_per_class = 10, la = 20, gamma = 0.1, margin = 0.01


    args = parser.parse_args()

    return args

def get_args_ssl_eval():
    parser = argparse.ArgumentParser()

    parser.add_argument('-cuda', '--cuda', default=False, action='store_true')
    parser.add_argument('-wandb', '--wandb', default=False, action='store_true')
    parser.add_argument('-seed', '--seed', default=402, type=int)
    parser.add_argument('--run_times', default=1, type=int)
    parser.add_argument('-trained_with_mltp_gpu', '--trained_with_mltp_gpu', default=False, action='store_true')
    parser.add_argument('--eval_mode', default='val', help="val or test", choices=['val', 'test', 'train'])
    parser.add_argument('--method_name', default='default', choices=SSL_MODELS) # does not support byol and simclr

    parser.add_argument('--pairwise_lbls', default=False, action='store_true')
    parser.add_argument('--eval_pairwise_hard_neg', default=False, action='store_true')


    parser.add_argument('-gpu', '--gpu_ids', default='', help="gpu ids used to train")  # before: default="0,1,2,3"

    parser.add_argument('-emb', '--emb_size', default=512, type=int)
    parser.add_argument('-b', '--batch_size', default=32, type=int)
    parser.add_argument('-w', '--workers', default=10, type=int)
    parser.add_argument('--pin_memory', default=False, action='store_true')

    parser.add_argument('-d', '--dataset', default=None, choices=eval_dataset_choices)
    parser.add_argument('--num_inst_per_class', default=5, type=int)
    parser.add_argument('--lnorm', default=False, action='store_true')

    parser.add_argument('--config_path', default='config/', help="config_path for datasets")
    parser.add_argument('--project_path', default='./', help="current project path")

    parser.add_argument('-num_of_dataset', '--num_of_dataset', type=int, default=4,
                        help="number of hotels val_datasets to go through")
    parser.add_argument('--baseline', default='proxy-anchor', choices=eval_BASELINE_MODELS)
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
    parser.add_argument('--aug_swap', type=int, default=1) # always set to 1 for no partitioning and swapping

    parser.add_argument('--ml_self_att', default=False, action='store_true')
    parser.add_argument('--ml_self_att_head_number', type=int, default=4)
    parser.add_argument('--ml_self_att_layers_to_use', type=int, default=4)

    parser.add_argument('--eval_metric', default='auc', choices=['auc', 'ret', 'conret'])


    parser.add_argument('--metric', default='cosine', choices=['cosine', 'euclidean'])


    args = parser.parse_args()

    return args
