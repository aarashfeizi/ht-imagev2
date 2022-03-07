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
               'softtriple']

BACKBONE_LIST = ['resnet50',
                 'deit',
                 'bninception']  # only implementing resnet50 :))

METRIC_LIST = ['cosine',
               'euclidean',
               'mlp']

def get_args():
    parser = argparse.ArgumentParser()

    # logistics
    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--gpu_ids', default='', help="gpu ids used to train")  # before: default="0,1,2,3"
    parser.add_argument('--seed', type=int, default=402, help="set random seed")
    parser.add_argument('--config_path', default='config/', help="config_path for datasets")
    parser.add_argument('--project_path', default='./', help="project_path")
    parser.add_argument('--xname', default='', help="extra name to be added to name")
    parser.add_argument('--save_model', default=False, action='store_true', help="save model or not")
    parser.add_argument('--no_validation', default=False, action='store_true', help="save model or not")




    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--draw_heatmaps', default=False, action='store_true')
    parser.add_argument('--workers', type=int, default=10, help="number of workers for data loading")
    parser.add_argument('--pin_memory', default=False, action='store_true', help="pinning memory for data loading")

    # data
    parser.add_argument('--dataset', default='hotels_small', choices=DATASET_LIST)
    parser.add_argument('--hard_triplet', default=False, action='store_true')

    # learning
    parser.add_argument('--ckpt_path', default=None, help="path to the checkpoint file")
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--bb_learning_rate', type=float, default=0.00001)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--emb_size', type=int, default=512)
    parser.add_argument('--num_inst_per_class', type=int, default=2)
    parser.add_argument('--k_inc_freq', type=int, default=0)
    parser.add_argument('--loss', default='pnpp', choices=LOSSES_LIST)
    parser.add_argument('--with_bce', default=False, action='store_true')
    parser.add_argument('--bce_weight', type=float, default=1.0)

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


def get_args_for_ordered_distance():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--gpu_ids', default='', help="gpu ids used to train")  # before: default="0,1,2,3"
    parser.add_argument('--config_path', default='config/', help="config_path for datasets")
    parser.add_argument('--project_path', default='./', help="project_path")



    parser.add_argument('--backbone', choices=['resnet50', 'resnet18'])

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