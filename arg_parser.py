import argparse

DATASET_LIST = ['hotels', 'hotels_small', 'cub']
LOSSES_LIST = ['pnpp', 'bce', 'trpl']
BACKBONE_LIST = ['resnet50', 'deit', 'bninception']  # only implementing resnet50 :))
METRIC_LIST = ['cosine', 'euclidean', 'mlp']

def get_args():
    parser = argparse.ArgumentParser()

    # logistics
    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--gpu_ids', default='', help="gpu ids used to train")  # before: default="0,1,2,3"
    parser.add_argument('--seed', type=int, default=402, help="set random seed")
    parser.add_argument('--config_path', default='config/', help="config_path for datasets")
    parser.add_argument('--project_path', default='./', help="project_path")


    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--workers', type=int, default=10, help="number of workers for data loading")
    parser.add_argument('--pin_memory', default=False, action='store_true', help="pinning memory for data loading")

    # data
    parser.add_argument('--dataset', default='hotels_small', choices=DATASET_LIST)

    # learning
    parser.add_argument('--ckpt_path', default=None, help="path to the checkpoint file")
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--bb_learning_rate', type=float, default=0.00001)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--emb_size', type=int, default=512)
    parser.add_argument('--num_inst_per_class', type=int, default=5)
    parser.add_argument('--loss', default='pnpp', choices=LOSSES_LIST)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--scale', type=float, default=3.0)
    parser.add_argument('--proxypncapp_lr', type=float, default=None)
    parser.add_argument('--metric', default='cosine', choices=METRIC_LIST)
    parser.add_argument('--backbone', default='resnet50', choices=BACKBONE_LIST)
    parser.add_argument('--lnorm', default=False, action='store_true', help="Layer norm BEFORE creating embeddings")


    args = parser.parse_args()

    return args