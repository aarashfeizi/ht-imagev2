import argparse
import os
import timm
import torch
from torch import nn

import utils
import arg_parser
import numpy as np
from tqdm import tqdm

def get_label_idx_orderings(net, loader, name, cuda=True):
    all_embs = []
    all_lbls = []

    with tqdm(total=len(loader), desc=f'Getting embeddings for {name}') as t:
        for i, (img, label) in enumerate(loader):
            if cuda:
                img = img.cuda()

            embeddings = net(img)
            all_embs.append(embeddings.cpu().detach().numpy())
            all_lbls.append(label.numpy())

            t.update()

    all_embs = np.concatenate(all_embs)
    all_lbls = np.concatenate(all_lbls)

    _, ordered_idxs, _ = utils.get_faiss_knn(all_embs, k=1000, gpu=cuda, metric='cosine')

    ordered_labels = []
    for idx, lbl in enumerate(all_lbls):
        ret_lbls = all_lbls[ordered_idxs[idx]]
        ordered_labels.append(ret_lbls)

    ordered_labels = np.concatenate(ordered_labels)

    return ordered_labels, ordered_idxs




if __name__ == '__main__':
    args = arg_parser.get_args_for_ordered_distance()
    dataset_config = utils.load_config(os.path.join(args.config_path, args.dataset + '.json'))

    all_args = utils.Global_Config_File(args=args, config_file=dataset_config, init_tb=False)

    net = timm.create_model(all_args.get('backbone'), pretrained=True, nun_classes=0)

    if all_args.get('cuda'):
        if all_args.get('gpu_ids') != '':
            os.environ["CUDA_VISIBLE_DEVICES"] = all_args.get('gpu_ids')

        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
        net.cuda()

    val_transforms, val_transforms_names = utils.TransformLoader(all_args).get_composed_transform(mode='val')

    eval_ldrs = []
    for i in range(0, all_args.get('num_of_dataset')):
        name = all_args.get(f'all_{all_args.get("eval_mode")}_files')[i]
        val_ldr = utils.get_data(all_args, mode=all_args.get('eval_mode'),
                                        file_name=name,
                                        transform=val_transforms,
                                        sampler_mode='db')

        label_orderings, idx_orderings = get_label_idx_orderings(net, val_ldr,
                                                                name=name,
                                                                cuda=all_args.get('cuda'))

        np.save(os.path.join(all_args.get('project_path'), f'{all_args.get("backbone")}_{name}_labels'), label_orderings)
        np.save(os.path.join(all_args.get('project_path'), f'{all_args.get("backbone")}_{name}_idxs'), idx_orderings)




