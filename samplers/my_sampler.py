import copy
import json
import random
from collections import defaultdict

import numpy as np

import utils
from samplers import RandomIdentitySampler


class KBatchSampler(RandomIdentitySampler):

    def prepare_batch(self):
        batch_idxs_dict = defaultdict(list)

        for label in self.labels:
            idxs = copy.deepcopy(self.data_dict[label])
            if self.pairwise_labels is not None: # use_pairwise_label should be true for training and KBatchSampler is only used for training
                idx_map = {c: idx for c, idx in enumerate(idxs)}
                pairwise_labels = self.pairwise_labels[idxs, :][:, idxs]
                possible_pairs = list(zip(*np.where(pairwise_labels == 1)))
                possible_pairs = set([tuple(map(idx_map.get, sorted(p))) for p in possible_pairs if p[0] != p[1]])

                if len(possible_pairs) == 0:
                    continue

                possible_pairs = [list(p) for p in possible_pairs]
                random.shuffle(possible_pairs)
                batch_idxs_dict[label] = possible_pairs
            else:
                if len(idxs) < self.K:
                    idxs.extend(np.random.choice(idxs, size=self.K - len(idxs), replace=True))
                random.shuffle(idxs)

                batch_idxs_dict[label] = [idxs[i * self.K: (i + 1) * self.K] for i in range(len(idxs) // self.K)]

        if set(self.labels) != set(list(batch_idxs_dict.keys())):
            print(f'{len(set(list(batch_idxs_dict.keys())) - set(self.labels))} labels are not being used!!')

        avai_labels = copy.deepcopy(list(batch_idxs_dict.keys()))
        return batch_idxs_dict, avai_labels


class BalancedTripletSampler(RandomIdentitySampler):
    """
    Produce batches of [Anchor, Positive, Negative]
    """

    def __init__(self, dataset, batch_size, num_instances, k_dec_freq, use_pairwise_label, **kwargs):
        super().__init__(dataset, batch_size, num_instances, k_dec_freq, use_pairwise_label, **kwargs)
        self.K = 2  # anchor and positive
        self.num_labels_per_batch = self.batch_size // 3 # batch of triplets
        self.max_iters = ((dataset.__len__() * 3) // batch_size)
        self.batch_indexes = None

    def prepare_batch(self):

        if self.batch_indexes:
            batch_idxs_dict = copy.deepcopy(self.batch_indexes)
            avai_labels = copy.deepcopy(sorted(list(self.batch_indexes.keys())))
        else:

            batch_idxs_dict = defaultdict(list)

            for label in self.labels:
                idxs = copy.deepcopy(self.data_dict[label])
                if (self.pairwise_labels is not None) and self.use_pairwise_label:
                    idx_map = {c: idx for c, idx in enumerate(idxs)}
                    pairwise_labels = self.pairwise_labels[idxs, :][:, idxs]
                    possible_pairs = list(zip(*np.where(pairwise_labels == 1)))
                    possible_pairs = set([tuple(map(idx_map.get, sorted(p))) for p in possible_pairs if p[0] != p[1]])

                    if len(possible_pairs) == 0:
                        continue

                    possible_pairs = [list(p) for p in possible_pairs]
                    random.shuffle(possible_pairs)
                    batch_idxs_dict[label] = possible_pairs
                else:
                    if len(idxs) < self.K:
                        idxs.extend(np.random.choice(idxs, size=self.K - len(idxs), replace=True))
                    random.shuffle(idxs)

                    batch_idxs_dict[label] = [idxs[i * self.K: (i + 1) * self.K] for i in range(len(idxs) // self.K)]

            for label in self.labels:
                if batch_idxs_dict.get(label, None) is None:
                    continue
                other_labels = list(set(self.labels) - set([label]))
                triplets = []
                for pair in batch_idxs_dict[label]:
                    neg_label = np.random.choice(other_labels, size=1)[0]
                    pair.extend(np.random.choice(self.data_dict[neg_label], size=1))
                    triplets.append(pair)

                    # add negative from same class, but with a pairwise_label of 0
                    if self.pairwise_labels is not None:
                        anchor = pair[0]
                        pos = pair[1]
                        pairwise_labels_for_anchor = self.pairwise_labels[anchor, :]
                        potential_negatives = np.where(pairwise_labels_for_anchor == 0)[0]
                        if len(potential_negatives) > 0:
                            negative = np.random.choice(potential_negatives, size=1)[0]
                            new_pair = tuple([anchor, pos, negative])
                            triplets.append(new_pair)

                batch_idxs_dict[label] = triplets

            if set(self.labels) != set(list(batch_idxs_dict.keys())):
                print(f'{len(set(self.labels) - set(list(batch_idxs_dict.keys())))} labels are not being used!!')

            avai_labels = copy.deepcopy(list(batch_idxs_dict.keys()))

        return batch_idxs_dict, avai_labels


class HardTripletSampler(RandomIdentitySampler):
    """
    Produce batches of [Anchor, Positive, Negative]
    for each anch, have the max sim(anch, neg)
    """

    def __init__(self, dataset, batch_size, num_instances, k_dec_freq, ordered_idxs=None, ordered_lbls=None, **kwargs):
        """

        :param ordered_idxs: ndarray, given a Dataset of size N, sim_indices is a ndarray of size N * K, representing the
         K nearest neighbors to each sample
        :param ordered_lbls: ndarray, given a Dataset of size N, sim_indices is a ndarray of size N * K, representing the
         labels of the K nearest neighbors to each sample
        """

        super().__init__(dataset, batch_size, num_instances, k_dec_freq, **kwargs)
        self.K = 2  # anchor and positive
        self.num_labels_per_batch = self.batch_size // 3 # batch of triplets
        self.max_iters = ((dataset.__len__() * 3) // batch_size)
        self.ordered_idxs = ordered_idxs
        self.ordered_lbls = ordered_lbls
        self.all_labels = np.array(dataset.label_list)
        self.__prepare_negs()

    def __prepare_negs(self):
        N, K = self.ordered_idxs.shape
        assert len(self.all_labels) == N
        all_labels = self.all_labels.repeat(K).reshape(N, K)
        self.pos_mask = (all_labels == self.ordered_lbls).astype(np.int64)
        negative_idxs_of_idxs = self.pos_mask.argmin(axis=1)
        y_idxs = np.array([i for i in range(N)])
        self.negative_idxs = self.ordered_idxs[y_idxs, negative_idxs_of_idxs]

    def prepare_batch(self):
        batch_idxs_dict = defaultdict(list)

        for label in self.labels:
            idxs = copy.deepcopy(self.data_dict[label])
            if len(idxs) < self.K:
                idxs.extend(np.random.choice(idxs, size=self.K - len(idxs), replace=True))
            random.shuffle(idxs)

            batch_idxs_dict[label] = [idxs[i * self.K: (i + 1) * self.K] for i in range(len(idxs) // self.K)]

        for label in self.labels:
            triplets = []
            for pair in batch_idxs_dict[label]:
                reversed_pair = pair[::-1]

                pair.extend([self.negative_idxs[pair[0]]])
                reversed_pair.extend([self.negative_idxs[reversed_pair[0]]])

                triplets.append(pair)
                triplets.append(reversed_pair)

            batch_idxs_dict[label] = triplets

        avai_labels = copy.deepcopy(self.labels)
        return batch_idxs_dict, avai_labels


class DataBaseSampler(RandomIdentitySampler):
    def __init__(self, dataset, batch_size, num_instances, k_dec_freq, **kwargs):
        super().__init__(dataset, batch_size, num_instances, k_dec_freq, **kwargs)
        self.batch_size = batch_size
        self.dataset_size = dataset.__len__()

    def prepare_batch(self):

        all_idxs = [i for i in range(self.dataset_size)]

        batch_idxs_list = []
        i = 0
        for i in range(self.max_iters + 1):
            idx_to_add = all_idxs[i * self.batch_size: (i + 1) * self.batch_size]
            if len(idx_to_add) > 0:
                batch_idxs_list.append(idx_to_add)

        return batch_idxs_list, None

    def __iter__(self):
        batch_idxs_list, _ = self.prepare_batch()
        for _ in range(len(batch_idxs_list)):
            batch = batch_idxs_list.pop(0)
            yield batch


class DrawHeatmapSampler(RandomIdentitySampler):
    def __init__(self, dataset, batch_size, num_instances, k_dec_freq, use_pairwise_label, idxes=None, **kwargs):
        super().__init__(dataset, batch_size, num_instances, k_dec_freq, use_pairwise_label, **kwargs)
        self.batch_size = 1
        self.batch_idxes = idxes
        if self.batch_idxes is None:
            self.batch_idxes = [i for i in range(20)] # in val1_small.csv is images/train/77/80534/travel_website/6997071.jpg

    def prepare_batch(self):
        batch_idxs_list = []
        for b in self.batch_idxes:
            batch_idxs_list.append([b])
        # all_idxs = []
        #
        # for label in self.labels:
        #     idxs = copy.deepcopy(self.data_dict[label])
        #     all_idxs.extend(idxs)
        #
        # batch_idxs_list = []
        #
        # for i in range(self.max_iters + 1):
        #     idx_to_add = all_idxs[i * self.batch_size: (i + 1) * self.batch_size]
        #     if len(idx_to_add) > 0:
        #         batch_idxs_list.append(idx_to_add)

        return batch_idxs_list, None

    def __iter__(self):
        batch_idxs_list, _ = self.prepare_batch()
        for _ in range(len(batch_idxs_list)):
            batch = batch_idxs_list.pop(0)
            yield batch

class Draw2XHeatmapSampler(BalancedTripletSampler):
    def __init__(self, dataset, batch_size, num_instances, k_dec_freq, use_pairwise_label, triplet_path='', **kwargs):
        super().__init__(dataset, batch_size, num_instances, k_dec_freq, use_pairwise_label, **kwargs)
        self.batch_size = 3
        self.num_labels_per_batch = self.batch_size // 3  # batch of triplets
        if triplet_path != '':
            self.batch_indexes = utils.load_json(triplet_path)
        else:
            self.batch_indexes = None

        self.max_iters = self.get_all_triplets()

    def get_all_triplets(self):
        n = 0
        for k, v in self.batch_indexes.items():
            n += len(v)

        return n