import copy
import random
from collections import defaultdict

import numpy as np

from samplers import RandomIdentitySampler


class KBatchSampler(RandomIdentitySampler):

    def prepare_batch(self):
        batch_idxs_dict = defaultdict(list)

        for label in self.labels:
            idxs = copy.deepcopy(self.data_dict[label])
            if len(idxs) < self.K:
                idxs.extend(np.random.choice(idxs, size=self.K - len(idxs), replace=True))
            random.shuffle(idxs)

            batch_idxs_dict[label] = [idxs[i * self.K: (i + 1) * self.K] for i in range(len(idxs) // self.K)]

        avai_labels = copy.deepcopy(self.labels)
        return batch_idxs_dict, avai_labels


class BalancedTripletSampler(RandomIdentitySampler):
    """
    Produce batches of [Anchor, Positive, Negative]
    """
    def __init__(self, dataset, batch_size, num_instances):
        super().__init__(dataset, batch_size, num_instances)
        self.K = 2 # anchor and positive
        self.max_iters = ((dataset.__len__() * 3) // batch_size)


    def prepare_batch(self):
        batch_idxs_dict = defaultdict(list)

        for label in self.labels:
            idxs = copy.deepcopy(self.data_dict[label])
            if len(idxs) < self.K:
                idxs.extend(np.random.choice(idxs, size=self.K - len(idxs), replace=True))
            random.shuffle(idxs)

            batch_idxs_dict[label] = [idxs[i * self.K: (i + 1) * self.K] for i in range(len(idxs) // self.K)]

        for label in self.labels:
            other_labels = list(set(self.labels) - set([label]))
            triplets = []
            for pair in batch_idxs_dict[label]:
                neg_label = np.random.choice(other_labels, size=1)[0]
                pair.extend(np.random.choice(self.data_dict[neg_label], size=1))
                triplets.append(pair)

            batch_idxs_dict[label] = triplets

        avai_labels = copy.deepcopy(self.labels)
        return batch_idxs_dict, avai_labels

class DataBaseSampler(RandomIdentitySampler):
    def __init__(self, dataset, batch_size, num_instances):
        super().__init__(dataset, batch_size, num_instances)
        self.batch_size = batch_size

    def prepare_batch(self):
        all_idxs = []

        for label in self.labels:
            idxs = copy.deepcopy(self.data_dict[label])
            all_idxs.extend(idxs)

        batch_idxs_list = []

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
