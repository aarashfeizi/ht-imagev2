import numpy as np
import pandas as pd
import torch


class Metric_Accuracy:

    def __init__(self):
        self.rights = 0
        self.wrongs = 0

    def update_acc(self, output, label, sigmoid=True):
        if sigmoid:
            pred = (torch.sigmoid(output) >= 0.5)
        else:
            pred = (output >= 0.5)

        batch_rights = sum(label.type(torch.int64) == pred.type(torch.int64)).cpu().numpy()

        self.rights += batch_rights
        self.wrongs += (label.shape[0] - batch_rights)

    def get_acc(self):
        return (self.rights / (self.rights + self.wrongs)) * 100

    def get_right_wrong(self):
        return {'right': self.rights, 'wrong': self.wrongs}

    def reset_acc(self):
        self.rights = 0
        self.wrongs = 0


class Accuracy_At_K():

    def __init__(self, ks=[1, 2, 4, 8], classes=np.array([]), pairwise=False):
        self.ks = ks
        self.k_valus = {i: 0 for i in self.ks}

        self.r_values = {i: 0 for i in self.ks}
        self.pairwise = pairwise

        self.n = 0

        self.classes = classes
        self.class_tot = len(self.classes)
        self.lbl2idx = {c: i for i, c in enumerate(self.classes)}

        self.per_class_k_valuese = {k: np.zeros(shape=self.class_tot) for k in self.ks}

        self.per_class_n = np.zeros(shape=self.class_tot)

    def update(self, lbl, ret_lbls):
        # all_lbl = sum(ret_lbls == lbl)

        for k in self.ks:
            if self.pairwise:
                target_lbl = 1.0
            else:
                target_lbl = lbl

            if target_lbl in ret_lbls[:k]:
                self.k_valus[k] += 1
                self.per_class_k_valuese[k][self.lbl2idx[lbl]] += 1

        self.n += 1
        self.per_class_n[self.lbl2idx[lbl]] += 1

    def __str__(self):
        output_str = ''
        metrics = self.get_all_metrics()

        for k, v in metrics.items():
            output_str += f'{k} = {v}\n'

        return output_str

    def get_all_metrics(self):
        output_dict = {}
        for k in self.ks:
            final_k = self.k_valus[k] / max(self.n, 1)
            output_dict[f'R@{k}'] = final_k

        return output_dict

    def get_per_class_metrics(self):

        assert sum(self.per_class_n) == self.n
        for k in self.ks:
            assert sum(self.per_class_k_valuese[k]) == self.k_valus[k]

        if self.n == 0:
            denom = [1 for _ in range(len(self.per_class_n))]
        else:
            denom = self.per_class_n

        d = {'label': self.classes,
             'n': self.per_class_n}

        for k in self.ks:
            d[f'k@{k}'] = (self.per_class_k_valuese[k] / denom)

        df = pd.DataFrame(data=d)

        return df


class MAPR():

    def __init__(self, classes=np.array([])):
        self.k1 = 0
        self.k5 = 0
        self.k10 = 0
        self.k100 = 0

        self.r1 = 0
        self.r5 = 0
        self.r10 = 0
        self.r100 = 0

        self.n = 0

        self.classes = classes
        self.class_tot = len(self.classes)
        self.lbl2idx = {c: i for i, c in enumerate(self.classes)}
        self.per_class_k1 = np.zeros(shape=self.class_tot)  # col1: kns
        self.per_class_k5 = np.zeros(shape=self.class_tot)  # col1: kns
        self.per_class_k10 = np.zeros(shape=self.class_tot)  # col1: kns
        self.per_class_k100 = np.zeros(shape=self.class_tot)  # col1: kns

        self.per_class_n = np.zeros(shape=self.class_tot)

    def update(self, lbl, ret_lbls):
        # all_lbl = sum(ret_lbls == lbl)
        if lbl == ret_lbls[0]:
            self.k1 += 1
            self.per_class_k1[self.lbl2idx[lbl]] += 1

        if lbl in ret_lbls[:5]:
            self.k5 += 1
            self.per_class_k5[self.lbl2idx[lbl]] += 1

        if lbl in ret_lbls[:10]:
            self.k10 += 1
            self.per_class_k10[self.lbl2idx[lbl]] += 1

        if lbl in ret_lbls[:100]:
            self.k100 += 1
            self.per_class_k100[self.lbl2idx[lbl]] += 1

        # self.r5 += (sum(ret_lbls[:5] == lbl) / all_lbl)
        # self.r10 += (sum(ret_lbls[:10] == lbl) / all_lbl)
        # self.r100 += (sum(ret_lbls[:100] == lbl) / all_lbl)

        self.n += 1
        self.per_class_n[self.lbl2idx[lbl]] += 1

    def __str__(self):
        k1, k5, k10, k100 = self.get_tot_metrics()

        return f'k@1 = {k1}\n' \
               f'k@5 = {k5}\n' \
               f'k@10 = {k10}\n' \
               f'k@100 = {k100}\n'
        # f'recall@1 = {r1}\n' \
        # f'recall@5 = {r5}\n' \
        # f'recall@10 = {r10}\n' \
        # f'recall@100 = {r100}\n'

    def get_tot_metrics(self):

        return (self.k1 / max(self.n, 1)), \
               (self.k5 / max(self.n, 1)), \
               (self.k10 / max(self.n, 1)), \
               (self.k100 / max(self.n, 1))

    def get_per_class_metrics(self):

        assert sum(self.per_class_n) == self.n
        assert sum(self.per_class_k1) == self.k1
        assert sum(self.per_class_k5) == self.k5
        assert sum(self.per_class_k10) == self.k10
        assert sum(self.per_class_k100) == self.k100

        if self.n == 0:
            denom = [1 for _ in range(len(self.per_class_n))]
        else:
            denom = self.per_class_n

        k1s, k5s, k10s, k100s = (self.per_class_k1 / denom), \
                                (self.per_class_k5 / denom), \
                                (self.per_class_k10 / denom), \
                                (self.per_class_k100 / denom)

        d = {'label': self.classes,
             'n': self.per_class_n,
             'k@1': k1s,
             'k@5': k5s,
             'k@10': k10s,
             'k@100': k100s}

        df = pd.DataFrame(data=d)

        return df

        # self.r1, self.r5, self.r10, self.r100


