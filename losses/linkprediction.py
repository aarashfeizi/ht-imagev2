import torch.nn as nn
import torch.nn.functional as F
import torch
import utils

class BCE_Loss(nn.Module):
    """
        Choose k nearest neighbors and calculate a BCE-Cross-Entropy loss
         on the anchor and each one of the k neighbors
    """

    def __init__(self, args):
        super(BCE_Loss, self).__init__()
        self.temperature = args.get('temperature')
        # self.bce_with_logit = torch.nn.BCEWithLogitsLoss()
        self.bce = nn.BCELoss()
        self.metric = args.get('loss_metric')
        self.mode = args.get('loss_mode')
        self.k_per_class = args.get('num_inst_per_class')

    def forward(self, batch, labels):
        if self.mode == 'emb':
            return self.forward_emb(batch, labels)
        elif self.mode == 'mbl': # multi-binary loss
            return self.forward_multi_bce(batch, labels)
        else: # 'bce'
            return self.forward_bce(batch, labels)

    def __get_positive_labels(self, bs):
        l = torch.zeros((bs, bs), dtype=torch.float32)
        for i in range(bs):
            x_idx = i // self.k_per_class * self.k_per_class
            y_idx = x_idx + i % self.k_per_class
            if i % self.k_per_class == 0:
                l[y_idx, x_idx] = 0
            else:
                l[y_idx, x_idx] = 1

        return l

    def forward_multi_bce(self, batch, labels):
        gpu = labels.device.type == 'cuda'
        if self.metric == 'euclidean':
            euc_distances = utils.pairwise_distance(batch, diag_to_max=False)  # between 0 and inf

            distances = euc_distances

            # preds = 2 * F.sigmoid(-euc_distances / self.temperature)  # between 0 and 1 (map inf to 0, and 0 to 1)

            # sorted_indices = euc_distances.argsort()[:, :-1]

        else: # 'cosine'
            batch = F.normalize(batch, p=2)
            cosine_sim = torch.matmul(batch, batch.T)  # between -1 and 1
            # min_value = cosine_sim.min().item() - 1
            # cosine_sim = cosine_sim.fill_diagonal_(min_value)
            distances = -cosine_sim

        bs = batch.shape[0]

        true_labels = (labels.repeat(bs).view(-1, bs) == labels.repeat_interleave(bs).view(-1, bs))  # boolean tensor
        true_labels = true_labels.type(torch.float32)
        negative_labels = 1 - true_labels
        positive_labels = self.__get_positive_labels(bs=bs)

        if gpu:
            positive_labels = positive_labels.cuda()

        loss1 = torch.sum(positive_labels * torch.exp(-distances), -1) # anch w/ pos similarity
        loss2 = torch.sum(negative_labels * torch.exp(-distances), -1) # anch w/ neg similarity
        preds = torch.sigmoid(loss1 / loss2)


        loss = -torch.log(loss1 / loss2)
        # loss = self.bce_with_logit(loss1 / loss2, )

        loss = loss.mean()

        # loss = self.bce_with_logit(neighbor_preds_, true_labels)
        # loss = self.bce(neighbor_preds_, true_labels)

        return loss

        # gpu = labels.device.type == 'cuda'
        #
        # mask_positive = utils.get_valid_positive_mask(labels, gpu)
        # pos_loss = ((1 - dot_product) * mask_positive.float()).sum(dim=1)
        # # positive_dist_idx = (cosine_sim * mask_positive.float())
        #
        # mask_negative = utils.get_valid_negative_mask(labels, gpu)
        # neg_loss = (F.relu(dot_product - self.margin) * mask_negative.float()).sum(dim=1)
        #
        # if self.l != 0:
        #     distances = utils.squared_pairwise_distances(batch)
        #     # import pdb
        #     # pdb.set_trace()
        #     idxs = distances.argsort()[:, 1]
        #     reg_loss = -1 * distances.gather(1, idxs.view(-1, 1)).mean()
        # else:
        #     reg_loss = None

        # if self.soft:
        #     loss = F.softplus(hardest_positive_dist - hardest_negative_dist)
        # else:
        #     loss = (hardest_positive_dist - hardest_negative_dist + self.margin).clamp(min=0)

    def forward_bce(self, batch, labels):
        # dot_product = torch.matmul(batch, batch.T)  # between -inf and +inf
        # min_value = dot_product.min().item() - 1
        # dot_product = dot_product.fill_diagonal_(min_value)

        # preds = F.sigmoid(dot_product) # between 0 and 1


        if self.metric == 'euclidean':
            euc_distances = utils.pairwise_distance(batch, diag_to_max=True)  # between 0 and inf

            preds = 2 * F.sigmoid(-euc_distances / self.temperature)  # between 0 and 1 (map inf to 0, and 0 to 1)

            sorted_indices = euc_distances.argsort()[:, :-1]

        else: # 'cosine'
            batch = F.normalize(batch, p=2)
            cosine_sim = torch.matmul(batch, batch.T)  # between -1 and 1
            min_value = cosine_sim.min().item() - 1
            cosine_sim = cosine_sim.fill_diagonal_(min_value)

            preds = (cosine_sim + 1) / 2  # between 0 and 1

            sorted_indices = (-cosine_sim).argsort()[:, :-1]

        k = min(self.k, sorted_indices.shape[1])

        if k == 0:
            k = sorted_indices.shape[1]  # if k=0, consider the whole batch

        neighbor_indices_ = sorted_indices[:, :k]
        indices = torch.tensor([[j for _ in range(k)] for j in range(len(labels))])
        neighbor_preds_ = preds[indices, neighbor_indices_]
        neighbor_labels_ = labels[neighbor_indices_]

        true_labels = (neighbor_labels_ == labels.repeat_interleave(k).view(-1, k))  # boolean tensor
        true_labels = true_labels.type(torch.float32)

        # loss = self.bce_with_logit(neighbor_preds_, true_labels)
        loss = self.bce(neighbor_preds_, true_labels)

        return loss

        # gpu = labels.device.type == 'cuda'
        #
        # mask_positive = utils.get_valid_positive_mask(labels, gpu)
        # pos_loss = ((1 - dot_product) * mask_positive.float()).sum(dim=1)
        # # positive_dist_idx = (cosine_sim * mask_positive.float())
        #
        # mask_negative = utils.get_valid_negative_mask(labels, gpu)
        # neg_loss = (F.relu(dot_product - self.margin) * mask_negative.float()).sum(dim=1)
        #
        # if self.l != 0:
        #     distances = utils.squared_pairwise_distances(batch)
        #     # import pdb
        #     # pdb.set_trace()
        #     idxs = distances.argsort()[:, 1]
        #     reg_loss = -1 * distances.gather(1, idxs.view(-1, 1)).mean()
        # else:
        #     reg_loss = None

        # if self.soft:
        #     loss = F.softplus(hardest_positive_dist - hardest_negative_dist)
        # else:
        #     loss = (hardest_positive_dist - hardest_negative_dist + self.margin).clamp(min=0)

    def forward_emb(self, batch, labels):

        euc_distances = utils.pairwise_distance(batch, diag_to_max=True)  # between 0 and inf

        euc_distances_sorted, sorted_indices = euc_distances.sort()


        sorted_indices = sorted_indices[:, :-1]
        sorted_euc_distances = euc_distances_sorted[:, :-1]

        k = min(self.k, sorted_euc_distances.shape[1])

        if k == 0:
            k = sorted_euc_distances.shape[1] # if k=0, consider the whole batch

        neighbor_indices_ = sorted_indices[:, :k]
        neighbor_distances_ = sorted_euc_distances[:, :k]

        neighbor_labels_ = labels[neighbor_indices_]

        true_labels = (neighbor_labels_ == labels.repeat_interleave(k).view(-1, k))  # boolean tensor
        true_labels = true_labels.type(torch.float32)

        # loss1 = torch.sum(true_labels * torch.exp(-neighbor_distances_), -1)
        # loss2 = torch.sum((1 - true_labels) * torch.exp(-neighbor_distances_), -1)
        # loss = -torch.log(loss1 / loss2)

        loss = torch.sum(-true_labels * F.log_softmax(-neighbor_distances_, dim=-1), dim=-1)

        loss = loss.mean()

        return loss

class Neighborhood_BCE_Loss(nn.Module):
    """
        Choose k nearest neighbors and calculate a BCE-Cross-Entropy loss
         on the anchor and each one of the k neighbors
    """

    def __init__(self, args):
        super(Neighborhood_BCE_Loss, self).__init__()
        self.k = args.get('')
        self.temperature = args.get('temperature')
        # self.bce_with_logit = torch.nn.BCEWithLogitsLoss()
        self.bce = nn.BCELoss()
        self.metric = args.get('loss_metric')
        self.mode = args.get('loss_mode')
        self.k_per_class = args.get('num_inst_per_class')

    def forward(self, batch, labels):
        if self.mode == 'emb':
            return self.forward_emb(batch, labels)
        elif self.mode == 'mbl': # multi-binary loss
            return self.forward_multi_bce(batch, labels)
        else: # 'bce'
            return self.forward_bce(batch, labels)

    def __get_positive_labels(self, bs):
        l = torch.zeros((bs, bs), dtype=torch.float32)
        for i in range(bs):
            x_idx = i // self.k_per_class * self.k_per_class
            y_idx = x_idx + i % self.k_per_class
            if i % self.k_per_class == 0:
                l[y_idx, x_idx] = 0
            else:
                l[y_idx, x_idx] = 1

        return l

    def forward_multi_bce(self, batch, labels):
        gpu = labels.device.type == 'cuda'
        if self.metric == 'euclidean':
            euc_distances = utils.pairwise_distance(batch, diag_to_max=False)  # between 0 and inf

            distances = euc_distances

            # preds = 2 * F.sigmoid(-euc_distances / self.temperature)  # between 0 and 1 (map inf to 0, and 0 to 1)

            # sorted_indices = euc_distances.argsort()[:, :-1]

        else: # 'cosine'
            batch = F.normalize(batch, p=2)
            cosine_sim = torch.matmul(batch, batch.T)  # between -1 and 1
            # min_value = cosine_sim.min().item() - 1
            # cosine_sim = cosine_sim.fill_diagonal_(min_value)
            distances = -cosine_sim

        bs = batch.shape[0]

        true_labels = (labels.repeat(bs).view(-1, bs) == labels.repeat_interleave(bs).view(-1, bs))  # boolean tensor
        true_labels = true_labels.type(torch.float32)
        negative_labels = 1 - true_labels
        positive_labels = self.__get_positive_labels(bs=bs)

        if gpu:
            positive_labels = positive_labels.cuda()

        loss1 = torch.sum(positive_labels * torch.exp(-distances), -1) # anch w/ pos similarity
        loss2 = torch.sum(negative_labels * torch.exp(-distances), -1) # anch w/ neg similarity
        preds = torch.sigmoid(loss1 / loss2)


        loss = -torch.log(loss1 / loss2)
        # loss = self.bce_with_logit(loss1 / loss2, )

        loss = loss.mean()

        # loss = self.bce_with_logit(neighbor_preds_, true_labels)
        # loss = self.bce(neighbor_preds_, true_labels)

        return loss

        # gpu = labels.device.type == 'cuda'
        #
        # mask_positive = utils.get_valid_positive_mask(labels, gpu)
        # pos_loss = ((1 - dot_product) * mask_positive.float()).sum(dim=1)
        # # positive_dist_idx = (cosine_sim * mask_positive.float())
        #
        # mask_negative = utils.get_valid_negative_mask(labels, gpu)
        # neg_loss = (F.relu(dot_product - self.margin) * mask_negative.float()).sum(dim=1)
        #
        # if self.l != 0:
        #     distances = utils.squared_pairwise_distances(batch)
        #     # import pdb
        #     # pdb.set_trace()
        #     idxs = distances.argsort()[:, 1]
        #     reg_loss = -1 * distances.gather(1, idxs.view(-1, 1)).mean()
        # else:
        #     reg_loss = None

        # if self.soft:
        #     loss = F.softplus(hardest_positive_dist - hardest_negative_dist)
        # else:
        #     loss = (hardest_positive_dist - hardest_negative_dist + self.margin).clamp(min=0)

    def forward_bce(self, batch, labels):
        # dot_product = torch.matmul(batch, batch.T)  # between -inf and +inf
        # min_value = dot_product.min().item() - 1
        # dot_product = dot_product.fill_diagonal_(min_value)

        # preds = F.sigmoid(dot_product) # between 0 and 1


        if self.metric == 'euclidean':
            euc_distances = utils.pairwise_distance(batch, diag_to_max=True)  # between 0 and inf

            preds = 2 * F.sigmoid(-euc_distances / self.temperature)  # between 0 and 1 (map inf to 0, and 0 to 1)

            sorted_indices = euc_distances.argsort()[:, :-1]

        else: # 'cosine'
            batch = F.normalize(batch, p=2)
            cosine_sim = torch.matmul(batch, batch.T)  # between -1 and 1
            min_value = cosine_sim.min().item() - 1
            cosine_sim = cosine_sim.fill_diagonal_(min_value)

            preds = (cosine_sim + 1) / 2  # between 0 and 1

            sorted_indices = (-cosine_sim).argsort()[:, :-1]

        k = min(self.k, sorted_indices.shape[1])

        if k == 0:
            k = sorted_indices.shape[1]  # if k=0, consider the whole batch

        neighbor_indices_ = sorted_indices[:, :k]
        indices = torch.tensor([[j for _ in range(k)] for j in range(len(labels))])
        neighbor_preds_ = preds[indices, neighbor_indices_]
        neighbor_labels_ = labels[neighbor_indices_]

        true_labels = (neighbor_labels_ == labels.repeat_interleave(k).view(-1, k))  # boolean tensor
        true_labels = true_labels.type(torch.float32)

        # loss = self.bce_with_logit(neighbor_preds_, true_labels)
        loss = self.bce(neighbor_preds_, true_labels)

        return loss

        # gpu = labels.device.type == 'cuda'
        #
        # mask_positive = utils.get_valid_positive_mask(labels, gpu)
        # pos_loss = ((1 - dot_product) * mask_positive.float()).sum(dim=1)
        # # positive_dist_idx = (cosine_sim * mask_positive.float())
        #
        # mask_negative = utils.get_valid_negative_mask(labels, gpu)
        # neg_loss = (F.relu(dot_product - self.margin) * mask_negative.float()).sum(dim=1)
        #
        # if self.l != 0:
        #     distances = utils.squared_pairwise_distances(batch)
        #     # import pdb
        #     # pdb.set_trace()
        #     idxs = distances.argsort()[:, 1]
        #     reg_loss = -1 * distances.gather(1, idxs.view(-1, 1)).mean()
        # else:
        #     reg_loss = None

        # if self.soft:
        #     loss = F.softplus(hardest_positive_dist - hardest_negative_dist)
        # else:
        #     loss = (hardest_positive_dist - hardest_negative_dist + self.margin).clamp(min=0)

    def forward_emb(self, batch, labels):

        euc_distances = utils.pairwise_distance(batch, diag_to_max=True)  # between 0 and inf

        euc_distances_sorted, sorted_indices = euc_distances.sort()


        sorted_indices = sorted_indices[:, :-1]
        sorted_euc_distances = euc_distances_sorted[:, :-1]

        k = min(self.k, sorted_euc_distances.shape[1])

        if k == 0:
            k = sorted_euc_distances.shape[1] # if k=0, consider the whole batch

        neighbor_indices_ = sorted_indices[:, :k]
        neighbor_distances_ = sorted_euc_distances[:, :k]

        neighbor_labels_ = labels[neighbor_indices_]

        true_labels = (neighbor_labels_ == labels.repeat_interleave(k).view(-1, k))  # boolean tensor
        true_labels = true_labels.type(torch.float32)

        # loss1 = torch.sum(true_labels * torch.exp(-neighbor_distances_), -1)
        # loss2 = torch.sum((1 - true_labels) * torch.exp(-neighbor_distances_), -1)
        # loss = -torch.log(loss1 / loss2)

        loss = torch.sum(-true_labels * F.log_softmax(-neighbor_distances_, dim=-1), dim=-1)

        loss = loss.mean()

        return loss