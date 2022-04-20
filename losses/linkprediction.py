import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class NormalBCELoss(nn.Module):
    """
    Do BCE loss on all possible pairs in batch
    """

    def __init__(self, args):
        super(NormalBCELoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, batch, labels, output_pred=None, train=True):
        assert output_pred is not None
        batch_bce_labels = utils.make_batch_bce_labels(labels)
        assert len(output_pred.flatten()) == len(batch_bce_labels.flatten())
        loss = self.bce_loss(output_pred.flatten(), batch_bce_labels.flatten())
        return loss


class HardBCELoss(NormalBCELoss):
    """
        If batch has k instances of each class, for each sample do BCE loss on (k - 1) hardest pairs
         (this causes the number of positive and negative pairs to be the same)
    """

    def __get_sims(self, batch):

        norm_embeddings = F.normalize(batch, p=2, dim=-1)
        if len(norm_embeddings.shape) == 2:
            sims = torch.matmul(norm_embeddings, norm_embeddings.T)
        else:  # todo make sure it works!!!!
            sims = (norm_embeddings * norm_embeddings.transpose(0, 1)).sum(dim=-1)
        # preds = (sims + 1) / 2  # maps (-1, 1) to (0, 1)
        #
        # preds = torch.clamp(preds, min=0.0, max=1.0)
        return sims

    def __get_mask(self, sims, batch_bce_labels):
        """

        :param sims: similarities
        :param batch_bce_labels: 0s and 1s, with -1s on the diagonal (should not choose itself as positive or negative)
        :return: mask of indecies, with equal positives and negatives, negatives are the most difficult ones
        """
        batch_size = batch_bce_labels.shape[0]
        col_index = sims.argsort(dim=1, descending=True)

        row_index = torch.tensor([[i for _ in range(batch_size)] for i in range(batch_size)])

        batch_bce_labels_reordered = batch_bce_labels[row_index, col_index]

        pos_index = col_index[batch_bce_labels_reordered == 1].reshape(batch_size, -1)
        neg_index = col_index[batch_bce_labels_reordered == 0].reshape(batch_size, -1)[:, :pos_index.shape[1]]

        mask_index = torch.cat([pos_index, neg_index], dim=1)

        return mask_index

    def forward(self, batch, labels, output_pred=None, train=True):
        assert output_pred is not None

        if train:
            batch_bce_labels = utils.make_batch_bce_labels(labels, diagonal_fill=-1)
            # sims = self.__get_sims(batch)
            sims = (output_pred * 2) - 1
            col_index = self.__get_mask(sims, batch_bce_labels)
            row_index = torch.tensor([[i for _ in range(col_index.shape[1])] for i in range(len(labels))])

            batch_bce_labels_chosen = batch_bce_labels[row_index, col_index]
            output_pred_chosen = output_pred[row_index, col_index]
        else:  # normal bce
            batch_bce_labels_chosen = utils.make_batch_bce_labels(labels)
            output_pred_chosen = output_pred

        assert len(output_pred_chosen.flatten()) == len(batch_bce_labels_chosen.flatten())

        loss = self.bce_loss(output_pred_chosen.flatten(), batch_bce_labels_chosen.flatten())
        return loss


# class BceAndOtherLoss(nn.Module):
#     """
#     Do BCE loss on all possible pairs in batch alongside another loss function
#     """
#
#     def __init__(self, args, other_loss):
#         super(BceAndOtherLoss, self).__init__()
#         self.bce_weight = args.get('bce_weight')
#         self.bce_loss = NormalBCELoss(args=args)
#         self.other_loss = other_loss
#
#     def forward(self, batch, labels, output_pred=None, train=True):
#         bce_loss_value = self.bce_loss(batch, labels, output_pred, train=train)
#         other_loss_value = self.other_loss(batch, labels.type(torch.int64))
#
#         loss = (self.bce_weight / self.bce_weight + 1) * bce_loss_value + \
#                (1 / self.bce_weight + 1) * other_loss_value
#
#         return loss
