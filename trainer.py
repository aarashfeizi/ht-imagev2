import numpy as np
from tqdm import tqdm
import torch

from SummaryWriter import SummaryWriter
from metrics import Metric_Accuracy

OPTIMIZERS = {'adam': torch.optim.Adam}

class Trainer:
    def __init__(self, args, loss, train_loader, val_loader, optimizer='adam', current_epoch=1):
        self.batch_size = args.get('batch_size')
        self.emb_size = args.get('emb_size')
        self.args = args
        self.cuda = args.get('cuda')
        self.epochs = args.get('epochs')
        self.current_epoch = current_epoch
        self.loss_name = args.get('loss')
        self.loss_function = loss
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer_name = optimizer
        self.optimizer = None
        self.tensorboard_path = args.get('tensorboard_path')
        self.tb_writer = SummaryWriter(self.tensorboard_path)

    def set_train_loader(self, train_loader):
        self.train_loader = train_loader

    def set_val_loader(self, val_loader):
        self.val_loader = val_loader

    def __set_optimizer(self, net):
        learnable_params = [{'params': net.encoder.rest.parameters(),
                             'lr': self.args.get('bb_learning_rate'),
                             'weight_decay': self.args.get('weight_decay'),
                             'new': False}]

        if net.encoder.last_conv is not None:
            learnable_params += [{'params': net.encoder.last_conv.parameters(),
                                  'lr': self.args.get('learning_rate'),
                                  'weight_decay': self.args.get('weight_decay'),
                                  'new': True}]


        self.optimizer = OPTIMIZERS[self.optimizer_name](params=learnable_params,
                                                         lr=self.args.get('learning_rate'),
                                                         weight_decay=self.args.get('weight_decay'))

    def __tb_draw_histograms(self, net):

        for name, param in net.named_parameters():
            if param.requires_grad:
                self.tb_writer.add_histogram(name, param.flatten(), self.current_epoch)

        self.tb_writer.flush()

    def __make_bce_labels(self, labels):
        """

        :param labels: e.g. [0, 0, 1, 1, 1, 2, 1, 2, 2, 3, 3]
        :return:
        """
        l_ = labels.repeat(len(labels)).reshape(-1, len(labels))
        l__ = labels.repeat_interleave(len(labels)).reshape(-1, len(labels))

        final_bce_labels = (l_ == l__).type(torch.float32)

        return final_bce_labels

    def __train_one_epoch(self, net):
        net.train()

        epoch_loss = 0

        acc = Metric_Accuracy()

        with tqdm(total=len(self.train_loader), desc=f'{self.current_epoch}/{self.epochs}') as t:
            for batch_id, (imgs, lbls) in enumerate(self.train_loader, 1):
                if self.cuda:
                    imgs = imgs.cuda()
                    lbls = lbls.cuda()

                preds, img_embeddings = net(imgs)
                bce_labels = self.__make_bce_labels(lbls)
                loss = self.loss_function(img_embeddings, lbls)

                acc.update_acc(preds.flatten(), bce_labels.flatten())

                epoch_loss += loss.item()


                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()

                t.set_postfix(loss=f'{epoch_loss / (batch_id) :.4f}',
                              train_acc=f'{acc.get_acc():.4f}'
                              )

                t.update()

        return epoch_loss, acc.get_acc()


    def train(self, net, val=True):
        self.__set_optimizer(net)

        if self.train_loader is None:
            raise Exception(f'train_loader is not initialized in trainer')

        if self.optimizer is None:
            raise Exception(f'optimizer is not initialized in trainer')

        starting_epoch = max(1, self.current_epoch)

        for epoch in range(starting_epoch, self.epochs):

            epoch_loss, epoch_acc = self.__train_one_epoch(net)

            print(f'Epoch {self.current_epoch}-> loss: ', epoch_loss, f', acc: ', epoch_acc)

            if val:
                with torch.no_grad:
                    val_loss, val_acc, val_embeddings = self.validate(net)

                    print(f'VALIDATION {self.current_epoch}-> val_loss: ', val_loss, f', val_acc: ', epoch_acc)

            self.current_epoch = epoch
            self.__tb_draw_histograms(net)

            # if self.scheduler:
            #     self.scheduler.step()
            # else:
            #     self.adaptive_scheduler.step(current_loss=val_loss, current_val=val_acc)

    def validate(self, net):
        net.eval()

        val_loss = 0

        acc = Metric_Accuracy()

        val_size = self.val_loader.dataset.__len__()

        embeddings = np.zeros((val_size, self.emb_size), dtype=torch.float32)

        with tqdm(total=len(self.val_loader), desc=f'{self.current_epoch} validating...') as t:
            for batch_id, (imgs, lbls) in enumerate(self.train_loader, 1):
                if self.cuda:
                    imgs = imgs.cuda()
                    lbls = lbls.cuda()

                preds, img_embeddings = net(imgs)
                bce_labels = self.__make_bce_labels(lbls)
                loss = self.loss_function(img_embeddings, lbls)

                begin_idx = (batch_id - 1) * self.batch_size
                end_idx = min(val_size, batch_id * self.batch_size)

                embeddings[begin_idx: end_idx, :] = img_embeddings.cpu().detach().numpy()

                acc.update_acc(preds.flatten(), bce_labels.flatten())

                val_loss += loss.item()

                t.set_postfix(loss=f'{val_loss / (batch_id) :.4f}',
                              val_acc=f'{acc.get_acc():.4f}'
                              )

                t.update()

        return val_loss, acc.get_acc(), embeddings
