from tqdm import tqdm
import torch

from metrics import Metric_Accuracy

OPTIMIZERS = {'adam': torch.optim.Adam}

class Trainer:
    def __init__(self, args, loss, train_loader, val_loader, optimizer='adam', current_epoch=1):
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
                labels = self.__make_bce_labels()
                loss = self.loss_function(img_embeddings, lbls)

                epoch_loss += loss.item()


                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()

                t.set_postfix(loss=f'{epoch_loss / (batch_id) :.4f}',
                              train_acc=f'{acc.get_acc():.4f}'
                              )

                t.update()

    def train(self, net, val=True):
        self.__set_optimizer(net)

        if self.train_loader is None:
            raise Exception(f'train_loader is not initialized in trainer')

        if self.optimizer is None:
            raise Exception(f'optimizer is not initialized in trainer')

        starting_epoch = max(1, self.current_epoch)

        for epoch in range(starting_epoch, self.epochs):

            loss = self.__train_one_epoch(net)

            self.current_epoch = epoch

