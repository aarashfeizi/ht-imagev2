import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import utils
from SummaryWriter import SummaryWriter
from metrics import Metric_Accuracy

OPTIMIZERS = {'adam': torch.optim.Adam}


class Trainer:
    """
    Class for training and validating model
    """

    def __init__(self, args, loss, train_loader, val_loader,
                 val_db_loader, optimizer='adam', current_epoch=0, force_new_dir=True):
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
        self.val_db_loader = val_db_loader
        self.optimizer_name = optimizer
        self.model_name = utils.get_model_name(self.args)
        self.optimizer = None
        self.tensorboard_path = None
        self.heatmap_loader = None
        if args.get('ckpt_path') is None:  # new model
            self.save_path = None
        else:  # loading pretrained model mode
            assert os.path.exists(args.get('ckpt_path'))
            assert not force_new_dir
            sp, _ = os.path.split(args.get('ckpt_path'))  # returns save_path_with_model_name and checkpoint_name
            _, mn = os.path.split(sp)  # returns root save_path and model_name
            self.model_name = mn
            self.save_path = sp

        self.tb_writer = None
        self.force = force_new_dir  # false if testing and the folder should already be there
        self.__set_tb_svdir()

    def __set_tb_svdir(self):
        self.tensorboard_path = os.path.join(self.args.get('tensorboard_path'), f'{self.model_name}')

        if self.save_path is None:
            self.save_path = os.path.join(self.args.get('save_path'), f'{self.model_name}')
            self.tensorboard_path = utils.make_dirs(self.tensorboard_path, force=self.force)
            self.save_path = utils.make_dirs(self.save_path, force=self.force)
        else:
            self.tensorboard_path = utils.make_dirs(self.tensorboard_path, force=False)
            print(f'Save_path set to {self.save_path} and model name set to {self.model_name} from checkpoint')

        self.tb_writer = SummaryWriter(self.tensorboard_path)

    def set_heatmap_loader(self, loader):
        self.heatmap_loader = loader

    def set_train_loader(self, train_loader):
        self.train_loader = train_loader

    def set_val_loader(self, val_loader):
        self.val_loader = val_loader

    def set_val_db_loader(self, val_db_loader):
        self.val_db_loader = val_db_loader

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

        if self.args.get('loss') == 'pnpp':
            assert self.args.get('PNPP_lr') is not None

            learnable_params += [{'params': self.loss_function.parameters(),
                                  'lr': self.args.get('PNPP_lr'),
                                  'new': True}]

        self.optimizer = OPTIMIZERS[self.optimizer_name](params=learnable_params,
                                                         lr=self.args.get('learning_rate'),
                                                         weight_decay=self.args.get('weight_decay'))

    def __tb_draw_histograms(self, net):

        for name, param in net.named_parameters():
            if param.requires_grad:
                self.tb_writer.add_histogram(name, param.flatten(), self.current_epoch)

        self.tb_writer.flush()

    def __tb_update_value(self, names_values):

        for (name, value) in names_values:
            self.tb_writer.add_scalar(name, value, self.current_epoch)

        self.tb_writer.flush()

    def __tb_draw_img(self, names_imgs):

        for i, (name, img) in enumerate(names_imgs, 1):
            self.tb_writer.add_image(f'{i}/{name}', img, global_step=self.current_epoch)

        self.tb_writer.flush()

    def draw_heatmaps(self, net):
        if self.heatmap_loader is None:
            raise Exception('self.heatmap_loader is not set in trainer.py!!')
        
        for i, (imgs, lbls, paths) in enumerate(self.heatmap_loader):
            if self.cuda:
                imgs = imgs.cuda()

            embeddings, activations = net.encoder(imgs, is_feat=True) # returns embeddings, [f1, f2, f3, f4]

            img_names = []

            for path in paths:

                _, img_name = os.path.split(path)
                img_name = img_name[:img_name.find('.')]
                img_names.append(img_name)

            org_imgs = []

            for path in paths:
                org_imgs.append(utils.transform_only_img(path))

            heatmaps = utils.get_all_heatmaps([activations], org_imgs)

            name_imgs = []
            for name, heatmap in zip(img_names, heatmaps):
                name_imgs.extend([(f'img_{name}/{n}', i) for n, i in heatmap.items()])

            self.__tb_draw_img(name_imgs)
            




    def __train_one_epoch(self, net):
        net.train()

        epoch_loss = 0

        acc = Metric_Accuracy()

        with tqdm(total=len(self.train_loader), desc=f'{self.current_epoch}/{self.epochs}') as t:
            for batch_id, (imgs, lbls) in enumerate(self.train_loader, 1):
                if self.cuda:
                    imgs = imgs.cuda()
                    lbls = lbls.cuda()

                preds, similarities, img_embeddings = net(imgs)
                bce_labels = utils.make_batch_bce_labels(lbls)

                loss = self.get_loss_value(img_embeddings, preds, lbls)

                if torch.isnan(loss):
                    raise Exception(f'Loss became NaN on iteration {batch_id} of epoch {self.current_epoch}! :(')

                acc.update_acc(preds.flatten(), bce_labels.flatten(), sigmoid=False)

                epoch_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()


                postfixes = {f'train_{self.loss_name}': f'{epoch_loss / (batch_id) :.4f}',
                             'train_acc': f'{acc.get_acc():.4f}'}
                t.set_postfix(**postfixes)

                t.update()

        return epoch_loss, acc.get_acc()

    def validate(self, net):
        net.eval()

        val_loss = 0

        acc = Metric_Accuracy()

        predicted_links = []
        true_links = []

        with tqdm(total=len(self.val_loader), desc=f'{self.current_epoch} validating...') as t:
            for batch_id, (imgs, lbls) in enumerate(self.val_loader, 1):
                if self.cuda:
                    imgs = imgs.cuda()
                    lbls = lbls.cuda()

                preds, similarities, img_embeddings = net(imgs)
                bce_labels = utils.make_batch_bce_labels(lbls)
                loss = self.get_loss_value(img_embeddings, preds, lbls)

                # equal numbers of positives and negatives
                balanced_preds = utils.balance_labels(preds.cpu().detach().numpy(), k=3)
                balanced_bce_labels = utils.balance_labels(bce_labels.cpu().detach().numpy(), k=3)

                predicted_links.extend(balanced_preds.numpy())
                true_links.extend(balanced_bce_labels.numpy())

                acc.update_acc(balanced_preds.flatten(), balanced_bce_labels.flatten(), sigmoid=False)

                val_loss += loss.item()

                postfixes = {f'val_{self.loss_name}': f'{val_loss / (batch_id) :.4f}',
                             'val_acc': f'{acc.get_acc():.4f}'}
                t.set_postfix(**postfixes)

                t.update()

        assert len(true_links) == len(predicted_links)
        auroc_score = roc_auc_score(true_links, predicted_links)

        return val_loss, acc.get_acc(), auroc_score

    def get_loss_value(self, embeddings, binary_predictions, lbls):
        if self.loss_name == 'bce':
            loss = self.loss_function(embeddings, lbls, output_pred=binary_predictions.flatten())
        else:
            loss = self.loss_function(embeddings, lbls.type(torch.int64))
        # elif self.loss_name == 'trpl':
        #     loss = self.loss_function(embeddings, lbls)
        return loss

    def get_embeddings(self, net):
        net.eval()
        val_size = self.val_db_loader.dataset.__len__()
        embeddings = np.zeros((val_size, self.emb_size), dtype=np.float32)
        classes = np.zeros((val_size,), dtype=np.float32)
        for batch_id, (imgs, lbls) in enumerate(self.val_db_loader):
            if self.cuda:
                imgs = imgs.cuda()

            preds, similarities, img_embeddings = net(imgs)

            begin_idx = batch_id * self.batch_size
            end_idx = min(val_size, (batch_id + 1) * self.batch_size)

            embeddings[begin_idx:end_idx, :] = img_embeddings.cpu().detach().numpy()
            classes[begin_idx:end_idx] = lbls.cpu().detach().numpy()

        return embeddings, classes

    def train(self, net, val=True):
        self.__set_optimizer(net)

        if self.train_loader is None:
            raise Exception(f'train_loader is not initialized in trainer')

        if self.optimizer is None:
            raise Exception(f'optimizer is not initialized in trainer')

        starting_epoch = max(1, self.current_epoch + 1)

        best_val_acc = -1

        # validate before training
        if val:
            with torch.no_grad():
                val_loss, val_acc, val_auroc_score = self.validate(net)

                embeddings, classes = self.get_embeddings(net)

                r_at_k_score = utils.get_recall_at_k(embeddings, classes,
                                                     metric='cosine',
                                                     sim_matrix=None)

                print(f'VALIDATION {self.current_epoch}-> val_loss: ', val_loss / len(self.val_loader),
                      f', val_acc: ', val_acc,
                      f', val_auroc: ', val_auroc_score,
                      f', val_R@K: ', r_at_k_score)

                list_for_tb = [('Val/Loss', val_loss / len(self.val_loader)),
                               ('Val/AUROC', val_auroc_score),
                               ('Val/Accuracy', val_acc)]

                for k, v in r_at_k_score.items():
                    list_for_tb.append((f'Val/{k}', v))

                self.__tb_update_value(list_for_tb)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                utils.save_model(net, self.current_epoch, best_val_acc, self.save_path)

        for epoch in range(starting_epoch, self.epochs + 1):

            self.current_epoch = epoch

            epoch_loss, epoch_acc = self.__train_one_epoch(net)

            print(f'Epoch {self.current_epoch}-> loss: ', epoch_loss / len(self.train_loader),
                  f', acc: ', epoch_acc)

            self.__tb_update_value([(f'Train/{self.loss_name}_Loss', epoch_loss / len(self.train_loader)),
                                    ('Train/Accuracy', epoch_acc)])

            if val:
                with torch.no_grad():
                    val_loss, val_acc, val_auroc_score = self.validate(net)

                    embeddings, classes = self.get_embeddings(net)

                    r_at_k_score = utils.get_recall_at_k(embeddings, classes,
                                                         metric='cosine',
                                                         sim_matrix=None)

                    # r_at_k_score.to_csv(
                    #     os.path.join(self.save_path, f'{self.args.get("dataset")}_{mode}_per_class_total_avg_k@n.csv'),
                    #     header=True,
                    #     index=False)

                    print(f'VALIDATION {self.current_epoch}-> val_loss: ', val_loss / len(self.val_loader),
                          f', val_acc: ', val_acc,
                          f', val_auroc: ', val_auroc_score,
                          f', val_R@K: ', r_at_k_score)

                    list_for_tb = [(f'Val/{self.loss_name}_Loss', val_loss / len(self.val_loader)),
                                   ('Val/AUROC', val_auroc_score),
                                   ('Val/Accuracy', val_acc)]

                    for k, v in r_at_k_score.items():
                        list_for_tb.append((f'Val/{k}', v))

                    self.__tb_update_value(list_for_tb)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    utils.save_model(net, self.current_epoch, best_val_acc, self.save_path)

            self.__tb_draw_histograms(net)

            if self.loss_name == 'pnpp':
                self.__tb_draw_histograms(self.loss_function)

            # if self.scheduler:
            #     self.scheduler.step()
            # else:
            #     self.adaptive_scheduler.step(current_loss=val_loss, current_val=val_acc)
