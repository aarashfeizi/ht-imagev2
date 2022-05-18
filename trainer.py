import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import losses
import utils
from SummaryWriter import SummaryWriter
from metrics import Metric_Accuracy
import optimizers


class Trainer:
    """
    Class for training and validating model
    """

    def __init__(self, args, loss, train_loader, val_loaders,
                 val_db_loaders, optimizer='adam', current_epoch=0, force_new_dir=True):
        self.batch_size = args.get('batch_size')
        self.emb_size = args.get('emb_size')
        self.args = args
        self.k_inc_freq = args.get('k_inc_freq')
        self.cuda = args.get('cuda')
        self.epochs = args.get('epochs')
        self.heatmap = args.get('draw_heatmaps')
        self.heatmap2x = args.get('draw_heatmaps2x')
        self.current_epoch = current_epoch
        self.cov_loss = None
        self.cov_loss_coefficient = args.get('cov_coef')
        self.var_loss_coefficient = args.get('var_coef')
        self.early_stopping_tol = args.get('early_stopping_tol')
        self.early_stopping_counter = 0

        if args.get('cov'):
            self.cov_loss = losses.covariance.COV_Loss(self.emb_size, static_mean=args.get('cov_static_mean'))

        self.loss_name = args.get('loss')
        self.loss_function = loss

        if args.get('with_bce'):
            self.bce_loss = losses.linkprediction.NormalBCELoss(args=args)
            self.bce_weight = args.get('bce_weight')
        else:
            self.bce_loss = None
            self.bce_weight = 0

        self.train_loader = train_loader
        self.val_loaders_dict = val_loaders
        self.val_db_loaders_dict = val_db_loaders
        self.optimizer_name = optimizer
        self.model_name = utils.get_model_name(self.args)
        self.optimizer = None
        self.tensorboard_path = None
        self.heatmap_loader = None
        self.heatmap2x_loader = None
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

    def set_heatmap2x_loader(self, loader):
        self.heatmap2x_loader = loader

    def set_train_loader(self, train_loader):
        self.train_loader = train_loader

    def set_val_loader(self, val_loaders):
        self.val_loaders_dict = val_loaders

    def set_val_db_loader(self, val_db_loader):
        self.val_db_loaders_dict = val_db_loader

    def __set_optimizer(self, net):

        if type(net) == torch.nn.DataParallel:
            netmod = net.module
        else:
            netmod = net

        if self.args.get('backbone') == 'resnet50':
            learnable_params = [{'params': netmod.encoder.rest.parameters(),
                                 'lr': self.args.get('learning_rate'),
                                 'weight_decay': self.args.get('weight_decay'),
                                 'new': False}]

            if netmod.encoder.last_conv is not None:
                learnable_params += [{'params': netmod.encoder.last_conv.parameters(),
                                      'lr': self.args.get('learning_rate') * self.args.get('new_lr_coef'),
                                      'weight_decay': self.args.get('weight_decay'),
                                      'new': True}]

            if netmod.final_projector is not None:
                learnable_params += [{'params': netmod.final_projector.parameters(),
                                      'lr': self.args.get('learning_rate') * self.args.get('new_lr_coef'),
                                      'weight_decay': self.args.get('weight_decay'),
                                      'new': True}]



            if len(netmod.projs) != 0:
                for p in netmod.projs:
                    learnable_params += [{'params': p.parameters(),
                                          'lr': self.args.get('learning_rate') * self.args.get('new_lr_coef'),
                                          'weight_decay': self.args.get('weight_decay'),
                                          'new': True}]
            if len(netmod.attQs) != 0:
                for p in netmod.attQs:
                    if p is not None:
                        learnable_params += [{'params': p.parameters(),
                                              'lr': self.args.get('learning_rate') * self.args.get('new_lr_coef'),
                                              'weight_decay': self.args.get('weight_decay'),
                                              'new': True}]
            if len(netmod.atts) != 0:
                for p in netmod.atts:
                    if p is not None:
                        learnable_params += [{'params': p.parameters(),
                                              'lr': self.args.get('learning_rate') * self.args.get('new_lr_coef'),
                                              'weight_decay': self.args.get('weight_decay'),
                                              'new': True}]
        else:
            learnable_params = [{'params': netmod.parameters(),
                                 'lr': self.args.get('learning_rate'),
                                 'weight_decay': self.args.get('weight_decay'),
                                 'new': False}]

        learnable_params += [{'params': self.loss_function.parameters(),
                              'lr': self.args.get('LOSS_lr'),
                              'new': True}]

        self.optimizer = optimizers.get_optimizer(args=self.args,
                                                  optimizer_name=self.optimizer_name,
                                                  learnable_params=learnable_params)

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
            self.tb_writer.add_image(f'{name}', img, global_step=self.current_epoch, dataformats='HWC')

        self.tb_writer.flush()



    def draw_heatmaps2x(self, net):
        if self.heatmap2x_loader is None:
            raise Exception('self.heatmap_loader is not set in trainer.py!!')

        for i, (imgs, lbls, paths) in enumerate(self.heatmap2x_loader):
            if self.cuda:
                imgs = imgs.cuda()

            e, activations = net.forward_with_pairwise_activations(imgs)  # returns embeddings, [f1, f2, f3, f4]

            img_names = []

            for path in paths:
                _, img_name = os.path.split(path)
                img_name = img_name[:img_name.find('.')]
                for path2 in paths:
                    _, img_name2 = os.path.split(path2)
                    img_name2 = img_name2[:img_name2.find('.')]
                    img_names.append(img_name + 'VS' + img_name2)

            org_imgs = []

            for path in paths:
                org_imgs.append(utils.transform_only_img(path))

            name_imgs = []
            heatmaps1 = {}
            heatmaps2 = {}
            if type(activations) is dict:
                for k, v in activations.items():  # 'org' and 'att'
                    b1, b2, _, _, _ = v[0].shape
                    assert b1 == b2
                    for i1 in range(b1):
                        for i2 in range(b2):
                            v_img1 = [temp[i1, i2:i2 + 1, :, :, :] for temp in v]
                            v_img2 = [temp[i2, i1:i1 + 1, :, :, :] for temp in v]
                            heatmaps1 = utils.get_all_heatmaps([v_img1], org_imgs[i1])
                            heatmaps2 = utils.get_all_heatmaps([v_img2], org_imgs[i2])

                    for name, heatmap1, heatmap2 in zip(img_names, heatmaps1, heatmaps2):
                        name_imgs.extend([(f'img_{name}_{k}/{n}', utils.concat_imgs(heatmap1[n], heatmaps2[n])) for n, _ in heatmap1.items()])

            # else:
            #     heatmaps = utils.get_all_heatmaps([activations], org_imgs)
            #
            #     for name, heatmap in zip(img_names, heatmaps):
            #         name_imgs.extend([(f'img_{name}/{n}', i) for n, i in heatmap.items()])

            self.__tb_draw_img(name_imgs)

    def draw_heatmaps(self, net):
        if self.heatmap_loader is None:
            raise Exception('self.heatmap_loader is not set in trainer.py!!')

        for i, (imgs, lbls, paths) in enumerate(self.heatmap_loader):
            if self.cuda:
                imgs = imgs.cuda()

            embeddings, activations = net.forward_with_activations(imgs) # returns embeddings, [f1, f2, f3, f4]

            img_names = []

            for path in paths:

                _, img_name = os.path.split(path)
                img_name = img_name[:img_name.find('.')]
                img_names.append(img_name)

            org_imgs = []

            for path in paths:
                org_imgs.append(utils.transform_only_img(path))

            name_imgs = []
            if type(activations) is dict:
                for k, v in activations.items():
                    heatmaps = utils.get_all_heatmaps([v], org_imgs)

                    for name, heatmap in zip(img_names, heatmaps):
                        name_imgs.extend([(f'img_{name}_{k}/{n}', i) for n, i in heatmap.items()])

            else:
                heatmaps = utils.get_all_heatmaps([activations], org_imgs)

                for name, heatmap in zip(img_names, heatmaps):
                    name_imgs.extend([(f'img_{name}/{n}', i) for n, i in heatmap.items()])

            self.__tb_draw_img(name_imgs)

    def __train_one_epoch(self, net):
        net.train()

        epoch_losses = None

        epoch_loss = 0

        acc = Metric_Accuracy()

        with tqdm(total=len(self.train_loader), desc=f'{self.current_epoch}/{self.epochs}') as t:
            for batch_id, (imgs, lbls) in enumerate(self.train_loader, 1):
                if self.cuda:
                    imgs = imgs.cuda()
                    lbls = lbls.cuda()

                if self.optimizer_name == 'adam':
                    utils.enable_running_stats(net)

                img_embeddings = net(imgs)
                preds, similarities = utils.get_preds(img_embeddings)
                bce_labels = utils.make_batch_bce_labels(lbls)

                loss, loss_items = self.get_loss_value(img_embeddings, preds, lbls)

                if torch.isnan(loss):
                    raise Exception(f'Loss became NaN on iteration {batch_id} of epoch {self.current_epoch}! :(')

                acc.update_acc(preds.flatten(), bce_labels.flatten(), sigmoid=False)

                epoch_loss += loss.item()

                if epoch_losses is None:
                    epoch_losses = loss_items
                else:
                    for k, v in loss_items.items():
                        epoch_losses[k] += v

                if self.optimizer_name != 'sam':
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                else:
                    loss.backward()
                    self.optimizer.first_step(zero_grad=True)

                    # second forward-backward step
                    utils.disable_running_stats(net)

                    img_embeddings_sam = net(imgs)
                    preds_sam, _ = utils.get_preds(img_embeddings_sam)

                    loss_sam, _ = self.get_loss_value(img_embeddings_sam, preds_sam, lbls)

                    loss_sam.backward()
                    self.optimizer.second_step(zero_grad=True)


                postfixes = {f'train_{self.loss_name}': f'{epoch_loss / (batch_id) :.4f}',
                             'train_acc': f'{acc.get_acc():.4f}'}
                t.set_postfix(**postfixes)

                t.update()

        if self.cov_loss and self.cov_loss.static_mean:
            self.cov_loss.reset_means()

        return epoch_losses, acc.get_acc()

    def validate(self, net, val_name, val_loader):
        if self.val_loaders_dict is None:
            raise Exception('val_loader is not set in trainer!')
        net.eval()

        val_losses = None
        val_loss = 0

        acc = Metric_Accuracy()

        predicted_links = []
        true_links = []

        with tqdm(total=len(val_loader), desc=f'{self.current_epoch} validating {val_name}...') as t:
            for batch_id, (imgs, lbls) in enumerate(val_loader, 1):
                if self.cuda:
                    imgs = imgs.cuda()
                    lbls = lbls.cuda()


                img_embeddings = net(imgs)
                preds, similarities = utils.get_preds(img_embeddings)
                bce_labels = utils.make_batch_bce_labels(lbls)
                loss, loss_items = self.get_loss_value(img_embeddings, preds, lbls, train=False)

                if val_losses is None:
                    val_losses = loss_items
                else:
                    for k, v in loss_items.items():
                        val_losses[k] += v

                # equal numbers of positives and negatives
                balanced_preds = utils.balance_labels(preds.cpu().detach().numpy(), k=3)
                balanced_bce_labels = utils.balance_labels(bce_labels.cpu().detach().numpy(), k=3)

                predicted_links.extend(balanced_preds.numpy())
                true_links.extend(balanced_bce_labels.numpy())

                acc.update_acc(balanced_preds.flatten(), balanced_bce_labels.flatten(), sigmoid=False)

                val_loss += loss.item()

                postfixes = {f'{val_name}_{self.loss_name}': f'{val_loss / (batch_id) :.4f}',
                             f'{val_name}_acc': f'{acc.get_acc():.4f}'}
                t.set_postfix(**postfixes)

                t.update()

        if self.cov_loss and self.cov_loss.static_mean:
            self.cov_loss.reset_means()

        assert len(true_links) == len(predicted_links)
        auroc_score = roc_auc_score(true_links, predicted_links)

        return val_losses, acc.get_acc(), auroc_score

    def get_loss_value(self, embeddings, binary_predictions, lbls, train=True):
        each_loss_item = {}
        if self.loss_name == 'bce' or self.loss_name == 'hardbce':
            loss = self.loss_function(embeddings, lbls, output_pred=binary_predictions, train=train)
        else:
            loss = self.loss_function(embeddings, lbls.type(torch.int64))

        each_loss_item[self.loss_name] = loss.item()

        if self.bce_loss:
            bce_loss_value = self.bce_loss(embeddings, lbls, output_pred=binary_predictions, train=train)

            loss = (self.bce_weight / self.bce_weight + 1) * bce_loss_value + \
                   (1 / self.bce_weight + 1) * loss

            each_loss_item['bce'] = bce_loss_value.item()
            each_loss_item[f'bce_{self.loss_name}'] = loss.item()

        if self.cov_loss:
            cov_loss_value, var_loss_value = self.cov_loss(embeddings)
            loss += self.cov_loss_coefficient * cov_loss_value
            loss += self.var_loss_coefficient * var_loss_value
            each_loss_item['cov'] = cov_loss_value.item()
            each_loss_item['var'] = var_loss_value.item()

        # elif self.loss_name == 'trpl':
        #     loss = self.loss_function(embeddings, lbls)
        return loss, each_loss_item

    def get_embeddings(self, net, data_loader=None):
        net.eval()
        if data_loader is None:
            data_loader = self.val_db_loaders_dict['val']

        val_size = data_loader.dataset.__len__()

        embeddings = np.zeros((val_size, self.emb_size), dtype=np.float32)
        classes = np.zeros((val_size,), dtype=np.float32)
        for batch_id, (imgs, lbls) in enumerate(data_loader):
            if self.cuda:
                imgs = imgs.cuda()

            img_embeddings = net(imgs)

            if len(img_embeddings.shape) == 3:
                img_embeddings = utils.get_diag_3d_tensor(img_embeddings)

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

        best_val_Rat1 = -1
        best_val_auroc_score = -1
        val_auroc_score = 0
        val_acc = 0

        if self.heatmap2x:
            self.draw_heatmaps2x(net)

        # validate before training
        if val:
            total_vals_Rat1 = 0.0
            total_vals_auroc = 0.0
            with torch.no_grad():

                for val_name, val_loader in self.val_loaders_dict.items():
                    if val_loader is None:
                        continue
                    capitalized_val_name = val_name[0].upper() + val_name[1:]

                    val_losses, val_acc, val_auroc_score = self.validate(net, capitalized_val_name, val_loader)

                    embeddings, classes = self.get_embeddings(net, data_loader=self.val_db_loaders_dict[val_name])

                    r_at_k_score = utils.get_recall_at_k(embeddings, classes,
                                                         metric='cosine',
                                                         sim_matrix=None)

                    all_val_losses = {lss_name: (lss / len(val_loader)) for lss_name, lss in val_losses.items()}

                    print(f'VALIDATION on {capitalized_val_name} {self.current_epoch}-> val_loss: ', all_val_losses,
                          f', val_acc: ', val_acc,
                          f', val_auroc: ', val_auroc_score,
                          f', val_R@K: ', r_at_k_score)

                    list_for_tb = [(f'{capitalized_val_name}/{lss_name}_Loss', lss / len(val_loader)) for lss_name, lss in
                                   val_losses.items()]
                    list_for_tb.append((f'{capitalized_val_name}/AUROC', val_auroc_score))
                    list_for_tb.append((f'{capitalized_val_name}/Accuracy', val_acc))
                    r_at_k_values = []
                    for k, v in r_at_k_score.items():
                        r_at_k_values.append(v)
                        list_for_tb.append((f'{capitalized_val_name}/{k}', v))

                    total_vals_Rat1 += r_at_k_values[0]
                    total_vals_auroc += val_auroc_score

                    self.__tb_update_value(list_for_tb)

                # if self.heatmap2x:
                #     self.draw_heatmaps2x(net)

                if self.heatmap:
                    self.draw_heatmaps(net)


            total_vals_Rat1 /= len(self.val_loaders_dict)
            total_vals_auroc /= len(self.val_loaders_dict)

            if total_vals_auroc > best_val_auroc_score:
                # best_val_acc = val_acc
                best_val_auroc_score = total_vals_auroc
                if self.args.get('save_model'):
                    utils.save_model(net, self.current_epoch, 'auc', self.save_path)
                else:
                    print('NOT SAVING MODEL!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            if total_vals_Rat1 > best_val_Rat1:
                # best_val_acc = val_acc
                best_val_Rat1 = total_vals_Rat1
                if self.args.get('save_model'):
                    utils.save_model(net, self.current_epoch, 'recall', self.save_path)
                else:
                    print('NOT SAVING MODEL!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        if self.epochs == 0:
            assert self.early_stopping_tol > 0
            max_epochs = 10000 + 1
        else:
            max_epochs = self.epochs + 1
        for epoch in range(starting_epoch, max_epochs):

            self.current_epoch = epoch

            epoch_losses, epoch_acc = self.__train_one_epoch(net)

            all_losses = {lss_name: (lss / len(self.train_loader)) for lss_name, lss in epoch_losses.items()}

            print(f'Epoch {self.current_epoch}-> loss: ', all_losses,
                  f', acc: ', epoch_acc)

            update_tb_losses = [(f'Train/{lss_name}_Loss', lss / len(self.train_loader)) for lss_name, lss in epoch_losses.items()]
            update_tb_losses.append(('Train/Accuracy', epoch_acc))

            self.__tb_update_value(update_tb_losses)

            total_vals_Rat1 = 0.0
            total_vals_auroc = 0.0
            if val:
                with torch.no_grad():

                    for val_name, val_loader in self.val_loaders_dict.items():
                        if val_loader is None:
                            continue
                        capitalized_val_name = val_name[0].upper() + val_name[1:]
                        val_losses, val_acc, val_auroc_score = self.validate(net, capitalized_val_name, val_loader)

                        embeddings, classes = self.get_embeddings(net, data_loader=self.val_db_loaders_dict[val_name])

                        r_at_k_score = utils.get_recall_at_k(embeddings, classes,
                                                             metric='cosine',
                                                             sim_matrix=None)

                        # r_at_k_score.to_csv(
                        #     os.path.join(self.save_path, f'{self.args.get("dataset")}_{mode}_per_class_total_avg_k@n.csv'),
                        #     header=True,
                        #     index=False)

                        all_val_losses = {lss_name: (lss / len(val_loader)) for lss_name, lss in val_losses.items()}

                        print(f'VALIDATION on {capitalized_val_name} {self.current_epoch}-> {capitalized_val_name}_loss: ', all_val_losses,
                              f', {capitalized_val_name}_acc: ', val_acc,
                              f', {capitalized_val_name}_auroc: ', val_auroc_score,
                              f', {capitalized_val_name}_R@K: ', r_at_k_score)

                        list_for_tb = [(f'{capitalized_val_name}/{lss_name}_Loss', lss / len(val_loader)) for lss_name, lss in
                                       val_losses.items()]
                        list_for_tb.append((f'{capitalized_val_name}/AUROC', val_auroc_score))
                        list_for_tb.append((f'{capitalized_val_name}/Accuracy', val_acc))
                        r_at_k_values = []
                        for k, v in r_at_k_score.items():
                            r_at_k_values.append(v)
                            list_for_tb.append((f'{capitalized_val_name}/{k}', v))

                        total_vals_Rat1 += r_at_k_values[0]
                        total_vals_auroc += val_auroc_score

                        self.__tb_update_value(list_for_tb)

                    if self.heatmap:
                        self.draw_heatmaps(net)

            total_vals_Rat1 /= len(self.val_loaders_dict)
            total_vals_auroc /= len(self.val_loaders_dict)

            if (val and total_vals_auroc > best_val_auroc_score) or \
                    (not val and epoch == max_epochs - 1):
                # best_val_acc = val_acc
                best_val_auroc_score = total_vals_auroc
                if self.args.get('save_model'):
                    utils.save_model(net, self.current_epoch, 'auc', self.save_path)
                else:
                    print('NOT SAVING MODEL!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

            if (val and total_vals_Rat1 >= best_val_Rat1) or \
                    (not val and epoch == max_epochs - 1):
                # best_val_acc = val_acc
                best_val_Rat1 = total_vals_Rat1
                self.early_stopping_counter = 0
                if self.args.get('save_model'):
                    utils.save_model(net, self.current_epoch, 'recall', self.save_path)
                else:
                    print('NOT SAVING MODEL!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            else:
                if self.early_stopping_tol > 0:
                    self.early_stopping_counter += 1

            self.__tb_draw_histograms(net)

            self.__tb_draw_histograms(self.loss_function)


            if (self.k_inc_freq != 0) and \
                    epoch % self.k_inc_freq == 0:

                self.args.set('num_inst_per_class', 2 * self.args.get('num_inst_per_class'))

                train_transforms, _ = utils.TransformLoader(self.args).get_composed_transform(
                    mode='train')
                train_loader = utils.get_data(self.args, mode='train', transform=train_transforms, sampler_mode='kbatch')
                self.train_loader = train_loader


            if self.early_stopping_tol > 0 and \
                    self.early_stopping_counter > self.early_stopping_tol:
                print(f'Early stoppping! {self.early_stopping_counter} epochs without r@1 improvement')
                break
            # if self.scheduler:
            #     self.scheduler.step()
            # else:
            #     self.adaptive_scheduler.step(current_loss=val_loss, current_val=val_acc)
