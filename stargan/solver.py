"""
solver.py

Author: Max Elliott

Solver class for training new models or previously saved models from a
checkpoint. Has capabilities for running extra auxiliary classifiers for tasks
that were never implemented (namely speaker classifiers and dimension classifiers).
Its generally safe to ignore anything in a "if self.use_speaker:"
or "if self.use_dimension:" block and these should be set to False by config.yaml.
"""
import random
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import audio_utils
import stargan.model as model
from stargan.logger import Logger
from stargan.sample_set import Sample_Set

from sklearn.metrics import accuracy_score


class Solver(object):

    def __init__(self, train_loader, test_loader, config, load_dir=None, recon_only=False):

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.sample_set = Sample_Set(config)
        self.config = config

        self.model_name = self.config['model']['name']
        self.model = model.StarGAN_emo_VC1(self.config, self.model_name)
        self.set_configuration()
        self.model = self.model

        self.recon_only = recon_only

        if load_dir is not None:
            self.load_checkpoint(load_dir)

    def load_checkpoint(self, load_dir):

        self.model.load(load_dir)
        self.config = self.model.config
        self.set_configuration()

    def set_configuration(self):

        # These are the INITIAL lr's. They are updated within the optimizers
        # over the training iterations
        self.g_lr = self.config['optimizer']['g_lr']
        self.d_lr = self.config['optimizer']['d_lr']
        self.emo_lr = self.config['optimizer']['emo_cls_lr']
        self.speaker_lr = self.config['optimizer']['speaker_cls_lr']
        self.dim_lr = self.config['optimizer']['dim_cls_lr']

        self.lambda_gp = self.config['loss']['lambda_gp']
        self.lambda_g_emo_cls = self.config['loss']['lambda_g_emo_cls']
        self.lambda_cycle = self.config['loss']['lambda_cycle']
        self.lambda_id = self.config['loss']['lambda_id']
        self.lambda_g_spk_cls = self.config['loss']['lambda_g_spk_cls']
        self.lambda_g_dim_cls = self.config['loss']['lambda_g_dim_cls']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.emo_loss_weights = torch.Tensor([4040./549, 4040./890, 4040./996, 4040./1605]).to(self.device)

        self.use_speaker = self.config['model']['use_speaker']
        self.use_dimension = self.config['model']['use_dimension']

        self.batch_size = self.config['model']['batch_size']

        self.num_iters = self.config['loss']['num_iters']
        self.num_iters_decay = self.config['loss']['num_iters_decay']
        self.resume_iters = self.config['loss']['resume_iters']
        self.current_iter = self.resume_iters

        # Number of D/emo_cls updates for each G update
        self.c_to_g_ratio = self.config['loss']['c_to_g_ratio']
        self.c_to_d_ratio = self.config['loss']['c_to_d_ratio']

        self.use_tensorboard = self.config['logs']['use_tensorboard']
        self.log_every = self.config['logs']['log_every']

        self.sample_dir = self.config['logs']['sample_dir']
        self.sample_every = self.config['logs']['sample_every']

        if 'test_every' in self.config['logs']:
            self.test_every = self.config['logs']['test_every']

        self.model_save_dir = self.config['logs']['model_save_dir']
        self.model_save_every = self.config['logs']['model_save_every']

        self.model_name = self.config['model']['name']
        self.model.name = self.model_name

        if self.use_tensorboard:
            self.logger = Logger(self.config['logs']['log_dir'], self.model_name)

    def set_classification_weights(self, weights):
        self.emo_loss_weights = weights.to(self.device)
        print("Set classification weights.")

    def train(self):
        """
        Main training loop
        """
        print('################ BEGIN TRAINING LOOP ################')

        start_iter = self.resume_iters + 1 # == 1 if new model

        self.update_lr(start_iter)

        # norm = Normalizer() #-------- ;;;WHAT DO I DO HERE ---------#
        data_iter = iter(self.train_loader)

        start_time = datetime.now()
        print('Started at {}'.format(start_time))

        # main training loop
        for i in range(start_iter, self.num_iters+1):

            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Iteration {:02}/{:02} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~".format(i,self.num_iters))
            print("Iteration {:02} lr = {:.6f}".format(i, self.model.d_optimizer.param_groups[0]['lr']))
            self.model.to_device(device=self.device)
            print("Device is ", self.device)
            print("Classifier device is ", self.model.emo_cls.device)
            self.model.set_train_mode()

            self.current_iter = i

            # Get data from data loader
            print('Getting mini-batch.')
            try:
                x, labels = next(data_iter)
            except:
                # print("In here")
                data_iter = iter(self.train_loader)
                x, labels = next(data_iter)

            x_real = x[0].to(device = self.device).unsqueeze(1)
            x_lens = x[1].to(device = self.device)
            print(f"solver.train: x_real size = {x_real.size()}")

            emo_labels = labels[:, 0].to(device=self.device)
            spk_labels = labels[:, 1].to(device=self.device)
            # ;;;;;;; GET DIM LABELS

            # Generate target domain labels randomly.
            num_emos = self.config['model']['num_classes']
            emo_targets = self.make_random_labels(num_emos, emo_labels.size(0))
            emo_targets = emo_targets.to(device=self.device)

            # one-hot versions of labels
            emo_labels_ones = F.one_hot(emo_labels, num_classes=num_emos).float().to(device=self.device)
            emo_targets_ones = F.one_hot(emo_targets, num_classes=num_emos).float().to(device=self.device)

            #############################################################
            #                    TRAIN CLASSIFIERS                      #
            #############################################################
            ce_weighted_loss_fn = nn.CrossEntropyLoss(weight=self.emo_loss_weights)
            ce_loss_fn = nn.CrossEntropyLoss()

            if self.config['loss']['train_classifier'] and not self.recon_only:
                print('Training Classifiers...')

                self.model.reset_grad()

                # Train with x_real
                preds_emo_real = self.model.emo_cls(x_real, x_lens)

                c_emo_real_loss = ce_weighted_loss_fn(preds_emo_real, emo_labels)

                c_emo_real_loss.backward()
                self.model.emo_cls_optimizer.step()

                if self.model.use_speaker:
                    self.model.reset_grad()

                    # Train with x_real
                    preds_speaker_real = self.model.speaker_cls(x_real, x_lens)

                    c_speaker_real_loss = ce_loss_fn(preds_speaker_real, spk_labels)

                    c_speaker_real_loss.backward()
                    self.model.speaker_cls_optimizer.step()

                if self.model.use_dimension:
                    self.model.reset_grad()

                    # Train with x_real
                    preds_dimension_real = self.model.dimension_cls(x_real, x_lens)

                    # ;;; DO FOR MULTILABEL
                    c_dimension_real_loss = ce_loss_fn(preds_dimension_real, dim_labels)

                    c_dimension_real_loss.backward()
                    self.model.speaker_cls_optimizer.step()
            else:
                print('No classifier training this run.')

            #############################################################
            #                    TRAIN DISCRIMINATOR                    #
            #############################################################
            if i % self.c_to_d_ratio == 0:
                print('Training Discriminator...')
                self.model.reset_grad()

                # Get results for x_fake
                x_fake = self.model.G(x_real, emo_targets_ones)
                # ;;; GET NEW X_LENS HERE

                # Get real/fake predictions
                d_preds_real = self.model.D(x_real, emo_labels_ones)
                d_preds_fake = self.model.D(x_fake.detach(), emo_targets_ones)
                # print(x_real.size())
                # print(x_fake.size())

                #C alculate loss
                grad_penalty = self.gradient_penalty(x_real, x_fake, emo_targets_ones)  # detach(), one hots?

                d_loss = -d_preds_real.mean() + d_preds_fake.mean() + \
                         self.lambda_gp * grad_penalty

                d_loss.backward()
                self.model.d_optimizer.step()
            else:
                print("No Discriminator update this iteration.")

            #############################################################
            #                      TRAIN GENERATOR                      #
            #############################################################
            if i % self.c_to_g_ratio == 0:
                print('Training Generator...')

                self.model.reset_grad()

                x_fake = self.model.G(x_real, emo_targets_ones)
                # ;;; GET NEW X_LENS HERE
                x_fake_lens = x_lens

                x_cycle = self.model.G(x_fake, emo_labels_ones)
                x_id = self.model.G(x_real, emo_labels_ones)
                d_preds_for_g = self.model.D(x_fake, emo_targets_ones)

                # x_cycle = self.make_equal_length(x_cycle, x_real)
                x_id = self.make_equal_length(x_id, x_real)

                l1_loss_fn = nn.L1Loss()

                loss_g_fake = - d_preds_for_g.mean()
                loss_cycle = l1_loss_fn(x_cycle, x_real)
                loss_id = l1_loss_fn(x_id, x_real)

                g_loss = loss_g_fake + self.lambda_id * loss_id + self.lambda_cycle * loss_cycle

                if not self.recon_only:
                    preds_emo_fake = self.model.emo_cls(x_fake, x_fake_lens)
                    loss_g_emo_cls = ce_weighted_loss_fn(preds_emo_fake, emo_targets)
                    g_loss += self.lambda_g_emo_cls * loss_g_emo_cls

                if self.use_speaker:

                    preds_spk_fake = self.model.speaker_cls(x_fake, x_fake_lens)
                    loss_g_spk_cls = ce_loss_fn(preds_spk_fake, spk_labels)
                    g_loss += self.lambda_g_spk_cls * loss_g_spk_cls

                if self.use_dimension:

                    preds_dim_fake = self.model.speaker_cls(x_fake, x_fake_lens)
                    loss_g_dim_cls = ce_loss_fn(preds_dim_fake, dim_labels)
                    g_loss += self.lambda_g_dim_cls * loss_g_dim_cls

                g_loss.backward()
                self.model.g_optimizer.step()
            else:
                print("No Generator update this iteration.")

            #############################################################
            #                  PRINTING/LOGGING/SAVING                  #
            #############################################################
            if i % self.log_every == 0:
                loss = {}
                loss['D/total_loss'] = d_loss.item()
                loss['G/total_loss'] = g_loss.item()
                loss['D/gradient_penalty'] = grad_penalty.item()
                loss['G/loss_cycle'] = loss_cycle.item()
                loss['G/loss_id'] = loss_id.item()
                loss['D/preds_real'] = d_preds_real.mean().item()
                loss['D/preds_fake'] = d_preds_fake.mean().item()
                if not self.recon_only:
                    loss['G/loss_g_emo_cls'] = loss_g_emo_cls.mean().item()

                if self.config['loss']['train_classifier'] and not self.recon_only:
                    loss['C/emo_real_loss'] = c_emo_real_loss.item()
                    if self.use_speaker:
                        loss['C/spk_real_loss'] = c_speaker_real_loss.item()
                        loss['G/spk_loss'] = loss_g_spk_cls.item()

                    if self.use_dimension:
                        loss['C/dim_real_loss'] = c_dimension_real_loss_real_loss.item()
                        loss['G/dim_loss'] = loss_g_dim_cls.item()

                for name, val in loss.items():
                    print("{:20} = {:.4f}".format(name, val))

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i)
            else:
                print("No log output this iteration.")

            # save checkpoint
            if i % self.model_save_every == 0:
                self.model.save(save_dir=self.model_save_dir, it=self.current_iter)
            else:
                print("No model saved this iteration.")

            if i % self.test_every == 0:
                self.test()
            # generate example samples from test set ;;; needs doing
            if i % self.sample_every == 0:
                if self.config['data']['type'] == 'mel':
                    self.sample_mel()
                else:
                    self.sample_world()

            # update learning rates
            self.update_lr(i)

            elapsed = datetime.now() - start_time
            print('{} elapsed. Iteration {:04} complete'.format(elapsed, i))

        self.model.save(save_dir=self.model_save_dir, it=self.current_iter)

    def test(self):

        # test_iter = iter(self.test_loader)
        print("Testing generator accuracy ...")
        self.model.set_eval_mode()

        real_preds = torch.rand(0).to(device=self.device, dtype=torch.long)
        fake_preds = torch.rand(0).to(device=self.device, dtype=torch.long)
        id_preds = torch.rand(0).to(device=self.device, dtype=torch.long)
        cycle_preds = torch.rand(0).to(device=self.device, dtype=torch.long)

        total_labels = torch.rand(0).to(device=self.device, dtype=torch.long)
        total_targets = torch.rand(0).to(device=self.device, dtype=torch.long)

        total_real = torch.rand(0).to(device=self.device, dtype=torch.float)
        total_id = torch.rand(0).to(device=self.device, dtype=torch.float)
        total_cycle = torch.rand(0).to(device=self.device, dtype=torch.float)
        l1_loss_fn = nn.L1Loss()

        for i, (x, labels) in enumerate(self.test_loader):

            x_real = x[0].to(device=self.device)
            x_lens = x[1].to(device=self.device)

            x_real = x_real.unsqueeze(1)

            emo_labels = labels[:, 0].to(device=self.device)
            spk_labels = labels[:, 1].to(device=self.device)

            # Generate target domain labels randomly.
            num_emos = self.config['model']['num_classes']
            emo_targets = self.make_random_labels(num_emos, emo_labels.size(0))
            emo_targets = emo_targets.to(device=self.device)

            # one-hot versions of labels
            emo_labels_ones = F.one_hot(emo_labels, num_classes=num_emos).float().to(device=self.device)
            emo_targets_ones = F.one_hot(emo_targets, num_classes=num_emos).float().to(device=self.device)

            with torch.no_grad():

                x_fake = self.model.G(x_real, emo_targets_ones)
                x_id = self.model.G(x_real, emo_labels_ones)
                x_cycle = self.model.G(x_fake, emo_labels_ones)

                new_lens = x_lens #;;; needs doing

                c_emo_real = self.model.emo_cls(x_real, x_lens)
                c_emo_fake = self.model.emo_cls(x_fake, x_lens)
                c_emo_id = self.model.emo_cls(x_id, x_lens)
                c_emo_cycle = self.model.emo_cls(x_cycle, x_lens)

                c_emo_real = torch.max(c_emo_real, dim = 1)[1]
                c_emo_fake = torch.max(c_emo_fake, dim = 1)[1]
                c_emo_id = torch.max(c_emo_id, dim = 1)[1]
                c_emo_cycle = torch.max(c_emo_cycle, dim = 1)[1]

                # D as well
                real_preds = torch.cat((real_preds, c_emo_real), dim=0)
                fake_preds = torch.cat((fake_preds, c_emo_fake), dim=0)
                id_preds = torch.cat((id_preds, c_emo_id), dim=0)
                cycle_preds = torch.cat((cycle_preds, c_emo_cycle), dim=0)

                total_real = torch.cat((total_real, x_real), dim=0)
                total_id = torch.cat((total_id, x_id), dim=0)
                total_cycle = torch.cat((total_cycle, x_cycle), dim=0)

                total_labels = torch.cat((total_labels, emo_labels), dim=0)
                total_targets = torch.cat((total_targets, emo_targets), dim=0)

        accuracy_real = accuracy_score(total_labels.cpu(), real_preds.cpu())
        accuracy_fake = accuracy_score(total_targets.cpu(), fake_preds.cpu())
        accuracy_id = accuracy_score(total_labels.cpu(), id_preds.cpu())
        accuracy_cycle = accuracy_score(total_labels.cpu(), cycle_preds.cpu())

        L1_loss_id = l1_loss_fn(total_id, total_real).item()
        L1_loss_cycle = l1_loss_fn(total_cycle, total_real).item()
        # print(L1_loss_id)
        # print(L1_loss_cycle)

        l = ["Accuracy_real", "Accuracy_fake", "Accuracy_id", "Accuracy_cycle", "L1_id", "L1_cycle"]

        print('{:20} = {:.3f}'.format(l[0], accuracy_real))
        print('{:20} = {:.3f}'.format(l[1], accuracy_fake))
        print('{:20} = {:.3f}'.format(l[2], accuracy_id))
        print('{:20} = {:.3f}'.format(l[3], accuracy_cycle))
        print('{:20} = {:.3f}'.format(l[4], L1_loss_id))
        print('{:20} = {:.3f}'.format(l[5], L1_loss_cycle))

        if self.use_tensorboard:
            self.logger.scalar_summary("Val/test_accuracy_real", accuracy_real, self.current_iter)
            self.logger.scalar_summary("Val/test_accuracy_fake", accuracy_fake, self.current_iter)
            self.logger.scalar_summary("Val/test_accuracy_id", accuracy_id, self.current_iter)
            self.logger.scalar_summary("Val/test_accuracy_cycle", accuracy_cycle, self.current_iter)
            self.logger.scalar_summary("Val/test_L1_id", L1_loss_id, self.current_iter)
            self.logger.scalar_summary("Val/test_L1_cycle", L1_loss_cycle, self.current_iter)

    def sample_mel(self):
        """
        Passes each performance sample through G for every target emotion. They
        are saved to 'config(sample_dir)/model_name/filename-<emo>to<trg>.png + .npy'
        """

        print("Saving mel samples...")

        self.model.to_device(device = self.device)
        self.model.set_eval_mode()

        # Make one-hot vector for each emotion category
        num_emos = self.config['model']['num_classes']
        emo_labels = torch.Tensor(range(0,num_emos)).long()
        emo_targets = F.one_hot(emo_labels, num_classes = num_emos).float().to(device = self.device)

        for tag, val in self.sample_set.get_set().items():
            # tag is filename, val is [mel, labels, spec]

            mel = val[0].unsqueeze(0).unsqueeze(0).to(device = self.device)
            labels = val[1]

            with torch.no_grad():
                # print(emo_targets)
                for i in range (0, emo_targets.size(0)):

                    fake = self.model.G(mel, emo_targets[i].unsqueeze(0))

                    filename_png = tag[0:-4] + "_" + str(int(labels[0].item())) + "to" + \
                                   str(emo_labels[i].item()) + '_i=' +\
                                   str(self.current_iter) + ".png"

                    filename_npy = tag[0:-4] + "_" + str(int(labels[0].item())) + "to" + \
                                   str(emo_labels[i].item()) + '_i=' +\
                                   str(self.current_iter) + ".npy"

                    fake = fake.squeeze()
                    audio_utils.save_spec_plot(fake.t(), self.model_name, filename_png)
                    audio_utils.save_spec(fake.t(), self.model_name, filename_npy)

    def sample_world(self):
        """
        Passes each performance sample through G for every target emotion. They
        are saved to 'config(sample_dir)/model_name/filename-<emo>to<trg>.png + .npy'
        """

        print("Saving world samples...")

        self.model.to_device(device = self.device)
        self.model.set_eval_mode()

        # Make one-hot vector for each emotion category
        num_emos = self.config['model']['num_classes']
        emo_labels = torch.Tensor(range(0,num_emos)).long()
        emo_targets = F.one_hot(emo_labels, num_classes = num_emos).float().to(device = self.device)

        for tag, val in self.sample_set.get_set().items():
            # tag is filename, val is [mel, labels, spec]

            f0_real = np.copy(val[0])
            ap_real = np.copy(val[1])
            sp = np.copy(val[2])
            coded_sp = torch.Tensor.clone(val[3])
            labels = torch.Tensor.clone(val[4])

            coded_sp = val[3].unsqueeze(0).unsqueeze(0).to(device = self.device)

            with torch.no_grad():

                for i in range(0, emo_targets.size(0)):

                    f0 = np.copy(f0_real)
                    ap = np.copy(ap_real)

                    fake = self.model.G(coded_sp, emo_targets[i].unsqueeze(0))

                    filename_wav = tag[0:-4] + "_" + str(int(labels[0].item())) + "to" + \
                                   str(emo_labels[i].item()) + '_i=' +\
                                   str(self.current_iter) + ".wav"

                    fake = fake.squeeze()
                    # print("Sampled size = ",fake.size())
                    # f = fake.data()
                    converted_sp = fake.cpu().numpy()
                    converted_sp = np.array(converted_sp, dtype = np.float64)

                    sample_length = converted_sp.shape[0]
                    if sample_length != ap.shape[0]:
                        ap = np.ascontiguousarray(ap[0:sample_length, :], dtype = np.float64)
                        f0 = np.ascontiguousarray(f0[0:sample_length], dtype = np.float64)

                    f0 = np.ascontiguousarray(f0[40:-40], dtype = np.float64)
                    ap = np.ascontiguousarray(ap[40:-40,:], dtype = np.float64)
                    converted_sp = np.ascontiguousarray(converted_sp[40:-40,:], dtype = np.float64)

                    # print("ap shape = ", val[1].shape)
                    # print("f0 shape = ", val[0].shape)

                    audio_utils.save_world_wav([f0, ap, sp, converted_sp], self.model_name, filename_wav)

    def update_lr(self, i):
        """Decay learning rates of the generator and discriminator and classifier."""
        if self.num_iters - self.num_iters_decay < i:
            decay_delta_d = self.d_lr / self.num_iters_decay
            decay_delta_g = self.g_lr / self.num_iters_decay
            decay_delta_emo = self.emo_lr / self.num_iters_decay

            decay_start = self.num_iters - self.num_iters_decay
            decay_iter = i - decay_start

            d_lr = self.d_lr - decay_iter * decay_delta_d
            g_lr = self.g_lr - decay_iter * decay_delta_g
            emo_lr = self.emo_lr - decay_iter * decay_delta_emo

            for param_group in self.model.g_optimizer.param_groups:
                param_group['lr'] = g_lr
            for param_group in self.model.d_optimizer.param_groups:
                param_group['lr'] = d_lr
            for param_group in self.model.emo_cls_optimizer.param_groups:
                param_group['lr'] = emo_lr

            # All use same lr now
            if self.use_speaker:
                for param_group in self.model.speaker_cls_optimizer.param_groups:
                    param_group['lr'] = emo_lr
            if self.use_dimension:
                for param_group in self.model.dimension_cls_optimizer.param_groups:
                    param_group['lr'] = emo_lr

    def make_random_labels(self, num_domains, num_labels):
        """
        Creates random labels for generator.
        num_domains: number of unique labels
        num_labels: total number of labels to generate
        """
        domain_list = np.arange(0, num_domains)
        # print(domain_list)
        labels = torch.zeros((num_labels))
        for i in range(0, num_labels):
            labels[i] = random.choice(domain_list).item()

        return labels.long()

    def gradient_penalty(self, x_real, x_fake, targets):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2.
        Taken from https://github.com/hujinsen/pytorch-StarGAN-VC"""
        # Compute loss for gradient penalty.
        alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
        x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
        out_src = self.model.D(x_hat, targets)

        weight = torch.ones(out_src.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=out_src,
                                   inputs=x_hat,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1)+ 1e-12)
        return torch.mean((dydx_l2norm-1)**2)

    def make_equal_length(self, x_out, x_real):
        ''' Needs implementing'''

        return x_out


if __name__ == '__main__':
    pass
