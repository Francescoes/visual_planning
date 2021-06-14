#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 12:32:19 2020

@author: petrapoklukar
"""

from algorithms import LBN_Algorithm
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import cv2
import torch
import algorithms.EarlyStopping as ES
from datetime import datetime

# ---
# ====================== Training functions ====================== #
# ---
class LBN_pushing(LBN_Algorithm):
    def __init__(self, opt):
        super().__init__(opt)


    def round_labels(self, x):

        # rescaled = x.cpu().numpy() * (self.data_max - self.data_min) + self.data_min
        rounded_labels = np.around(x.cpu().numpy()).astype(int)

        # Filter out of the range labels because MSE can be out
        cropped_rounded_labels = np.maximum(0.0, np.minimum(rounded_labels, 1.0))
        # assert(np.all(cropped_rounded_labels) >= self.data_min)
        # assert(np.all(cropped_rounded_labels) <= self.data_max)
        return cropped_rounded_labels.astype(int)


    def get_box_center_from_x_y(self, array):
        """
        Returns the center coordinates corresponding to the box where the LBN
        prediction x and y are pointing to. It assumes top left corner = (0,0),
        x increasing positively downwards and y towards the right.
        """
        x, y = array[1], array[0]
        # cx_vec = [55,115,185] # wrt the image coordinates (x positive towards right)
        # cy_vec = [87,140,190] # wrt the image coordinates (y positive towards down)
        cx_vec = [35,100,160,220] # wrt the image coordinates (x positive towards right)
        cy_vec = [40,130,220,300] # wrt the image coordinates (y positive towards down)

        cx = cx_vec[x]
        cy = cy_vec[y]
        return (cx,cy)


    # def plot_prediction(self, img1, img2, pred_coords_scaled, coords_scaled,
    #                         split='train', n_subplots=3, new_save_path=None):
    #     """Plots the LBN predictions on the given (no-)action pair."""
    #     img1 = self.vae.decoder(img1)[0]
    #     img2 = self.vae.decoder(img2)[0]
    #
    #     # Descale coords back to the original size
    #     pred_coords = self.descale_coords(pred_coords_scaled.detach())
    #     coords = self.descale_coords(coords_scaled)
    #
    #     plt.figure(1)
    #     for i in range(n_subplots):
    #         # Start state predictions and ground truth
    #         plt.subplot(n_subplots, 2, 2*i+1)
    #         pred_pick_xy = self.get_box_center_from_x_y(pred_coords[i][:2])
    #         actual_pick_xy = self.get_box_center_from_x_y(coords[i][:2])
    #         state1_img = (img1[i].detach().cpu().numpy().transpose(1, 2, 0).copy() * 255).astype(np.uint8)
    #         marked_img1 = cv2.circle(state1_img, tuple(pred_pick_xy), 10, (255, 0, 0), -1)
    #         marked_img1 = cv2.circle(marked_img1, tuple(actual_pick_xy), 15, (0, 255, 0), 4)
    #         fig=plt.imshow(marked_img1)
    #
    #         # Start state predicted height for the robot and ground truth
    #         pred_pick_height = round(pred_coords_scaled[i][2].detach().item())
    #         plt.title('State 1, h_pred {0}'.format(pred_pick_height))
    #         fig.axes.get_xaxis().set_visible(False)
    #         fig.axes.get_yaxis().set_visible(False)
    #
    #         # End state predictions and ground truth
    #         plt.subplot(n_subplots, 2, 2*i+2)
    #         pred_place_xy = self.get_box_center_from_x_y(pred_coords[i][3:])
    #         actual_place_xy = self.get_box_center_from_x_y(cinput_uvoords[i][3:])
    #
    #         state2_img = (img2[i].detach().cpu().numpy().transpose(1, 2, 0).copy() * 255).astype(np.uint8)
    #         marked_img2 = cv2.circle(state2_img, tuple(pred_place_xy), 10, (255, 0, 0), -1)
    #         marked_img2 = cv2.circle(marked_img2, tuple(actual_place_xy), 15, (0, 255, 0), 4)
    #         fig=plt.imshow(marked_img2)
    #         plt.title('State 2')
    #         fig.axes.get_xaxis().set_visible(False)
    #         fig.axes.get_yaxis().set_visible(False)
    #
    #     if new_save_path:
    #         plt.savefig(new_save_path)
    #     else:
    #         plt.savefig(self.save_path + '_Predictions' + split + str(self.current_epoch))
    #     plt.clf()
    #     plt.close('all')
    #     # cv2.destroyAllWindows()


    def train(self, train_dataset, test_dataset, num_workers=0, chpnt_path=''):
        """Trains an LBN model with given hyperparameters."""
        dataloader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True,
                num_workers=num_workers, drop_last=False)
        n_data = len(train_dataset)

        self.data_min = train_dataset.min
        self.data_max = train_dataset.max

        print(('\nPrinting model specifications...\n' +
               ' *- Path to the model: {0}\n' +
               ' *- Training dataset: {1}\n' +
               ' *- Number of training samples: {2}\n' +
               ' *- Number of epochs: {3}\n' +
               ' *- Batch size: {4}\n'
               ).format(self.model_path, train_dataset.dataset_name, n_data,
                   self.epochs, self.batch_size))

        if chpnt_path:
            # Pick up the last epochs specs
            self.load_checkpoint(chpnt_path)

        else:
            # Initialise the model
            self.model = self.init_model()
            self.start_epoch, self.lr = self.lr_schedule.pop(0)
            try:
                self.lr_update_epoch, self.new_lr = self.lr_schedule.pop(0)
            except:
                self.lr_update_epoch, self.new_lr = self.start_epoch - 1, self.lr
            self.optimiser = self.init_optimiser()
            self.training_losses = []
            self.valid_losses = []
            self.epoch_losses = []
            self.epoch_times = []
            self.training_time = 0
            print((' *- Learning rate: {0}\n' +
                   ' *- Next lr update at {1} to the value {2}\n' +
                   ' *- Remaining lr schedule: {3}'
                   ).format(self.lr, self.lr_update_epoch, self.new_lr,
                   self.lr_schedule))

        self.load_vae()
        es = ES.EarlyStopping(patience=20)
        num_parameters = self.count_parameters()
        self.opt['num_parameters'] = num_parameters
        print(' *- Model parameter/training samples: {0}'.format(
                num_parameters/len(train_dataset)))
        print(' *- Model parameters: {0}'.format(num_parameters))

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                spacing = 1
                print('{0:>2}{1}\n\t of dimension {2}'.format('', name, spacing),
                      list(param.shape))

        print('\nStarting to train the model...\n' )
        training_start = datetime.now()
        for self.current_epoch in range(self.start_epoch, self.epochs):
            self.model.train()
            self.update_learning_rate(self.optimiser)
            epoch_loss = np.zeros(4)
            epoch_start = datetime.now()

            for batch_idx, (img1, img2, labels) in enumerate(dataloader):
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                labels = labels.float().to(self.device)

                # LBNet loss
                pred_labels = self.model(img1, img2)

                (the_loss, inputXloss, inputYloss) = self.compute_loss(pred_labels, labels)

                epoch_loss += self.format_loss([the_loss, inputXloss, inputYloss])

                # Optimise the model
                self.optimiser.zero_grad()
                the_loss.backward()
                self.optimiser.step()

                # Monitoring the learning
                epoch_loss += self.format_loss([the_loss, inputXloss, inputYloss])

            # Monitor the time
            epoch_end = datetime.now()
            epoch_time = epoch_end - epoch_start
            self.epoch_times.append(epoch_time)
            self.training_time = training_start - epoch_end

            # Plot learning curves
            epoch_loss /= len(dataloader)
            epoch_loss[-1] = int(self.current_epoch)
            self.epoch_losses.append(epoch_loss)
            self.plot_model_loss()

            valid_loss = self.compute_test_loss(test_dataset)
            valid_loss[-1] = int(self.current_epoch)
            self.valid_losses.append(valid_loss)
            self.plot_learning_curve()
            self.plot_epoch_time()

            # Update the best model
            try:
                if es.keep_best(valid_loss[0]):
                    self.best_model= {
                            'model': self.model,
                            'epoch': self.current_epoch,
                            'train_loss': epoch_loss[0],
                            'valid_loss': valid_loss[0]
                        }
                    print(' *- New best model at epoch ', self.current_epoch)
            except AssertionError:
                break

            # Update the checkpoint only if there was no early stopping
            self.save_checkpoint(epoch_loss[0])

            # Print current loss values every epoch
            if (self.current_epoch + 1) % self.console_print == 0:
                print('Epoch {0}: [{1}]'.format(self.current_epoch, epoch_time))
                print('   Train loss: {0:.3f} inputX: {1:.3f} inputY: {2:.3f} '.format(*epoch_loss))
                print('   Valid loss: {0:.3f} inputX: {1:.3f} inputY: {2:.3f} '.format(*valid_loss))
                print('   LR: {0:.6e}\n'.format(self.lr))

            # Print validation results when specified
            if (self.current_epoch + 1) % self.snapshot == 0:

                # # Plot LBN predictions
                # self.plot_prediction(img1, img2, pred_coords, coords)
                self.model.eval()

                # Plot training and validation loss
                self.save_checkpoint(epoch_loss[0], keep=True)

                # Write logs
                self.save_logs(train_dataset, test_dataset)
                self.plot_snapshot_loss()

        print('Training completed.')
        training_end = datetime.now()
        self.training_time = training_end - training_start

        self.plot_model_loss()
        self.plot_epoch_time()
        self.model.eval()

        # Save the model
        self.save_checkpoint(epoch_loss[0], keep=True)
        torch.save(self.best_model['model'].state_dict(), self.model_path)
        torch.save(self.model.state_dict(), self.save_path + '_lastModel.pt')
        self.save_logs(train_dataset, test_dataset)

        # Plot predetermined test images for a fair comparisson among models
        self.plot_test_images(test_dataset)


    def score_model(
            self, model_name, path_to_valid_dataset, path_to_result_file,
            load_checkpoint=False, path_to_chpnt='', suffix='', noise=False):
        """Scores a trained LBN model on the test set."""

        # Load the dataprint("##################",coords_array)
        with open(path_to_valid_dataset, 'rb') as f:
            valid_data = pickle.load(f)
            print(' *- Loaded data from ', path_to_valid_dataset)

        if type(valid_data) == dict:
            threshold_min = valid_data['min']
            threshold_max = valid_data['max']
            valid_data = valid_data['data']
        else:
            threshold_min, threshold_max = 0., 1.

        self.data_min = threshold_min
        self.data_max = threshold_max

        # Load the trained vae network
        self.load_vae()

        # Load the trained lbn network
        self.model = self.init_model()
        if load_checkpoint:
            checkpoint = torch.load(path_to_chpnt, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(' *- LBN loaded from a checkpoint.')
        else:
            path_to_trained_LBNmodel = './models/{0}/lbnet_model.pt'.format(model_name)
            self.model.load_state_dict(torch.load(path_to_trained_LBNmodel,
                                                  map_location=self.device))
        self.model.eval()
        assert(not self.vae.training)
        assert(not self.model.training)

        # Init the counters
        pred_label_list = []
        actual_label_list = []
        n_points = 0

        # Categorical
        label_1_score, label_2_score = 0, 0
        pick_score, label_score, total_score = 0, 0, 0

        # MSE
        label_1_reg_score, label_2_reg_score = 0, 0
        total_reg_score = 0

        # Compute the error
        for item in valid_data:
            n_points += 1

            # (img1, img2, 1, input_uv, output_uv)
            img1 = item[0]
            img2 = item[1]

            input_uv = item[3].reshape(-1, 1)
            labels_array_scaled =  np.array(input_uv).astype('float32')

            # # Scale the rest
            # labels_array_scaled = (labels_array - threshold_min)/(threshold_max - threshold_min)

            assert(np.all(labels_array_scaled) >= 0. and np.all(labels_array_scaled) <= 1.)

            if noise:
                # Add noise to the labels
                tiny_noise = np.random.uniform(-0.1, 0.1, size=(2, 1))
                tiny_noise[2] = 0.
                noisy_labels_array_scaled = labels_array_scaled + tiny_noise
                new_noisy_normalised_labels = np.maximum(0., np.minimum(noisy_labels_array_scaled, 1.))
                labels = torch.from_numpy(new_noisy_normalised_labels).transpose(1, 0) # shape (1, 2)
            else:
                labels = torch.from_numpy(labels_array_scaled).transpose(1, 0) # shape (1, 2)

            # VAE forward pass
            img1 = img1.to(self.device).unsqueeze(0).float()
            img2 = img2.to(self.device).unsqueeze(0).float()
            enc_mean1, _ = self.vae.encoder(img1)
            enc_mean2, _ = self.vae.encoder(img2)

            # Get the predictions from the LBN
            pred_labels = self.model(enc_mean1, enc_mean2)
            pred_labels_np = self.round_labels(pred_labels.detach()).squeeze()
            labels_np = self.round_labels(labels).squeeze() # shape (2, )

            # Append for histogram plots
            pred_label_list.append(pred_labels_np.reshape(-1, 1))
            actual_label_list.append(labels_np.reshape(-1, 1))

            # Compute the mse loss and log the resutls
            (the_loss, inputXloss, inputYloss) = self.compute_loss(pred_labels.float(), labels.float().to(self.device))

            # MSE loss
            label_1_reg_score += inputXloss.item()
            label_2_reg_score += inputYloss.item()
            total_reg_score += the_loss.item()

            # Categorical loss on individual labels
            correct_label_1 = 1 if pred_labels_np[0] == labels_np[0] else 0
            correct_label_2 = 1 if pred_labels_np[1] == labels_np[1] else 0

            label_1_score += correct_label_1
            label_2_score += correct_label_2
            label_score += (correct_label_1 + correct_label_2 )//2

            # Categorical loss on start and end predictions
            correct_pick = 1 if sum(abs(pred_labels_np[:2] - labels_np[:2])) == 0 else 0

            pick_score += correct_pick
            total_score += correct_pick

            # --- Plot some for visual inspection
            # if n_points % 100 == 0:
            #     self.plot_prediction(
            #             enc_mean1, enc_mean2, pred_labels, labels,
            #             split='valid' + str(n_points), n_subplots=1)

        # Normalise the MSE errors
        label_1_reg_score /= n_points
        label_2_reg_score /= n_points
        total_reg_score /= n_points

        # Normalise the categorical errors
        label_1_score /= n_points
        label_2_score /= n_points
        label_score /= n_points

        results_d = {
                'model_name': model_name,
                'n_points': n_points,
                'random_seed': self.random_seed,

                'label_1_avgdisterrror': round(label_1_reg_score, 2),
                'label_2_avgdisterrror': round(label_2_reg_score, 2),
                'total_avgdisterrror': round(total_reg_score, 2),

                'label_1_score': round(label_1_score, 2),
                'label_2_score': round(label_2_score, 2),
                'label_score': round(label_score, 2),
                'pick_score_per': round(pick_score/n_points, 2),
                'total_score_per': round(total_score/n_points, 2),
                }

        print('\nValidation scores:')
        print(results_d)
        print('\n')
        import pandas as pd
        df = pd.DataFrame.from_dict([results_d])
        df.to_csv(path_to_result_file, header=None, index=False, mode='a')
        return results_d
