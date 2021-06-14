#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 13:16:31 2020

@author: petrapoklukar
"""

from algorithms import APN_Algorithm
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import cv2
import torch
import algorithms.EarlyStopping as ES
from datetime import datetime
import os


class EncoderAPN_folding(APN_Algorithm):
    def __init__(self, opt):
        super().__init__(opt)
        self.norm_param_d = None
        self.img_size = None
        self.data_min = None
        self.data_max = None    
        
        
    def load_norm_param_d(self, path_to_norm_param_d, random_seed):
        """Loads the dict with normalisation parameters"""
        
        path = path_to_norm_param_d + str(random_seed) + '.pkl'
        with open(path, 'rb') as f:
            self.norm_param_d = pickle.load(f)
        
        self.path_to_norm_param_d = path_to_norm_param_d
        self.norm_mean = np.array([
                self.norm_param_d['inputX_mu'], self.norm_param_d['inputY_mu'], 0, 
                self.norm_param_d['outputX_mu'], self.norm_param_d['outputY_mu']])
        self.norm_std = np.array([
                self.norm_param_d['inputX_std'], self.norm_param_d['inputY_std'], 1, 
                self.norm_param_d['outputX_std'], self.norm_param_d['outputY_std']])
        self.norm_mean_tensor = torch.from_numpy(self.norm_mean)
        self.norm_std_tensor = torch.from_numpy(self.norm_std)
    
    
    def descale_coords(self, x):
        """
        Descales the coordinates from [0, 1] interval back to the original 
        image size.
        """
        rescaled = x * (self.data_max - self.data_min) + self.data_min
        rounded_coords = np.around(rescaled).astype(int)
       
        # Filter out of the range coordinates because MSE can be out
        cropped_rounded_coords = np.maximum(
                self.data_min, np.minimum(rounded_coords, self.data_max))
        
        assert(np.all(cropped_rounded_coords) >= self.data_min)
        assert(np.all(cropped_rounded_coords) <= self.data_max)
        return cropped_rounded_coords.astype(int)
    
    
    def denormalise(self, x):
        """
        Denormalises the coordinates with the mean and std of the training
        split. The resulting coordinates are in the [0, 1] interval.
        """
        denormalised = x.cpu().numpy() * self.norm_std + self.norm_mean
        assert(np.all(denormalised) >= 0.)
        assert(np.all(denormalised) <= 1.)
        return denormalised
    
    
    def normalise(self, x):
        """
        Normalises the coordinates with the mean and std of the training
        split.
        """
        normalised = (x - self.norm_mean_tensor) / self.norm_std_tensor
        return normalised
    
    
    def plot_prediction(self, img1, img2, pred_coords_norm, coords_norm, 
                        split='train', n_subplots=3, new_save_path=None):
        """Plots the APN predictions on the given (no-)action pair."""
        
        # Denormalise & descale coords back to the original size
        pred_coords_denorm = self.denormalise(pred_coords_norm.detach())
        coords_denorm = self.denormalise(coords_norm.detach())
        
        pred_coords = self.descale_coords(pred_coords_denorm)
        coords = self.descale_coords(coords_denorm)

        # If there are outliers
        pad_left = abs(int(self.data_min))
        pad_right = abs(int(self.data_max)) - self.img_size
        
        plt.figure(1)        
        for i in range(n_subplots):
            plt.subplot(n_subplots, 2, 2*i+1)
            
            # Start state predictions and ground truth
            pred_pick_xy = pred_coords[i][:2] + abs(int(self.data_min))
            actual_pick_xy = coords[i][:2] + abs(int(self.data_min))
            state1_img = (img1[i].detach().cpu().numpy().transpose(1, 2, 0).copy() * 255).astype(np.uint8)
            state1_img_padded = cv2.copyMakeBorder(
                    state1_img, pad_left, pad_right, pad_left, pad_right, cv2.BORDER_CONSTANT)
            marked_img1 = cv2.circle(state1_img_padded, tuple(pred_pick_xy), 10, 
                                     (255, 0, 0), -1)
            marked_img1 = cv2.circle(marked_img1, tuple(actual_pick_xy), 15, 
                                     (0, 255, 0), 4)
            fig=plt.imshow(marked_img1)
            
            # Start state predicted height for the robot and ground truth
            pred_pick_height = round(pred_coords_denorm[i][2])
            actual_pick_height = round(coords_denorm[i][2])
            plt.title('State 1, \nh_pred {0}/h_true {1}'.format(
                    pred_pick_height, actual_pick_height))
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)

            # End state predictions and ground truth
            plt.subplot(n_subplots, 2, 2*i+2)
            pred_place_xy = pred_coords[i][3:] + abs(int(self.data_min))
            actual_place_xy = coords[i][3:] + abs(int(self.data_min))
            state2_img = (img2[i].detach().cpu().numpy().transpose(1, 2, 0).copy() * 255).astype(np.uint8)
            state2_img_padded = cv2.copyMakeBorder(
                    state2_img, pad_left, pad_right, pad_left, pad_right, cv2.BORDER_CONSTANT)
            marked_img2 = cv2.circle(state2_img_padded, tuple(pred_place_xy), 10, 
                                     (255, 0, 0), -1)
            marked_img2 = cv2.circle(marked_img2, tuple(actual_place_xy), 15, 
                                     (0, 255, 0), 4)
            fig = plt.imshow(marked_img2)
            plt.title('State 2')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
        
        if new_save_path: 
            if 'valid' in split:
                new_save_path += '_pick{0}_place{1}'.format(
                        str(actual_pick_xy), str(actual_place_xy))
            plt.savefig(new_save_path)
        else:
            plt.savefig(self.save_path + '_Predictions' + split + str(self.current_epoch))
        plt.clf()
        plt.close()
        plt.close('all')
        # cv2.destroyAllWindows()
        
        
    def train(self, train_dataset, test_dataset, num_workers=0, chpnt_path=''):  
        """Trains an APN model with given the hyperparameters."""    
        dataloader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True, 
                num_workers=num_workers, drop_last=False)
        n_data = len(train_dataset)
        self.data_min = train_dataset.min
        self.data_max = train_dataset.max
        self.img_size = train_dataset.img_size
        assert(train_dataset.img_size == test_dataset.img_size)
        
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
            self.valid_losses = []
            self.epoch_losses = []
            self.epoch_times = []
            self.training_time = 0
            print((' *- Learning rate: {0}\n' + 
                   ' *- Next lr update at {1} to the value {2}\n' + 
                   ' *- Remaining lr schedule: {3}'
                   ).format(self.lr, self.lr_update_epoch, self.new_lr, 
                   self.lr_schedule))            
        
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
            epoch_loss = np.zeros(7)
            epoch_start = datetime.now()
            
            for batch_idx, (img1, img2, coords) in enumerate(dataloader):
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                coords = coords.float().to(self.device) 
                
                # APNet loss
                pred_coords = self.model(img1, img2) 
                (the_loss, inputXloss, inputYloss, inputHloss, outputXloss, 
                 outputYloss) = self.compute_loss(pred_coords, coords) 

                epoch_loss += self.format_loss([
                        the_loss, inputXloss, inputYloss, 
                        inputHloss, outputXloss, outputYloss]) 
    
                # Optimise the model 
                self.optimiser.zero_grad()
                the_loss.backward()
                self.optimiser.step()                

                # Monitoring the learning 
                epoch_loss += self.format_loss([the_loss, inputXloss, inputYloss, 
                                                inputHloss, outputXloss, outputYloss]) 
             
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
                print('   Train loss: {0:.3f} inputX: {1:.3f} inputY: {2:.3f} inputH: {3:.3f} outputX: {4:.3f} outputY: {5:.3f}'.format(*epoch_loss))
                print('   Valid loss: {0:.3f} inputX: {1:.3f} inputY: {2:.3f} inputH: {3:.3f} outputX: {4:.3f} outputY: {5:.3f}'.format(*valid_loss))
                print('   LR: {0:.6e}\n'.format(self.lr))
                
            
            # Print validation results when specified
            if (self.current_epoch + 1) % self.snapshot == 0:
                
                # Plot APN predictions
                self.plot_prediction(img1, img2, pred_coords, coords)
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
        
        # Save the best performing model
        best_model_dict = {
                'model': self.best_model['model'].state_dict(), 
                'trained_params': {
                        'data_min': self.data_min,
                        'data_max': self.data_max,
                        'norm_mean': self.norm_mean,
                        'norm_std': self.norm_std
                        }
                }
        torch.save(best_model_dict, self.save_path + '_bestModelAll.pt') 
        
        # Save the last model        
        torch.save(self.model.state_dict(), self.save_path + '_lastModel.pt')
        last_model_dict = {
                'model': self.model.state_dict(),
                'trained_params': {
                        'data_min': self.data_min,
                        'data_max': self.data_max,
                        'norm_mean': self.norm_mean,
                        'norm_std': self.norm_std
                        }
                }
        torch.save(last_model_dict, self.save_path + '_lastModelAll.pt')
        self.save_logs(train_dataset, test_dataset)
        
        # Plot predetermined test images for a fair comparisson among models
        self.plot_test_images(test_dataset)
    
    
    def score_model(
            self, model_name, path_to_valid_dataset, path_to_result_file, 
            random_seed, load_checkpoint=False, path_to_chpnt='', suffix=''):
        """Scores a trained model on the test set."""
        
        # Load the data 
        path = path_to_valid_dataset + '.pkl'
        with open(path, 'rb') as f:
            valid_data_dict = pickle.load(f)
            threshold_min = valid_data_dict['min']
            threshold_max = valid_data_dict['max']
            valid_data = valid_data_dict['data']

        self.data_min = threshold_min
        self.data_max = threshold_max

        print(' *- Loaded data from: ', path_to_valid_dataset)
        print(' *- Loaded normalisation parameters from: ',  self.path_to_norm_param_d)
        print(' *- Chosen thresholds: ', threshold_min, threshold_max)
        
        # Load the trained vae network
        self.load_vae()
        
        # Load the trained apn network
        self.model = self.init_model()
        if load_checkpoint:
            checkpoint = torch.load(path_to_chpnt, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(' *- APN loaded from a checkpoint.')
        else:
            path_to_trained_APNmodel = './models/{0}/encoderapnet_model.pt'.format(model_name)
            self.model.load_state_dict(torch.load(path_to_trained_APNmodel, 
                                                  map_location=self.device))
        self.model.eval()
        assert(not self.vae.training)
        assert(not self.model.training)
        if not self.img_size:
            self.img_size = valid_data[0][0].shape[0]

        # Init the counters        
        n_points = 0
        pickX_reg_score, pickY_reg_score, height_reg_score = 0, 0, 0
        placeX_reg_score, placeY_reg_score, total_reg_score = 0, 0, 0
        picks_list = []
        places_list = []
        
        pred_coord_list = []
        actual_coord_list = []
        
        # Compute the error
        for img1, img2, coords_unnorm in valid_data:

            n_points += 1

            # Normalise the targets with the parameters of the training split
            norm_mean = torch.from_numpy(self.norm_mean.squeeze())
            norm_std = torch.from_numpy(self.norm_std.squeeze())
            coords_norm = (coords_unnorm - norm_mean)/norm_std
            coords = coords_norm.unsqueeze(0).float().to(self.device) # add BS 1

            # VAE forward pass
            img1 = img1.to(self.device).unsqueeze(0).float()
            img2 = img2.to(self.device).unsqueeze(0).float()        

            dec_mean1, _, _, _ = self.vae(img1)
            dec_mean2, _, _, _ = self.vae(img2)
            
            # Get the predictions from the APN
            pred_coords = self.model(dec_mean1, dec_mean2).detach() # (1, 5)
            
            # Compute the mse loss and log the resutls
            (the_loss, inputXloss, inputYloss, inputHloss, outputXloss, 
                 outputYloss) = self.compute_loss(pred_coords.float(), coords) 
            
            pickX_reg_score += inputXloss.item()
            pickY_reg_score += inputYloss.item()
            placeX_reg_score += outputXloss.item()
            placeY_reg_score += outputYloss.item()
            total_reg_score += the_loss.item()
            
             # --- Separately save cases with high error
            if (inputXloss.item() > 0.2 or inputYloss.item() > 0.2 or 
                outputXloss.item() > 0.2 or outputYloss.item() > 0.2):
                failure_dir = '{0}/best_model_failures{1}'.format(self.opt['exp_dir'], 
                               suffix)
                if (not os.path.isdir(failure_dir)):
                    os.makedirs(failure_dir)
                
                new_save_path = '{0}/{1}'.format(failure_dir, str(n_points))
                self.plot_prediction(
                        dec_mean1, dec_mean2, pred_coords, coords,
                        split='valid' + str(n_points), n_subplots=1, 
                        new_save_path=new_save_path)
                
                new_save_path_originals = '{0}/{1}originals'.format(
                        failure_dir, str(n_points))
                plt.figure(1)        
                plt.subplot(1, 2, 1)
                fig=plt.imshow(img1.squeeze().cpu().numpy().transpose(1, 2, 0))
                plt.title('Original state1 img')
                fig.axes.get_xaxis().set_visible(False)
                fig.axes.get_yaxis().set_visible(False)
        
                plt.subplot(1, 2, 2)
                fig=plt.imshow(img2.squeeze().cpu().numpy().transpose(1, 2, 0))
                plt.title('Original state2 img')
                fig.axes.get_xaxis().set_visible(False)
                fig.axes.get_yaxis().set_visible(False)
                
                plt.savefig(new_save_path_originals)
                plt.clf()
                plt.close()
            
            # --- Save some cases for visual inspection
            if n_points % 5 == 0:
                self.plot_prediction(
                        dec_mean1, dec_mean2, pred_coords, coords,
                        split='valid' + str(n_points), n_subplots=1)
            
            # --- Compare the distribution of predictions vs ground truth
            # Descale and denormalise the coordinates back to the original form
            pred_coords_denorm = self.denormalise(pred_coords.detach())
            coords_denorm = self.denormalise(coords.detach())
            
            pred_coords_descaled = self.descale_coords(pred_coords_denorm)
            coords_descaled = self.descale_coords(coords_denorm)
        
            # Append for histogram plots
            pred_coord_list.append(pred_coords_descaled.reshape(-1, 1))
            actual_coord_list.append(coords_descaled.reshape(-1, 1))
            
            pred_pick_xy = pred_coords_descaled[0][:2] + abs(int(self.data_min))
            actual_pick_xy = coords_descaled[0][:2] + abs(int(self.data_min))
            pred_place_xy = pred_coords_descaled[0][3:] + abs(int(self.data_min))
            actual_place_xy = coords_descaled[0][3:] + abs(int(self.data_min))
            
            picks_list.append([tuple(pred_pick_xy), tuple(actual_pick_xy)])
            places_list.append([tuple(pred_place_xy), tuple(actual_place_xy)])
            
#            pad_left = abs(int(self.data_min))
#            pad_right = abs(int(self.data_max)) - self.img_size
        
        # ---  PLOTS OF DISTRIBUTION OF THE PREDICTIONS 
        picks_image = np.zeros(shape=[500, 500, 3], dtype=np.uint8)
        places_image = np.zeros(shape=[500, 500, 3], dtype=np.uint8)
        all_image = np.zeros(shape=[500, 500, 3], dtype=np.uint8)
        
        picks_image = np.zeros(shape=[500, 500, 3], dtype=np.uint8)
        plt.figure(1)        
        for i in range(len(picks_list)):            
            picks_image = cv2.circle(picks_image, picks_list[i][0], 10, (255, 0, 0), -1)
            picks_image = cv2.circle(picks_image, picks_list[i][1], 15, (144, 238, 144), 4)
        fig=plt.imshow(picks_image)
        plt.savefig('./models/{0}/{1}'.format(model_name, 'All_valid_picks' + suffix))
        plt.clf()
        plt.close()
        
        picks_image = np.zeros(shape=[500, 500, 3], dtype=np.uint8)
        plt.figure(1)        
        for i in range(len(picks_list)):            
            picks_image = cv2.circle(picks_image, (picks_list[i][0][0], 100), 10, (255, 0, 0), -1)
            picks_image = cv2.circle(picks_image, (picks_list[i][1][0], 120), 15, (144, 238, 144), 4)
        fig=plt.imshow(picks_image)
        plt.savefig('./models/{0}/{1}'.format(model_name, 'All_valid_picks_x' + suffix))
        plt.clf()
        plt.close()
        
        picks_image = np.zeros(shape=[500, 500, 3], dtype=np.uint8)
        plt.figure(1)        
        for i in range(len(picks_list)):            
            picks_image = cv2.circle(picks_image, (100, picks_list[i][0][1]), 10, (255, 0, 0), -1)
            picks_image = cv2.circle(picks_image, (120, picks_list[i][1][1]), 15, (144, 238, 144), 4)
        fig=plt.imshow(picks_image)
        plt.savefig('./models/{0}/{1}'.format(model_name, 'All_valid_picks_y' + suffix))
        plt.clf()
        plt.close()
        
        places_image = np.zeros(shape=[500, 500, 3], dtype=np.uint8)
        plt.figure(2)        
        for i in range(len(places_list)):            
            places_image = cv2.circle(places_image, places_list[i][0], 10, (255, 0, 0), -1)
            places_image = cv2.circle(places_image, places_list[i][1], 15, (144, 238, 144), 4)
        fig=plt.imshow(places_image)
        plt.savefig('./models/{0}/{1}'.format(model_name, 'All_valid_places' + suffix))
        plt.clf()
        plt.close()
        
        places_image = np.zeros(shape=[500, 500, 3], dtype=np.uint8)
        plt.figure(2)        
        for i in range(len(picks_list)):            
            places_image = cv2.circle(places_image, (places_list[i][0][0], 100), 10, (255, 0, 0), -1)
            places_image = cv2.circle(places_image, (places_list[i][1][0], 120), 15, (144, 238, 144), 4)
        fig=plt.imshow(places_image)
        plt.savefig('./models/{0}/{1}'.format(model_name, 'All_valid_places_x' + suffix))
        plt.clf()
        plt.close()
        
        places_image = np.zeros(shape=[500, 500, 3], dtype=np.uint8)
        plt.figure(2)        
        for i in range(len(picks_list)):            
            places_image = cv2.circle(places_image, (100, places_list[i][0][1]), 10, (255, 0, 0), -1)
            places_image = cv2.circle(places_image, (120, places_list[i][1][1]), 15, (144, 238, 144), 4)
        fig=plt.imshow(places_image)
        plt.savefig('./models/{0}/{1}'.format(model_name, 'All_valid_places_y' + suffix))
        plt.clf()
        plt.close()
        
        plt.figure(3)      
        for i in range(len(places_list)):            
            all_image = cv2.circle(all_image, places_list[i][0], 10, (255, 0, 0), -1)
            all_image = cv2.circle(all_image, places_list[i][1], 15, (144, 238, 144), 4)
            all_image = cv2.circle(all_image, picks_list[i][0], 10, (255, 0, 0), -1)
            all_image = cv2.circle(all_image, picks_list[i][1], 15, (144, 238, 144), 4)
        fig=plt.imshow(all_image)
        plt.savefig('./models/{0}/{1}'.format(model_name, 'All_valid_everything' + suffix))
        plt.clf()
        plt.close()
        
        
        # Visualise the distribution        
        pred_data = np.concatenate(pred_coord_list, axis=1)
        actual_data = np.concatenate(actual_coord_list, axis=1) # (5, n_valid)

        plt.figure(1)        
        plt.hist(pred_data[0, :], bins=20, label='pred', color='red', alpha = 0.5)
        plt.hist(actual_data[0, :], bins=20, label='true', color='green', alpha = 0.5)
        plt.title('pickX')
        plt.show()
        plt.savefig('./models/{0}/{1}'.format(model_name, 'AllValidHist_picks_x' + suffix))
        plt.clf()
        plt.close()
        
        plt.figure(2)        
        plt.hist(pred_data[1, :], bins=20, label='pred', color='red', alpha = 0.5)
        plt.hist(actual_data[1, :], bins=20, label='true', color='green', alpha = 0.5)
        plt.title('pickY')
        plt.show()
        plt.savefig('./models/{0}/{1}'.format(model_name, 'AllValidHist_picks_y' + suffix))
        plt.clf()
        plt.close()        
        
        plt.figure(3)        
        plt.hist(pred_data[3, :], bins=20, label='pred', color='red', alpha = 0.5)
        plt.hist(actual_data[3, :], bins=20, label='true', color='green', alpha = 0.5)
        plt.title('placeX')
        plt.show()
        plt.savefig('./models/{0}/{1}'.format(model_name, 'AllValidHist_place_x' + suffix))
        plt.clf()
        plt.close()
        
        plt.figure(4)        
        plt.hist(pred_data[4, :], bins=20, label='pred', color='red', alpha = 0.5)
        plt.hist(actual_data[4, :], bins=20, label='true', color='green', alpha = 0.5)
        plt.title('placeY')
        plt.show()
        plt.savefig('./models/{0}/{1}'.format(model_name, 'AllValidHist_place_y' + suffix))
        plt.clf()
        plt.close()
        # ----- END OF DISTRIBUTION PLOTS 

                        
        results_d = {
                'model_name': model_name,
                'n_points': n_points, 
                'random_seed': self.random_seed,
                'pickXmse': round(pickX_reg_score/n_points, 2), 
                'pickYmse': round(pickY_reg_score/n_points, 2), 
                'heightmse': round(height_reg_score/n_points, 2), 
                'placeXmse': round(placeX_reg_score/n_points, 2),
                'placeYmse': round(placeY_reg_score/n_points, 2),
                'total_score_mse': round(total_reg_score/n_points, 2)
                }
        
        print('\nValidation scores:\n {0}\n'.format(results_d))
        import pandas as pd
        df = pd.DataFrame.from_dict([results_d])
        df.to_csv(path_to_result_file, header=None, index=False, mode='a')
        return results_d


    def score_model_rawHoldout(
            self, model_name, path_to_valid_dataset, path_to_result_file, 
            random_seed, load_checkpoint=False, path_to_chpnt='', suffix='', 
            load_best=False):
        """Scores a trained model on the test set."""
        
        # Load the data 
        path = path_to_valid_dataset + '.pkl'
        with open(path, 'rb') as f:
            valid_data_dict = pickle.load(f)
            threshold_min = valid_data_dict['min']
            threshold_max = valid_data_dict['max']
            valid_data = valid_data_dict['data']

        self.data_min = threshold_min
        self.data_max = threshold_max

        print(' *- Loaded data from: ', path_to_valid_dataset)
        print(' *- Loaded normalisation parameters from: ',  self.path_to_norm_param_d)
        print(' *- Chosen thresholds: ', threshold_min, threshold_max)
        
        # Load the trained vae network
        self.load_vae()
        
        # Load the trained apn network
        self.model = self.init_model()
        if load_checkpoint:
            checkpoint = torch.load(path_to_chpnt, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(' *- APN loaded from a checkpoint: ', path_to_chpnt)
        elif load_best:
            path_to_trained_APNmodel = './models/{0}/encoderapnet_bestModelAll.pt'.format(model_name)
            self.model.load_state_dict(torch.load(path_to_trained_APNmodel, 
                                                  map_location=self.device)['model'])
        else:
            path_to_trained_APNmodel = './models/{0}/encoderapnet_model.pt'.format(model_name)
            self.model.load_state_dict(torch.load(path_to_trained_APNmodel, 
                                                  map_location=self.device))
        print(' *- APN loaded from ', path_to_trained_APNmodel)
        self.model.eval()
        assert(not self.vae.training)
        assert(not self.model.training)
        if not self.img_size:
            self.img_size = valid_data[0][0].size(-1)
        
        # Init the counters        
        n_points = 0
        pickX_reg_score, pickY_reg_score, height_reg_score = 0, 0, 0
        placeX_reg_score, placeY_reg_score, total_reg_score = 0, 0, 0
        picks_list, places_list = [], []
        
        pred_coord_list = []
        actual_coord_list = []
        
        pickX_reg_score1, pickY_reg_score1, height_reg_score1 = 0, 0, 0
        placeX_reg_score1, placeY_reg_score1, total_reg_score1 = 0, 0, 0
        # Compute the error
        for img1, img2, coords_unnorm in valid_data:

            n_points += 1

            # Normalise the targets with the parameters of the training split
            norm_mean = torch.from_numpy(self.norm_mean.squeeze())
            norm_std = torch.from_numpy(self.norm_std.squeeze())
            coords_norm = (coords_unnorm - norm_mean)/norm_std
            coords = coords_norm.unsqueeze(0).float().to(self.device) # add BS 1

            # VAE forward pass
            img1 = img1.to(self.device).unsqueeze(0).float()
            img2 = img2.to(self.device).unsqueeze(0).float()        

            dec_mean1, _, _, _ = self.vae(img1)
            dec_mean2, _, _, _ = self.vae(img2)
            
            # Get the predictions from the APN
            pred_coords = self.model(dec_mean1, dec_mean2).detach() # (1, 5)
            pred_coords_denorm = self.denormalise(pred_coords.detach()) # denormalise the predictions to be in [0, 1]

            true_coords_unnorm = coords_unnorm.unsqueeze(0).float().to(self.device) # add BS 1

            print('\n Check pred_coords_denorm vs true_coords_unnorm')
            print(pred_coords_denorm.shape, true_coords_unnorm.shape)
            print(pred_coords_denorm.max(), true_coords_unnorm.max())
            print(pred_coords_denorm.min(), true_coords_unnorm.min())
            # Compute the mse loss and log the resutls
            (the_loss, inputXloss, inputYloss, inputHloss, outputXloss, 
                 outputYloss) = self.compute_loss(
                 torch.from_numpy(pred_coords_denorm).float().to(self.device), true_coords_unnorm) 
            print(pred_coords_denorm, true_coords_unnorm)

            pickX_reg_score += inputXloss.item()
            pickY_reg_score += inputYloss.item()
            placeX_reg_score += outputXloss.item()
            placeY_reg_score += outputYloss.item()
            height_reg_score += inputHloss.item()
            total_reg_score += the_loss.item()
            print('\n Check the scores')
            print(pickX_reg_score, pickY_reg_score, height_reg_score, placeX_reg_score, placeY_reg_score)
            print('\n')


            pred_coords_denorm_descaled = self.descale_coords(pred_coords_denorm)
            true_coords_unnorm_descaled = self.descale_coords(true_coords_unnorm.detach().cpu().numpy())
            print('\n Check pred_coords_denorm_descaled vs true_coords_unnorm_descaled')
            print(pred_coords_denorm_descaled.shape, true_coords_unnorm_descaled.shape)
            print(pred_coords_denorm_descaled.max(), true_coords_unnorm_descaled.max())
            print(pred_coords_denorm_descaled.min(), true_coords_unnorm_descaled.min())
            print(pred_coords_denorm_descaled, true_coords_unnorm_descaled)
            print('\n')


            (the_loss1, inputXloss1, inputYloss1, inputHloss1, outputXloss1, 
                 outputYloss1) = self.compute_loss(
                 torch.from_numpy(pred_coords_denorm_descaled).float().to(self.device), 
                 torch.from_numpy(true_coords_unnorm_descaled).float().to(self.device))

            pickX_reg_score1 += inputXloss1.item()
            pickY_reg_score1 += inputYloss1.item()
            placeX_reg_score1 += outputXloss1.item()
            placeY_reg_score1 += outputYloss1.item()
            height_reg_score1 += inputHloss1.item()
            total_reg_score1 += the_loss1.item()
            print('\n Check the scores on pixels')
            print(pickX_reg_score1, pickY_reg_score1, placeX_reg_score1, placeY_reg_score1)
            print('\n')

            
            # --- Separately save cases with high error
            if (inputXloss.item() > 0.2 or inputYloss.item() > 0.2 or 
                outputXloss.item() > 0.2 or outputYloss.item() > 0.2):
                failure_dir = '{0}/best_model_failures{1}'.format(self.opt['exp_dir'], 
                               suffix)
                if (not os.path.isdir(failure_dir)):
                    os.makedirs(failure_dir)
                
                new_save_path = '{0}/{1}'.format(failure_dir, str(n_points))
                self.plot_prediction(
                        dec_mean1, dec_mean2, pred_coords, coords,
                        split='valid' + str(n_points), n_subplots=1, 
                        new_save_path=new_save_path)
                
                new_save_path_originals = '{0}/{1}originals'.format(
                        failure_dir, str(n_points))
                plt.figure(1)        
                plt.subplot(1, 2, 1)
                fig=plt.imshow(img1.squeeze().cpu().numpy().transpose(1, 2, 0))
                plt.title('Original state1 img')
                fig.axes.get_xaxis().set_visible(False)
                fig.axes.get_yaxis().set_visible(False)
        
                plt.subplot(1, 2, 2)
                fig=plt.imshow(img2.squeeze().cpu().numpy().transpose(1, 2, 0))
                plt.title('Original state2 img')
                fig.axes.get_xaxis().set_visible(False)
                fig.axes.get_yaxis().set_visible(False)
                
                plt.savefig(new_save_path_originals)
                plt.clf()
                plt.close()
            
            # --- Save some cases for visual inspection
            if n_points % 5 == 0:
                self.plot_prediction(
                        dec_mean1, dec_mean2, pred_coords, coords,
                        split='valid' + str(n_points), n_subplots=1)
            
            # --- Compare the distribution of predictions vs ground truth
            # Descale and denormalise the coordinates back to the original form
            pred_coords_denorm = self.denormalise(pred_coords.detach())
            coords_denorm = self.denormalise(coords.detach())
            
            pred_coords_descaled = self.descale_coords(pred_coords_denorm)
            coords_descaled = self.descale_coords(coords_denorm)

            # Append for histogram plots
            pred_coord_list.append(pred_coords_descaled.reshape(-1, 1))
            actual_coord_list.append(coords_descaled.reshape(-1, 1))
            
            pred_pick_xy = pred_coords_descaled[0][:2] + abs(int(self.data_min))
            actual_pick_xy = coords_descaled[0][:2] + abs(int(self.data_min))
            pred_place_xy = pred_coords_descaled[0][3:] + abs(int(self.data_min))
            actual_place_xy = coords_descaled[0][3:] + abs(int(self.data_min))
            
            picks_list.append([tuple(pred_pick_xy), tuple(actual_pick_xy)])
            places_list.append([tuple(pred_place_xy), tuple(actual_place_xy)])
            
#            pad_left = abs(int(self.data_min))
#            pad_right = abs(int(self.data_max)) - self.img_size
        
        # ---  PLOTS OF DISTRIBUTION OF THE PREDICTIONS 
        picks_image = np.zeros(shape=[500, 500, 3], dtype=np.uint8)
        places_image = np.zeros(shape=[500, 500, 3], dtype=np.uint8)
        all_image = np.zeros(shape=[500, 500, 3], dtype=np.uint8)
        
        picks_image = np.zeros(shape=[500, 500, 3], dtype=np.uint8)
        plt.figure(1)        
        for i in range(len(picks_list)):            
            picks_image = cv2.circle(picks_image, picks_list[i][0], 10, (255, 0, 0), -1)
            picks_image = cv2.circle(picks_image, picks_list[i][1], 15, (144, 238, 144), 4)
        fig=plt.imshow(picks_image)
        plt.savefig('./models/{0}/{1}'.format(model_name, 'All_valid_picks' + suffix))
        plt.clf()
        plt.close()
        
        picks_image = np.zeros(shape=[500, 500, 3], dtype=np.uint8)
        plt.figure(1)        
        for i in range(len(picks_list)):            
            picks_image = cv2.circle(picks_image, (picks_list[i][0][0], 100), 10, (255, 0, 0), -1)
            picks_image = cv2.circle(picks_image, (picks_list[i][1][0], 120), 15, (144, 238, 144), 4)
        fig=plt.imshow(picks_image)
        plt.savefig('./models/{0}/{1}'.format(model_name, 'All_valid_picks_x' + suffix))
        plt.clf()
        plt.close()
        
        picks_image = np.zeros(shape=[500, 500, 3], dtype=np.uint8)
        plt.figure(1)        
        for i in range(len(picks_list)):            
            picks_image = cv2.circle(picks_image, (100, picks_list[i][0][1]), 10, (255, 0, 0), -1)
            picks_image = cv2.circle(picks_image, (120, picks_list[i][1][1]), 15, (144, 238, 144), 4)
        fig=plt.imshow(picks_image)
        plt.savefig('./models/{0}/{1}'.format(model_name, 'All_valid_picks_y' + suffix))
        plt.clf()
        plt.close()
        
        places_image = np.zeros(shape=[500, 500, 3], dtype=np.uint8)
        plt.figure(2)        
        for i in range(len(places_list)):            
            places_image = cv2.circle(places_image, places_list[i][0], 10, (255, 0, 0), -1)
            places_image = cv2.circle(places_image, places_list[i][1], 15, (144, 238, 144), 4)
        fig=plt.imshow(places_image)
        plt.savefig('./models/{0}/{1}'.format(model_name, 'All_valid_places' + suffix))
        plt.clf()
        plt.close()
        
        places_image = np.zeros(shape=[500, 500, 3], dtype=np.uint8)
        plt.figure(2)        
        for i in range(len(picks_list)):            
            places_image = cv2.circle(places_image, (places_list[i][0][0], 100), 10, (255, 0, 0), -1)
            places_image = cv2.circle(places_image, (places_list[i][1][0], 120), 15, (144, 238, 144), 4)
        fig=plt.imshow(places_image)
        plt.savefig('./models/{0}/{1}'.format(model_name, 'All_valid_places_x' + suffix))
        plt.clf()
        plt.close()
        
        places_image = np.zeros(shape=[500, 500, 3], dtype=np.uint8)
        plt.figure(2)        
        for i in range(len(picks_list)):            
            places_image = cv2.circle(places_image, (100, places_list[i][0][1]), 10, (255, 0, 0), -1)
            places_image = cv2.circle(places_image, (120, places_list[i][1][1]), 15, (144, 238, 144), 4)
        fig=plt.imshow(places_image)
        plt.savefig('./models/{0}/{1}'.format(model_name, 'All_valid_places_y' + suffix))
        plt.clf()
        plt.close()
        
        plt.figure(3)      
        for i in range(len(places_list)):            
            all_image = cv2.circle(all_image, places_list[i][0], 10, (255, 0, 0), -1)
            all_image = cv2.circle(all_image, places_list[i][1], 15, (144, 238, 144), 4)
            all_image = cv2.circle(all_image, picks_list[i][0], 10, (255, 0, 0), -1)
            all_image = cv2.circle(all_image, picks_list[i][1], 15, (144, 238, 144), 4)
        fig=plt.imshow(all_image)
        plt.savefig('./models/{0}/{1}'.format(model_name, 'All_valid_everything' + suffix))
        plt.clf()
        plt.close()
        
        
        # Visualise the distribution        
        pred_data = np.concatenate(pred_coord_list, axis=1)
        actual_data = np.concatenate(actual_coord_list, axis=1) # (5, n_valid)

        plt.figure(1)        
        plt.hist(pred_data[0, :], bins=20, label='pred', color='red', alpha = 0.5)
        plt.hist(actual_data[0, :], bins=20, label='true', color='green', alpha = 0.5)
        plt.title('pickX')
        plt.show()
        plt.savefig('./models/{0}/{1}'.format(model_name, 'AllValidHist_picks_x' + suffix))
        plt.clf()
        plt.close()
        
        plt.figure(2)        
        plt.hist(pred_data[1, :], bins=20, label='pred', color='red', alpha = 0.5)
        plt.hist(actual_data[1, :], bins=20, label='true', color='green', alpha = 0.5)
        plt.title('pickY')
        plt.show()
        plt.savefig('./models/{0}/{1}'.format(model_name, 'AllValidHist_picks_y' + suffix))
        plt.clf()
        plt.close()        
        
        plt.figure(3)        
        plt.hist(pred_data[3, :], bins=20, label='pred', color='red', alpha = 0.5)
        plt.hist(actual_data[3, :], bins=20, label='true', color='green', alpha = 0.5)
        plt.title('placeX')
        plt.show()
        plt.savefig('./models/{0}/{1}'.format(model_name, 'AllValidHist_place_x' + suffix))
        plt.clf()
        plt.close()
        
        plt.figure(4)        
        plt.hist(pred_data[4, :], bins=20, label='pred', color='red', alpha = 0.5)
        plt.hist(actual_data[4, :], bins=20, label='true', color='green', alpha = 0.5)
        plt.title('placeY')
        plt.show()
        plt.savefig('./models/{0}/{1}'.format(model_name, 'AllValidHist_place_y' + suffix))
        plt.clf()
        plt.close()
        # ----- END OF DISTRIBUTION PLOTS 
            
        pickX_reg_score /= n_points
        pickY_reg_score /= n_points
        height_reg_score /= n_points
        placeX_reg_score /= n_points
        placeY_reg_score /= n_points
        total_reg_score /= n_points

        pickX_reg_score1 /= n_points
        pickY_reg_score1 /= n_points
        height_reg_score1 /= n_points
        placeX_reg_score1 /= n_points
        placeY_reg_score1 /= n_points
        total_reg_score1 /= n_points
        print('\n\n\n', pickX_reg_score1, pickY_reg_score1, height_reg_score1, placeX_reg_score1, placeY_reg_score1)
            
        results_d = {
                'model_name': model_name,
                'n_points': n_points, 
                'random_seed': self.random_seed,
                'pickXmse': round(pickX_reg_score, 2), 
                'pickYmse': round(pickY_reg_score, 2), 
                'heightmse': round(height_reg_score, 2), 
                'placeXmse': round(placeX_reg_score, 2),
                'placeYmse': round(placeY_reg_score, 2),
                'total_score_mse': round(total_reg_score, 2)
                }
        results_d1 = {
                'model_name': model_name,
                'n_points': n_points, 
                'random_seed': self.random_seed,
                'pickXmse': round(pickX_reg_score1, 2), 
                'pickYmse': round(pickY_reg_score1, 2), 
                'heightmse': round(height_reg_score1, 2), 
                'placeXmse': round(placeX_reg_score1, 2),
                'placeYmse': round(placeY_reg_score1, 2),
                'total_score_mse': round(total_reg_score1, 2)
                }
        
        print('\nValidation scores:')
        print(results_d)
        print('\n')
        import pandas as pd
        df = pd.DataFrame.from_dict([results_d])
        df.to_csv(path_to_result_file, header=None, index=False, mode='a')
        df1 = pd.DataFrame.from_dict([results_d1])
        df1.to_csv(path_to_result_file, header=None, index=False, mode='a')
        return results_d