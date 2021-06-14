#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 20:07:02 2020

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
import os
from datetime import datetime

# ---
# ====================== Training functions ====================== #
# ---
class APN_stacking_dev(APN_Algorithm):
    def __init__(self, opt):
        super().__init__(opt)
            

    def descale_coords(self, x):
        """
        Descales the coordinates from [0, 1] interval back to the original 
        image size.
        """
        # print('\n\n IN descale_coords')
        # print('coords original: ', x, x.shape)
        # print('data min/max ', self.data_min, self.data_max)
        rescaled = x.cpu().numpy() * (self.data_max - self.data_min) + self.data_min
        # print('coords rescaled: ', rescaled, rescaled.shape)

        rounded_coords = np.around(rescaled).astype(int)
        # print('coords rounded: ', rounded_coords, rounded_coords.shape)

        # Filter out of the range coordinates because MSE can be out
        cropped_rounded_coords = np.maximum(self.data_min, np.minimum(rounded_coords, self.data_max))
        # print('coords cropped rounded: ', cropped_rounded_coords, cropped_rounded_coords.shape)

        # print(rounded_coords)
        # print(cropped_rounded_coords)
        assert(np.all(cropped_rounded_coords) >= self.data_min)
        assert(np.all(cropped_rounded_coords) <= self.data_max)
        return cropped_rounded_coords.astype(int)
    
    
    def get_box_center_from_x_y(self, array):
        """
        Returns the center coordinates corresponding to the box where the APN
        prediction x and y are pointing to. It assumes top left corner = (0,0), 
        x increasing positively downwards and y towards the right.
        """
        x, y = array[1], array[0]
        cx_vec = [55,115,185] #according to image coordinates (x positive towards right)
        cy_vec = [87,140,190] #according to image coordinates (y positive towards down)
        cx = cx_vec[y]
        if x == 2:
            cy = 195
        else:
            cy = cy_vec[x]
        return (cx,cy)


    def plot_prediction(self, img1, img2, pred_coords_scaled, coords_scaled, 
                            split='train', n_subplots=3, new_save_path=None):
        """Plots the APN predictions on the given (no-)action pair."""
        img1 = self.vae.decoder(img1)[0]
        img2 = self.vae.decoder(img2)[0]

        # Descale coords back to the original size
        # print('\n\n In plot_prediction')
        # print('pred_coords_scaled ', pred_coords_scaled.shape, pred_coords_scaled)
        pred_coords = self.descale_coords(pred_coords_scaled.detach())
        # print('pred_coords ', pred_coords.shape, pred_coords)

        # print('coords_scaled ', coords_scaled.shape, coords_scaled)        
        coords = self.descale_coords(coords_scaled)
        # print('coords ', coords.shape, coords)
        
        plt.figure(1)        
        for i in range(n_subplots):
            # Start state predictions and ground truth
            plt.subplot(n_subplots, 2, 2*i+1)


            pred_pick_xy = self.get_box_center_from_x_y(pred_coords[i][:2])
            actual_pick_xy = self.get_box_center_from_x_y(coords[i][:2])
            # print(pred_pick_xy, actual_pick_xy)

            state1_img = (img1[i].detach().cpu().numpy().transpose(1, 2, 0).copy() * 255).astype(np.uint8)
            marked_img1 = cv2.circle(state1_img, tuple(pred_pick_xy), 10, (255, 0, 0), -1)
            marked_img1 = cv2.circle(marked_img1, tuple(actual_pick_xy), 15, (0, 255, 0), 4)
            fig=plt.imshow(marked_img1)
            
            # Start state predicted height for the robot and ground truth
            pred_pick_height = round(pred_coords_scaled[i][2].detach().item())
            plt.title('State 1, h_pred {0}'.format(pred_pick_height))
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)

            # End state predictions and ground truth
            plt.subplot(n_subplots, 2, 2*i+2)            
            pred_place_xy = self.get_box_center_from_x_y(pred_coords[i][3:])
            actual_place_xy = self.get_box_center_from_x_y(coords[i][3:])
            
            state2_img = (img2[i].detach().cpu().numpy().transpose(1, 2, 0).copy() * 255).astype(np.uint8)
            marked_img2 = cv2.circle(state2_img, tuple(pred_place_xy), 10, (255, 0, 0), -1)
            marked_img2 = cv2.circle(marked_img2, tuple(actual_place_xy), 15, (0, 255, 0), 4)
            fig=plt.imshow(marked_img2)
            plt.title('State 2')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
        
        if new_save_path: 
            plt.savefig(new_save_path)
        else:
            plt.savefig(self.save_path + '_Predictions' + split + str(self.current_epoch))
        plt.clf()
        plt.close('all')
        # cv2.destroyAllWindows()    


    def train(self, train_dataset, test_dataset, num_workers=0, chpnt_path=''):  
        """Trains an APN model with given hyperparameters."""    
##        Debugging & Testing
#        import torch.utils.data as data
#        train_sampler = data.SubsetRandomSampler(
#            np.random.choice(list(range(len(train_dataset))), 
#                             100, replace=False)) 
#        TODO: shuffle, sampler
        
        dataloader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True, 
                num_workers=num_workers, drop_last=True)#, sampler=train_sampler)
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
        torch.save(self.model.state_dict(), self.save_path + '_lastModel.pt')
        self.save_logs(train_dataset, test_dataset)
        
        # Plot predetermined test images for a fair comparisson among models
        self.plot_test_images(test_dataset)
    
    
    def score_model_on_test_split(
            self, model_name, path_to_valid_dataset, path_to_result_file, 
            random_seed, load_checkpoint=False, path_to_chpnt='', noise=False):
        """Scores a trained model on the test set which is preprocessed."""
        
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        print(' *- Random seed set to ', random_seed)
        
        # Load the data        
        with open(path_to_valid_dataset, 'rb') as f:
            valid_data_dict = pickle.load(f)
            threshold_min = valid_data_dict['min']
            threshold_max = valid_data_dict['max']
            valid_data = valid_data_dict['data']
            print(' *- Loaded data from ', path_to_valid_dataset)
             
        self.data_min = threshold_min
        self.data_max = threshold_max
        
        # Load the trained vae network
        self.load_vae()
        
        # Load the trained apn network
        self.model = self.init_model()
        if load_checkpoint:
            path_to_trained_APNmodel = './models/{0}/apnet_model.pt'.format(model_name)
            checkpoint = torch.load(path_to_trained_APNmodel, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(' *- APN loaded from a checkpoint: ', path_to_trained_APNmodel)
        else:
            self.model.load_state_dict(torch.load(path_to_chpnt, 
                                                  map_location=self.device))
            print(' *- APN loaded: ', path_to_trained_APNmodel)
            
        self.model.eval()
        assert(not self.vae.training)
        assert(not self.model.training)
        if not self.img_size:
            self.img_size = valid_data[0][0].size(-1)

        # Init the counters        
        pred_coord_list = []
        actual_coord_list = []
        
        n_points = 0
        pick_score, place_score, total_score = 0, 0, 0
        pickX_score, pickY_score = 0, 0
        placeX_score, placeY_score = 0, 0
        coord_score = 0
        
        pickX_reg_score, pickY_reg_score = 0, 0
        placeX_reg_score, placeY_reg_score, total_reg_score = 0, 0, 0
        
        # Compute the error
        for img1, img2, coords in valid_data:
            n_points += 1
            
            # VAE forward pass
            img1 = img1.to(self.device).unsqueeze(0).float()
            img2 = img2.to(self.device).unsqueeze(0).float()
            enc_mean1, _ = self.vae.encoder(img1)
            enc_mean2, _ = self.vae.encoder(img2)
            
            # Get the predictions from the APN
            pred_coords = self.model(enc_mean1, enc_mean2)
            coords = coords.unsqueeze(0).to(self.device)

            pred_coords_np = self.descale_coords(pred_coords.detach()).squeeze()
            coords_np = self.descale_coords(coords).squeeze() # shape (5, )
            
            # Append for histogram plots
            pred_coord_list.append(pred_coords_np.reshape(-1, 1))
            actual_coord_list.append(coords_np.reshape(-1, 1))
            
            # Compute the mse loss and log the resutls
            (the_loss, inputXloss, inputYloss, inputHloss, outputXloss, 
                 outputYloss) = self.compute_loss(pred_coords.float(), coords.float()) 
            
            pickX_reg_score += inputXloss.item()
            pickY_reg_score += inputYloss.item()
            placeX_reg_score += outputXloss.item()
            placeY_reg_score += outputYloss.item()
            total_reg_score += the_loss.item()
            
            correct_pickX = 1 if pred_coords_np[0] == coords_np[0] else 0 
            correct_pickY = 1 if pred_coords_np[1] == coords_np[1] else 0 
            correct_placeX = 1 if pred_coords_np[3] == coords_np[3] else 0
            correct_placeY = 1 if pred_coords_np[4] == coords_np[4] else 0
            
            pickX_score += correct_pickX
            pickY_score += correct_pickY
            placeX_score += correct_placeX
            placeY_score += correct_placeY
            coord_score += (correct_pickX + correct_pickY + correct_placeX + correct_placeY)//4
            
            correct_pick = 1 if sum(abs(pred_coords_np[:2] - coords_np[:2])) == 0 else 0 
            correct_place = 1 if sum(abs(pred_coords_np[3:] - coords_np[3:])) == 0 else 0
            
            pick_score += correct_pick
            place_score += correct_place
            total_score += (correct_pick + correct_place)//2
            
            # --- Separately save cases with high error
            if correct_pick == 0 or correct_place == 0:
                failure_dir = '{0}/best_model_failures'.format(self.opt['exp_dir'])
                if (not os.path.isdir(failure_dir)):
                    os.makedirs(failure_dir)
                    
                new_save_path = '{0}/{1}_pick{2}_place{3}'.format(
                        failure_dir, str(n_points), str(correct_pick),
                        str(correct_place))
                self.plot_prediction(
                        enc_mean1, enc_mean2, pred_coords, coords,
                        split='valid' + str(n_points), n_subplots=1, 
                        new_save_path=new_save_path)
                
                new_save_path_originals = '{0}/{1}originals_pick{2}_place{3}'.format(
                        failure_dir, str(n_points), str(correct_pick),
                        str(correct_place))
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
            if n_points % 100 == 0:
                self.plot_prediction(
                        enc_mean1, enc_mean2, pred_coords, coords,
                        split='valid' + str(n_points), n_subplots=1)
        
        # --- Compare the distribution of predictions vs ground truth
        # Descale and denormalise the coordinates back to the original form
        pred_data = np.concatenate(pred_coord_list, axis=1)
        actual_data = np.concatenate(actual_coord_list, axis=1) # (5, n_valid)

        plt.figure(1)        
        plt.hist(pred_data[0, :], bins=20, label='pred', color='red', alpha = 0.5)
        plt.hist(actual_data[0, :], bins=20, label='true', color='green', alpha = 0.5)
        plt.title('pickX')
        plt.show()
        plt.savefig('./models/{0}/{1}'.format(model_name, 'All_valid_picks_x'))
        plt.clf()
        plt.close()
        
        plt.figure(2)        
        plt.hist(pred_data[1, :], bins=20, label='pred', color='red', alpha = 0.5)
        plt.hist(actual_data[1, :], bins=20, label='true', color='green', alpha = 0.5)
        plt.title('pickY')
        plt.show()
        plt.savefig('./models/{0}/{1}'.format(model_name, 'All_valid_picks_y'))
        plt.clf()
        plt.close()        
        
        plt.figure(3)        
        plt.hist(pred_data[3, :], bins=20, label='pred', color='red', alpha = 0.5)
        plt.hist(actual_data[3, :], bins=20, label='true', color='green', alpha = 0.5)
        plt.title('placeX')
        plt.show()
        plt.savefig('./models/{0}/{1}'.format(model_name, 'All_valid_place_x'))
        plt.clf()
        plt.close()
        
        plt.figure(4)        
        plt.hist(pred_data[4, :], bins=20, label='pred', color='red', alpha = 0.5)
        plt.hist(actual_data[4, :], bins=20, label='true', color='green', alpha = 0.5)
        plt.title('placeY')
        plt.show()
        plt.savefig('./models/{0}/{1}'.format(model_name, 'All_valid_place_y'))
        plt.clf()
        plt.close()
        
        # Normalise the errors
        pickX_reg_score /= n_points
        pickY_reg_score /= n_points
        placeX_reg_score /= n_points
        placeY_reg_score /= n_points
        total_reg_score /= n_points
        
        pickX_score /= n_points
        pickY_score /= n_points
        placeX_score /= n_points
        placeY_score /= n_points
        coord_score /= n_points
                
        results_d = {
                'model_name': model_name,
                'n_points': n_points, 
                'random_seed': self.random_seed,
                
                'pickX_avgdisterrror': round(pickX_reg_score, 2), 
                'pickY_avgdisterrror': round(pickY_reg_score, 2), 
                'placeX_avgdisterrror': round(placeX_reg_score, 2),
                'placeY_avgdisterrror': round(placeY_reg_score, 2),
                'total_avgdisterrror': round(total_reg_score, 2),
                
                'pickX_score': round(pickX_score, 2),
                'pickY_score': round(pickY_score, 2),
                'placeX_score': round(placeX_score, 2),
                'placeY_score': round(placeY_score, 2),
                'coord_score': round(coord_score, 2),
                
                'pick_score_per': round(pick_score/n_points, 2), 
                'place_score_per': round(place_score/n_points, 2),
                'total_score_per': round(total_score/n_points, 2),
                }
        
        print('\nValidation scores:')
        print(results_d)
        print('\n')
        import pandas as pd
        df = pd.DataFrame.from_dict([results_d])
        df.to_csv(path_to_result_file, header=None, index=False, mode='a')
        return results_d
    
    
    def score_model(
            self, model_name, path_to_valid_dataset, path_to_result_file, 
            load_checkpoint=False, path_to_chpnt='', suffix='', noise=False):
        """Scores a trained model on the test set."""
        
        # Load the data        
        with open(path_to_valid_dataset, 'rb') as f:
            valid_data = pickle.load(f)
            print(' *- Loaded data from ', path_to_valid_dataset)
            
        if type(valid_data) == dict:
            threshold_min = valid_data['min']
            threshold_max = valid_data['max']
            valid_data = valid_data['data']
        else:            
            threshold_min, threshold_max = 0., 2. 
        self.data_min = threshold_min
        self.data_max = threshold_max
        print(' *- Thresholds ', threshold_min, threshold_max)
        
        # Load the trained vae network
        self.load_vae()
        
        # Load the trained apn network
        self.model = self.init_model()
        if load_checkpoint:
            checkpoint = torch.load(path_to_chpnt, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(' *- APN loaded from a checkpoint.')
        else:
            path_to_trained_APNmodel = './models/{0}/apnet_model.pt'.format(model_name)
            self.model.load_state_dict(torch.load(path_to_trained_APNmodel, 
                                                  map_location=self.device))
            print(' *- APN loaded: ', path_to_trained_APNmodel)
        self.model.eval()
        assert(not self.vae.training)
        assert(not self.model.training)

        # Init the counters        
        pred_coord_list = []
        actual_coord_list = []
        
        n_points = 0
        pick_score, place_score, total_score = 0, 0, 0
        pickX_score, pickY_score = 0, 0
        placeX_score, placeY_score = 0, 0
        coord_score = 0
        
        pickX_reg_score, pickY_reg_score = 0, 0
        placeX_reg_score, placeY_reg_score, total_reg_score = 0, 0, 0

        # Compute the error
        for (img1, img2, coords) in valid_data:
            n_points += 1
            #(img1, img2, 1, input_uv, output_uv)
            # img1 = item[0]
            # img2 = item[1]
            # input_uv = item[3].reshape(-1, 1)
            # input_h = np.array(1).reshape(-1, 1)
            # output_uv = item[4].reshape(-1, 1)
            # coords_array = np.concatenate([input_uv, input_h, output_uv]).astype('float32')
            
            # # Scale the rest
            # coords_array_scaled = (coords_array - threshold_min)/(threshold_max - threshold_min)
            # coords_array_scaled[2] = 1.0 # just a dummy value for height
            # assert(np.all(coords_array_scaled) >= 0. and np.all(coords_array_scaled) <= 1.)
            
            # if noise:
            #     # Add noise to the coordinates
            #     tiny_noise = np.random.uniform(-0.1, 0.1, size=(5, 1))
            #     tiny_noise[2] = 0.
            #     noisy_coords_array_scaled = coords_array_scaled + tiny_noise
            #     new_noisy_normalised_coords = np.maximum(0., np.minimum(noisy_coords_array_scaled, 1.))
            #     coords = torch.from_numpy(new_noisy_normalised_coords).transpose(1, 0) # shape (1, 5)
            # else:
            #     coords = torch.from_numpy(coords_array_scaled).transpose(1, 0) # shape (1, 5)
            
            # VAE forward pass
            img1 = img1.to(self.device).unsqueeze(0).float() #[1, 3, 256, 256]
            img2 = img2.to(self.device).unsqueeze(0).float() #[1, 3, 256, 256]
            
            enc_mean1, _ = self.vae.encoder(img1) #[1, 12]
            enc_mean2, _ = self.vae.encoder(img2) #[1, 12]
            
            # Get the predictions from the APN
            pred_coords = self.model(enc_mean1, enc_mean2) #[1, 5], e.g. tensor([[1.1689, 0.7339, 1.0000, 1.8545, 1.1014]])
            
            # Compute the mse loss and log the resutls
            coords = coords.unsqueeze(0).to(self.device) #[1, 5], e.g. tensor([[0.5000, 1.0000, 1.0000, 1.0000, 1.0000]])

            (the_loss, inputXloss, inputYloss, inputHloss, outputXloss, 
                 outputYloss) = self.compute_loss(pred_coords.float(), coords.float()) 
            print('\n', the_loss, inputXloss, inputYloss, inputHloss, outputXloss, 
                 outputYloss)

            # print('Going into descale.')
            
            # print('pred_coords ', pred_coords, pred_coords.shape)
            # Descale coords back to [0, 1, 2]
            pred_coords_np = self.descale_coords(pred_coords.detach()).squeeze() # shape (5, )
            # print('pred_coords_np ', pred_coords_np, pred_coords_np.shape)

            # print('coords_np ', coords, coords.shape)
            coords_np = self.descale_coords(coords).squeeze() # shape (5, )
            # print('coords_np ', coords_np, coords_np.shape)
            # print('\n End descale;\n')
            
            
            
            # Append for histogram plots
            pred_coord_list.append(pred_coords_np.reshape(-1, 1))
            actual_coord_list.append(coords_np.reshape(-1, 1))
            
            pickX_reg_score += inputXloss.item()
            pickY_reg_score += inputYloss.item()
            placeX_reg_score += outputXloss.item()
            placeY_reg_score += outputYloss.item()
            total_reg_score += the_loss.item()
            
            
            # print(pred_coords_np, coords_np)
            correct_pickX = 1 if pred_coords_np[0] == coords_np[0] else 0 
            # print('\n', pred_coords_np[0] == coords_np[0], correct_pickX)

            correct_pickY = 1 if pred_coords_np[1] == coords_np[1] else 0
            # print('\n', pred_coords_np[1] == coords_np[1], correct_pickY) 

            correct_placeX = 1 if pred_coords_np[3] == coords_np[3] else 0
            # print('\n', pred_coords_np[3] == coords_np[3], correct_placeX) 

            correct_placeY = 1 if pred_coords_np[4] == coords_np[4] else 0
            # print('\n', pred_coords_np[4] == coords_np[4], correct_placeY) 
            # print('Checking correct moves \n', correct_pickX, correct_pickY, correct_placeX, correct_placeY)
            
            # print('\n', pickX_score, pickY_score, placeX_score, placeY_score)
            pickX_score += correct_pickX
            pickY_score += correct_pickY
            placeX_score += correct_placeX
            placeY_score += correct_placeY
            # print('\n', pickX_score, pickY_score, placeX_score, placeY_score)
            coord_score += (correct_pickX + correct_pickY + correct_placeX + correct_placeY)//4
            # print('\n', coord_score)
            
            # print('\n\n\n\n\n', pred_coords_np, '\n', coords_np)
            correct_pick = 1 if sum(abs(pred_coords_np[:2] - coords_np[:2])) == 0 else 0 
            # print('\n', correct_pick, sum(abs(pred_coords_np[:2] - coords_np[:2])))

            
            correct_place = 1 if sum(abs(pred_coords_np[3:] - coords_np[3:])) == 0 else 0
            # print('\n', correct_place, sum(abs(pred_coords_np[3:] - coords_np[3:])))

            # print(pick_score, place_score, total_score)
            pick_score += correct_pick
            place_score += correct_place
            total_score += (correct_pick + correct_place)//2
            # print(pick_score, place_score, total_score)
            

            # --- Separately save cases with high error
            if correct_pick == 0 or correct_place == 0:
                failure_dir = '{0}/best_model_failures'.format(self.opt['exp_dir'])
                if (not os.path.isdir(failure_dir)):
                    os.makedirs(failure_dir)
                    
                new_save_path = '{0}/{1}_pick{2}_place{3}'.format(
                        failure_dir, str(n_points), str(correct_pick),
                        str(correct_place))
                self.plot_prediction(
                        enc_mean1, enc_mean2, pred_coords, coords,
                        split='valid' + str(n_points), n_subplots=1, 
                        new_save_path=new_save_path)
                
                new_save_path_originals = '{0}/{1}originals_pick{2}_place{3}'.format(
                        failure_dir, str(n_points), str(correct_pick),
                        str(correct_place))
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
            
            # --- Plot some for visual inspection
            if n_points % 100 == 0:
                self.plot_prediction(
                        enc_mean1, enc_mean2, pred_coords, coords,
                        split='valid' + str(n_points), n_subplots=1)
        
        # --- Compare the distribution of predictions vs ground truth
        # Descale and denormalise the coordinates back to the original form      
        pred_data = np.concatenate(pred_coord_list, axis=1)
        actual_data = np.concatenate(actual_coord_list, axis=1) # (5, n_valid)

        plt.figure(1)        
        plt.hist(pred_data[0, :], bins=20, label='pred', color='red', alpha = 0.5)
        plt.hist(actual_data[0, :], bins=20, label='true', color='green', alpha = 0.5)
        plt.title('pickX')
        plt.show()
        plt.savefig('./models/{0}/{1}'.format(model_name, 'All_valid_picks_x' + suffix))
        plt.clf()
        plt.close()
        
        plt.figure(2)        
        plt.hist(pred_data[1, :], bins=20, label='pred', color='red', alpha = 0.5)
        plt.hist(actual_data[1, :], bins=20, label='true', color='green', alpha = 0.5)
        plt.title('pickY')
        plt.show()
        plt.savefig('./models/{0}/{1}'.format(model_name, 'All_valid_picks_y' + suffix))
        plt.clf()
        plt.close()        
        
        plt.figure(3)        
        plt.hist(pred_data[3, :], bins=20, label='pred', color='red', alpha = 0.5)
        plt.hist(actual_data[3, :], bins=20, label='true', color='green', alpha = 0.5)
        plt.title('placeX')
        plt.show()
        plt.savefig('./models/{0}/{1}'.format(model_name, 'All_valid_place_x' + suffix))
        plt.clf()
        plt.close()
        
        plt.figure(4)        
        plt.hist(pred_data[4, :], bins=20, label='pred', color='red', alpha = 0.5)
        plt.hist(actual_data[4, :], bins=20, label='true', color='green', alpha = 0.5)
        plt.title('placeY')
        plt.show()
        plt.savefig('./models/{0}/{1}'.format(model_name, 'All_valid_place_y' + suffix))
        plt.clf()
        plt.close()
        
        # Normalise the errors
        pickX_reg_score /= n_points
        pickY_reg_score /= n_points
        placeX_reg_score /= n_points
        placeY_reg_score /= n_points
        total_reg_score /= n_points
        
        pickX_score /= n_points
        pickY_score /= n_points
        placeX_score /= n_points
        placeY_score /= n_points
        coord_score /= n_points
                
        results_d = {
                'model_name': model_name + suffix,
                'n_points': n_points, 
                'random_seed': self.random_seed,
                
                'pickX_avgdisterrror': round(pickX_reg_score, 2), 
                'pickY_avgdisterrror': round(pickY_reg_score, 2), 
                'placeX_avgdisterrror': round(placeX_reg_score, 2),
                'placeY_avgdisterrror': round(placeY_reg_score, 2),
                'total_avgdisterrror': round(total_reg_score, 2),
                
                'pickX_score': round(pickX_score, 2),
                'pickY_score': round(pickY_score, 2),
                'placeX_score': round(placeX_score, 2),
                'placeY_score': round(placeY_score, 2),
                'coord_score': round(coord_score, 2),
                
                'pick_score_per': round(pick_score/n_points, 2), 
                'place_score_per': round(place_score/n_points, 2),
                'total_score_per': round(total_score/n_points, 2),
                }
        
        print('\nValidation scores:')
        print(results_d)
        print('\n')
        import pandas as pd
        df = pd.DataFrame.from_dict([results_d])
        df.to_csv(path_to_result_file, header=None, index=False, mode='a')
        return results_d

