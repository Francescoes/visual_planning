#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 09:31:39 2020

@author: petrapoklukar
"""
from importlib.machinery import SourceFileLoader
import os
import torch

class VAECheckParameters_VAE_Algorithm():
    def __init__(self, opt):
      # Save the whole config
      self.opt = opt

      # Training parameters
      self.batch_size = opt['batch_size']
      self.epochs = opt['epochs']
      self.current_epoch = None
      self.snapshot = opt['snapshot']
      self.console_print = opt['console_print']
      self.lr_schedule = opt['lr_schedule']
      self.init_lr_schedule = opt['lr_schedule']

      # Beta scheduling
      self.beta = opt['beta_min']
      self.beta_range = opt['beta_max'] - opt['beta_min'] + 1
      self.beta_steps = opt['beta_steps'] - 1
      self.beta_idx = 0
      

      # Gamma scheduling
      self.gamma_warmup = opt['gamma_warmup']
      self.gamma = 0 if self.gamma_warmup > 0 else opt['gamma_min']
      self.gamma_min = opt['gamma_min']
      self.gamma_idx = 0
      self.gamma_update_step = (opt['gamma_max'] - opt['gamma_min']) / opt['gamma_steps']
      self.gamma_update_epoch_step = (self.epochs - self.gamma_warmup - 1) / opt['gamma_steps']

      # Action loss parameters
      self.min_dist_samples = opt['min_dist_samples']
      self.weight_dist_loss = opt['weight_dist_loss']
      self.distance_type = opt['distance_type'] if 'distance_type' in opt.keys() else '2'
      self.batch_dist_dict = {}
      self.epoch_dist_dict = {}
      self.min_epochs = opt['min_epochs'] if 'min_epochs' in opt.keys() else 499
      self.max_epochs = opt['max_epochs'] if 'max_epochs' in opt.keys() else 499

      # Other parameters
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      self.opt['device'] = self.device
    
    
    def update_learning_rate(self):
        """Annealing schedule for the learning rate."""
        if self.current_epoch == self.lr_update_epoch:            
          self.lr = self.new_lr
          print('Epoch {0} update LR: {1}'.format(self.current_epoch, self.lr))
          try:
              self.lr_update_epoch, self.new_lr = self.lr_schedule.pop(0)
          except:
              print('\t *- LR schedule ended')
          print('\t *- LR schedule:', self.lr_schedule)
            
    def update_beta(self):
        """Annealing schedule for the KL term."""
        beta_current_step = (self.beta_idx + 1.0) / self.beta_steps
        epoch_to_update = beta_current_step * self.epochs
        if self.current_epoch > epoch_to_update and beta_current_step <= 1:
            self.beta = beta_current_step * self.beta_range
            self.beta_idx += 1
            print('Epoch {0} update beta: {1}, beta_idx {2}, beta_current_step {3}'.format(
                self.current_epoch, self.beta, self.beta_idx, beta_current_step))
    
    
    def update_gamma(self):
        """Annealing schedule for the distance term."""
        epoch_to_update = self.gamma_idx * self.gamma_update_epoch_step + self.gamma_warmup
        if (self.current_epoch + 1) > epoch_to_update:
            self.gamma = self.gamma_min + self.gamma_idx * self.gamma_update_step
            self.gamma_idx += 1
            print('Epoch {0} update gamma: {1}, gamma_idx {2}'.format(
                self.current_epoch, self.gamma, self.gamma_idx))
            
    def simulate_train(self):
      self.start_epoch, self.lr = self.lr_schedule.pop(0)
      try:
          self.lr_update_epoch, self.new_lr = self.lr_schedule.pop(0)
      except:
          self.lr_update_epoch, self.new_lr = self.start_epoch - 1, self.lr

      for self.current_epoch in range(self.start_epoch, self.epochs):
        self.update_beta()
        self.update_gamma()
        self.update_learning_rate()
  

class VAECheckParameters_VAE_Algorithm_extension_dev():
    def __init__(self, opt):
        # Save the whole config
        self.opt = opt

        # Training parameters
        self.batch_size = opt['batch_size']
        self.epochs = opt['epochs']
        self.current_epoch = None
        self.loss_fn = opt['loss_fn']
        self.snapshot = opt['snapshot']
        self.console_print = opt['console_print']
        self.lr_schedule = opt['lr_schedule']
        self.init_lr_schedule = opt['lr_schedule']


        # Beta scheduling
        self.beta = opt['beta_min']
        self.beta_range = opt['beta_max'] - opt['beta_min'] 
        self.beta_steps = opt['beta_steps']
        self.beta_idx = 0
        self.beta_max_epoch = opt['beta_max_epoch']

        # Gamma scheduling
        self.gamma_warmup = opt['gamma_warmup']
        self.gamma = 0 if self.gamma_warmup > 0 else opt['gamma_min']
        self.gamma_min = opt['gamma_min']
        self.gamma_idx = 0
        self.gamma_update_step = (opt['gamma_max'] - opt['gamma_min']) / opt['gamma_steps']
        self.gamma_update_epoch_step = (self.epochs - self.gamma_warmup - 1) / opt['gamma_steps']

        # Action loss parameters
        self.min_dist = opt['min_dist']
        self.min_dist_step = opt['min_dist_step']
        self.min_dist_epoch_update = opt['min_dist_epoch_update']
#        self.weight_dist_loss = opt['weight_dist_loss']
        self.distance_type = opt['distance_type'] if 'distance_type' in opt.keys() else '2'
        self.batch_dist_dict = {}
        self.epoch_dist_dict = {}
        self.min_epochs = opt['min_epochs'] if 'min_epochs' in opt.keys() else 499
        self.max_epochs = opt['max_epochs'] if 'max_epochs' in opt.keys() else 499

    def update_learning_rate(self):
        """Annealing schedule for the learning rate."""
        if self.current_epoch == self.lr_update_epoch:            
          self.lr = self.new_lr
          print('Epoch {0} update LR: {1}'.format(self.current_epoch, self.lr))
          try:
              self.lr_update_epoch, self.new_lr = self.lr_schedule.pop(0)
          except:
              print('\t *- LR schedule ended')
          print('\t *- LR schedule:', self.lr_schedule)


    def update_beta(self):
        """Annealing schedule for the KL term."""
        beta_current_step = (self.beta_idx + 1.0) / self.beta_steps
        epoch_to_update = beta_current_step * self.beta_max_epoch
        if self.current_epoch > epoch_to_update and beta_current_step <= 1:
            self.beta = beta_current_step * self.beta_range
            self.beta_idx += 1
            print('Epoch {0} update beta: {1}, beta_idx {2}, beta_current_step {3}'.format(
                self.current_epoch, self.beta, self.beta_idx, beta_current_step))


    def update_gamma(self):
        """Annealing schedule for the distance term."""
        epoch_to_update = self.gamma_idx * self.gamma_update_epoch_step + self.gamma_warmup
        if (self.current_epoch + 1) > epoch_to_update:
            self.gamma = self.gamma_min + self.gamma_idx * self.gamma_update_step
            self.gamma_idx += 1
            print('Epoch {0} update gamma: {1}, gamma_idx {2}'.format(
                self.current_epoch, self.gamma, self.gamma_idx))
            
    def simulate_train(self):
      self.start_epoch, self.lr = self.lr_schedule.pop(0)
      try:
          self.lr_update_epoch, self.new_lr = self.lr_schedule.pop(0)
      except:
          self.lr_update_epoch, self.new_lr = self.start_epoch - 1, self.lr

      for self.current_epoch in range(self.start_epoch, self.epochs):
        self.update_beta()
        self.update_gamma()
        self.update_learning_rate()
        

if __name__ == '__main__':
  exp_vae = 'UnityToy4BHardLight2500_TinyResNet_md5_wd40_ld32_s100_new_logs_L1'
  vae_config_file = os.path.join('../', 'configs', exp_vae + '.py')
  vae_config = SourceFileLoader(exp_vae, vae_config_file).load_module().config 
  c = VAECheckParameters_VAE_Algorithm(vae_config['vae_opt'])
  c.simulate_train()
    
#  exp_vae = 'UnityToy4BHardLight2500_TinyResNet_ld32_L1_b0t1_g115_mdu5'
#  vae_config_file = os.path.join('../', 'configs', exp_vae + '.py')
#  vae_config = SourceFileLoader(exp_vae, vae_config_file).load_module().config 
#  c = VAECheckParameters_VAE_Algorithm_extension_dev(vae_config['vae_opt'])
#  c.simulate_train()