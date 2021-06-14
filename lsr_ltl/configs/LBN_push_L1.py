#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 09:55:26 2020
@author: petrapoklukar
"""

batch_size = 64

config = {}

# set the parameters related to the training and testing set
data_train_opt = {}
data_train_opt['batch_size'] = batch_size
data_train_opt['dataset_name'] = 'label_VAE_push_v1'
data_train_opt['split'] = 'train'
data_train_opt['dtype'] = 'latent'
data_train_opt['img_size'] = 256
data_train_opt['task_name'] = 'push'

data_test_opt = {}
data_test_opt['batch_size'] = batch_size
data_test_opt['epoch_size'] = None
data_test_opt['dataset_name'] = 'label_VAE_push_v1'
data_test_opt['split'] = 'test'
data_test_opt['dtype'] = 'latent'
data_test_opt['img_size'] = 256
data_test_opt['task_name'] = 'push'

data_valid_opt = {}
data_valid_opt['path_to_data'] = './LTL_data/push/evaluation_push_labels.pkl'
data_valid_opt['path_to_result_file'] = './models/LBN_push_evaluation_results.txt'
data_valid_opt['split'] = 'valid'
data_valid_opt['dtype'] = 'original'
data_valid_opt['img_size'] = 256

data_original_opt = {}
data_original_opt['path_to_original_dataset'] = './LTL_data/push/'
data_original_opt['original_dataset_name'] = 'push_labels'
data_original_opt['path_to_original_train_data'] = './LTL_data/push/train_push_labels.pkl'
data_original_opt['path_to_original_test_data'] = './LTL_data/push/test_push_labels.pkl'
data_original_opt['generate_from_checkpoint'] = False
data_original_opt['n_samples'] = 2

config['data_train_opt'] = data_train_opt
config['data_test_opt'] = data_test_opt
config['data_valid_opt'] = data_valid_opt
config['data_original_opt'] = data_original_opt

model_opt = {
    'filename': 'lbnet',
    'vae_name': 'VAE_push_v1', # to decode samples with
    'model_module': 'LBN_MLP',
    'model_class': 'LBNet',

    'device': 'cpu',
    'shared_imgnet_dims': [64, 64],
    'separated_imgnet_dims': [64, 16],
    'shared_reprnet_dims': [16*2, 16, 2],
    'dropout': None,
    'weight_init': 'normal_init',

    'epochs': 200,
    'batch_size': batch_size,
    'lr_schedule':  [(0, 1e-2), (50, 1e-3), (150, 1e-4)],
    'snapshot': 20,
    'console_print': 5,
    'optim_type': 'Adam',
    'random_seed': 1201,
    'num_workers': 4,
}

config['model_opt'] = model_opt
config['algorithm_type'] = 'LBN_pushing'
